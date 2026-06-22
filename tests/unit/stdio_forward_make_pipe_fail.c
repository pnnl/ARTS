/* SPDX-License-Identifier: Apache-2.0
 *
 * T196 — make_pipe failure path: -1 return + no fd/slot leak.
 *
 * arts_stdio_forwarder_make_pipe returns -1 on pipe()/malloc/pthread_create
 * failure, and on each failure branch it must clean up everything it had
 * allocated so far so the launcher's "output is never discarded, just falls
 * back to the inherited fd" contract holds without leaking resources.
 *
 * This drives the pipe() failure branch by lowering RLIMIT_NOFILE so no new fd
 * can be opened, then asserts:
 *   - make_pipe returns -1 (does not fabricate a fd), and
 *   - no file descriptor is leaked (open-fd count is identical before/after),
 *     which proves the pipe-fail branch left no half-open pipe, and
 *   - no node was pushed onto g_forwarders (the static stack head is unchanged
 *     -> shutdown_all then joins zero threads and does not hang/double-free).
 *
 * #include's stdio_forward.c for the static g_forwarders stack head.
 */
#define ARTS_SYSTEM_PRINT_H
#define ARTS_WARN(...) ((void)0)
#include "../../libs/src/core/transport/stdio_forward.c"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <unistd.h>

static int count_open_fds(void) {
  DIR *d = opendir("/proc/self/fd");
  if (!d) {
    return -1;
  }
  int n = 0;
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (e->d_name[0] != '.') {
      n++;
    }
  }
  closedir(d);
  return n; /* includes the DIR's own fd; consistent across both calls */
}

int main(void) {
  /* Sink must be a valid FILE* (we never reach the reader). */
  FILE *sink = tmpfile();
  if (!sink) {
    (void)fprintf(stderr, "FAIL make_pipe_fail: tmpfile\n");
    return 1;
  }

  forwarder_slot_t *head_before =
      __atomic_load_n(&g_forwarders, __ATOMIC_ACQUIRE);

  /* Lower the soft fd limit to the current usage so pipe() can open no new fd
   * -> EMFILE. */
  struct rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) != 0) {
    (void)fprintf(stderr, "FAIL make_pipe_fail: getrlimit\n");
    return 1;
  }
  struct rlimit saved = rl;

  int fds_before = count_open_fds();
  if (fds_before < 0) {
    (void)fprintf(stderr, "FAIL make_pipe_fail: count fds\n");
    return 1;
  }

  /* Set the soft cap to the number of fds currently open (no headroom). */
  rl.rlim_cur = (rlim_t)fds_before;
  if (setrlimit(RLIMIT_NOFILE, &rl) != 0) {
    (void)fprintf(stderr, "FAIL make_pipe_fail: setrlimit\n");
    return 1;
  }

  int rc = arts_stdio_forwarder_make_pipe(0, "stdout", sink);

  /* Restore the limit before any further allocation / reporting. */
  (void)setrlimit(RLIMIT_NOFILE, &saved);

  int fds_after = count_open_fds();

  forwarder_slot_t *head_after =
      __atomic_load_n(&g_forwarders, __ATOMIC_ACQUIRE);

  if (rc != -1) {
    (void)fprintf(
        stderr, "FAIL make_pipe_fail: expected -1 on pipe() failure, got %d\n",
        rc);
    if (rc >= 0) {
      close(rc);
    }
    return 1;
  }
  if (fds_after != fds_before) {
    (void)fprintf(stderr,
                  "FAIL make_pipe_fail: fd leak: before=%d after=%d "
                  "(failure branch left a pipe fd open)\n",
                  fds_before, fds_after);
    return 1;
  }
  if (head_after != head_before) {
    (void)fprintf(stderr,
                  "FAIL make_pipe_fail: a slot was pushed despite failure "
                  "(slot leak / phantom thread)\n");
    return 1;
  }

  /* shutdown_all must be a clean no-op (nothing pushed). */
  arts_stdio_forwarder_shutdown_all();

  fclose(sink);
  printf("PASS stdio_forward_make_pipe_fail: -1 returned, no fd/slot leak\n");
  return 0;
}
