/* SPDX-License-Identifier: Apache-2.0
 *
 * T197 — forwarder_thread_main sink_fd<0 early-out.
 *
 * If fileno(sink) is negative the reader cannot forward anywhere, so the thread
 * must immediately close its read_fd, set read_fd = -1, and return NULL — no
 * read loop, no leaked fd, clean thread exit.
 *
 * A FILE* opened with fmemopen() has NO backing file descriptor, so fileno()
 * returns -1 (EBADF) — exactly the degenerate sink the branch guards against.
 *
 * The test captures the node (it is the freshly-pushed g_forwarders head),
 * joins the reader thread itself (join is the happens-before edge for the
 * thread's read_fd = -1 store), and asserts:
 *   - read_fd == -1 (the early-out closed it), and
 *   - the original read_fd is no longer open (no fd leak).
 * It then tears the single node down the same way shutdown_all would, minus the
 * second join (the thread is already joined).
 *
 * #include's stdio_forward.c for the static node type / thread body / globals.
 */
#define ARTS_SYSTEM_PRINT_H
#define ARTS_WARN(...) ((void)0)
#include "../../libs/src/core/transport/stdio_forward.c"

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
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
  return n;
}

int main(void) {
  /* No-backing-fd sink => fileno() < 0. */
  static char membuf[64];
  FILE *sink = fmemopen(membuf, sizeof(membuf), "w");
  if (!sink) {
    (void)fprintf(stderr, "FAIL fileno_negative: fmemopen\n");
    return 1;
  }
  if (fileno(sink) >= 0) {
    (void)fprintf(stderr,
                  "FAIL fileno_negative: fmemopen sink has a real fd (%d); "
                  "cannot exercise the sink_fd<0 branch on this libc\n",
                  fileno(sink));
    return 1;
  }

  /* Baseline fd count taken with the sink already open and no forwarder pipe;
   * after the whole lifecycle the count must return to this. */
  int fds_before = count_open_fds();

  int wfd = arts_stdio_forwarder_make_pipe(0, "stdout", sink);
  if (wfd < 0) {
    (void)fprintf(stderr, "FAIL fileno_negative: make_pipe -1\n");
    return 1;
  }

  /* The just-pushed node is the stack head.  We only dereference its `thread`
   * (set before publish, never written by the reader) before joining; the
   * reader's read_fd store is read ONLY after the join happens-before edge. */
  forwarder_slot_t *node = __atomic_load_n(&g_forwarders, __ATOMIC_ACQUIRE);
  if (!node) {
    (void)fprintf(stderr, "FAIL fileno_negative: no node published\n");
    return 1;
  }
  pthread_t th = node->thread;

  /* Join the reader ourselves: it must have taken the early-out and exited
   * (it never blocks in read, since it returns before the loop). */
  (void)pthread_join(th, NULL);

  /* Now (post-join) it is safe to read the field the reader wrote. */
  if (node->read_fd != -1) {
    (void)fprintf(stderr,
                  "FAIL fileno_negative: read_fd=%d after early-out, "
                  "expected -1 (branch did not close/clear it)\n",
                  node->read_fd);
    return 1;
  }

  /* Tear down the single node ourselves (thread already joined; mirror
   * shutdown_all's free, skipping the now-illegal second join). */
  __atomic_store_n(&g_forwarders, NULL, __ATOMIC_RELEASE);
  if (node->read_fd >= 0) {
    close(node->read_fd);
  }
  free(node);

  close(wfd);

  /* Leak check: read-end closed by the early-out, write-end closed above,
   * node freed -> fd count is back to baseline. */
  int fds_after = count_open_fds();
  if (fds_after != fds_before) {
    (void)fprintf(stderr, "FAIL fileno_negative: fd leak: before=%d after=%d\n",
                  fds_before, fds_after);
    return 1;
  }

  fclose(sink);
  printf("PASS stdio_forward_fileno_negative: early-out closed read_fd, "
         "clean exit, no fd leak\n");
  return 0;
}
