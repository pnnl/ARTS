/* SPDX-License-Identifier: Apache-2.0
 *
 * T201 — raw-fd anti-deadlock (the load-bearing design reason for the module).
 *
 * forwarder_thread_main deliberately uses raw read(2)/write(2) instead of
 * fdopen()+fgets/fputs.  fdopen would register the forwarder fd as a FILE* in
 * glibc's global stream chain; while the reader blocks in fgets->read it would
 * hold that FILE*'s internal lock.  Any thread on the master then calling
 * fflush(NULL) (which iterates the whole stream chain taking each FILE* lock)
 * would deadlock against the blocked reader.  Raw fd I/O keeps the forwarder's
 * pipe OUT of the FILE* chain, so fflush(NULL) cannot block on it.
 *
 * This test pins that invariant: a forwarder is parked in read(2) on a pipe
 * whose write-end is held open with NO data, while another thread spins on
 * fflush(NULL) + buffered fprintf to a heap FILE*.  If the forwarder ever
 * entered the stream chain (a regression to fdopen/fgets), fflush(NULL) would
 * hang.  A watchdog alarm bounds the test: completion within the budget == no
 * deadlock; the alarm firing == FAIL.
 *
 * #include's stdio_forward.c only for cluster symmetry (public API used here).
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define ARTS_SYSTEM_PRINT_H
#define ARTS_WARN(...) ((void)0)
#include "../../libs/src/core/transport/stdio_forward.c"

#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { FLUSH_ITERS = 200000 };

static _Atomic int g_flusher_done;

static void watchdog(int sig) {
  (void)sig;
  /* async-signal-safe: write directly, then _exit nonzero. */
  static const char msg[] =
      "FAIL fflush_no_deadlock: WATCHDOG fired — fflush(NULL) blocked while a "
      "forwarder was parked in read (forwarder fd entered the FILE* chain)\n";
  (void)write(STDERR_FILENO, msg, sizeof(msg) - 1);
  _exit(2);
}

static void *flusher(void *arg) {
  (void)arg;
  /* A heap FILE* registered in glibc's stream chain, with buffered output, so
   * fflush(NULL) has a non-trivial chain to iterate and a buffer to drain. */
  char *backing = NULL;
  size_t backing_sz = 0;
  FILE *mem = open_memstream(&backing, &backing_sz);
  if (!mem) {
    return NULL;
  }
  for (int i = 0; i < FLUSH_ITERS; i++) {
    fprintf(mem, "tick %d\n", i); /* buffered write into the chain */
    if (fflush(NULL) != 0) {      /* iterate+lock the whole stream chain */
      break;
    }
  }
  fclose(mem);
  free(backing);
  atomic_store(&g_flusher_done, 1);
  return NULL;
}

int main(void) {
  /* Watchdog: if anything deadlocks, fail deterministically instead of hanging
   * the whole test suite. */
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = watchdog;
  sigaction(SIGALRM, &sa, NULL);
  alarm(20);

  FILE *sink = tmpfile();
  if (!sink) {
    (void)fprintf(stderr, "FAIL fflush_no_deadlock: tmpfile\n");
    return 1;
  }

  /* Park a forwarder in read(2): make_pipe spawns the reader; we keep the
   * write-end OPEN and write nothing, so the reader blocks in read forever
   * (until we close the write-end at teardown). */
  int wfd = arts_stdio_forwarder_make_pipe(0, "stdout", sink);
  if (wfd < 0) {
    (void)fprintf(stderr, "FAIL fflush_no_deadlock: make_pipe -1\n");
    return 1;
  }

  pthread_t fth;
  if (pthread_create(&fth, NULL, flusher, NULL) != 0) {
    (void)fprintf(stderr, "FAIL fflush_no_deadlock: flusher create\n");
    return 1;
  }

  /* The main thread also hammers fflush(NULL) to widen the window. */
  for (int i = 0; i < FLUSH_ITERS; i++) {
    if (fflush(NULL) != 0) {
      break;
    }
  }

  pthread_join(fth, NULL);

  if (!atomic_load(&g_flusher_done)) {
    (void)fprintf(stderr, "FAIL fflush_no_deadlock: flusher did not finish\n");
    return 1;
  }

  /* No deadlock occurred (we reached here before the alarm).  Now release the
   * parked reader: close the write-end so it sees EOF, then shutdown joins. */
  alarm(0);
  close(wfd);
  arts_stdio_forwarder_shutdown_all();
  fclose(sink);

  printf("PASS stdio_forward_fflush_no_deadlock: forwarder parked in read, "
         "%d x fflush(NULL) on 2 threads, no deadlock\n",
         FLUSH_ITERS);
  return 0;
}
