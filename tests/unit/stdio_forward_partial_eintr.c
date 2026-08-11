/* SPDX-License-Identifier: Apache-2.0
 *
 * T195 — stdio_forward inner write-loop: partial write + EINTR retry.
 *
 * forwarder_thread_main's inner loop must drain ALL n bytes of every read
 * chunk to the sink fd, retrying on partial writes (w < n-off) and on EINTR.
 * This test forces both conditions simultaneously:
 *
 *   - Back-pressure / partial writes: the sink is the write-end of a pipe whose
 *     read-end is drained SLOWLY (small reads with brief sleeps) by a helper
 *     thread.  The pipe buffer fills, so the forwarder's write(2) returns short
 *     counts repeatedly — exercising the `off += w` partial-write loop.
 *   - EINTR: a SIGALRM timer fires repeatedly on the forwarder thread (signal
 *     unblocked only there), so a blocked read(2)/write(2) returns -1/EINTR;
 *     the loop must `continue` and lose no bytes.
 *
 * Correctness: every byte arrives at the drainer exactly once, in order
 * (single pipe, single forwarder => order IS guaranteed here, unlike the
 * multi-source case).  Mismatch => a real copy-loop defect (do not relax).
 *
 * #include's stdio_forward.c for the static reader-thread body.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define ARTS_SYSTEM_PRINT_H
#define ARTS_WARN(...) ((void)0)
#include "../../libs/src/core/transport/stdio_forward.c"

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

enum { PAYLOAD = 60013 }; /* > 4096, prime-ish; multiple read chunks */

static void alarm_handler(int sig) {
  (void)sig; /* no-op: just interrupt I/O */
}

static int g_sink_read_fd; /* read-end of the sink pipe, drained slowly */
static _Atomic int g_drain_done;
static unsigned char g_drained[PAYLOAD + 32];
static _Atomic size_t g_drained_n;

static void *drainer(void *arg) {
  (void)arg;
  for (;;) {
    unsigned char tmp[512];
    ssize_t r = read(g_sink_read_fd, tmp, sizeof(tmp));
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    if (r == 0) {
      break; /* EOF: forwarder closed its write-end (sink fd) -> done */
    }
    size_t cur = atomic_load(&g_drained_n);
    if (cur + (size_t)r <= sizeof(g_drained)) {
      memcpy(g_drained + cur, tmp, (size_t)r);
      atomic_store(&g_drained_n, cur + (size_t)r);
    }
    /* Slow the drain to keep the pipe full and force partial writes upstream.
     */
    struct timespec ts = {0, 200000}; /* 0.2 ms */
    nanosleep(&ts, NULL);
  }
  atomic_store(&g_drain_done, 1);
  return NULL;
}

int main(void) {
  /* Sink = write-end of a pipe; drainer owns the read-end. */
  int sinkpipe[2];
  if (pipe(sinkpipe) != 0) {
    (void)fprintf(stderr, "FAIL partial_eintr: sink pipe()\n");
    return 1;
  }
  g_sink_read_fd = sinkpipe[0];
  /* Shrink the pipe buffer so back-pressure hits quickly (best-effort). */
  (void)fcntl(sinkpipe[1], F_SETPIPE_SZ, 4096);

  FILE *sink = fdopen(sinkpipe[1], "w");
  if (!sink) {
    (void)fprintf(stderr, "FAIL partial_eintr: fdopen sink\n");
    return 1;
  }

  pthread_t dth;
  if (pthread_create(&dth, NULL, drainer, NULL) != 0) {
    (void)fprintf(stderr, "FAIL partial_eintr: drainer create\n");
    return 1;
  }

  /* Install SIGALRM (no SA_RESTART) so it interrupts I/O. */
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = alarm_handler;
  sigaction(SIGALRM, &sa, NULL);

  /* The forwarder thread is created inside make_pipe; the process-wide signal
   * disposition + an interval timer will deliver SIGALRM to whichever thread
   * is running.  Use a fast repeating real-time timer to pepper the copy loop
   * with EINTR while it is blocked in read/write. */
  int wfd = arts_stdio_forwarder_make_pipe(0, "stdout", sink);
  if (wfd < 0) {
    (void)fprintf(stderr, "FAIL partial_eintr: make_pipe -1\n");
    return 1;
  }

  struct itimerval it;
  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = 500;
  it.it_interval.tv_sec = 0;
  it.it_interval.tv_usec = 500; /* every 0.5 ms */
  setitimer(ITIMER_REAL, &it, NULL);

  unsigned char *src = (unsigned char *)malloc(PAYLOAD);
  if (!src) {
    (void)fprintf(stderr, "FAIL partial_eintr: malloc\n");
    return 1;
  }
  for (size_t i = 0; i < PAYLOAD; i++) {
    src[i] = (unsigned char)((i * 131u + 11u) & 0xFFu);
  }

  /* Feed the source pipe; this write may also be interrupted (we retry). */
  size_t off = 0;
  while (off < PAYLOAD) {
    ssize_t w = write(wfd, src + off, PAYLOAD - off);
    if (w < 0) {
      if (errno == EINTR) {
        continue;
      }
      (void)fprintf(stderr, "FAIL partial_eintr: feed write errno=%d\n", errno);
      return 1;
    }
    off += (size_t)w;
  }
  close(wfd); /* data-then-EOF for the forwarder */

  /* shutdown_all joins the forwarder thread; it returns once the forwarder hit
   * EOF on its read-end and drained everything to the sink. */
  arts_stdio_forwarder_shutdown_all();

  /* Forwarder is gone => its sink fd (sinkpipe[1], owned by `sink` FILE*) must
   * be closed so the drainer sees EOF.  fclose(sink) closes that fd. */
  fclose(sink);

  /* Stop the timer before joining the drainer. */
  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = 0;
  it.it_interval.tv_sec = 0;
  it.it_interval.tv_usec = 0;
  setitimer(ITIMER_REAL, &it, NULL);

  pthread_join(dth, NULL);

  size_t got = atomic_load(&g_drained_n);
  if (got != PAYLOAD) {
    (void)fprintf(stderr,
                  "FAIL partial_eintr: drained %zu bytes, expected %d "
                  "(byte loss in partial-write/EINTR loop)\n",
                  got, PAYLOAD);
    return 1;
  }
  if (memcmp(g_drained, src, PAYLOAD) != 0) {
    (void)fprintf(stderr,
                  "FAIL partial_eintr: byte mismatch / reordering in copy\n");
    return 1;
  }

  free(src);
  printf("PASS stdio_forward_partial_eintr: %d bytes in order under "
         "back-pressure + EINTR\n",
         PAYLOAD);
  return 0;
}
