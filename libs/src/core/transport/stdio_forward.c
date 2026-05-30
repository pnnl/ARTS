/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
#include "arts/transport/stdio_forward.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "arts/system/print.h"

/* Per-stream forwarder state, one heap node per forwarded stream, pushed onto
 * a lock-free Treiber stack.  A launcher forwards both stdout and stderr for
 * every non-master rank, so the live forwarder count grows with the node
 * count.  Heap nodes (rather than a fixed array) keep each node's address
 * stable for the lifetime of its reader thread — the thread holds a pointer
 * to its own node — and impose no ceiling, so a rank's output is never
 * silently dropped once some fixed bound is exceeded.  Nodes are only ever
 * pushed (make_pipe) and drained exactly once at teardown (shutdown_all via
 * a single atomic exchange), so the stack is ABA-free without a lock; reader
 * threads touch their own node only, never the stack head. */
typedef struct forwarder_slot_s {
  struct forwarder_slot_s *next;
  int read_fd;
  pthread_t thread;
  FILE *sink;
} forwarder_slot_t;

static forwarder_slot_t *g_forwarders;

static void *forwarder_thread_main(void *arg) {
  forwarder_slot_t *f = (forwarder_slot_t *)arg;
  /* Use raw read(2)/write(2) instead of fdopen+fgets/fputs.  fdopen
   * registers the new FILE* in glibc's global stream chain; while the
   * forwarder is blocked in fgets→read it holds that FILE*'s internal
   * lock, so any worker thread on rank 0 (the master process) that
   * calls fflush(NULL) — e.g. graph500's FLUSH macro = fflush(0) —
   * deadlocks iterating the chain.  Children avoid this only because
   * execv() wipes glibc's stream chain; rank 0 owns the forwarder FILE*
   * itself and hangs.  Raw fd I/O keeps the pipes outside the FILE*
   * chain entirely. */
  int sink_fd = fileno(f->sink);
  if (sink_fd < 0) {
    close(f->read_fd);
    f->read_fd = -1;
    return NULL;
  }
  char buf[4096];
  for (;;) {
    ssize_t n = read(f->read_fd, buf, sizeof(buf));
    if (n <= 0) {
      if (n < 0 && errno == EINTR) {
        continue;
      }
      break; /* EOF or unrecoverable error */
    }
    /* Drain to sink_fd; loop in case of partial writes / EINTR. */
    size_t off = 0;
    while (off < (size_t)n) {
      ssize_t w = write(sink_fd, buf + off, (size_t)n - off);
      if (w < 0) {
        if (errno == EINTR) {
          continue;
        }
        goto done;
      }
      off += (size_t)w;
    }
  }
done:
  close(f->read_fd);
  f->read_fd = -1;
  return NULL;
}

int arts_stdio_forwarder_make_pipe(unsigned int rank, const char *stream_label,
                                   FILE *sink) {
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    ARTS_WARN("stdio_forward: pipe() failed for rank %u %s: %s", rank,
              stream_label, strerror(errno));
    return -1;
  }

  forwarder_slot_t *slot = (forwarder_slot_t *)malloc(sizeof(*slot));
  if (!slot) {
    ARTS_WARN("stdio_forward: out of memory for rank %u %s", rank,
              stream_label);
    close(pipefd[0]);
    close(pipefd[1]);
    return -1;
  }
  slot->next = NULL;
  slot->read_fd = pipefd[0];
  slot->sink = sink;

  /* Start the reader before publishing the node: the thread only touches its
   * own node (never the list), so it needs no list membership to run, and
   * keeping an unstarted node off the list means shutdown never joins a
   * thread that was never created. */
  int err = pthread_create(&slot->thread, NULL, forwarder_thread_main, slot);
  if (err != 0) {
    ARTS_WARN("stdio_forward: pthread_create failed for rank %u %s: %s", rank,
              stream_label, strerror(err));
    free(slot);
    close(pipefd[0]);
    close(pipefd[1]);
    return -1;
  }

  /* Treiber push: publish the fully-built node with a release CAS so a
   * concurrent drain that acquires it sees the node complete. */
  forwarder_slot_t *head = __atomic_load_n(&g_forwarders, __ATOMIC_RELAXED);
  do {
    slot->next = head;
  } while (!__atomic_compare_exchange_n(&g_forwarders, &head, slot, false,
                                        __ATOMIC_RELEASE, __ATOMIC_RELAXED));

  return pipefd[1];
}

void arts_stdio_forwarder_shutdown_all(void) {
  /* Detach the whole stack with one acquire exchange, then join + free: the
   * reader threads touch only their own node, and our caller guarantees all
   * children have exited (so every pipe is at EOF and every reader has left
   * its loop) before calling this.  Idempotent — a second call exchanges out
   * an already-NULL head and the drain loop runs zero times. */
  forwarder_slot_t *list = __atomic_exchange_n(&g_forwarders, NULL, __ATOMIC_ACQUIRE);

  while (list) {
    forwarder_slot_t *next = list->next;
    /* EOF arrives when the child's write-end closes (child process exit or
     * explicit close); our caller guarantees that happened before calling
     * this function, so the reader has already left its read loop. */
    (void)pthread_join(list->thread, NULL);
    if (list->read_fd >= 0) {
      close(list->read_fd);
    }
    free(list);
    list = next;
  }
}
