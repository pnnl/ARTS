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

/* Per-pipe forwarder state: owned by the module, kept in a fixed-size
 * array to avoid dynamic resize locking.  MAX_FORWARDERS = 2 streams
 * (stdout, stderr) * MAX_CHILDREN.  Node count is bounded by ARTS's
 * routing-table size; 64 is ample for foreseeable runs. */
#define ARTS_MAX_FORWARDERS 128

typedef struct {
  bool active;
  int read_fd;
  pthread_t thread;
  unsigned int rank;
  const char *stream_label;
  FILE *sink;
} forwarder_slot_t;

static forwarder_slot_t g_forwarders[ARTS_MAX_FORWARDERS];
static pthread_mutex_t g_forwarders_lock = PTHREAD_MUTEX_INITIALIZER;

static void *forwarder_thread_main(void *arg) {
  forwarder_slot_t *f = (forwarder_slot_t *)arg;
  FILE *in = fdopen(f->read_fd, "r");
  if (!in) {
    close(f->read_fd);
    f->read_fd = -1;
    return NULL;
  }
  char buf[4096];
  /* Pass child's lines through unchanged: remote stdout → master stdout,
   * remote stderr → master stderr.  Source-side arts_printf / ARTS_*
   * already include the rank in their own prefix, so no additional
   * tagging is done here.  Line-buffered fgets + single fputs keeps
   * concurrent threads from interleaving mid-line (glibc per-FILE
   * lock). */
  while (fgets(buf, sizeof(buf), in)) {
    fputs(buf, f->sink);
    fflush(f->sink);
  }
  fclose(in); /* closes read_fd */
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

  pthread_mutex_lock(&g_forwarders_lock);
  forwarder_slot_t *slot = NULL;
  for (int i = 0; i < ARTS_MAX_FORWARDERS; i++) {
    if (!g_forwarders[i].active) {
      slot = &g_forwarders[i];
      break;
    }
  }
  if (!slot) {
    pthread_mutex_unlock(&g_forwarders_lock);
    ARTS_WARN("stdio_forward: no free slot (max=%d) for rank %u %s",
              ARTS_MAX_FORWARDERS, rank, stream_label);
    close(pipefd[0]);
    close(pipefd[1]);
    return -1;
  }
  slot->active = true;
  slot->read_fd = pipefd[0];
  slot->rank = rank;
  slot->stream_label = stream_label;
  slot->sink = sink;

  int err = pthread_create(&slot->thread, NULL, forwarder_thread_main, slot);
  if (err != 0) {
    ARTS_WARN("stdio_forward: pthread_create failed for rank %u %s: %s", rank,
              stream_label, strerror(err));
    slot->active = false;
    slot->read_fd = -1;
    pthread_mutex_unlock(&g_forwarders_lock);
    close(pipefd[0]);
    close(pipefd[1]);
    return -1;
  }
  pthread_mutex_unlock(&g_forwarders_lock);

  return pipefd[1];
}

void arts_stdio_forwarder_shutdown_all(void) {
  pthread_mutex_lock(&g_forwarders_lock);
  for (int i = 0; i < ARTS_MAX_FORWARDERS; i++) {
    if (!g_forwarders[i].active) {
      continue;
    }
    pthread_t thread = g_forwarders[i].thread;
    pthread_mutex_unlock(&g_forwarders_lock);
    /* Join without holding the lock — thread is reading its own slot
     * (rank/stream_label/sink) which we don't touch; and the loop
     * variable `i` is stack-local so re-taking the lock after join is
     * safe.  EOF arrives when the child's write-end closes (child
     * process exit or explicit close); our caller guarantees that
     * happened before calling this function. */
    (void)pthread_join(thread, NULL);
    pthread_mutex_lock(&g_forwarders_lock);
    g_forwarders[i].active = false;
    g_forwarders[i].read_fd = -1;
  }
  pthread_mutex_unlock(&g_forwarders_lock);
}
