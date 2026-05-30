/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/
#include "arts/system/debug.h"

#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "arts.h"
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/runtime_state.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"

/* Pending EDT dump — invoked from signal handlers and on demand.  Walks
 * every per-worker route table plus the remote route table, prints every
 * EDT entry with depc_needed > 0 (i.e., still waiting on dependencies).
 * Not strictly async-signal-safe (uses fprintf), but we are tearing the
 * process down anyway and the diagnostic value outweighs the risk. */
void arts_dump_pending_edts(void) {
  fprintf(stderr, "\n[ARTS] === Pending EDT dump (rank %u) ===\n",
          arts_global_rank_id);
  unsigned int total_pending = 0;
  unsigned int total_zero = 0;
  unsigned int n_tables =
      arts_node_info.total_thread_count + ARTS_REMOTE_ROUTE_SHARDS;
  for (unsigned int t = 0; t < n_tables; t++) {
    arts_route_table_t *rt;
    if (t < arts_node_info.total_thread_count) {
      rt = arts_node_info.route_table[t];
    } else {
      rt = arts_node_info
               .remote_route_table[t - arts_node_info.total_thread_count];
    }
    if (!rt)
      continue;
    arts_route_table_iterator_t iter;
    arts_reset_route_table_iterator(&iter, rt);
    arts_route_item_t *item;
    while ((item = arts_route_table_iterate(&iter)) != NULL) {
      if (arts_guid_get_kind(item->key) != ARTS_GUID_EDT)
        continue;
      /* Debug dump: peek the cb (unsafe-by-design, sanity-filtered below). */
      arts_shared_ptr_t h = arts_atomic_shared_load(&item->value);
      if (!h)
        continue;
      void *data = arts_shared_get(h);
      arts_shared_release(&h);
      if (!data)
        continue;
      struct arts_edt_s *edt = (struct arts_edt_s *)data;
      unsigned int needed = edt->depc_needed;
      unsigned int depc = edt->depc;
      /* Sanity-filter stale entries: route_table slots persist after EDT
       * free, so atomic_load can return a pointer into freed memory. */
      if (edt->guid != item->key)
        continue;
      if (edt->guid == 0)
        continue;
      if (depc > 100)
        continue;
      if (needed > depc)
        continue;
      if (needed > 0) {
        fprintf(stderr,
                "  [PENDING] guid=%lu home=%u depc=%u depc_needed=%u "
                "epoch=%lu arts_id=%lu func=%p (table=%u)\n",
                (uint64_t)edt->guid,
                arts_guid_get_rank(edt->guid), edt->depc, needed,
                (uint64_t)edt->epoch_guid, edt->arts_id, (void *)edt->func_ptr,
                t);
        /* depv layout: [arts_edt_s header | paramv u64s | depv
         * arts_edt_dep_t[]] */
        arts_edt_dep_t *depv =
            (arts_edt_dep_t *)((char *)edt + sizeof(struct arts_edt_s) +
                               (edt->paramc * sizeof(uint64_t)));
        for (unsigned int s = 0; s < depc; s++) {
          unsigned int g_home = 0;
          unsigned int g_type = 0;
          if (depv[s].guid != 0) {
            g_home = arts_guid_get_rank(depv[s].guid);
            g_type = arts_guid_get_kind(depv[s].guid);
          }
          const char *state = "NULL_GUID";
          if (depv[s].guid != 0) {
            state = depv[s].ptr ? "satisfied" : "NOT_SATISFIED";
          }
          (void)fprintf(
              stderr,
              "    slot[%u] guid=%lu (home=%u type=%u) ptr=%p mode=%d %s\n", s,
              (uint64_t)depv[s].guid, g_home, g_type, depv[s].ptr,
              (int)depv[s].mode, state);
        }
        total_pending++;
      } else {
        total_zero++;
      }
    }
  }
  fprintf(
      stderr,
      "[ARTS] Total pending EDTs: %u (depc_needed>0), %u with deps satisfied\n",
      total_pending, total_zero);
  fflush(stderr);
}

// Async-signal-safe signal name lookup (strsignal() is NOT safe).
static const char *signal_name(int sig) {
  switch (sig) {
  case SIGSEGV:
    return "SIGSEGV";
  case SIGBUS:
    return "SIGBUS";
  case SIGFPE:
    return "SIGFPE";
  case SIGTERM:
    return "SIGTERM";
  case SIGINT:
    return "SIGINT";
  case SIGALRM:
    return "SIGALRM";
  case SIGHUP:
    return "SIGHUP";
  default:
    return "UNKNOWN";
  }
}

// Async-signal-safe: write an unsigned int as decimal digits to stderr.
static void write_uint(unsigned int val) {
  char buf[16];
  int pos = (int)sizeof(buf);
  if (val == 0) {
    buf[--pos] = '0';
  } else {
    while (val > 0) {
      buf[--pos] = (char)('0' + (val % 10));
      val /= 10;
    }
  }
  (void)write(STDERR_FILENO, buf + pos, (size_t)(sizeof(buf) - (size_t)pos));
}

static void write_backtrace(void) {
  void *frames[32];
  int depth = backtrace(frames, 32);
  backtrace_symbols_fd(frames, depth, STDERR_FILENO);
}

// Crash signals (SIGSEGV, SIGBUS, SIGFPE) — unrecoverable, re-raise for core.
static void arts_crash_signal_handler(int sig) {
  const char *pre = "\n[ARTS] Crashed: ";
  (void)write(STDERR_FILENO, pre, strlen(pre));
  const char *name = signal_name(sig);
  (void)write(STDERR_FILENO, name, strlen(name));
  const char *mid = " (rank ";
  (void)write(STDERR_FILENO, mid, strlen(mid));
  write_uint(arts_global_rank_id);
  const char *post = ") — stack trace:\n";
  (void)write(STDERR_FILENO, post, strlen(post));

  write_backtrace();

  struct sigaction sa;
  sa.sa_handler = SIG_DFL;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(sig, &sa, NULL);
  (void)raise(sig);
}

// Termination signals (SIGTERM, SIGINT, SIGALRM, SIGHUP) — graceful exit.
//
// IMPORTANT: only async-signal-safe operations here. The previous version
// called arts_dump_pending_edts() (fprintf + locks) which could deadlock
// against a worker that already held stdio/malloc locks, leaving the
// process hung instead of dying — observed as zombie ranks surviving
// SIGTERM. Diagnostic dumps are now opt-in via a flag the main thread
// inspects on its own (signal-safe) path.
static void arts_term_signal_handler(int sig) {
  const char *pre = "\n[ARTS] Killed by ";
  (void)write(STDERR_FILENO, pre, strlen(pre));
  const char *name = signal_name(sig);
  (void)write(STDERR_FILENO, name, strlen(name));
  const char *mid = " (rank ";
  (void)write(STDERR_FILENO, mid, strlen(mid));
  write_uint(arts_global_rank_id);
  const char *post = ")\n";
  (void)write(STDERR_FILENO, post, strlen(post));

  /* backtrace_symbols_fd is async-signal-safe per glibc; keep it. */
  write_backtrace();

  struct sigaction sa;
  sa.sa_handler = SIG_DFL;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(sig, &sa, NULL);
  (void)raise(sig);
}

/* Termination signals handled by the dedicated sigwait watcher thread
 * (see arts_install_signal_watcher_thread). Listed once so the install
 * routine and watcher loop agree on the set. */
static const int arts_term_sigs[] = {SIGTERM, SIGINT, SIGALRM, SIGHUP};
static const unsigned arts_term_sig_count =
    sizeof(arts_term_sigs) / sizeof(arts_term_sigs[0]);

void arts_install_signal_handlers(void) {
  // Crash handlers — backtrace + re-raise for core dump.
  struct sigaction crash_sa;
  crash_sa.sa_handler = arts_crash_signal_handler;
  sigemptyset(&crash_sa.sa_mask);
  crash_sa.sa_flags = 0;
  int crash_sigs[] = {SIGSEGV, SIGBUS, SIGFPE};
  for (int i = 0; i < 3; i++) {
    sigaction(crash_sigs[i], &crash_sa, NULL);
  }

  /* Termination signals are handled by the sigwait watcher thread (see
   * arts_install_signal_watcher_thread).  We still install
   * arts_term_signal_handler as a *fallback*: if a signal somehow
   * reaches a non-watcher thread (e.g. before the mask was inherited,
   * or after the watcher exits), we at least print a backtrace and
   * re-raise SIG_DFL to die cleanly. */
  struct sigaction term_sa;
  term_sa.sa_handler = arts_term_signal_handler;
  sigemptyset(&term_sa.sa_mask);
  term_sa.sa_flags = 0;
  for (unsigned i = 0; i < arts_term_sig_count; i++) {
    sigaction(arts_term_sigs[i], &term_sa, NULL);
  }

  // Ignore SIGPIPE — let socket writes fail with EPIPE instead of killing us.
  signal(SIGPIPE, SIG_IGN); // NOLINT(cert-err33-c)
}

/*--- Signal-watcher thread -------------------------------------------------*/

/* Dedicated thread that owns SIGTERM/SIGINT/SIGALRM/SIGHUP delivery via
 * sigwait().  Lets graceful-shutdown code paths (which may grab locks
 * and call fprintf) run in a normal thread context instead of inside an
 * async-unsafe signal handler.  Wakes worker/sender/receiver threads via
 * arts_enter_shutdown_state(true), which broadcasts the cluster-wide
 * SHUTDOWN_MSG and clears their alive flags.
 *
 * Wake-on-stop: arts_stop_signal_watcher_thread sets stop_flag and
 * sends SIGRTMIN to break out of sigwait. */
static pthread_t arts_signal_watcher;
static atomic_bool arts_signal_watcher_started = ATOMIC_VAR_INIT(false);
static atomic_bool arts_signal_watcher_stop = ATOMIC_VAR_INIT(false);

static void *arts_signal_watcher_main(void *arg) {
  (void)arg;

  sigset_t wait_set;
  sigemptyset(&wait_set);
  for (unsigned i = 0; i < arts_term_sig_count; i++) {
    sigaddset(&wait_set, arts_term_sigs[i]);
  }
  sigaddset(&wait_set, SIGRTMIN);

  unsigned hits = 0;
  while (1) {
    int sig = 0;
    int rc = sigwait(&wait_set, &sig);
    if (rc != 0) {
      /* sigwait should not normally fail; if it does, bail to avoid a
       * tight error loop. */
      (void)fprintf(stderr,
                    "[ARTS] signal watcher: sigwait failed (rc=%d), exiting\n",
                    rc);
      break;
    }
    if (atomic_load(&arts_signal_watcher_stop)) {
      break; /* normal shutdown poke */
    }
    if (sig == SIGRTMIN) {
      /* Spurious wake — re-check stop_flag at top of loop. */
      continue;
    }

    hits++;
    if (hits == 1) {
      (void)fprintf(stderr,
                    "\n[ARTS] signal watcher: caught %d on rank %u, "
                    "initiating graceful shutdown\n",
                    sig, arts_global_rank_id);
      /* Now safe to call non-async-signal-safe code: we are in a normal
       * thread context.  Broadcast SHUTDOWN_MSG to peers and stop our
       * own workers; main thread's arts_thread_main_join will return,
       * launcher cleanup will reap children, normal teardown runs. */
      arts_enter_shutdown_state(true);
      /* Stay in the loop: a second signal forces exit. */
    } else {
      (void)fprintf(stderr,
                    "\n[ARTS] signal watcher: second signal %d, forcing exit\n",
                    sig);
      _exit(128 + sig);
    }
  }
  return NULL;
}

void arts_install_signal_watcher_thread(void) {
  if (atomic_exchange(&arts_signal_watcher_started, true)) {
    return; /* already installed */
  }

  /* Block the term signals on the calling (main) thread so that every
   * subsequently spawned thread inherits the block.  Only the watcher
   * thread will receive them via sigwait. */
  sigset_t block_set;
  sigemptyset(&block_set);
  for (unsigned i = 0; i < arts_term_sig_count; i++) {
    sigaddset(&block_set, arts_term_sigs[i]);
  }
  sigaddset(&block_set, SIGRTMIN);
  pthread_sigmask(SIG_BLOCK, &block_set, NULL);

  if (pthread_create(&arts_signal_watcher, NULL, arts_signal_watcher_main,
                     NULL) != 0) {
    (void)fprintf(stderr, "[ARTS] signal watcher: pthread_create failed; "
                          "falling back to sigaction handlers\n");
    /* Unblock so the legacy sigaction handler can still fire. */
    pthread_sigmask(SIG_UNBLOCK, &block_set, NULL);
    atomic_store(&arts_signal_watcher_started, false);
  }
}

void arts_stop_signal_watcher_thread(void) {
  if (!atomic_load(&arts_signal_watcher_started)) {
    return;
  }
  if (atomic_exchange(&arts_signal_watcher_stop, true)) {
    return; /* already stopping */
  }
  /* Wake watcher out of sigwait.  pthread_kill targets only the watcher
   * thread, so other threads' masks (which also block SIGRTMIN) are
   * irrelevant. */
  pthread_kill(arts_signal_watcher, SIGRTMIN);
  pthread_join(arts_signal_watcher, NULL);
  atomic_store(&arts_signal_watcher_started, false);
}

#ifndef __APPLE__

#include <sys/prctl.h>
#include <sys/resource.h>

void arts_turn_on_core_dumps(void) {
  (void)prctl(PR_SET_DUMPABLE, 1);

  struct rlimit limit;
  limit.rlim_cur = RLIM_INFINITY;
  limit.rlim_max = RLIM_INFINITY;
  if (setrlimit(RLIMIT_CORE, &limit) != 0) {
    ARTS_INFO("Failed to force core dumps");
  }
}

#else

void arts_turn_on_core_dumps(void) {
  ARTS_INFO("Core dumps not supported on OS X.");
}

#endif
