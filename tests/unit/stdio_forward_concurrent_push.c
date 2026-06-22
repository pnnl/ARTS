/* SPDX-License-Identifier: Apache-2.0
 *
 * T199 — concurrent Treiber push onto g_forwarders (lost-push / ABA).
 *
 * make_pipe publishes its node with a release-CAS Treiber push.  The launcher
 * calls it single-threaded in practice, but the CAS must be correct under
 * contention.  K threads each push M nodes concurrently; afterwards the stack
 * must contain EXACTLY K*M distinct nodes — no push lost to a CAS race, no node
 * duplicated/cycled by ABA.  Then a single shutdown_all must join+free every
 * one of them without hanging.
 *
 * Verification:
 *   - every make_pipe returns a valid write-end fd (counted atomically),
 *   - walking g_forwarders before shutdown yields exactly K*M nodes, all with
 *     distinct addresses (a lost push -> fewer; ABA/cycle -> a repeat or an
 *     infinite walk, bounded by a guard),
 *   - shutdown_all then drains them all (head == NULL afterwards).
 *
 * Each thread keeps a private FILE* sink (tmpfile) and writes nothing, then the
 * write-ends are closed before shutdown so readers hit EOF and join cleanly.
 *
 * #include's stdio_forward.c for the static node type / stack head.
 */
#define ARTS_SYSTEM_PRINT_H
#define ARTS_WARN(...) ((void)0)
#include "../../libs/src/core/transport/stdio_forward.c"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

enum { K = 8, M = 64, TOTAL = K * M };

static atomic_int g_start;
static _Atomic int g_made;
static int g_wfds[TOTAL];
static _Atomic int g_wfd_idx;
static FILE *g_sinks[K];

static void *pusher(void *arg) {
  long id = (long)arg;
  FILE *sink = g_sinks[id];
  /* Spin-gate so all K threads hammer the CAS at once. */
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
    /* spin */
  }
  for (int i = 0; i < M; i++) {
    int wfd = arts_stdio_forwarder_make_pipe((unsigned int)id, "stdout", sink);
    if (wfd >= 0) {
      atomic_fetch_add_explicit(&g_made, 1, memory_order_relaxed);
      int slot = atomic_fetch_add_explicit(&g_wfd_idx, 1, memory_order_relaxed);
      g_wfds[slot] = wfd;
    } else {
      /* Resource exhaustion would corrupt the exact-count assertion; record a
       * sentinel so main can distinguish a real lost-push from an OOM/EMFILE.
       */
      atomic_fetch_sub_explicit(&g_made, 1000000, memory_order_relaxed);
    }
  }
  return NULL;
}

int main(void) {
  for (int i = 0; i < K; i++) {
    g_sinks[i] = tmpfile();
    if (!g_sinks[i]) {
      (void)fprintf(stderr, "FAIL concurrent_push: tmpfile %d\n", i);
      return 1;
    }
  }

  pthread_t th[K];
  for (long i = 0; i < K; i++) {
    if (pthread_create(&th[i], NULL, pusher, (void *)i) != 0) {
      (void)fprintf(stderr, "FAIL concurrent_push: pthread_create %ld\n", i);
      return 1;
    }
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);
  for (int i = 0; i < K; i++) {
    pthread_join(th[i], NULL);
  }

  int made = atomic_load_explicit(&g_made, memory_order_relaxed);
  if (made != TOTAL) {
    (void)fprintf(stderr,
                  "FAIL concurrent_push: made=%d, expected %d "
                  "(negative => a make_pipe hit resource exhaustion)\n",
                  made, TOTAL);
    return 1;
  }

  /* Walk the stack: count nodes + check for duplicate addresses (ABA/cycle).
   * Bound the walk so an accidental cycle terminates instead of hanging. */
  int count = 0;
  forwarder_slot_t *p = __atomic_load_n(&g_forwarders, __ATOMIC_ACQUIRE);
  /* Collect addresses to detect a cycle/duplicate (O(n^2) is fine for 512). */
  forwarder_slot_t *seen[TOTAL + 8];
  while (p) {
    if (count >= TOTAL + 4) {
      (void)fprintf(stderr,
                    "FAIL concurrent_push: stack walk exceeded %d nodes "
                    "(cycle / duplicated node from ABA)\n",
                    TOTAL);
      return 1;
    }
    for (int j = 0; j < count; j++) {
      if (seen[j] == p) {
        (void)fprintf(stderr,
                      "FAIL concurrent_push: duplicate node %p in stack "
                      "(ABA)\n",
                      (void *)p);
        return 1;
      }
    }
    seen[count] = p;
    count++;
    p = p->next;
  }
  if (count != TOTAL) {
    (void)fprintf(stderr,
                  "FAIL concurrent_push: stack has %d nodes, expected %d "
                  "(lost push under CAS contention)\n",
                  count, TOTAL);
    return 1;
  }

  /* Close all write-ends so readers EOF, then join+free everything. */
  int nfds = atomic_load_explicit(&g_wfd_idx, memory_order_relaxed);
  for (int i = 0; i < nfds; i++) {
    close(g_wfds[i]);
  }
  arts_stdio_forwarder_shutdown_all();

  if (__atomic_load_n(&g_forwarders, __ATOMIC_ACQUIRE) != NULL) {
    (void)fprintf(stderr, "FAIL concurrent_push: head not NULL after "
                          "shutdown\n");
    return 1;
  }

  for (int i = 0; i < K; i++) {
    fclose(g_sinks[i]);
  }
  printf("PASS stdio_forward_concurrent_push: %d threads x %d = %d nodes, "
         "exact, all joined\n",
         K, M, TOTAL);
  return 0;
}
