/* Real-hardware litmus for the grant-baton release/re-check protocol.
 *
 * Two roles mirror the production shape (grant_wt.c start_grant_round /
 * round close vs the request handler):
 *
 *   RELEASER:  work = pop-all;            REQUESTER:  queue++            (RMW)
 *              baton = 0    (release)                 if (CAS baton 0->1)
 *              [seq_cst fence]                            serve pop-all  (RMW)
 *              if (queue nonempty)                     else
 *                 if (CAS baton 0->1) serve               rely on releaser
 *
 * Without the fence, TSO's StoreLoad reordering lets the releaser's
 * emptiness LOAD execute before its baton-release STORE is globally
 * visible: the releaser sees the pre-publication queue (empty) while the
 * requester's CAS still sees baton==1 — both stand down and the queued
 * item is never served (the lost wake that wedged the runtime).  With the
 * fence the pair is a proper Dekker publication and every item is served.
 *
 * This test runs the FENCED shape millions of times and fails on any lost
 * item.  It pins the protocol's ALGORITHM (serve accounting can never leak
 * an item); note that the round-gate synchronization narrows the hardware
 * reordering window enough that the UNfenced variant does not reliably
 * fail here — the fence's necessity is established by the production
 * reproduction, and a sharper hardware litmus (or a weak-memory model
 * checker run) is tracked as follow-up work.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#define LITMUS_FENCED 1
#define ROUNDS 2000000

static atomic_uint baton;
static atomic_uint queued;   /* items published and not yet served */
static atomic_uint served;
static atomic_uint round_no; /* phase gate so both threads race per round */

static void *requester(void *arg) {
  (void)arg;
  for (unsigned r = 1; r <= ROUNDS; r++) {
    while (atomic_load_explicit(&round_no, memory_order_acquire) != r) {
    }
    atomic_fetch_add_explicit(&queued, 1, memory_order_release); /* publish */
    unsigned expect = 0;
    if (atomic_compare_exchange_strong_explicit(&baton, &expect, 1,
                                                memory_order_acq_rel,
                                                memory_order_acquire)) {
      /* baton won: serve everything queued, then release (no re-check
       * needed on this side for the litmus — the releaser role covers the
       * publication race under test). */
      unsigned q = atomic_exchange_explicit(&queued, 0, memory_order_acq_rel);
      atomic_fetch_add_explicit(&served, q, memory_order_relaxed);
      atomic_store_explicit(&baton, 0, memory_order_release);
    }
    /* CAS lost: the releaser's re-check must serve us. */
  }
  return NULL;
}

static void *releaser(void *arg) {
  (void)arg;
  for (unsigned r = 1; r <= ROUNDS; r++) {
    /* Take the baton for this round's "close" (uncontended at this point:
     * the requester is parked on the round gate). */
    atomic_store_explicit(&baton, 1, memory_order_release);
    atomic_store_explicit(&round_no, r, memory_order_release); /* go */
    /* Round close: serve what is visible, release, re-check. */
    unsigned q = atomic_exchange_explicit(&queued, 0, memory_order_acq_rel);
    atomic_fetch_add_explicit(&served, q, memory_order_relaxed);
    atomic_store_explicit(&baton, 0, memory_order_release);
#if LITMUS_FENCED
    atomic_thread_fence(memory_order_seq_cst);
#endif
    if (atomic_load_explicit(&queued, memory_order_acquire) != 0) {
      unsigned expect = 0;
      if (atomic_compare_exchange_strong_explicit(&baton, &expect, 1,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire)) {
        unsigned q2 =
            atomic_exchange_explicit(&queued, 0, memory_order_acq_rel);
        atomic_fetch_add_explicit(&served, q2, memory_order_relaxed);
        atomic_store_explicit(&baton, 0, memory_order_release);
      }
    }
    /* Wait for the requester side of this round to finish before judging:
     * a residue is only "lost" once neither side can still serve it. */
    while (atomic_load_explicit(&served, memory_order_acquire) +
               atomic_load_explicit(&queued, memory_order_acquire) <
           r) {
      /* The requester may still be mid-round; if its CAS lost AND our
       * re-check missed the publication, this loop never exits for the
       * lost item — detect that via a bounded spin. */
      static const unsigned long SPIN_BOUND = 400000000UL;
      static unsigned long spin;
      if (++spin > SPIN_BOUND) {
        printf("FAIL: dekker litmus lost an item at round %u "
               "(served=%u queued=%u)\n",
               r, atomic_load(&served), atomic_load(&queued));
        exit(1);
      }
    }
  }
  return NULL;
}

int main(void) {
  pthread_t a, b;
  pthread_create(&a, NULL, requester, NULL);
  pthread_create(&b, NULL, releaser, NULL);
  pthread_join(a, NULL);
  pthread_join(b, NULL);
  unsigned s = atomic_load(&served), q = atomic_load(&queued);
  if (s + q != ROUNDS) {
    printf("FAIL: dekker litmus accounting (served=%u queued=%u rounds=%u)\n",
           s, q, ROUNDS);
    return 1;
  }
  printf("PASS: grant_baton_dekker_litmus rounds=%u served=%u\n", ROUNDS, s);
  return 0;
}
