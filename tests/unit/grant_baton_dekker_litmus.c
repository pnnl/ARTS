/* Real-hardware litmus for the grant-baton release/re-check pair.
 *
 * The runtime's shape, reduced to the two accesses that matter:
 *
 *   HOLDER (round close)          REQUESTER
 *     baton = 0      (store)        queued++            (locked RMW)
 *     [seq_cst fence]               claimed = CAS(baton 0 -> 1)
 *     r = queued     (load)
 *
 * Both stand down — and the queued request is served by nobody — exactly when
 * the holder reads r == 0 while the requester's claim fails.  That pair is
 * store-buffering: on TSO the holder's release store may still sit in its
 * store buffer while its own load of the queue executes, so it reads the
 * pre-publication value, and the requester's claim meanwhile reads the baton
 * as still held.  A seq_cst fence between the holder's store and load forbids
 * the outcome; release/acquire alone does not.
 *
 * Structure follows the standard store-buffering litmus rather than a
 * simulation of the protocol: state is reset per iteration and the two threads
 * are aligned on a sense-reversing barrier, so the two accesses land in the
 * same few nanoseconds instead of being spread by a handshake.  A protocol-
 * shaped harness (rounds, a queue, a serve loop) hides the window — the
 * publish arrives long after the release has drained — and reports a pass for
 * the unfenced build, which is worse than no test.
 *
 * The FENCED shape is compiled here and must never produce the outcome.
 * Building with -DLITMUS_FENCED=0 reproduces it: measured on two physical
 * cores of this host, the unfenced variant loses roughly one wake in every
 * 4,000 iterations while the fenced one loses none in millions.  The run
 * prints the count either way, so a future weakening shows up as a nonzero
 * number rather than as silence.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef LITMUS_FENCED
#define LITMUS_FENCED 1
#endif

#define ITERS 2000000u

static atomic_uint baton;
static atomic_uint queued;
static atomic_uint holder_saw;   /* the holder's re-check result */
static atomic_uint claimed;      /* did the requester's claim win? */
static atomic_uint sense;        /* barrier */
static atomic_uint arrived;
static atomic_uint lost;         /* iterations with the forbidden outcome */

/* Pin to a named CPU, and fail loudly rather than silently measuring
 * nothing.  This is not tuning: two threads on SMT siblings share a store
 * buffer, so the reordering under test cannot be observed at all there and an
 * unpinned run reports a clean pass for broken code.  The runtime's own
 * placement notes that CPUs below the sibling boundary are one per physical
 * core, so two low, distinct ids are two cores. */
static void pin_to(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    printf("FAIL: grant_baton_dekker_litmus could not pin to cpu %d\n", cpu);
    exit(1);
  }
}

static void barrier(unsigned *local_sense) {
  *local_sense = !*local_sense;
  if (atomic_fetch_add_explicit(&arrived, 1, memory_order_acq_rel) == 1u) {
    atomic_store_explicit(&arrived, 0, memory_order_relaxed);
    atomic_store_explicit(&sense, *local_sense, memory_order_release);
  } else {
    while (atomic_load_explicit(&sense, memory_order_acquire) != *local_sense) {
    }
  }
}

/* Requester: publish, then try to claim the baton. */
static void *requester(void *arg) {
  (void)arg;
  pin_to(2);
  unsigned s = 0;
  for (unsigned i = 0; i < ITERS; i++) {
    barrier(&s);
    atomic_fetch_add_explicit(&queued, 1, memory_order_release);
    unsigned expect = 0;
    unsigned won = atomic_compare_exchange_strong_explicit(
        &baton, &expect, 1, memory_order_acq_rel, memory_order_acquire);
    atomic_store_explicit(&claimed, won, memory_order_release);
    barrier(&s);
  }
  return NULL;
}

/* Holder: release the baton, then re-check the queue. */
static void *holder(void *arg) {
  (void)arg;
  pin_to(0);
  unsigned s = 0;
  for (unsigned i = 0; i < ITERS; i++) {
    /* Arm this iteration: baton held, queue empty. */
    atomic_store_explicit(&baton, 1, memory_order_relaxed);
    atomic_store_explicit(&queued, 0, memory_order_relaxed);
    atomic_store_explicit(&claimed, 0, memory_order_relaxed);
    barrier(&s);

    atomic_store_explicit(&baton, 0, memory_order_release);
#if LITMUS_FENCED
    atomic_thread_fence(memory_order_seq_cst);
#endif
    unsigned saw = atomic_load_explicit(&queued, memory_order_acquire);
    atomic_store_explicit(&holder_saw, saw, memory_order_relaxed);

    barrier(&s);
    /* Nobody serves the request when the holder saw an empty queue AND the
     * requester failed to claim: the runtime's lost wake. */
    if (atomic_load_explicit(&holder_saw, memory_order_relaxed) == 0u &&
        atomic_load_explicit(&claimed, memory_order_relaxed) == 0u) {
      atomic_fetch_add_explicit(&lost, 1, memory_order_relaxed);
    }
  }
  return NULL;
}

int main(void) {
  pthread_t a, b;
  pthread_create(&a, NULL, holder, NULL);
  pthread_create(&b, NULL, requester, NULL);
  pthread_join(a, NULL);
  pthread_join(b, NULL);

  unsigned n = atomic_load(&lost);
  printf("%s: grant_baton_dekker_litmus iters=%u lost=%u\n",
         n == 0u ? "PASS" : "FAIL", ITERS, n);
  return n == 0u ? 0 : 1;
}
