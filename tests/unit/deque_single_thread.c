/* SPDX-License-Identifier: Apache-2.0
 *
 * T019 — single-thread functional + edge coverage of the Chase-Lev deque
 * (libs/src/core/utils/deque.c).  No concurrency: this pins the owner-side
 * contract that the race tests (T020-T022) build on.
 *
 * Properties exercised:
 *   1. LIFO via push_front / pop_front: items come back newest-first; the
 *      deque drains to empty (pop_front returns NULL) and stays empty.
 *   2. FIFO via push_front / pop_back: the steal end returns oldest-first
 *      (single-threaded, so pop_back's CAS always succeeds against a quiet
 *      top).  Drains to empty.
 *   3. Segment grow + doubling + modulo wrap: pushing past the initial
 *      capacity forces grow_circular_array (size*2); after several grows the
 *      whole sequence is still recovered in order, proving the [t,b) copy and
 *      i%size indexing across multiple segments is correct.  We start with a
 *      small deque so the index wraps within a segment too.
 *   4. Interleaved push/pop_front then push again reuses slots via modulo.
 *
 * Edge cases (suspected-bug pins — record, do NOT fix):
 *   - B054 (deque.c:109): arts_deque_full uses `size - 1` on unsigned size.
 *     For a size==1 deque, size-1 == 0, so full() == (bottom >= top), which is
 *     true immediately after init (bottom==top==1).  We pin that a size==1
 *     deque reports full at once, and that the FIRST push therefore grows the
 *     array (push_front's own `b >= a->size - 1 + t` trigger) — the size-1
 *     arithmetic is load-bearing and the documented near-underflow boundary.
 *     A size==0 deque would make push_front's `a->size - 1` wrap to UINT_MAX
 *     (never grows) and then `i % 0` is UB; we therefore do NOT call
 *     push/pop on a size==0 deque (that is the undefined path).  We instead
 *     pin the size==1 boundary which is the smallest DEFINED capacity and
 *     show its full()==true-at-init behavior.
 */

#include "arts/utils/deque.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Encode small integers as non-NULL pointers (NULL is the empty sentinel). */
static inline void *enc(uint64_t v) { return (void *)(uintptr_t)(v + 1); }
static inline uint64_t dec(void *p) { return (uint64_t)(uintptr_t)p - 1; }

static int fail(const char *msg) {
  (void)fprintf(stderr, "FAIL deque_single_thread: %s\n", msg);
  return 1;
}

int main(void) {
  /* ---- 1. LIFO: push_front N, pop_front returns reverse order. ---- */
  {
    struct arts_deque_s *d = arts_deque_new(8);
    const uint64_t N = 5;
    for (uint64_t i = 0; i < N; i++) {
      arts_deque_push_front(d, enc(i), 0);
    }
    for (uint64_t i = 0; i < N; i++) {
      void *o = arts_deque_pop_front(d);
      if (o == NULL)
        return fail("LIFO: premature empty");
      uint64_t got = dec(o);
      uint64_t want = N - 1 - i; /* newest first */
      if (got != want) {
        (void)fprintf(
            stderr, "LIFO: pos %" PRIu64 " got %" PRIu64 " want %" PRIu64 "\n",
            i, got, want);
        return fail("LIFO order wrong");
      }
    }
    if (arts_deque_pop_front(d) != NULL)
      return fail("LIFO: not empty");
    /* repeated pop on empty stays empty (bottom restored to top). */
    if (arts_deque_pop_front(d) != NULL)
      return fail("LIFO: empty unstable");
    arts_deque_delete(d);
  }

  /* ---- 2. FIFO: push_front N, pop_back returns insertion order. ---- */
  {
    struct arts_deque_s *d = arts_deque_new(8);
    const uint64_t N = 5;
    for (uint64_t i = 0; i < N; i++) {
      arts_deque_push_front(d, enc(i), 0);
    }
    for (uint64_t i = 0; i < N; i++) {
      void *o = arts_deque_pop_back(d);
      if (o == NULL)
        return fail("FIFO: premature empty");
      uint64_t got = dec(o);
      if (got != i) { /* oldest first */
        (void)fprintf(stderr, "FIFO: pos %" PRIu64 " got %" PRIu64 "\n", i,
                      got);
        return fail("FIFO order wrong");
      }
    }
    if (arts_deque_pop_back(d) != NULL)
      return fail("FIFO: not empty");
    arts_deque_delete(d);
  }

  /* ---- 3. Grow + wrap: push far past the initial capacity (forces several
   *        doublings), pop_back drains all in FIFO order. ---- */
  {
    const unsigned int init = 2;
    struct arts_deque_s *d = arts_deque_new(init);
    const uint64_t N = 1000; /* 2 -> 4 -> 8 -> ... many grows */
    for (uint64_t i = 0; i < N; i++) {
      arts_deque_push_front(d, enc(i), 0);
    }
    /* pop_back (FIFO) should return 0,1,2,...,N-1 — proves the grow copy of
     * the live window [t,b) and the i%size indexing across segments. */
    for (uint64_t i = 0; i < N; i++) {
      void *o = arts_deque_pop_back(d);
      if (o == NULL)
        return fail("grow: premature empty");
      if (dec(o) != i) {
        (void)fprintf(stderr, "grow: pos %" PRIu64 " got %" PRIu64 "\n", i,
                      dec(o));
        return fail("grow: value lost/reordered across segments");
      }
    }
    if (arts_deque_pop_back(d) != NULL)
      return fail("grow: not empty");
    arts_deque_delete(d);
  }

  /* ---- 3b. Grow then LIFO drain (pop_front across grown segments). ---- */
  {
    struct arts_deque_s *d = arts_deque_new(2);
    const uint64_t N = 500;
    for (uint64_t i = 0; i < N; i++) {
      arts_deque_push_front(d, enc(i), 0);
    }
    for (uint64_t i = 0; i < N; i++) {
      void *o = arts_deque_pop_front(d);
      if (o == NULL)
        return fail("grow-LIFO: premature empty");
      uint64_t want = N - 1 - i;
      if (dec(o) != want)
        return fail("grow-LIFO: order wrong");
    }
    if (arts_deque_pop_front(d) != NULL)
      return fail("grow-LIFO: not empty");
    arts_deque_delete(d);
  }

  /* ---- 4. Interleaved push/pop_front, then re-push: modulo slot reuse. ----
   */
  {
    struct arts_deque_s *d = arts_deque_new(4);
    arts_deque_push_front(d, enc(10), 0);
    arts_deque_push_front(d, enc(11), 0);
    if (dec(arts_deque_pop_front(d)) != 11)
      return fail("interleave a");
    arts_deque_push_front(d, enc(12), 0);
    if (dec(arts_deque_pop_front(d)) != 12)
      return fail("interleave b");
    if (dec(arts_deque_pop_front(d)) != 10)
      return fail("interleave c");
    if (arts_deque_pop_front(d) != NULL)
      return fail("interleave d");
    arts_deque_delete(d);
  }

  /* ---- 5. B054 boundary: size==1 deque.  size-1 == 0 so full()==(b>=t),
   *        true at init.  First push therefore must grow. ---- */
  {
    struct arts_deque_s *d = arts_deque_new(1);
    if (!arts_deque_full(d)) {
      return fail(
          "B054 pin: size==1 deque expected full() at init (size-1==0)");
    }
    /* push must still succeed (it grows internally on the same condition). */
    arts_deque_push_front(d, enc(7), 0);
    void *o = arts_deque_pop_front(d);
    if (o == NULL || dec(o) != 7)
      return fail("size==1: push/pop broken");
    arts_deque_delete(d);
  }

  printf("PASS deque_single_thread: LIFO/FIFO/grow/wrap/size==1 boundary OK\n");
  return 0;
}

/* ── libc-backed shims so the test links deque.c without the ARTS runtime ── */
void *arts_calloc_align(size_t nmemb, size_t size, size_t align) {
  void *p = NULL;
  size_t total = nmemb * size;
  if (align < sizeof(void *))
    align = sizeof(void *);
  /* round total up to a multiple of align (posix_memalign requirement). */
  size_t rem = total % align;
  if (rem)
    total += align - rem;
  if (posix_memalign(&p, align, total) != 0)
    return NULL;
  memset(p, 0, total);
  return p;
}
void arts_free(void *ptr) { free(ptr); }
uint64_t arts_atomic_cswap_u64(volatile uint64_t *destination, uint64_t old_val,
                               uint64_t swap_in) {
  __atomic_compare_exchange_n(destination, &old_val, swap_in, false,
                              __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return old_val; /* updated to the observed value on failure */
}
