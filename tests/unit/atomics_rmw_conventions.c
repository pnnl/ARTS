/* SPDX-License-Identifier: Apache-2.0
 *
 * T241 — single-thread pinning of the exact return-value conventions of the
 * arts_atomic_* RMW wrappers in libs/src/core/utils/atomics.c.
 *
 * The load-bearing contract is
 * that the *_add / *_sub / *_add_u64 / *_sub_u64 family return the NEW
 * (post-op) value, while fetch_add / fetch_sub / fetch_and / swap return the
 * OLD (pre-op) value, and cswap returns the PRIOR value (not a bool).  These
 * two conventions sit one keyword apart in the source and are trivially
 * confusable; a single-thread test pins them against constant inputs so any
 * future "fix" that flips a __sync_add_and_fetch into __sync_fetch_and_add (or
 * vice-versa) is caught.
 *
 * Pure-unit: links only against atomics.c + libc.  No ARTS runtime.
 */

#include "arts/utils/atomics.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static int g_fail;

#define CHECK_U(expr, want)                                                    \
  do {                                                                         \
    unsigned int got_ = (expr);                                                \
    unsigned int want_ = (want);                                               \
    if (got_ != want_) {                                                       \
      (void)fprintf(stderr,                                                    \
                    "FAIL atomics_rmw_conventions: %s == %u, expected %u\n",   \
                    #expr, got_, want_);                                       \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

#define CHECK_U64(expr, want)                                                  \
  do {                                                                         \
    uint64_t got_ = (expr);                                                    \
    uint64_t want_ = (want);                                                   \
    if (got_ != want_) {                                                       \
      (void)fprintf(stderr,                                                    \
                    "FAIL atomics_rmw_conventions: %s == %" PRIu64             \
                    ", expected %" PRIu64 "\n",                                \
                    #expr, got_, want_);                                       \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

int main(void) {
  /* ---- 32-bit: add returns NEW, fetch_add returns OLD. ---- */
  {
    unsigned int v = 10;
    CHECK_U(arts_atomic_add(&v, 5), 15);       /* returns post-add */
    CHECK_U(v, 15);                            /* and stored value updated */
    CHECK_U(arts_atomic_fetch_add(&v, 5), 15); /* returns pre-add (old) */
    CHECK_U(v, 20);
  }

  /* ---- 32-bit: sub returns NEW. ---- */
  {
    unsigned int v = 20;
    CHECK_U(arts_atomic_sub(&v, 7), 13); /* returns post-sub */
    CHECK_U(v, 13);
  }

  /* ---- 32-bit swap returns OLD. ---- */
  {
    unsigned int v = 0xABCD;
    CHECK_U(arts_atomic_swap(&v, 0x1234), 0xABCD); /* returns old */
    CHECK_U(v, 0x1234);
  }

  /* ---- 32-bit cswap returns PRIOR value, success and failure. ---- */
  {
    unsigned int v = 100;
    /* success: prior == expected, store happens */
    CHECK_U(arts_atomic_cswap(&v, 100, 200), 100);
    CHECK_U(v, 200);
    /* failure: prior != expected, no store, returns the actual prior */
    CHECK_U(arts_atomic_cswap(&v, 999, 7), 200);
    CHECK_U(v, 200);
  }

  /* ---- 32-bit read is a plain acquire load. ---- */
  {
    unsigned int v = 0xFEED;
    CHECK_U(arts_atomic_read(&v), 0xFEED);
  }

  /* ---- 64-bit: add_u64 returns NEW, fetch_add_u64 returns OLD. ---- */
  {
    uint64_t v = 1000;
    CHECK_U64(arts_atomic_add_u64(&v, 100), 1100); /* new */
    CHECK_U64(v, 1100);
    CHECK_U64(arts_atomic_fetch_add_u64(&v, 100), 1100); /* old */
    CHECK_U64(v, 1200);
  }

  /* ---- 64-bit: sub_u64 returns NEW, fetch_sub_u64 returns OLD. ---- */
  {
    uint64_t v = 1200;
    CHECK_U64(arts_atomic_sub_u64(&v, 200), 1000); /* new */
    CHECK_U64(v, 1000);
    CHECK_U64(arts_atomic_fetch_sub_u64(&v, 200), 1000); /* old */
    CHECK_U64(v, 800);
  }

  /* ---- 64-bit swap returns OLD. ---- */
  {
    uint64_t v = 0xDEADBEEFull;
    CHECK_U64(arts_atomic_swap_u64(&v, 0xC0FFEEull), 0xDEADBEEFull);
    CHECK_U64(v, 0xC0FFEEull);
  }

  /* ---- 64-bit cswap_u64 returns PRIOR, success and failure. ---- */
  {
    uint64_t v = 0xAAAAull;
    CHECK_U64(arts_atomic_cswap_u64(&v, 0xAAAAull, 0xBBBBull), 0xAAAAull);
    CHECK_U64(v, 0xBBBBull);
    CHECK_U64(arts_atomic_cswap_u64(&v, 0x0ull, 0x1ull), 0xBBBBull); /* fail */
    CHECK_U64(v, 0xBBBBull);
  }

  /* ---- 64-bit fetch_and returns OLD; AND is the real operation despite the
   * misleading parameter name "add_val" in the source. ---- */
  {
    uint64_t v = 0xF0F0F0F0ull;
    CHECK_U64(arts_atomic_fetch_and_u64(&v, 0xFFFF0000ull),
              0xF0F0F0F0ull);    /* returns old */
    CHECK_U64(v, 0xF0F00000ull); /* and AND-ed, not added */
  }

  /* ---- 64-bit read is a plain acquire load. ---- */
  {
    uint64_t v = 0x123456789Aull;
    CHECK_U64(arts_atomic_read_u64(&v), 0x123456789Aull);
  }

  if (g_fail) {
    return 1;
  }
  printf("PASS atomics_rmw_conventions: add/sub=new, fetch_*/swap/and=old, "
         "cswap=prior pinned\n");
  return 0;
}
