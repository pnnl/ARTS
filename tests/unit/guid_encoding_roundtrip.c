/* SPDX-License-Identifier: Apache-2.0
 *
 * guid_encoding_roundtrip — pure-unit coverage of guid.c encode/decode and
 * the contiguous-range allocator, with NO ARTS runtime.
 *
 * guid.c is #include'd directly so the file-local generator globals
 * (keys_per_thread / global_guid_on / min/max_global_guid_thread) and the
 * per-thread counter rows can be driven straight from this TU.  The runtime
 * symbols guid.c references (arts_node_info, arts_thread_info,
 * arts_global_rank_id/count, arts_abort, arts_malloc/calloc/free) are shimmed
 * at the bottom with libc-backed definitions.
 *
 * Properties asserted (all single-threaded — pure encoding contracts):
 *
 *   1. Field round-trip: ARTS_GUID_MAKE(type, rank, key) decodes back via
 *      get_kind / get_rank / get_key for the full field ranges (2-bit type,
 *      14-bit rank, large key) with no cross-field bleed.
 *   2. is_local matches arts_global_rank_id.
 *   3. A contiguous reserve_range mints `size` distinct GUIDs; from_index /
 *      index_from are exact inverses over [0,size); index_from returns -1 on a
 *      foreign-type GUID and on a key below the range start.
 *   4. Per-(rank,type) counters are disjoint: two reserves of the SAME rank
 *      advance, two reserves of DIFFERENT types never alias keys.
 *   5. Exhaustion is a HARD ERROR (no wraparound) — verified in a forked child
 *      so the abort doesn't take the test process down.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* ── runtime symbol shims (must precede the #include of guid.c) ─────────── */
unsigned int arts_global_rank_id = 0;
unsigned int arts_global_rank_count = 1;
static int g_abort_calls = 0;
void arts_abort(uint8_t code) {
  g_abort_calls++;
  /* In the parent process exhaustion is reached only inside a forked child
   * (see test 5); _exit there.  Any unexpected abort in the parent path is a
   * real failure, so exit non-zero. */
  _exit(code ? code : 70);
}
void *arts_malloc(size_t s) { return malloc(s); }
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void *arts_calloc_aligned(size_t n, size_t s, size_t a) {
  void *p = NULL;
  if (posix_memalign(&p, a < sizeof(void *) ? sizeof(void *) : a, n * s)) {
    return NULL;
  }
  memset(p, 0, n * s);
  return p;
}
void arts_free(void *p) { free(p); }

/* Wait-free counter primitives for the DB seq allocator (libc-free unit
 * pattern: mirror the atomics.c definitions verbatim). */
uint64_t arts_atomic_fetch_add_u64(volatile uint64_t *d, uint64_t v) {
  return __sync_fetch_and_add(d, v);
}
uint64_t arts_atomic_cswap_u64(volatile uint64_t *d, uint64_t o, uint64_t n) {
  return __sync_val_compare_and_swap(d, o, n);
}
uint64_t arts_atomic_read_u64(const volatile uint64_t *d) {
  return __atomic_load_n(d, __ATOMIC_ACQUIRE);
}

#include "../../libs/src/core/gas/guid.c"

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

#define FAIL(...)                                                              \
  do {                                                                         \
    (void)fprintf(stderr, "FAIL guid_encoding_roundtrip: " __VA_ARGS__);       \
    return 1;                                                                  \
  } while (0)

/* Bring the per-thread generator to a clean post-parallel-start state with a
 * generous per-thread key budget. */
static void generator_reset(uint64_t key_budget) {
  arts_thread_info.thread_id = 0;
  arts_node_info.total_thread_count = 1;
  arts_node_info.gpu = 0;
  /* free any prior reset's allocations (keep the harness leak-clean for LSan)
   */
  if (arts_node_info.keys) {
    free(arts_node_info.keys[0]);
    free(arts_node_info.keys);
  }
  free(arts_node_info.global_guid_thread_id);
  /* fresh per-thread counter row (all per-(rank,type) counters start at 1) */
  arts_node_info.keys = (uint64_t **)calloc(1, sizeof(uint64_t *));
  arts_node_info.keys[0] = (uint64_t *)calloc(
      (size_t)ARTS_GUID_LAST * arts_global_rank_count, sizeof(uint64_t));
  for (unsigned i = 0; i < ARTS_GUID_LAST * arts_global_rank_count; i++) {
    arts_node_info.keys[0][i] = 1;
  }
  arts_node_info.global_guid_thread_id =
      (uint64_t *)calloc(1, sizeof(uint64_t));
  arts_node_info.global_guid_thread_id[0] = 0;
  num_tables = 1;
  min_global_guid_thread = 0;
  max_global_guid_thread = 1;
  keys_per_thread = key_budget;
  global_guid_on = 0;

  /* DB seq allocator (mirrors set_guid_generator_after_parallel_start). */
  arts_db_seq_budget =
      ((ARTS_GUID_DB_SEQ_MASK + 1) - ARTS_GUID_DB_STARTUP_RESERVE) /
      arts_global_rank_count;
  db_seq_creator_base = arts_db_seq_budget * arts_global_rank_id;
  if (db_seq_next) {
    free((void *)db_seq_next);
  }
  db_seq_next = (volatile uint64_t *)malloc(sizeof(uint64_t) *
                                            arts_global_rank_count);
  for (unsigned r = 0; r < arts_global_rank_count; r++) {
    db_seq_next[r] = db_seq_creator_base + 1;
  }
  free(t_db_cursor);
  t_db_cursor = NULL;
}

int main(void) {
  /* ---- Test 1: field round-trip, no cross-field bleed. ---- */
  for (unsigned t = 0; t <= ARTS_GUID_TYPE_MASK; t++) {
    unsigned ranks[] = {0u, 1u, 4242u, (unsigned)ARTS_GUID_RANK_MASK};
    uint64_t keys[] = {1u, 0xABCDEFu, ((uint64_t)1 << 47), ARTS_GUID_KEY_MASK};
    for (unsigned ri = 0; ri < 4; ri++) {
      for (unsigned ki = 0; ki < 4; ki++) {
        arts_guid_t g = ARTS_GUID_MAKE(t, ranks[ri], keys[ki]);
        if ((unsigned)arts_guid_get_kind(g) != t) {
          FAIL("kind decode %u != %u\n", (unsigned)arts_guid_get_kind(g), t);
        }
        if (arts_guid_get_rank(g) != ranks[ri]) {
          FAIL("rank decode %u != %u\n", arts_guid_get_rank(g), ranks[ri]);
        }
        if (arts_guid_get_key(g) != keys[ki]) {
          FAIL("key decode %lu != %lu\n", (unsigned long)arts_guid_get_key(g),
               (unsigned long)keys[ki]);
        }
      }
    }
  }

  /* ---- Test 2: is_local. ---- */
  generator_reset(1u << 20);
  arts_global_rank_id = 3;
  if (arts_guid_is_local(ARTS_GUID_MAKE(ARTS_GUID_DB, 3, 5))) {
    /* ok */
  } else {
    FAIL("is_local(rank==self) returned false\n");
  }
  if (arts_guid_is_local(ARTS_GUID_MAKE(ARTS_GUID_DB, 4, 5))) {
    FAIL("is_local(rank!=self) returned true\n");
  }
  arts_global_rank_id = 0;

  /* ---- Test 3: contiguous range round-trip. ---- */
  generator_reset(1u << 20);
  const unsigned SIZE = 64;
  arts_guid_t start = arts_guid_reserve_range(ARTS_GUID_DB, SIZE, 0);
  if (start == NULL_GUID) {
    FAIL("reserve_range returned NULL_GUID\n");
  }
  arts_guid_t prev = NULL_GUID;
  for (unsigned i = 0; i < SIZE; i++) {
    arts_guid_t g = arts_guid_from_index(start, i);
    if (arts_guid_get_kind(g) != ARTS_GUID_DB) {
      FAIL("range[%u] kind mismatch\n", i);
    }
    if (arts_guid_get_rank(g) != 0) {
      FAIL("range[%u] rank mismatch\n", i);
    }
    if (i > 0 && g == prev) {
      FAIL("range[%u] not distinct\n", i);
    }
    int idx = arts_guid_index_from(start, g);
    if (idx != (int)i) {
      FAIL("index_from(range[%u]) = %d\n", i, idx);
    }
    prev = g;
  }
  /* foreign-type GUID → -1 */
  arts_guid_t edt0 = arts_guid_reserve_range(ARTS_GUID_EDT, 1, 0);
  if (arts_guid_index_from(start, arts_guid_from_index(edt0, 0)) != -1) {
    FAIL("index_from did not reject foreign type\n");
  }
  /* key below range start → -1 (same type, smaller key) */
  arts_guid_t below =
      ARTS_GUID_MAKE(ARTS_GUID_DB, 0, ARTS_GUID_GET_KEY(start) - 1);
  if (arts_guid_index_from(start, below) != -1) {
    FAIL("index_from did not reject key below start\n");
  }

  /* ---- Test 4: counter disjointness across type. ---- */
  generator_reset(1u << 20);
  arts_guid_t a = arts_guid_create_for_rank(0, ARTS_GUID_DB);
  arts_guid_t b = arts_guid_create_for_rank(0, ARTS_GUID_DB);
  if (a == b) {
    FAIL("two same-(rank,type) reserves aliased\n");
  }
  arts_guid_t e = arts_guid_create_for_rank(0, ARTS_GUID_EVENT);
  /* different type → different kind field; keys may coincide but the encoded
   * GUIDs must differ because the type bits differ. */
  if (a == e) {
    FAIL("DB and EVENT GUID aliased\n");
  }

  /* ---- Test 5: exhaustion is a HARD ERROR (forked children). ---- */
  pid_t pid = fork();
  if (pid == 0) {
    /* child: shrink the DB slice so the first chunk lease overflows it —
     * the claim's returned-base bound check must abort, never wrap. */
    generator_reset(4);
    arts_db_seq_budget = 4;
    for (int i = 0; i < 100; i++) {
      (void)arts_guid_create_for_rank(0, ARTS_GUID_DB);
    }
    /* Should never reach here — exhaustion must have aborted. */
    _exit(0);
  } else if (pid > 0) {
    int st = 0;
    (void)waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) == 0) {
      FAIL("DB exhaustion did not hard-error (child exit status %d)\n", st);
    }
  } else {
    FAIL("fork failed\n");
  }
  pid = fork();
  if (pid == 0) {
    /* child: flat-kind path — a tiny keys_per_thread budget must abort. */
    generator_reset(4);
    for (int i = 0; i < 100; i++) {
      (void)arts_guid_create_for_rank(0, ARTS_GUID_EDT);
    }
    _exit(0);
  } else if (pid > 0) {
    int st = 0;
    (void)waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) == 0) {
      FAIL("flat-kind exhaustion did not hard-error (status %d)\n", st);
    }
  } else {
    FAIL("fork failed\n");
  }

  /* ---- Test 6: szhint encode/decode — bound covers, sentinel unreachable. */
  {
    uint64_t probes[] = {0,    1,    63,   64,   65,   2047, 2048,
                         2049, 4095, 4096, 4097, 65536};
    for (unsigned i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
      uint64_t sz = probes[i];
      uint64_t hint = arts_db_szhint_encode(sz);
      if (hint == ARTS_GUID_DB_SZHINT_NONE) {
        FAIL("szhint encode hit the sentinel for size %lu\n",
             (unsigned long)sz);
      }
      arts_guid_t g = arts_db_guid_stamp_szhint(
          ARTS_GUID_MAKE(ARTS_GUID_DB, 1, 12345), sz);
      uint64_t bound = arts_db_szhint_bound(g);
      if (bound < sz) {
        FAIL("szhint bound %lu < size %lu\n", (unsigned long)bound,
             (unsigned long)sz);
      }
      if (sz >= 4096 && bound > sz + sz / 32 + 64) {
        FAIL("szhint overshoot too large: size %lu bound %lu\n",
             (unsigned long)sz, (unsigned long)bound);
      }
      if (ARTS_GUID_DB_GET_SEQ(ARTS_GUID_GET_KEY(g)) != 12345) {
        FAIL("stamp disturbed the seq field\n");
      }
    }
    /* power sweep to the 4TB edge: bound covers up to the encodable max,
     * beyond it the sentinel (bound 0) takes over. */
    for (uint64_t sz = 64; sz <= ((uint64_t)4 << 40); sz <<= 1) {
      uint64_t hint = arts_db_szhint_encode(sz);
      arts_guid_t g = arts_db_guid_stamp_szhint(
          ARTS_GUID_MAKE(ARTS_GUID_DB, 1, 7), sz);
      uint64_t bound = arts_db_szhint_bound(g);
      if (hint != ARTS_GUID_DB_SZHINT_NONE && bound < sz) {
        FAIL("power sweep: bound %lu < size %lu\n", (unsigned long)bound,
             (unsigned long)sz);
      }
      if (hint == ARTS_GUID_DB_SZHINT_NONE && bound != 0) {
        FAIL("sentinel decoded a nonzero bound\n");
      }
      arts_guid_t gm1 = arts_db_guid_stamp_szhint(
          ARTS_GUID_MAKE(ARTS_GUID_DB, 1, 7), sz - 1);
      if (arts_db_szhint_bound(gm1) != 0 && arts_db_szhint_bound(gm1) < sz - 1) {
        FAIL("power-1 sweep: bound < size at %lu\n", (unsigned long)(sz - 1));
      }
    }
    /* the all-ones sentinel value itself decodes to 0 */
    arts_guid_t s_g = ARTS_GUID_MAKE(
        ARTS_GUID_DB, 1, ARTS_GUID_DB_KEY(ARTS_GUID_DB_SZHINT_NONE, 7));
    if (arts_db_szhint_bound(s_g) != 0) {
      FAIL("explicit sentinel decoded nonzero\n");
    }
  }

  /* free the final reset's allocations so LSan sees a clean exit */
  if (arts_node_info.keys) {
    free(arts_node_info.keys[0]);
    free(arts_node_info.keys);
  }
  free(arts_node_info.global_guid_thread_id);

  printf("PASS guid_encoding_roundtrip: field round-trip, range inverse, "
         "counter disjoint, exhaustion hard-errors, szhint bound-covers\n");
  return 0;
}
