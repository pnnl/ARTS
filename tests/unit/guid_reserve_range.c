/* SPDX-License-Identifier: Apache-2.0
 *
 * guid_reserve_range — pure-unit coverage of arts_guid_reserve_range and the
 * labeled-vs-auto collision-avoidance invariant.
 *
 * Properties:
 *
 *   1. A ROUND_ROBIN reserve_range claims the SAME span [base, base+stride)
 *      on every home's counter, so a subsequent AUTO mint (which leases a
 *      chunk from the same shared counter) can never produce a key inside
 *      the labeled span.  We reserve a distributed range, expand it via
 *      from_index, then auto-mint on every rank and assert NO auto GUID
 *      equals any labeled GUID.
 *
 *   2. A single-rank reserve_range claims `size` CONSECUTIVE seqs in one
 *      fetch-add on the shared counter — contiguity holds and later mints
 *      land strictly above the range.
 *
 *   3. DB range members inherit the range's exact szhint bits (the sentinel),
 *      and index_from round-trips every member.
 *
 * No runtime: guid.c is #include'd with libc-backed shims.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

unsigned int arts_global_rank_id = 0;
unsigned int arts_global_rank_count = 4;
void arts_abort(uint8_t code) { _exit(code ? code : 70); }
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
    (void)fprintf(stderr, "FAIL guid_reserve_range: " __VA_ARGS__);            \
    return 1;                                                                  \
  } while (0)

static void generator_reset(uint64_t key_budget) {
  arts_thread_info.thread_id = 0;
  arts_node_info.total_thread_count = 1;
  arts_node_info.gpu = 0;
  if (arts_node_info.keys) {
    free(arts_node_info.keys[0]);
    free(arts_node_info.keys);
  }
  free(arts_node_info.global_guid_thread_id);
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

static void generator_free(void) {
  if (arts_node_info.keys) {
    free(arts_node_info.keys[0]);
    free(arts_node_info.keys);
    arts_node_info.keys = NULL;
  }
  free(arts_node_info.global_guid_thread_id);
  arts_node_info.global_guid_thread_id = NULL;
  free((void *)db_seq_next);
  db_seq_next = NULL;
  free(t_db_cursor);
  t_db_cursor = NULL;
}

int main(void) {
  /* ---- 1. labeled-vs-auto collision avoidance. ---- */
  generator_reset(1u << 20);
  const unsigned RSIZE = 40; /* spans several keys per rank (stride=10) */
  arts_guid_t dist =
      arts_guid_reserve_range(ARTS_GUID_DB, RSIZE, ARTS_HINT_ROUND_ROBIN);
  if (dist == NULL_GUID) {
    FAIL("distributed reserve_range returned NULL_GUID\n");
  }
  if (arts_guid_get_rank(dist) != ARTS_DISTRIBUTED_RANK) {
    FAIL("distributed range rank tag wrong: %u\n", arts_guid_get_rank(dist));
  }

  /* materialize all labeled GUIDs */
  arts_guid_t labeled[RSIZE];
  for (unsigned i = 0; i < RSIZE; i++) {
    labeled[i] = arts_guid_from_index(dist, i);
  }

  /* now auto-mint on EVERY rank; none may collide with a labeled GUID */
  for (unsigned r = 0; r < arts_global_rank_count; r++) {
    for (int k = 0; k < 8; k++) {
      arts_guid_t auto_g = arts_guid_create_for_rank(r, ARTS_GUID_DB);
      for (unsigned i = 0; i < RSIZE; i++) {
        if (auto_g == labeled[i]) {
          FAIL("auto GUID on rank %u collided with labeled[%u]\n", r, i);
        }
      }
    }
  }

  /* ---- 2. single-rank range: contiguous, exclusive. ---- */
  generator_reset(1u << 20);
  const unsigned NEED = 8;
  arts_guid_t rstart = arts_guid_reserve_range(ARTS_GUID_DB, NEED, 2);
  if (rstart == NULL_GUID) {
    FAIL("single-rank reserve_range returned NULL_GUID\n");
  }
  for (unsigned i = 1; i < NEED; i++) {
    if (arts_guid_get_key(arts_guid_from_index(rstart, i)) !=
        arts_guid_get_key(rstart) + i) {
      FAIL("range not contiguous at %u\n", i);
    }
  }
  /* a later mint on the same home must land strictly above the range */
  arts_guid_t after = arts_guid_create_for_rank(2, ARTS_GUID_DB);
  if (ARTS_GUID_DB_GET_SEQ(ARTS_GUID_GET_KEY(after)) <
      ARTS_GUID_DB_GET_SEQ(ARTS_GUID_GET_KEY(rstart)) + NEED) {
    FAIL("auto mint landed inside the reserved single-rank range\n");
  }

  /* ---- 3. szhint inheritance + index_from round-trip. ---- */
  for (unsigned i = 0; i < NEED; i++) {
    arts_guid_t g = arts_guid_from_index(rstart, i);
    if (ARTS_GUID_DB_GET_SZHINT(ARTS_GUID_GET_KEY(g)) !=
        ARTS_GUID_DB_SZHINT_NONE) {
      FAIL("range member %u lost the sentinel szhint\n", i);
    }
    if (arts_guid_index_from(rstart, g) != (int)i) {
      FAIL("index_from round-trip broke at %u\n", i);
    }
  }

  generator_free();
  printf("PASS guid_reserve_range: labeled-vs-auto collision-free, "
         "contiguous single-rank claim, sentinel inheritance + round-trip\n");
  return 0;
}
