/* SPDX-License-Identifier: Apache-2.0
 *
 * route_table_db_shard — the DB-kind route-table shard decision is a pure
 * function of (kind, seq), agreed on by install and lookup for EVERY DB key
 * source, and stable across the parallel-start flip:
 *
 *   1. Pre-flip (arts_db_seq_budget == 0): every DB key resolves to the
 *      shared remote shard table (legacy gate is also still closed).
 *   2. Startup-countdown keys (top seq region) resolve to the SAME remote
 *      shard before and after the flip.
 *   3. Post-flip self-created keys resolve to the local per-thread table at
 *      (seq >> CHUNK_BITS) % num_tables — chunk-mates share a table.
 *   4. Foreign-creator keys resolve to the remote shard; the SAME key viewed
 *      from the creating rank's identity resolves local (today's
 *      local/remote split semantics, recovered arithmetically).
 *   5. Distributed-range members and single-rank reserved members follow the
 *      same arithmetic (creator slice), sentinel szhint notwithstanding.
 *   6. Flat kinds (EDT/EVENT) keep the legacy keys_per_thread partition.
 *
 * No runtime: route_table.c + guid.c + shared.c #include'd; OoO drain no-op.
 */

#include <stdatomic.h>
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

struct arts_route_item_s;
void arts_ooo_drain(struct arts_route_item_s *s) { (void)s; }
void arts_ooo_redrive_all(struct arts_route_item_s *s) { (void)s; }
void arts_ooo_free_all(struct arts_route_item_s *s) { (void)s; }

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
#include "../../libs/src/core/gas/route_table.c"
#include "../../libs/src/core/utils/shared.c"

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

#define FAIL(...)                                                              \
  do {                                                                         \
    (void)fprintf(stderr, "FAIL route_table_db_shard: " __VA_ARGS__);          \
    return 1;                                                                  \
  } while (0)

#define NTABLES 3u

static void rt_init(void) {
  arts_thread_info.thread_id = 0;
  arts_node_info.total_thread_count = NTABLES;
  arts_node_info.gpu = 0;
  arts_node_info.keys =
      (uint64_t **)calloc(NTABLES, sizeof(uint64_t *));
  arts_node_info.keys[0] = (uint64_t *)calloc(
      (size_t)ARTS_GUID_LAST * arts_global_rank_count, sizeof(uint64_t));
  for (unsigned i = 0; i < ARTS_GUID_LAST * arts_global_rank_count; i++) {
    arts_node_info.keys[0][i] = 1;
  }
  arts_node_info.global_guid_thread_id =
      (uint64_t *)calloc(NTABLES, sizeof(uint64_t));
  num_tables = NTABLES;
  min_global_guid_thread = 0;
  max_global_guid_thread = NTABLES;
  keys_per_thread = 0; /* pre-flip */
  global_guid_on = 0;
  arts_db_seq_budget = 0; /* pre-flip */
  arts_node_info.route_table =
      (arts_route_table_t **)calloc(NTABLES, sizeof(arts_route_table_t *));
  for (unsigned t = 0; t < NTABLES; t++) {
    arts_node_info.route_table[t] = arts_new_route_table(256, 4);
  }
  for (int s = 0; s < ARTS_REMOTE_ROUTE_SHARDS; s++) {
    arts_node_info.remote_route_table[s] = arts_new_route_table(64, 4);
  }
}

static void flip_on(void) {
  arts_db_seq_budget =
      ((ARTS_GUID_DB_SEQ_MASK + 1) - ARTS_GUID_DB_STARTUP_RESERVE) /
      arts_global_rank_count;
  db_seq_creator_base = arts_db_seq_budget * arts_global_rank_id;
  db_seq_next = (volatile uint64_t *)malloc(sizeof(uint64_t) *
                                            arts_global_rank_count);
  for (unsigned r = 0; r < arts_global_rank_count; r++) {
    db_seq_next[r] = db_seq_creator_base + 1;
  }
  keys_per_thread =
      (((uint64_t)1 << ARTS_GUID_KEY_BITS) /
       ((uint64_t)NTABLES * arts_global_rank_count));
}

int main(void) {
  rt_init();

  /* ---- 1+2. pre-flip: startup-countdown key -> remote; stable across flip. */
  global_guid_on = ((uint64_t)1) << ARTS_GUID_KEY_BITS;
  arts_guid_t startup_db = arts_guid_create_for_rank(0, ARTS_GUID_DB);
  global_guid_on = 0;
  arts_route_table_t *pre = arts_get_route_table(startup_db);
  int is_remote = 0;
  for (int s = 0; s < ARTS_REMOTE_ROUTE_SHARDS; s++) {
    if (pre == arts_node_info.remote_route_table[s]) {
      is_remote = 1;
    }
  }
  if (!is_remote) {
    FAIL("pre-flip startup DB key did not resolve to a remote shard\n");
  }
  flip_on();
  if (arts_get_route_table(startup_db) != pre) {
    FAIL("startup DB key changed tables across the flip\n");
  }

  /* ---- 3. self-created keys: local table, chunk-granular agreement. ---- */
  arts_guid_t a = arts_guid_create_for_rank(1, ARTS_GUID_DB);
  arts_guid_t b = arts_guid_create_for_rank(1, ARTS_GUID_DB);
  uint64_t seq_a = ARTS_GUID_DB_GET_SEQ(ARTS_GUID_GET_KEY(a));
  arts_route_table_t *ta = arts_get_route_table(a);
  if (ta != arts_node_info
                .route_table[(seq_a >> ARTS_GUID_DB_CHUNK_BITS) % NTABLES]) {
    FAIL("self key not at (seq>>CHUNK)%%tables\n");
  }
  if (arts_get_route_table(b) != ta) {
    FAIL("chunk-mates split across tables\n");
  }

  /* ---- 4. foreign-creator key: remote here, local at the creator. ---- */
  uint64_t foreign_seq = arts_db_seq_budget * 2 + 5; /* rank 2's slice */
  arts_guid_t foreign = ARTS_GUID_MAKE(
      ARTS_GUID_DB, 0, ARTS_GUID_DB_KEY(ARTS_GUID_DB_SZHINT_NONE, foreign_seq));
  arts_route_table_t *ft = arts_get_route_table(foreign);
  is_remote = 0;
  for (int s = 0; s < ARTS_REMOTE_ROUTE_SHARDS; s++) {
    if (ft == arts_node_info.remote_route_table[s]) {
      is_remote = 1;
    }
  }
  if (!is_remote) {
    FAIL("foreign-creator DB key did not resolve remote\n");
  }
  arts_global_rank_id = 2; /* view the same key as its creator */
  db_seq_creator_base = arts_db_seq_budget * 2;
  arts_route_table_t *at_creator = arts_get_route_table(foreign);
  if (at_creator !=
      arts_node_info
          .route_table[(foreign_seq >> ARTS_GUID_DB_CHUNK_BITS) % NTABLES]) {
    FAIL("creator's view of its own key is not local\n");
  }
  arts_global_rank_id = 0;
  db_seq_creator_base = 0;

  /* ---- 5. reserved members follow the same arithmetic. ---- */
  arts_guid_t range = arts_guid_reserve_range(ARTS_GUID_DB, 8, 3);
  arts_guid_t member = arts_guid_from_index(range, 3);
  uint64_t mseq = ARTS_GUID_DB_GET_SEQ(ARTS_GUID_GET_KEY(member));
  if (arts_get_route_table(member) !=
      arts_node_info
          .route_table[(mseq >> ARTS_GUID_DB_CHUNK_BITS) % NTABLES]) {
    FAIL("reserved member not in the creator-slice local table\n");
  }

  /* ---- 6. flat kinds keep the legacy partition. ---- */
  arts_guid_t edt =
      ARTS_GUID_MAKE(ARTS_GUID_EDT, 0, keys_per_thread + 1); /* thread 1 */
  if (arts_get_route_table(edt) != arts_node_info.route_table[1]) {
    FAIL("flat-kind key left the keys_per_thread partition\n");
  }

  printf("PASS route_table_db_shard: pre/post-flip stability, chunk-granular "
         "locals, arithmetic creator recovery, flat kinds untouched\n");
  return 0;
}
