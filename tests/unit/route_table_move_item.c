/* SPDX-License-Identifier: Apache-2.0
 *
 * route_table_move_item — arts_route_table_move_item success + loss path
 * (census 18-gas GAP 5; suspected bug B043).
 *
 * move_item relocates the SAME cb (no re-wrap, no extra ref) from old_key's
 * slot to new_key's slot, preserving the single-owner invariant across a GUID
 * change (DB rename / type-copy):
 *
 *   1. SUCCESS: new_key empty → old_key is vacated (becomes absent), the cb is
 *      CAS-installed into new_key (lookup new returns the moved object, lookup
 *      old returns NULL), the object is NOT freed (single cb, just relocated),
 *      and exactly one slot owns it.
 *
 *   2. LOSS (the SURPRISING contract — B043): new_key already occupied → the
 *      exchange-out of old_key has ALREADY vacated it, and the moved cb is
 *      released.  So a FAILED move DESTROYS old_key's object (its deleter runs)
 *      while new_key keeps its prior occupant.  This is the documented db.c
 *      contract ("false == old gone").  Pin it: after a losing move, old_key is
 *      absent AND its object's deleter ran exactly once, new_key unchanged.
 *
 *   3. old_key absent → move returns false, no effect, no free.
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
unsigned int arts_global_rank_count = 1;
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

static _Atomic int g_deletes;
static void *g_last_deleted;
static void test_deleter(void *obj) {
  atomic_fetch_add_explicit(&g_deletes, 1, memory_order_relaxed);
  g_last_deleted = obj;
  free(obj);
}

#define FAIL(...)                                                              \
  do {                                                                         \
    (void)fprintf(stderr, "FAIL route_table_move_item: " __VA_ARGS__);         \
    return 1;                                                                  \
  } while (0)

static void rt_init(void) {
  arts_thread_info.thread_id = 0;
  arts_node_info.total_thread_count = 1;
  arts_node_info.gpu = 0;
  arts_node_info.keys = (uint64_t **)calloc(1, sizeof(uint64_t *));
  arts_node_info.keys[0] = (uint64_t *)calloc(
      (size_t)ARTS_GUID_LAST * arts_global_rank_count, sizeof(uint64_t));
  for (unsigned i = 0; i < ARTS_GUID_LAST * arts_global_rank_count; i++) {
    arts_node_info.keys[0][i] = 1;
  }
  arts_node_info.global_guid_thread_id =
      (uint64_t *)calloc(1, sizeof(uint64_t));
  num_tables = 1;
  min_global_guid_thread = 0;
  max_global_guid_thread = 1;
  keys_per_thread = 1u << 20;
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
  arts_node_info.route_table =
      (arts_route_table_t **)calloc(1, sizeof(arts_route_table_t *));
  arts_node_info.route_table[0] = arts_new_route_table(1024, 10);
  for (int s = 0; s < ARTS_REMOTE_ROUTE_SHARDS; s++) {
    arts_node_info.remote_route_table[s] = arts_new_route_table(64, 10);
  }
}

static int slot_present(arts_guid_t g, void *expect) {
  arts_shared_ptr_t h = arts_route_table_lookup(g);
  void *got = h ? arts_shared_get(h) : NULL;
  if (h) {
    arts_shared_release(&h);
  }
  return got == expect;
}

int main(void) {
  rt_init();
  arts_route_table_register_deleter(ARTS_GUID_DB, test_deleter);

  /* ---- 1. SUCCESS: move into an empty new_key. ---- */
  {
    arts_guid_t old_g = arts_guid_reserve(ARTS_GUID_DB, 0);
    arts_guid_t new_g = arts_guid_reserve(ARTS_GUID_DB, 0);
    int *obj = (int *)malloc(sizeof(int));
    *obj = 0xABC;
    arts_route_table_install(obj, old_g, 0, false);
    atomic_store(&g_deletes, 0);

    if (!arts_route_table_move_item(old_g, new_g)) {
      FAIL("1: move into empty new_key returned false\n");
    }
    if (atomic_load(&g_deletes) != 0) {
      FAIL("1: object freed during a SUCCESSFUL move (cb was re-wrapped?)\n");
    }
    if (!slot_present(new_g, obj)) {
      FAIL("1: new_key does not hold the moved object\n");
    }
    if (!slot_present(old_g, NULL)) {
      FAIL("1: old_key not vacated after successful move\n");
    }
    /* clean: destroy via new_g, frees the single cb exactly once. */
    if (!arts_route_table_set_destroyed(new_g)) {
      FAIL("1: set_destroyed(new) returned false\n");
    }
    if (atomic_load(&g_deletes) != 1) {
      FAIL("1: object freed %d times after destroy, want 1\n",
           atomic_load(&g_deletes));
    }
  }

  /* ---- 2. LOSS (B043): new_key occupied → old destroyed, new kept. ---- */
  {
    arts_guid_t old_g = arts_guid_reserve(ARTS_GUID_DB, 0);
    arts_guid_t new_g = arts_guid_reserve(ARTS_GUID_DB, 0);
    int *old_obj = (int *)malloc(sizeof(int));
    *old_obj = 0x111;
    int *new_obj = (int *)malloc(sizeof(int));
    *new_obj = 0x222;
    arts_route_table_install(old_obj, old_g, 0, false);
    arts_route_table_install(new_obj, new_g, 0, false);
    atomic_store(&g_deletes, 0);
    g_last_deleted = NULL;

    bool moved = arts_route_table_move_item(old_g, new_g);
    if (moved) {
      FAIL("2: move into OCCUPIED new_key returned true\n");
    }
    /* B043 contract: old_key is vacated AND its object destroyed. */
    if (!slot_present(old_g, NULL)) {
      FAIL("2: B043 — old_key NOT vacated on a failed move\n");
    }
    if (atomic_load(&g_deletes) != 1 || g_last_deleted != old_obj) {
      FAIL("2: B043 — old_obj not destroyed exactly once on failed move "
           "(deletes=%d last=%p old=%p)\n",
           atomic_load(&g_deletes), g_last_deleted, (void *)old_obj);
    }
    /* new_key keeps its prior occupant, untouched. */
    if (!slot_present(new_g, new_obj)) {
      FAIL("2: new_key occupant changed by a failed move\n");
    }
    /* teardown the survivor. */
    arts_route_table_set_destroyed(new_g);
    if (atomic_load(&g_deletes) != 2) {
      FAIL("2: new_obj not freed on teardown\n");
    }
  }

  /* ---- 3. old_key absent → false, no effect. ---- */
  {
    arts_guid_t old_g =
        arts_guid_reserve(ARTS_GUID_DB, 0); /* never installed */
    arts_guid_t new_g = arts_guid_reserve(ARTS_GUID_DB, 0);
    atomic_store(&g_deletes, 0);
    if (arts_route_table_move_item(old_g, new_g)) {
      FAIL("3: move from an absent old_key returned true\n");
    }
    if (atomic_load(&g_deletes) != 0) {
      FAIL("3: a no-op move freed something\n");
    }
    if (!slot_present(new_g, NULL)) {
      FAIL("3: new_key spuriously populated\n");
    }
  }

  printf("PASS route_table_move_item: success relocates single cb; LOSS path "
         "vacates+destroys old_key (B043 contract pinned); absent→false\n");
  return 0;
}
