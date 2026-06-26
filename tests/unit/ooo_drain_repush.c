/* SPDX-License-Identifier: Apache-2.0
 *
 * ooo_drain_repush — OoO drain re-push vs ooo_list Treiber under reuse.
 *
 * Properties under test (ooo.c arts_ooo_drain + dispatch_or_defer MISS path):
 *
 *  (1) "Save next BEFORE dispatch."  arts_ooo_drain reads head->next into a
 *      local BEFORE calling dispatch_or_defer, because on a MISS the dispatch
 *      RE-PUSHES the very same node (overwriting head->link.next as it links to
 *      the new chain top).  If the walk re-read head->next AFTER dispatch it
 *      would follow the re-push chain and either spin or lose the rest of the
 *      snapshot.  We drive a drain whose every node MISSes (slot value == NULL)
 *      so every node is re-pushed mid-walk, then assert the snapshot was walked
 *      to completion exactly once (each node visited once, none lost).
 *
 *  (2) Re-push-onto-same-stack ABA safety under allocator address reuse.  The
 *      ooo_list is consumed ONLY by whole-chain reverse_drain (a single
 *      atomic_exchange); there is deliberately NO pop_one.  A push only links
 *      to "whatever is on top now" and never caches head->next for a CAS, so
 *      re-pushing a node back onto the SAME stack is ABA-safe even when freed
 *      payloads' addresses are recycled.  We interleave a MISS-drain (which
 *      re-pushes) with a concurrent producer thread that pushes fresh payloads
 *      (forcing the allocator to recycle addresses of nodes that were freed on
 *      HIT in prior rounds).  Correctness criterion across the whole run: every
 *      distinct payload id is dispatched EXACTLY ONCE and the chain ends empty
 *      (no loss, no duplicate, no corruption).
 *
 * Each payload carries a unique 32-bit id in its args blob; the dispatch
 * handler records the id into a global seen[] bitmap and flags any duplicate.
 *
 * Standalone: #includes ooo.c, links shared.c, single fixed slot, libc shims.
 */

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/utils/shared.h"

/* ── Dispatch accounting: record id, flag duplicates ────────────────────── */
#define MAX_IDS 200000
static _Atomic unsigned char g_seen[MAX_IDS];
static _Atomic uint64_t g_dispatched;
static _Atomic uint64_t g_dup;

static void recorder(void *item, void *args) {
  (void)item;
  uint32_t id = *(const uint32_t *)args;
  if (id < MAX_IDS) {
    unsigned char prev =
        atomic_fetch_add_explicit(&g_seen[id], 1, memory_order_relaxed);
    if (prev != 0) {
      atomic_fetch_add_explicit(&g_dup, 1, memory_order_relaxed);
    }
  }
  atomic_fetch_add_explicit(&g_dispatched, 1, memory_order_relaxed);
}

void arts_handler_event_satisfy_slot(void *i, void *a) { recorder(i, a); }
void arts_handler_edt_satisfy_slot(void *i, void *a) { recorder(i, a); }
void arts_handler_event_add_dependence(void *i, void *a) { recorder(i, a); }
void arts_handler_edt_destroy(void *i, void *a) { recorder(i, a); }
void arts_handler_event_destroy(void *i, void *a) { recorder(i, a); }
void arts_handler_db_destroy(void *i, void *a) { recorder(i, a); }
void arts_db_acquire_replay_dep(void *i, void *a) { recorder(i, a); }
void arts_handler_db_snapshot_request(void *i, void *a) { recorder(i, a); }
void arts_handler_db_writeback(void *i, void *a) { recorder(i, a); }
#if defined(ARTS_PROTOCOL_LOCK)
void arts_handler_db_lock_request(void *i, void *a) { recorder(i, a); }
void arts_handler_db_lock_release(void *i, void *a) { recorder(i, a); }
#elif defined(ARTS_TIMING_EAGER) || defined(ARTS_TIMING_LAZY)
void arts_handler_db_ownership_request(void *i, void *a) { recorder(i, a); }
#endif

/* ── Single fixed slot ────────────────────────────────────────────────── */
static arts_route_item_t g_slot;
void arts_route_table_reserve_or_lookup(arts_guid_t key,
                                        arts_route_item_t **out) {
  (void)key;
  *out = &g_slot;
}

void *arts_malloc(size_t size) { return malloc(size); }
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void arts_free(void *p) { free(p); }

#include "../../libs/src/core/ooo.c"

static int g_obj = 0x1234;
static void noop_deleter(void *o) { (void)o; }

/* ───────────────────────────────────────────────────────────────────────
 * PART 1 — deterministic single-thread "save next before dispatch".
 *
 * Pre-fill the chain with M payloads while value == NULL.  Call arts_ooo_drain
 * once: every node MISSes (value still NULL) so every node is re-pushed.  The
 * snapshot must be walked to completion — exactly M distinct ids deferred, and
 * after the drain the chain still holds exactly M nodes (the re-pushed ones).
 * Then publish value and drain again: now every node HITs and dispatches once.
 * Verify each of the M ids dispatched exactly once.
 * ──────────────────────────────────────────────────────────────────────── */
static int part1_save_next(void) {
  const int M = 1000;
  atomic_store_explicit(&g_slot.value, (arts_shared_slot_t){0},
                        memory_order_relaxed);
  arts_lf_stack_init(&g_slot.ooo_list);
  for (int id = 0; id < M; id++) {
    atomic_store_explicit(&g_seen[id], 0, memory_order_relaxed);
  }
  atomic_store_explicit(&g_dispatched, 0, memory_order_relaxed);
  atomic_store_explicit(&g_dup, 0, memory_order_relaxed);

  /* Defer M payloads (all MISS — value NULL — so they push, no dispatch). */
  for (uint32_t id = 0; id < (uint32_t)M; id++) {
    arts_ooo_dispatch_or_defer(&g_slot, NULL, OOO_EVENT_SATISFY_SLOT, &id,
                               sizeof(id));
  }
  if (atomic_load_explicit(&g_dispatched, memory_order_relaxed) != 0) {
    (void)fprintf(stderr, "FAIL part1: pre-install defer dispatched early\n");
    return 1;
  }

  /* Drain with value STILL NULL: every node MISSes mid-walk and is re-pushed.
   * The walk must save next-before-dispatch and complete the snapshot. */
  arts_ooo_drain(&g_slot);
  if (atomic_load_explicit(&g_dispatched, memory_order_relaxed) != 0) {
    (void)fprintf(stderr, "FAIL part1: MISS-drain dispatched %" PRIu64 "\n",
                  atomic_load_explicit(&g_dispatched, memory_order_relaxed));
    return 1;
  }
  /* Count re-pushed nodes by detaching (then put them back via reverse so a
   * subsequent drain can dispatch them). */
  arts_lf_link_t *chain = arts_lf_stack_reverse_drain(&g_slot.ooo_list);
  int repushed = 0;
  for (arts_lf_link_t *p = chain; p;) {
    arts_lf_link_t *nx = atomic_load_explicit(&p->next, memory_order_relaxed);
    repushed++;
    p = nx;
  }
  if (repushed != M) {
    (void)fprintf(stderr,
                  "FAIL part1: re-pushed %d nodes after MISS-drain (want %d) — "
                  "snapshot truncated/looped (save-next regression)\n",
                  repushed, M);
    return 1;
  }
  /* Re-attach the detached chain so we can dispatch it now. */
  for (arts_lf_link_t *p = chain; p;) {
    arts_lf_link_t *nx = atomic_load_explicit(&p->next, memory_order_relaxed);
    arts_lf_stack_push(&g_slot.ooo_list, p);
    p = nx;
  }

  /* Publish value and drain: every node HITs and dispatches once. */
  arts_shared_ptr_t cb = arts_shared_make(&g_obj, noop_deleter);
  arts_atomic_shared_store(&g_slot.value, cb);
  arts_ooo_drain(&g_slot);

  uint64_t got = atomic_load_explicit(&g_dispatched, memory_order_relaxed);
  if (got != (uint64_t)M) {
    (void)fprintf(stderr,
                  "FAIL part1: HIT-drain dispatched %" PRIu64 " (want %d)\n",
                  got, M);
    return 1;
  }
  if (atomic_load_explicit(&g_dup, memory_order_relaxed) != 0) {
    (void)fprintf(stderr, "FAIL part1: duplicate dispatch detected\n");
    return 1;
  }
  for (int id = 0; id < M; id++) {
    if (atomic_load_explicit(&g_seen[id], memory_order_relaxed) != 1) {
      (void)fprintf(stderr, "FAIL part1: id %d seen %u times (want 1)\n", id,
                    atomic_load_explicit(&g_seen[id], memory_order_relaxed));
      return 1;
    }
  }
  arts_shared_ptr_t old =
      arts_atomic_shared_exchange(&g_slot.value, (arts_shared_ptr_t)NULL);
  if (old) {
    arts_shared_release(&old);
  }
  printf("[part1] PASS  save-next-before-dispatch: %d nodes re-pushed + "
         "dispatched once\n",
         M);
  return 0;
}

/* ───────────────────────────────────────────────────────────────────────
 * PART 2 — concurrent re-push vs producer push under address reuse.
 *
 * One "miss-drainer" thread repeatedly drains the slot while value == NULL
 * (each drain re-pushes whatever it snapshots).  N producer threads push fresh
 * payloads concurrently.  After producers finish, value is published and a
 * final drain dispatches everything.  Across the whole run every distinct id
 * must dispatch EXACTLY once (re-push must not lose or duplicate a node, and
 * recycled addresses must not corrupt the Treiber chain).
 * ──────────────────────────────────────────────────────────────────────── */
#define P2_PRODUCERS 6
#define P2_PER_PRODUCER 20000

static atomic_int g_p2_gate;
static _Atomic int g_p2_producers_done;
static _Atomic int g_drainer_stop;

typedef struct {
  uint32_t base; /* first id this producer emits */
  int count;
} p2_producer_ctx_t;

static void *p2_producer_fn(void *vp) {
  p2_producer_ctx_t *c = (p2_producer_ctx_t *)vp;
  while (atomic_load_explicit(&g_p2_gate, memory_order_acquire) == 0) {
  }
  for (int k = 0; k < c->count; k++) {
    uint32_t id = c->base + (uint32_t)k;
    arts_ooo_dispatch_or_defer(&g_slot, NULL, OOO_EVENT_SATISFY_SLOT, &id,
                               sizeof(id));
  }
  atomic_fetch_add_explicit(&g_p2_producers_done, 1, memory_order_release);
  return NULL;
}

/* Continuously drain while value == NULL — forces concurrent re-push of any
 * snapshot it grabs (every node MISSes since value stays NULL until the end).
 */
static void *p2_drainer_fn(void *vp) {
  (void)vp;
  while (atomic_load_explicit(&g_p2_gate, memory_order_acquire) == 0) {
  }
  while (atomic_load_explicit(&g_drainer_stop, memory_order_acquire) == 0) {
    arts_ooo_drain(&g_slot);
  }
  return NULL;
}

static int part2_concurrent_repush(void) {
  const int TOTAL = P2_PRODUCERS * P2_PER_PRODUCER;
  if (TOTAL > MAX_IDS) {
    (void)fprintf(stderr, "FAIL part2: TOTAL %d > MAX_IDS\n", TOTAL);
    return 1;
  }
  atomic_store_explicit(&g_slot.value, (arts_shared_slot_t){0},
                        memory_order_relaxed);
  arts_lf_stack_init(&g_slot.ooo_list);
  for (int id = 0; id < TOTAL; id++) {
    atomic_store_explicit(&g_seen[id], 0, memory_order_relaxed);
  }
  atomic_store_explicit(&g_dispatched, 0, memory_order_relaxed);
  atomic_store_explicit(&g_dup, 0, memory_order_relaxed);
  atomic_init(&g_p2_gate, 0);
  atomic_store_explicit(&g_p2_producers_done, 0, memory_order_relaxed);
  atomic_store_explicit(&g_drainer_stop, 0, memory_order_relaxed);

  pthread_t prod[P2_PRODUCERS];
  p2_producer_ctx_t pctx[P2_PRODUCERS];
  for (int i = 0; i < P2_PRODUCERS; i++) {
    pctx[i].base = (uint32_t)(i * P2_PER_PRODUCER);
    pctx[i].count = P2_PER_PRODUCER;
    if (pthread_create(&prod[i], NULL, p2_producer_fn, &pctx[i]) != 0) {
      (void)fprintf(stderr, "FAIL part2: pthread_create producer %d\n", i);
      return 1;
    }
  }
  pthread_t drainer;
  if (pthread_create(&drainer, NULL, p2_drainer_fn, NULL) != 0) {
    (void)fprintf(stderr, "FAIL part2: pthread_create drainer\n");
    return 1;
  }

  atomic_store_explicit(&g_p2_gate, 1, memory_order_release);

  for (int i = 0; i < P2_PRODUCERS; i++) {
    pthread_join(prod[i], NULL);
  }
  /* Stop the miss-drainer (it has been re-pushing snapshots all along). */
  atomic_store_explicit(&g_drainer_stop, 1, memory_order_release);
  pthread_join(drainer, NULL);

  /* Nothing should have dispatched yet (value stayed NULL → all MISS). */
  if (atomic_load_explicit(&g_dispatched, memory_order_relaxed) != 0) {
    (void)fprintf(stderr,
                  "FAIL part2: %" PRIu64
                  " dispatched while value NULL (impossible on MISS path)\n",
                  atomic_load_explicit(&g_dispatched, memory_order_relaxed));
    return 1;
  }

  /* Publish value and drain everything that survived the concurrent re-pushes.
   * A few final drains in case a node was re-pushed onto a fresh chain by the
   * last MISS-drain. */
  arts_shared_ptr_t cb = arts_shared_make(&g_obj, noop_deleter);
  arts_atomic_shared_store(&g_slot.value, cb);
  for (int i = 0; i < 4; i++) {
    arts_ooo_drain(&g_slot);
  }

  uint64_t got = atomic_load_explicit(&g_dispatched, memory_order_relaxed);
  uint64_t dup = atomic_load_explicit(&g_dup, memory_order_relaxed);
  if (got != (uint64_t)TOTAL) {
    (void)fprintf(stderr,
                  "FAIL part2: dispatched %" PRIu64 " (want %d) — node lost or "
                  "duplicated under concurrent re-push\n",
                  got, TOTAL);
    return 1;
  }
  if (dup != 0) {
    (void)fprintf(stderr, "FAIL part2: %" PRIu64 " duplicate dispatches\n",
                  dup);
    return 1;
  }
  for (int id = 0; id < TOTAL; id++) {
    if (atomic_load_explicit(&g_seen[id], memory_order_relaxed) != 1) {
      (void)fprintf(stderr, "FAIL part2: id %d seen %u (want 1)\n", id,
                    atomic_load_explicit(&g_seen[id], memory_order_relaxed));
      return 1;
    }
  }
  arts_lf_link_t *leftover = arts_lf_stack_drain(&g_slot.ooo_list);
  if (leftover != NULL) {
    (void)fprintf(stderr, "FAIL part2: chain not empty after final drains\n");
    return 1;
  }
  arts_shared_ptr_t old =
      arts_atomic_shared_exchange(&g_slot.value, (arts_shared_ptr_t)NULL);
  if (old) {
    arts_shared_release(&old);
  }
  printf("[part2] PASS  concurrent re-push: %d ids, each dispatched once\n",
         TOTAL);
  return 0;
}

int main(void) {
  if (part1_save_next() != 0) {
    return 1;
  }
  if (part2_concurrent_repush() != 0) {
    return 1;
  }
  printf("PASS ooo_drain_repush: save-next + re-push ABA all clean\n");
  return 0;
}
