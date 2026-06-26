/* SPDX-License-Identifier: Apache-2.0
 *
 * ooo_toctou_rescue — OoO engine push-then-install TOCTOU rescue.
 *
 * Property under test (ooo.c arts_ooo_dispatch_or_defer MISS path):
 *   A producer that observes slot->value == NULL pushes its payload onto the
 *   slot's ooo_list, then executes a seq_cst fence and re-loads slot->value.
 *   If an installer published `value` after the producer's initial NULL load
 *   but before/around the push, the producer's own post-push re-check MUST
 *   observe the install and drain the slot — so the just-pushed node is never
 *   stranded.  Symmetrically, the installer's own drain (after publishing
 *   value) detaches whatever was already pushed.  Between the two, EVERY
 *   pushed payload is dispatched EXACTLY ONCE and freed (no loss, no double).
 *
 * Interleaving driven:
 *   For each round, N producer threads each call dispatch_or_defer on the
 *   SAME freshly-reset slot while one installer thread publishes value and
 *   drains.  A start-gate releases all threads simultaneously to maximise the
 *   window where a producer loads NULL, then the installer publishes, then the
 *   producer pushes (the exact stranding window the fence+reload rescues).
 *   The dispatch handler increments a per-round atomic counter; after join we
 *   assert dispatched == pushed and the slot's ooo_list is empty.
 *
 * This is a standalone unit test: it #includes the runtime's ooo.c so the
 * file-static g_ooo_table + engine bodies are compiled in, links shared.c for
 * the cb shared-ptr slot, and provides a single-slot
 * arts_route_table_reserve_or_lookup plus libc malloc shims.  No ARTS runtime.
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

/* ── Dispatch accounting ──────────────────────────────────────────────────
 * Every g_ooo_table[kind] handler routes here.  We only ever defer one kind
 * in this test (the model-agnostic OOO_EVENT_SATISFY_SLOT), but all table
 * entries point at the same recorder so the build's per-config table is
 * satisfied regardless of protocol. */
static _Atomic uint64_t g_dispatched;

static void recorder(void *item, void *args) {
  (void)item;
  (void)args;
  atomic_fetch_add_explicit(&g_dispatched, 1, memory_order_relaxed);
}

/* The g_ooo_table static initializer in ooo.c references these handler symbols
 * by name (the model-agnostic set + the active model's OOO_DB_* set).  Provide
 * every one as the same recorder so any build links.  Signatures must match
 * arts_ooo_handler_fn_t exactly. */
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

/* ── Single fixed slot the test fully controls ────────────────────────────
 * The OoO engine reaches a slot's ooo_list / value through
 * arts_route_table_reserve_or_lookup; we override it to hand back one static
 * slot so the test drives value publish/destroy directly. */
static arts_route_item_t g_slot;

void arts_route_table_reserve_or_lookup(arts_guid_t key,
                                        arts_route_item_t **out) {
  (void)key;
  *out = &g_slot;
}

/* libc-backed allocator shims (ooo.c payloads + shared.c cb pool). */
void *arts_malloc(size_t size) { return malloc(size); }
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void arts_free(void *p) { free(p); }

/* Pull in the engine under test (compiles g_ooo_table + the bodies). */
#include "../../libs/src/core/ooo.c"

/* A dummy object the installer publishes into the slot. */
static int g_obj = 0xABCD;
static void noop_deleter(void *o) { (void)o; }

#define ROUNDS 4000
#define PRODUCERS 6

typedef struct {
  atomic_int *gate;
  ooo_kind_t kind;
} producer_ctx_t;

static void *producer_fn(void *vp) {
  producer_ctx_t *c = (producer_ctx_t *)vp;
  while (atomic_load_explicit(c->gate, memory_order_acquire) == 0) {
    /* spin */
  }
  /* Fresh-entry dispatch_or_defer: HIT dispatches inline, MISS pushes +
   * post-push rescue.  Either way the op must be dispatched exactly once. */
  uint64_t dummy_args = 0;
  arts_ooo_dispatch_or_defer(&g_slot, NULL, c->kind, &dummy_args,
                             sizeof(dummy_args));
  return NULL;
}

typedef struct {
  atomic_int *gate;
} installer_ctx_t;

static void *installer_fn(void *vp) {
  installer_ctx_t *c = (installer_ctx_t *)vp;
  while (atomic_load_explicit(c->gate, memory_order_acquire) == 0) {
    /* spin */
  }
  /* Publish value (the install) then drain — the create-handler protocol. */
  arts_shared_ptr_t cb = arts_shared_make(&g_obj, noop_deleter);
  arts_atomic_shared_store(&g_slot.value, cb);
  arts_ooo_drain(&g_slot);
  return NULL;
}

int main(void) {
  /* Choose a model-agnostic kind that is always present in g_ooo_table. */
  const ooo_kind_t kind = OOO_EVENT_SATISFY_SLOT;

  for (int r = 0; r < ROUNDS; r++) {
    /* Reset the slot to the pre-install state (value NULL, empty chain). */
    atomic_store_explicit(&g_slot.value, (arts_shared_slot_t){0},
                          memory_order_relaxed);
    arts_lf_stack_init(&g_slot.ooo_list);
    __atomic_store_n(&g_slot.gen, 0, __ATOMIC_RELAXED);
    atomic_store_explicit(&g_dispatched, 0, memory_order_relaxed);

    atomic_int gate;
    atomic_init(&gate, 0);

    pthread_t prod[PRODUCERS];
    producer_ctx_t pctx[PRODUCERS];
    for (int i = 0; i < PRODUCERS; i++) {
      pctx[i].gate = &gate;
      pctx[i].kind = kind;
      if (pthread_create(&prod[i], NULL, producer_fn, &pctx[i]) != 0) {
        (void)fprintf(stderr, "FAIL: pthread_create producer %d\n", i);
        return 1;
      }
    }
    pthread_t inst;
    installer_ctx_t ictx = {.gate = &gate};
    if (pthread_create(&inst, NULL, installer_fn, &ictx) != 0) {
      (void)fprintf(stderr, "FAIL: pthread_create installer\n");
      return 1;
    }

    atomic_store_explicit(&gate, 1, memory_order_release);

    for (int i = 0; i < PRODUCERS; i++) {
      pthread_join(prod[i], NULL);
    }
    pthread_join(inst, NULL);

    /* After all producers + the installer's drain, there must be NO node left
     * stranded on the chain: a final drain must dispatch nothing. */
    arts_ooo_drain(&g_slot);

    uint64_t got = atomic_load_explicit(&g_dispatched, memory_order_relaxed);
    if (got != (uint64_t)PRODUCERS) {
      (void)fprintf(stderr,
                    "FAIL round %d: dispatched %" PRIu64
                    " (want %d) — a pushed "
                    "node was stranded or double-dispatched\n",
                    r, got, PRODUCERS);
      return 1;
    }
    /* Chain must be empty now. */
    arts_lf_link_t *leftover = arts_lf_stack_drain(&g_slot.ooo_list);
    if (leftover != NULL) {
      (void)fprintf(stderr, "FAIL round %d: ooo_list not empty after drain\n",
                    r);
      return 1;
    }
    /* Release the slot's install ref so the cb pool is balanced. */
    arts_shared_ptr_t old =
        arts_atomic_shared_exchange(&g_slot.value, (arts_shared_ptr_t)NULL);
    if (old) {
      arts_shared_release(&old);
    }
  }

  printf("PASS ooo_toctou_rescue: %d rounds x %d producers, no strand/dup\n",
         ROUNDS, PRODUCERS);
  return 0;
}
