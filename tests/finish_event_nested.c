/* tests/finish_event_nested.c
 *
 * Rigorously exercises nested finish events (Mechanism A — auto-chain).
 *
 * An outer finish event F1 is created in main_edt.  Several "outer" leaves and
 * one "orchestrator" EDT join F1.  The orchestrator, while running under F1,
 * creates an inner finish event F2 — which the runtime auto-chains under F1
 * (INCR F1 + dep F2->DECR F1) — and spawns "inner" leaves joined to F2.  A
 * successor EDT depends on F1.
 *
 * Because F2 is auto-chained under F1, F1 cannot drain (and the successor
 * cannot fire) until BOTH the F1-direct work (outer leaves + orchestrator)
 * AND the entire F2 sub-scope (inner leaves) have completed.  The successor
 * asserts that all of them ran first; if auto-chain were broken, F1 would
 * drain before the inner leaves finished and the assert would fail.
 */
#include "arts.h"

#include <stdatomic.h>

#define OUTER_LEAVES 3
#define INNER_LEAVES 5

static atomic_int g_outer_done;
static atomic_int g_inner_done;
static atomic_int g_orch_done;
static volatile int g_ok = 0;

void outer_leaf(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  atomic_fetch_add(&g_outer_done, 1);
}

void inner_leaf(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  atomic_fetch_add(&g_inner_done, 1);
}

/* Runs under F1; creates F2 (auto-chained under F1) and fills it with leaves.
 */
void orchestrator(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t f2 = arts_event_create(&fh); /* auto-chains under ambient F1 */
  arts_edt_hint_t ih = ARTS_EDT_HINT_DEFAULTS;
  ih.finish_event = f2;
  for (int i = 0; i < INNER_LEAVES; i++) {
    arts_edt_create(inner_leaf, 0, NULL, 0, &ih);
  }
  /* No wait: orchestrator's completion releases F2's creator-token; F2 drains
   * when its inner leaves finish; F2's fire DECRs F1 via the auto-chain. */
  atomic_fetch_add(&g_orch_done, 1);
}

/* Fires only when F1 drains — must observe ALL work complete. */
void successor(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  int outer = atomic_load(&g_outer_done);
  int inner = atomic_load(&g_inner_done);
  int orch = atomic_load(&g_orch_done);
  if (outer == OUTER_LEAVES && inner == INNER_LEAVES && orch == 1) {
    g_ok = 1;
    arts_printf("OK nested: outer=%d inner=%d orch=%d all drained before F1 "
                "successor\n",
                outer, inner, orch);
  } else {
    arts_printf("FAIL nested: outer=%d (want %d) inner=%d (want %d) orch=%d "
                "(want 1)\n",
                outer, OUTER_LEAVES, inner, INNER_LEAVES, orch);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  atomic_store(&g_outer_done, 0);
  atomic_store(&g_inner_done, 0);
  atomic_store(&g_orch_done, 0);

  arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t f1 = arts_event_create(&fh);

  arts_edt_hint_t oh = ARTS_EDT_HINT_DEFAULTS;
  oh.finish_event = f1;
  for (int i = 0; i < OUTER_LEAVES; i++) {
    arts_edt_create(outer_leaf, 0, NULL, 0, &oh);
  }
  arts_edt_create(orchestrator, 0, NULL, 0, &oh);

  arts_edt_hint_t sh = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t s = arts_edt_create(successor, 0, NULL, 1, &sh);
  arts_add_dependence(f1, s, 0, DB_MODE_NULL);
  /* main_edt returns; F1's creator-token is released at its completion. */
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_ok ? 0 : 1;
}
