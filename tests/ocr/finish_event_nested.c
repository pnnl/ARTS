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
 *
 * Each leaf writes a 1 into its own private DB slot via a RW dep; the
 * orchestrator similarly marks its own slot.  The successor receives a RO dep
 * on the summary DB (which the orchestrator/leaves have already released by
 * the time F1 fires) and verifies all slots are set.  No global cross-EDT
 * state; successor calls arts_abort(1) on mismatch.
 */
#include "arts.h"

/* Total slots: OUTER_LEAVES outer + INNER_LEAVES inner + 1 orch */
#define OUTER_LEAVES 3
#define INNER_LEAVES 5
/* slots 0..(OUTER_LEAVES-1): outer leaves
 * slots OUTER_LEAVES..(OUTER_LEAVES+INNER_LEAVES-1): inner leaves
 * slot  OUTER_LEAVES+INNER_LEAVES: orchestrator */
#define ORCH_SLOT (OUTER_LEAVES + INNER_LEAVES)
#define N_SLOTS (OUTER_LEAVES + INNER_LEAVES + 1)

/* outer_leaf: depv[0] = counter DB (RW), paramv[0] = slot index */
void outer_leaf(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *c = (int *)depv[0].ptr;
  if (c == NULL) {
    arts_printf("FAIL: outer_leaf got NULL counter DB\n");
    arts_abort(1);
  }
  c[(int)paramv[0]] = 1;
}

/* inner_leaf: depv[0] = counter DB (RW), paramv[0] = slot index */
void inner_leaf(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *c = (int *)depv[0].ptr;
  if (c == NULL) {
    arts_printf("FAIL: inner_leaf got NULL counter DB\n");
    arts_abort(1);
  }
  c[(int)paramv[0]] = 1;
}

/* orchestrator: depv[0] = counter DB (RW).
 * Runs under F1; creates F2 (auto-chained under F1) and fills it with inner
 * leaves, each of which receives a RW dep on the counter DB. */
void orchestrator(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  arts_guid_t cdb = depv[0].guid;
  int *c = (int *)depv[0].ptr;
  if (c == NULL) {
    arts_printf("FAIL: orchestrator got NULL counter DB\n");
    arts_abort(1);
  }

  arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t f2 = arts_event_create(&fh); /* auto-chains under ambient F1 */
  arts_edt_hint_t ih = ARTS_EDT_HINT_DEFAULTS;
  ih.finish_event = f2;
  for (int i = 0; i < INNER_LEAVES; i++) {
    uint64_t slot = (uint64_t)(OUTER_LEAVES + i);
    arts_guid_t il = arts_edt_create(inner_leaf, 1, &slot, 1, &ih);
    arts_add_dependence(cdb, il, 0, DB_MODE_RW);
  }
  /* Mark the orchestrator's own slot before returning. */
  c[ORCH_SLOT] = 1;
  /* No wait: orchestrator's completion releases F2's creator-token; F2 drains
   * when its inner leaves finish; F2's fire DECRs F1 via the auto-chain. */
}

/* successor: depv[0] = F1 event (DB_MODE_NULL), depv[1] = counter DB (RO).
 * Fires only when F1 drains — must observe ALL work complete. */
void successor(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *c = (int *)depv[1].ptr;
  if (c == NULL) {
    arts_printf("FAIL nested: counter DB ptr is NULL\n");
    arts_abort(1);
  }
  int outer = 0;
  for (int i = 0; i < OUTER_LEAVES; i++) {
    outer += c[i];
  }
  int inner = 0;
  for (int i = 0; i < INNER_LEAVES; i++) {
    inner += c[OUTER_LEAVES + i];
  }
  int orch = c[ORCH_SLOT];
  if (outer != OUTER_LEAVES || inner != INNER_LEAVES || orch != 1) {
    arts_printf("FAIL nested: outer=%d (want %d) inner=%d (want %d) orch=%d "
                "(want 1)\n",
                outer, OUTER_LEAVES, inner, INNER_LEAVES, orch);
    arts_abort(1);
  }
  arts_printf("OK nested: outer=%d inner=%d orch=%d all drained before F1 "
              "successor\n",
              outer, inner, orch);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* Counter DB: N_SLOTS ints, zero-initialised by the creator before release.
   * Leaves and orchestrator each mark their own slot via RW dep. */
  int *c = NULL;
  arts_guid_t cdb = arts_db_create((void **)&c, sizeof(int) * N_SLOTS, ARTS_DB,
                                   ARTS_DB_PROP_NONE, NULL);
  for (int i = 0; i < N_SLOTS; i++) {
    c[i] = 0;
  }
  arts_db_release(cdb, DB_MODE_RW);

  arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t f1 = arts_event_create(&fh);

  arts_edt_hint_t oh = ARTS_EDT_HINT_DEFAULTS;
  oh.finish_event = f1;
  for (int i = 0; i < OUTER_LEAVES; i++) {
    uint64_t slot = (uint64_t)i;
    arts_guid_t ol = arts_edt_create(outer_leaf, 1, &slot, 1, &oh);
    arts_add_dependence(cdb, ol, 0, DB_MODE_RW);
  }
  arts_guid_t orch = arts_edt_create(orchestrator, 0, NULL, 1, &oh);
  arts_add_dependence(cdb, orch, 0, DB_MODE_RW);

  /* successor: slot 0 = F1 (NULL), slot 1 = counter DB (RO). */
  arts_edt_hint_t sh = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t s = arts_edt_create(successor, 0, NULL, 2, &sh);
  arts_add_dependence(f1, s, 0, DB_MODE_NULL);
  arts_add_dependence(cdb, s, 1, DB_MODE_RO);
  /* main_edt returns; F1's creator-token is released at its completion. */
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
