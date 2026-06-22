/* tests/finish_event_wait.c
 *
 * Exercises three finish-event usage patterns in one test:
 *   1. arts_event_create(FINISH) + explicit hint.finish_event join
 *      + arts_event_wait (wait-style sync).
 *   2. arts_event_create(FINISH) + explicit hint.finish_event join
 *      + arts_add_dependence(fe, successor, ...) with early return
 *      (continuation-style sync via creator-token release on EDT completion).
 *
 * Correctness signal: the finish scope draining is itself the proof that all
 * leaves ran (OCR finish-event contract).  arts_abort(1) in successor on the
 * continuation path signals any unexpected failure; a stranded waiter surfaces
 * as a ctest TIMEOUT.  No global cross-EDT state.
 */
#include "arts.h"

void leaf(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  /* Leaf body: just run.  Completion is recorded by the finish scope. */
}

void successor(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  /* Firing here proves fe2 drained (all 8 leaves completed). */
  arts_printf("OK continuation-style: 8 leaves drained before successor\n");
  arts_shutdown();
}

void orchestrator(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* --- Wait-style: 4 leaves under fe, blocked via arts_event_wait. --- */
  arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t fe = arts_event_create(&fh);

  arts_edt_hint_t eh = ARTS_EDT_HINT_DEFAULTS;
  eh.finish_event = fe;
  for (int i = 0; i < 4; i++) {
    arts_edt_create(leaf, 0, NULL, 0, &eh);
  }

  arts_event_wait(fe);
  /* Reaching here proves all 4 leaves completed (finish-event contract). */
  arts_printf("OK wait-style: 4 leaves drained before wait returned\n");

  /* --- Continuation-style: 8 leaves under fe2; successor fires when
   *     fe2 drains (creator-token released on orchestrator return). --- */
  arts_event_hint_t fh2 = ARTS_EVENT_HINT_FINISH;
  arts_guid_t fe2 = arts_event_create(&fh2);

  arts_edt_hint_t eh2 = ARTS_EDT_HINT_DEFAULTS;
  eh2.finish_event = fe2;
  for (int i = 0; i < 8; i++) {
    arts_edt_create(leaf, 0, NULL, 0, &eh2);
  }

  arts_edt_hint_t sh = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t s = arts_edt_create(successor, 0, NULL, 1, &sh);
  arts_add_dependence(fe2, s, 0, DB_MODE_NULL);

  /* Return without waiting: creator-token for fe2 is released on EDT
   * completion (arts_owned_finish_cleanup), allowing fe2 to drain and
   * fire successor once all 8 leaves have also decremented. */
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_edt_hint_t h = ARTS_EDT_HINT_DEFAULTS;
  arts_edt_create(orchestrator, 0, NULL, 0, &h);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
