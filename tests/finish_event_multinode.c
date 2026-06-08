/* tests/finish_event_multinode.c
 *
 * Exercises the cross-node proxy-LATCH allocation inserted in
 * arts_remote_handle_edt_move.  A finish-root EDT spawns one leaf per
 * rank; at least one leaf lands on a remote rank, exercising the proxy
 * path.  The termination EDT fires only after every leaf's DECR
 * propagates back to the finish_event via the local proxy chain. */
#include "arts.h"

static volatile int term_ran = 0;

void term_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  term_ran = 1;
  arts_printf("PASS: cross-node finish-scope drained, termination EDT fired\n");
  arts_shutdown();
}

void leaf_on_other_rank(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  arts_printf("leaf running on rank %u\n", arts_get_current_rank());
}

void finish_root(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  /* Spawn one leaf per rank.  At least one is cross-node (rank != home),
   * exercising the proxy-allocation path in arts_remote_handle_edt_move. */
  unsigned int n = arts_get_total_ranks();
  for (unsigned int r = 0; r < n; r++) {
    arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
    hint.rank = r;
    arts_edt_create(leaf_on_other_rank, 0, NULL, 0, &hint);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  arts_guid_t term = arts_edt_create(term_edt, 0, NULL, 1, NULL);
  arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t fe = arts_event_create(&fh);
  arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
  hint.finish_event = fe;
  arts_edt_create(finish_root, 0, NULL, 0, &hint);
  arts_add_dependence(fe, term, 0, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return term_ran ? 0 : 1;
}
