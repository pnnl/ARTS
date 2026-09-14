#include "arts.h"

static void launch(uint64_t n, arts_guid_t result, unsigned int rank);

static void leaf(uint32_t pc, const uint64_t *pv, uint32_t dc,
                 arts_edt_dep_t dv[]) {
  (void)pc;
  (void)dc;
  (void)dv;
  arts_edt_set_result(pv[0]);
}

static void sum(uint32_t pc, const uint64_t *pv, uint32_t dc,
                arts_edt_dep_t dv[]) {
  (void)pc;
  (void)dc;
  if (dv[0].ptr != NULL || dv[1].ptr != NULL ||
      dv[0].mode != DB_MODE_NULL || dv[1].mode != DB_MODE_NULL) {
    arts_printf("FAIL: scalar return acquired storage\n");
    arts_abort(1);
  }
  arts_edt_set_result(dv[0].guid + dv[1].guid);
  arts_event_destroy(pv[0]);
  arts_event_destroy(pv[1]);
}

static void branch(uint32_t pc, const uint64_t *pv, uint32_t dc,
                   arts_edt_dep_t dv[]) {
  (void)pc;
  (void)dc;
  (void)dv;
  unsigned int rank = arts_get_current_rank();
  arts_guid_t left = arts_event_create(&ARTS_EVENT_HINT_ONCE);
  arts_guid_t right = arts_event_create(&ARTS_EVENT_HINT_ONCE);
  uint64_t inputs[] = {left, right};
  arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
  hint.rank = rank;
  hint.output_event = pv[1];
  arts_guid_t continuation = arts_edt_create(sum, 2, inputs, 2, &hint);
  arts_add_dependence(left, continuation, 0, DB_MODE_NULL);
  arts_add_dependence(right, continuation, 1, DB_MODE_NULL);
  unsigned int next = (rank + 1) % arts_get_total_ranks();
  launch(pv[0] - 1, left, next);
  launch(pv[0] - 2, right, rank);
}

static void launch(uint64_t n, arts_guid_t result, unsigned int rank) {
  uint64_t args[] = {n, result};
  arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
  hint.rank = rank;
  if (n < 2)
    hint.output_event = result;
  arts_edt_create(n < 2 ? leaf : branch, 2, args, 0, &hint);
}

static void check(uint32_t pc, const uint64_t *pv, uint32_t dc,
                  arts_edt_dep_t dv[]) {
  (void)pc;
  (void)dc;
  if (dv[0].guid != 144 || dv[0].ptr != NULL ||
      dv[0].mode != DB_MODE_NULL) {
    arts_printf("FAIL: recursive scalar return was %lu\n", dv[0].guid);
    arts_abort(1);
  }
  arts_event_destroy(pv[0]);
  arts_printf("PASS: pre-created continuations receive recursive scalar returns\n");
  arts_shutdown();
}

void main_edt(uint32_t pc, const uint64_t *pv, uint32_t dc,
              arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  (void)dv;
  arts_guid_t result = arts_event_create(&ARTS_EVENT_HINT_ONCE);
  uint64_t args[] = {result};
  arts_guid_t sink = arts_edt_create(check, 1, args, 1, NULL);
  arts_add_dependence(result, sink, 0, DB_MODE_NULL);
  launch(12, result, 0);
}

int main(int argc, char **argv) { return arts_rt(argc, argv) != 0; }
