#include "arts.h"

static void check(uint32_t pc, const uint64_t *pv, uint32_t dc,
                  arts_edt_dep_t dv[]) {
  (void)pc;
  (void)dc;
  if (dv[0].guid != pv[0] || dv[0].ptr != NULL ||
      dv[0].mode != DB_MODE_NULL) {
    arts_printf("FAIL: event payload policy was not preserved\n");
    arts_abort(1);
  }
}

void main_edt(uint32_t pc, const uint64_t *pv, uint32_t dc,
              arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  (void)dv;
  for (unsigned int rank = 0; rank < arts_get_total_ranks(); ++rank) {
    for (unsigned int discard = 0; discard < 2; ++discard) {
      for (unsigned int late = 0; late < 2; ++late) {
        arts_guid_t finish = arts_event_create(&ARTS_EVENT_HINT_FINISH);
        arts_event_hint_t event_hint = ARTS_EVENT_HINT_LATCH(2);
        event_hint.rank = rank;
        event_hint.discard_data = discard != 0;
        arts_guid_t event = arts_event_create(&event_hint);
        uint64_t expected = discard ? NULL_GUID : UINT64_MAX;
        arts_edt_hint_t edt_hint = ARTS_EDT_HINT_DEFAULTS;
        edt_hint.rank = (rank + 1) % arts_get_total_ranks();
        edt_hint.finish_event = finish;
        arts_guid_t sink = arts_edt_create(check, 1, &expected, 1, &edt_hint);
        if (!late)
          arts_add_dependence(event, sink, 0, DB_MODE_NULL);
        arts_event_satisfy(event, UINT64_MAX);
        arts_event_satisfy(event, UINT64_MAX);
        if (late)
          arts_add_dependence(event, sink, 0, DB_MODE_NULL);
        arts_event_wait(finish);
        arts_event_destroy(event);
      }
    }
  }
  arts_printf("PASS: event payload policy survives remote creation and binding\n");
  arts_shutdown();
}

int main(int argc, char **argv) { return arts_rt(argc, argv) != 0; }
