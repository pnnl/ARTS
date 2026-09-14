#include "arts.h"

static void check_value(uint32_t pc, const uint64_t *pv, uint32_t dc,
                        arts_edt_dep_t dv[]) {
  (void)pc;
  (void)dc;
  if (dv[0].guid != (arts_guid_t)pv[0] || dv[0].ptr != NULL ||
      dv[0].mode != DB_MODE_NULL) {
    arts_printf("FAIL: event value lost its payload or acquired a DB\n");
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
    uint64_t *value;
    arts_guid_t data = arts_db_create((void **)&value, sizeof(*value),
                                      ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
    *value = 42;
    arts_db_release(data, DB_MODE_RW);
    arts_guid_t opaque_event = arts_event_create(&ARTS_EVENT_HINT_ONCE);
    uint64_t payloads[] = {0, 42, (uint64_t)data, (uint64_t)opaque_event,
                           UINT64_MAX, UINT64_C(0x4000000000000001)};
    for (unsigned int i = 0; i < sizeof(payloads) / sizeof(payloads[0]); ++i) {
      arts_guid_t finish = arts_event_create(&ARTS_EVENT_HINT_FINISH);
      arts_event_hint_t hint = ARTS_EVENT_HINT_ONCE;
      hint.rank = rank;
      arts_guid_t source = arts_event_create(&hint);
      arts_guid_t sink = arts_edt_create(
          check_value, 1, &payloads[i], 2,
          &(arts_edt_hint_t){.rank = rank, .finish_event = finish});
      arts_add_dependence(source, sink, 0, DB_MODE_NULL);
      arts_event_satisfy(source, (arts_guid_t)payloads[i]);
      if (i == 2)
        arts_db_destroy(data);
      arts_edt_satisfy_slot(sink, 1, NULL_GUID, DB_MODE_NULL);
      arts_event_wait(finish);
      arts_event_destroy(source);
    }
    arts_event_destroy(opaque_event);
  }
  arts_printf("PASS: explicit event edges deliver opaque values without acquire\n");
  arts_shutdown();
}

int main(int argc, char **argv) { return arts_rt(argc, argv) != 0; }
