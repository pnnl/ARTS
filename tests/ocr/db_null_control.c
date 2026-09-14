#include "arts.h"

static void check_control(uint32_t pc, const uint64_t *pv, uint32_t dc,
                          arts_edt_dep_t dv[]) {
  (void)pc;
  (void)dc;
  if (dv[0].guid != (arts_guid_t)pv[0] || dv[0].ptr != NULL ||
      dv[0].mode != DB_MODE_NULL) {
    arts_printf("FAIL: NULL dependence changed GUID, mode, or pointer\n");
    arts_abort(1);
  }
}

void main_edt(uint32_t pc, const uint64_t *pv, uint32_t dc,
              arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  (void)dv;
  for (unsigned int retired = 0; retired < 2; ++retired) {
    for (unsigned int rank = 0; rank < arts_get_total_ranks(); ++rank) {
      uint64_t *value;
      arts_guid_t data = arts_db_create((void **)&value, sizeof(*value),
                                        ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
      *value = 42;
      arts_db_release(data, DB_MODE_RW);
      arts_guid_t finish = arts_event_create(&ARTS_EVENT_HINT_FINISH);
      arts_guid_t source = arts_event_create(&ARTS_EVENT_HINT_ONCE);
      uint64_t args[] = {(uint64_t)data};
      arts_guid_t sink = arts_edt_create(
          check_control, 1, args, 2,
          &(arts_edt_hint_t){.rank = rank, .finish_event = finish});
      arts_add_dependence(source, sink, 0, DB_MODE_NULL);
      arts_event_satisfy(source, data);
      if (retired)
        arts_db_destroy(data);
      /* The sink becomes ready only after the optional retirement. */
      arts_add_dependence(NULL_GUID, sink, 1, DB_MODE_NULL);
      arts_event_wait(finish);
      arts_event_destroy(source);
      if (!retired)
        arts_db_destroy(data);
    }
  }
  arts_printf("PASS: live and retired NULL control dependencies\n");
  arts_shutdown();
}

int main(int argc, char **argv) { return arts_rt(argc, argv) != 0; }
