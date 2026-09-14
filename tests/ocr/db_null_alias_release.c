#include "arts.h"

static void check_release(uint32_t pc, const uint64_t *pv, uint32_t dc,
                          arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  arts_guid_t db = dv[1].guid;
  if (dv[0].guid != db || dv[0].ptr != NULL ||
      dv[0].mode != DB_MODE_NULL || dv[1].ptr == NULL) {
    arts_printf("FAIL: opaque GUID was acquired or hid the DB dependency\n");
    arts_abort(1);
  }
  *(uint64_t *)dv[1].ptr = 17;
  arts_db_release(db, DB_MODE_RW);
  if (dv[0].guid != db || dv[0].mode != DB_MODE_NULL ||
      dv[1].ptr != NULL || dv[1].guid != NULL_GUID) {
    arts_printf("FAIL: DB release selected the opaque-value slot\n");
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
    arts_guid_t db = arts_db_create((void **)&value, sizeof(*value),
                                    ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
    *value = 0;
    arts_db_release(db, DB_MODE_RW);
    arts_guid_t finish = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
    hint.rank = rank;
    hint.finish_event = finish;
    arts_guid_t sink = arts_edt_create(check_release, 0, NULL, 2, &hint);
    arts_edt_satisfy_slot(sink, 0, db, DB_MODE_NULL);
    arts_add_dependence(db, sink, 1, DB_MODE_RW);
    arts_event_wait(finish);
    arts_db_destroy(db);
  }
  arts_printf("PASS: releasing a DB preserves equal opaque values\n");
  arts_shutdown();
}

int main(int argc, char **argv) { return arts_rt(argc, argv) != 0; }
