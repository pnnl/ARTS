/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file runtime_teardown_leak.c
/// @brief ASan/LSan teardown-leak guard: a full bring-up + workload + clean
///        shutdown must free everything arts_runtime_node_init allocated.
///
/// arts_runtime_global_cleanup mirrors arts_runtime_node_init: it must free the
/// per-thread arrays (deque/route_table/local_spin/thread_roles/keys/counter
/// sub-arrays/capture lists), the 8 remote route-table shards, the per-thread
/// local route tables, the shared scratch buf, the object-counter node storage,
/// and (non-C++) the event_dep_pool.  A missing free in cleanup, or an alloc in
/// node_init with no matching cleanup, shows up as an LSan leak at process
/// exit. The event_dep_pool is allocated-but-currently-unused (a documented
/// dead allocation) and must still be destroyed without leaking.
///
/// To make the teardown meaningful we exercise the allocating paths: create and
/// release several DBs (route-table inserts + object storage), create events
/// and EDTs, and run a small RW/RO chain, then shut down.  Under an ASan/LSan
/// build a teardown leak fails the test; otherwise it is a clean functional
/// smoke.
///
/// runtime_single (ASan/LSan), all configs (protocol-agnostic).

#include "arts.h"

#include <stdatomic.h>

#define N_DBS 6

void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *d = (int *)depv[0].ptr;
  bool ok = (d != NULL && d[0] == 7);
  if (ok) {
    arts_printf("  PASS: reader saw 7\n");
  } else {
    arts_printf("  FAIL: reader mismatch\n");
  }
  arts_shutdown();
}

void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *d = (int *)depv[0].ptr;
  if (d) {
    d[0] = 7;
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== runtime_teardown_leak ===\n");

  /* Several short-lived DBs to populate route tables + object storage. */
  for (int i = 0; i < N_DBS; i++) {
    void *p = NULL;
    arts_guid_t db =
        arts_db_create(&p, 64, ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
    ((int *)p)[0] = i;
    arts_db_release(db, DB_MODE_RW);
    arts_db_destroy(db);
  }

  /* A live RW/RO chain through an event + EDTs so allocation/free paths run. */
  void *p = NULL;
  arts_guid_t db =
      arts_db_create(&p, sizeof(int), ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
  ((int *)p)[0] = 0;
  arts_db_release(db, DB_MODE_RW);

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t w =
      arts_edt_create(writer_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, w, 0, DB_MODE_RW);

  arts_guid_t r =
      arts_edt_create(reader_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(db, r, 0, DB_MODE_RO);

  arts_printf("PASS runtime_teardown_leak: workload launched\n");
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  arts_printf("TEARDOWN_LEAK_DONE\n");
  return 0;
}
