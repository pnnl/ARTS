/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

#include "arts.h"
#include <stdio.h>

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_shutdown();
}

static void once_receiver(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: ONCE event delivered to waiter\n");
}

static void idem_late_receiver(uint32_t paramc, const uint64_t *paramv,
                               uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: IDEMPOTENT late dep fires immediately\n");
}

static void sticky_late_receiver(uint32_t paramc, const uint64_t *paramv,
                                 uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: STICKY late dep fires immediately\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== event_hint_presets ===\n");

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  /* Test 1: ONCE — add dep first, then satisfy */
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_ONCE;
    arts_guid_t ev = arts_event_create(&h);
    arts_guid_t waiter =
        arts_edt_create(once_receiver, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ev, waiter, 0, DB_MODE_RW);
    arts_event_satisfy(ev, NULL_GUID);
  }

  /* Test 2: IDEMPOTENT — satisfy first, then add late dep */
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_IDEMPOTENT;
    arts_guid_t ev = arts_event_create(&h);
    arts_event_satisfy(ev, NULL_GUID);
    arts_guid_t late =
        arts_edt_create(idem_late_receiver, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ev, late, 0, DB_MODE_RW);
  }

  /* Test 3: STICKY — satisfy first, then add late dep */
  {
    arts_event_hint_t h = ARTS_EVENT_HINT_STICKY;
    arts_guid_t ev = arts_event_create(&h);
    arts_event_satisfy(ev, NULL_GUID);
    arts_guid_t late =
        arts_edt_create(sticky_late_receiver, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ev, late, 0, DB_MODE_RW);
  }

  /* Test 4: labeled-GUID event */
  {
    arts_guid_t reserved = arts_guid_reserve(ARTS_GUID_EVENT, 0);
    arts_event_hint_t h = ARTS_EVENT_HINT_ONCE;
    h.guid = reserved;
    arts_guid_t ev = arts_event_create(&h);
    bool ok = (ev == reserved);
    arts_printf("  %s: labeled-GUID event honors pre-reserved GUID\n",
                ok ? "PASS" : "FAIL");
    arts_event_satisfy(ev, NULL_GUID);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
