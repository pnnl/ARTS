/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

#include "arts.h"
#include <stdio.h>
#include <stdint.h>

#define SENTINEL 0xABCD1234ULL
#define DB_SIZE (sizeof(uint64_t))

static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc;
  const uint64_t *data = (const uint64_t *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == SENTINEL);
  arts_printf("  %s: labeled-GUID DB data accessible via reserved GUID\n",
              ok ? "PASS" : "FAIL");
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc; (void)depv;
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc; (void)depv;

  arts_printf("=== db_labeled_guid ===\n");

  arts_guid_t reserved = arts_guid_reserve(ARTS_GUID_DB, 0);

  uint64_t *ptr = (uint64_t *)arts_db_create_with_guid(
      reserved, DB_SIZE, ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  ptr[0] = SENTINEL;
  arts_db_release(reserved, DB_MODE_RW);

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  arts_guid_t r = arts_edt_create(reader_edt, 0, NULL, 1,
                                  &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(reserved, r, 0, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
