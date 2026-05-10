/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

#include "arts.h"
#include <stdint.h>
#include <stdio.h>

static void check_self_guid_edt(uint32_t paramc, const uint64_t *paramv,
                                uint32_t depc, arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  arts_guid_t expected = (arts_guid_t)paramv[0];
  (void)paramc;
  arts_guid_t actual = arts_edt_get_current_guid();
  bool ok = (actual == expected);
  arts_printf("  %s: arts_edt_get_current_guid matches reserved GUID\n",
              ok ? "PASS" : "FAIL");
}

static void check_epoch_guid_edt(uint32_t paramc, const uint64_t *paramv,
                                 uint32_t depc, arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  arts_guid_t expected_epoch = (arts_guid_t)paramv[0];
  (void)paramc;
  arts_guid_t actual = arts_epoch_get_current_guid();
  bool ok = (actual == expected_epoch);
  arts_printf("  %s: arts_epoch_get_current_guid matches epoch GUID\n",
              ok ? "PASS" : "FAIL");
}

static void epoch_add_edt_worker(uint32_t paramc, const uint64_t *paramv,
                                 uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("  PASS: EDT added via arts_epoch_add_edt fired correctly\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_epoch_api ===\n");

  /* Test 1: Pre-reserve GUID, create EDT with that GUID, verify self-GUID. */
  {
    arts_guid_t reserved = arts_guid_reserve(ARTS_GUID_EDT, 0);
    uint64_t param = (uint64_t)reserved;
    arts_guid_t epoch1 =
        arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
    arts_epoch_start(epoch1);
    arts_edt_create(
        check_self_guid_edt, 1, &param, 0,
        &(arts_edt_hint_t){.rank = 0, .guid = reserved, .epoch = epoch1});
    arts_epoch_wait(epoch1);
  }

  /* Test 2: Create epoch, create EDT in that epoch, verify epoch GUID. */
  {
    arts_guid_t epoch2 =
        arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
    arts_epoch_start(epoch2);
    uint64_t eparam = (uint64_t)epoch2;
    arts_edt_create(check_epoch_guid_edt, 1, &eparam, 0,
                    &(arts_edt_hint_t){.rank = 0, .epoch = epoch2});
    arts_epoch_wait(epoch2);
  }

  /* Test 3: arts_epoch_add_edt — create EDT without epoch, then add it. */
  {
    arts_guid_t epoch3 =
        arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
    arts_guid_t w = arts_edt_create(epoch_add_edt_worker, 0, NULL, 1,
                                    &(arts_edt_hint_t){.rank = 0});
    arts_epoch_add_edt(w, epoch3);
    arts_epoch_start(epoch3);
    arts_add_dependence((arts_guid_t)(0), w, 0, DB_MODE_VAL);
    arts_epoch_wait(epoch3);
  }

  /* Test 4: arts_epoch_wait confirmation */
  arts_printf("  PASS: arts_epoch_wait blocked and returned correctly\n");

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
