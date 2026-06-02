/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

#include "arts.h"
#include <stdint.h>
#include <stdio.h>

#define DB_SIZE (sizeof(uint64_t) * 4)

static void ro_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc;
  const uint64_t *data = (const uint64_t *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 0xDEADULL && data[1] == 0xBEEFULL);
  arts_printf("  %s: RO dep sees written values\n", ok ? "PASS" : "FAIL");
}

static void rw_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc;
  uint64_t *data = (uint64_t *)depv[0].ptr;
  if (data) { data[0] = 0xCAFEULL; data[1] = 0xBABEULL; }
}

static void rw_verifier(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc;
  const uint64_t *data = (const uint64_t *)depv[0].ptr;
  bool ok = (data != NULL && data[0] == 0xCAFEULL && data[1] == 0xBABEULL);
  arts_printf("  %s: RW modification visible via RO after RW releases\n",
              ok ? "PASS" : "FAIL");
}

static void val_receiver(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc;
  uint64_t received = (uint64_t)depv[0].guid;
  bool ok = (received == 0x42ULL);
  arts_printf("  %s: VAL dep delivers raw value (got 0x%lx)\n",
              ok ? "PASS" : "FAIL", (unsigned long)received);
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc; (void)depv;
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc; (void)paramv; (void)depc; (void)depv;

  arts_printf("=== edt_dep_variants ===\n");

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), shut, 0);
  arts_epoch_start(epoch);

  /* Test 1: RO */
  void *ptr1 = NULL;
  arts_guid_t db1 = arts_db_create(&ptr1, DB_SIZE, ARTS_DB,
                                   ARTS_DB_PROP_NONE, NULL);
  ((uint64_t *)ptr1)[0] = 0xDEADULL;
  ((uint64_t *)ptr1)[1] = 0xBEEFULL;
  arts_db_release(db1, DB_MODE_RW);

  arts_guid_t r1 = arts_edt_create(ro_reader, 0, NULL, 1,
                                   &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(db1, r1, 0, DB_MODE_RO);

  /* Test 2: RW chain — writer modifies, verifier (inner-epoch finish) reads */
  void *ptr2 = NULL;
  arts_guid_t db2 = arts_db_create(&ptr2, DB_SIZE, ARTS_DB,
                                   ARTS_DB_PROP_NONE, NULL);
  ((uint64_t *)ptr2)[0] = 0ULL;
  arts_db_release(db2, DB_MODE_RW);

  /* verifier: depc=2 (slot 0=DB RO dep, slot 1=inner epoch signal) */
  arts_guid_t v2 = arts_edt_create(rw_verifier, 0, NULL, 2,
                                   &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(db2, v2, 0, DB_MODE_RO);

  arts_guid_t inner2 = arts_epoch_create(arts_get_current_rank(), v2, 1);
  arts_epoch_start(inner2);

  arts_guid_t w2 = arts_edt_create(rw_writer, 0, NULL, 1,
                                   &(arts_edt_hint_t){.rank = 0, .epoch = inner2});
  arts_add_dependence(db2, w2, 0, DB_MODE_RW);
  (void)w2;

  /* Test 3: VAL — raw value 0x42 as dependency */
  arts_guid_t v3 = arts_edt_create(val_receiver, 0, NULL, 1,
                                   &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence((arts_guid_t)(0x42ULL), v3, 0, DB_MODE_VAL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
