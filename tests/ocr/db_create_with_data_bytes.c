/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/// @file db_create_with_data_bytes.c
/// @brief T057 — byte-exact DB payload after acquire (strengthen the
///        scalar-only db_create_with_data integration test).
///
/// Exercises the install memcpy path (buffer.c `arts_db_buf_install` with a
/// non-NULL data_payload) end-to-end: create-time the creator fills the DB
/// buffer with a non-trivial byte pattern; a consumer EDT then acquires the DB
/// RO and asserts EVERY byte matches with memcmp (not just a scalar sample).
///
/// Covers buffer-census gap #1 (the memcpy branch's byte correctness was never
/// checked — Phase 1-4 of coherence_buffer_test all zero-init).  Several sizes
/// are exercised (including an unaligned tail and a single-byte DB) so the
/// memcpy length handling is fully driven.  Config-agnostic: the create+RO read
/// path is identical under all 6 build configs (single node — no transfer).
///
/// A stranded waiter is caught by the ctest TIMEOUT (no in-test watchdog).

#include "arts.h"

#include <stdint.h>
#include <string.h>

#define NUM_CASES 4u

/// Per-case parameters live in paramv:
///   paramv[0] = db_guid (to recompute the expected pattern)
///   paramv[1] = db_size in bytes
///   paramv[2] = case id (for the message)
#define PV_DB 0
#define PV_SIZE 1
#define PV_ID 2

/// Deterministic byte pattern keyed by the DB guid + byte index, so each case
/// has distinct bytes and a stale/zeroed/short copy is caught.
static unsigned char pattern_byte(uint64_t key, uint64_t i) {
  return (unsigned char)((key * 1315423911u) ^ (i * 2654435761u) ^ (i << 3));
}

/// Consumer: RO-acquire the DB and assert every byte matches the pattern.
void check_bytes(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t key = paramv[PV_DB];
  uint64_t size = paramv[PV_SIZE];
  unsigned int id = (unsigned int)paramv[PV_ID];
  const unsigned char *data = (const unsigned char *)depv[0].ptr;
  if (data == NULL) {
    arts_printf("  FAIL: case %u RO acquire returned NULL\n", id);
    arts_abort(1);
  }
  for (uint64_t i = 0; i < size; i++) {
    unsigned char want = pattern_byte(key, i);
    if (data[i] != want) {
      arts_printf("  FAIL: case %u byte %lu = 0x%02x want 0x%02x\n", id,
                  (unsigned long)i, data[i], want);
      arts_abort(1);
    }
  }
  arts_printf("  PASS: case %u byte-exact %lu bytes\n", id,
              (unsigned long)size);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_create_with_data_bytes ===\n");

  /* Sizes chosen to exercise: sub-word, exact-word, unaligned tail, and a
   * large buffer that spans well past the 64-byte buffer header. */
  const uint64_t sizes[NUM_CASES] = {1u, 7u, 64u, 1000u};

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  for (unsigned int c = 0; c < NUM_CASES; c++) {
    uint64_t size = sizes[c];
    arts_guid_t g = arts_guid_reserve(ARTS_GUID_DB, 0);
    unsigned char *p = (unsigned char *)arts_db_create_with_guid(
        g, size, ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
    for (uint64_t i = 0; i < size; i++) {
      p[i] = pattern_byte((uint64_t)g, i);
    }
    arts_db_release(g, DB_MODE_RW);

    uint64_t pv[3] = {(uint64_t)g, size, c};
    arts_guid_t e =
        arts_edt_create(check_bytes, 3, pv, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(g, e, 0, DB_MODE_RO);
  }

  arts_event_wait(fe);
  arts_printf("PASS: db_create_with_data_bytes %u cases byte-exact\n",
              NUM_CASES);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
