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
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/

/// @file mrsw_empty_map_transfer.c
/// @brief MRSW EAGER empty-map owner->owner transfer header
/// (runtime_multinode).
///
/// arts_db_send_ownership_response always emits an 8-byte map header even when
/// the owner-side dedup map is empty (EAGER's last_sent_version is always NULL,
/// so it ships an empty 8-byte header).  The receiver always reconstructs
/// map_size>=8 from that header; if the empty header were omitted, the
/// receiver's data_size = total - sizeof(hdr) - map_size would UNDERFLOW.  This
/// test drives an EAGER owner->owner transfer of a TINY DB (so data_size is
/// itself near zero, maximizing the underflow risk if the map header is
/// mishandled).
///
/// Construction: a small DB (one byte) is RW-acquired in turn by a chain of
/// EDTs on different ranks, forcing repeated owner->owner ownership transfers.
/// Each writer increments the single byte (mod 256); the final reader asserts
/// the byte equals the number of writes mod 256.  A map_size/data_size
/// underflow corrupts the transferred payload -> wrong final byte or a
/// SIGSEGV/hang (ctest TIMEOUT).
///
/// EAGER + MRSW only (the empty-8-byte-header invariant is the EAGER ship path;
/// LAZY serializes a real map).  Requires >= 2 ranks; SKIPs on 1n.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW) || !defined(ARTS_TIMING_EAGER)

int main(void) {
  printf("SKIP mrsw_empty_map_transfer: MRSW+EAGER-only\n");
  return 0;
}

#else

#define N_WRITES 64u

static void byte_inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint8_t *d = (uint8_t *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (uint8_t)(d[0] + 1u);
  }
}

static void final_check_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint8_t *d = (const uint8_t *)depv[0].ptr;
  uint8_t v = (d != NULL) ? d[0] : 0u;
  uint8_t expect = (uint8_t)(N_WRITES & 0xffu);
  if (v != expect) {
    (void)fprintf(stderr, "FAIL: final byte %u != %u (map/data underflow)\n",
                  (unsigned)v, (unsigned)expect);
    arts_abort(1);
  }
  arts_printf("PASS: mrsw_empty_map_transfer final=%u\n", (unsigned)expect);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mrsw_empty_map_transfer ===\n");

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2u) {
    arts_printf("SKIP: mrsw_empty_map_transfer requires >= 2 ranks\n");
    arts_shutdown();
    return;
  }

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint8_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint8_t *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  /* final_check reads depv[0].ptr (db) -> db is slot 0 (RO); the writers'
   * finish event fe is slot 1 (NULL).  depc=2 so the passive immediate db
   * satisfy does NOT fire final_check early — it also waits for fe (every
   * byte_inc writer done).  (Wiring both deps at slot 0 with depc=1 let the db
   * satisfy fire final_check before any writer ran -> it read the initial 0.)
   */
  arts_guid_t fin = arts_edt_create(final_check_edt, 0, NULL, 2,
                                    &(arts_edt_hint_t){.rank = 0});
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(db, fin, 0, DB_MODE_RO);
  arts_add_dependence(fe, fin, 1, DB_MODE_NULL);

  /* Each write on a different rank forces an owner->owner EAGER transfer of the
   * tiny DB; the DB RW chain serializes them. */
  for (unsigned int i = 0; i < N_WRITES; i++) {
    unsigned int rank = (i + 1u) % nranks; /* walk the ring, off home */
    arts_guid_t w =
        arts_edt_create(byte_inc_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = rank, .finish_event = fe});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRSW && ARTS_TIMING_EAGER */
