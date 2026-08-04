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

/// @file val_snapshot_request.c
/// @brief Differential RO snapshot serve: HOME home-master vs OWNER redirect.
///
/// arts_handler_db_snapshot_request (home side) diverges by placement:
///   - HOME: home holds the canonical buffer (kept current by synchronous
///     PUBLISH), serves the snapshot itself, and dedups foreign requesters
///     via update_cached_version_max (NO_DATA when the requester's watermark is
///     current).
///   - OWNER: home does NOT hold data; it records the requester in cached_ranks,
///     loads rw_holder (acquire), and sends REDIRECT_RO to the owner — even
///     mid-transfer rw_holder names the OLD owner which still serves the
///     redirect from its retained buffer+map.
/// Either path MUST deliver the current value to every RO reader.
///
/// SCENARIO.  A home RW DataBlock is first written by a REMOTE RW owner (so
/// home is NOT the owner and the RO path must go through snapshot_request).
/// Then multiple RO readers — on home AND on other ranks, including the SAME
/// reader rank twice to hit the dedup/watermark branch — each must observe the
/// owner's value.  Repeated over several writer/reader rounds to advance the
/// version watermark.  A stale or NULL snapshot => arts_abort; a stranded RO
/// waiter => ctest TIMEOUT.
///
/// VAL-only, runs under both placements (the contrast is the whole point — the
/// scenario is built once and the build dir selects the serve path).  Clean
/// skip otherwise.  Needs 2+ ranks for a remote owner / real redirect; 1n
/// serves locally and still passes.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_VAL)

int main(void) {
  printf("SKIP val_snapshot_request: VAL-only\n");
  return 0;
}

#else

#define ROUNDS 32u
#define VAL_BASE 0x5A000000u

static void wr_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

static void rd_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr,
                  "FAIL val_snapshot_request: expected 0x%x got 0x%x\n",
                  expect, d ? d[0] : 0u);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

#if defined(ARTS_WRITE_POLICY_WT)
  const char *placement = "HOME";
#else
  const char *placement = "OWNER";
#endif
  arts_printf("=== val_snapshot_request (%s) ===\n", placement);

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W = (nranks > 1) ? 1u : 0u;

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  for (unsigned int r = 0; r < ROUNDS; r++) {
    uint64_t v = (uint64_t)(VAL_BASE + r);

    /* Remote RW owner writes -> home is not the owner. */
    arts_guid_t ew = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w = arts_edt_create(
        wr_edt, 1, &v, 1, &(arts_edt_hint_t){.rank = W, .finish_event = ew});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
    arts_event_wait(ew);

    /* RO readers on home + a remote, plus the remote reader rank repeated to
     * hit the dedup/watermark branch (HOME update_cached_version_max /
     * OWNER monotonic redirect watermark). */
    unsigned int rdr = (nranks > 2) ? 2u : ((nranks > 1) ? 1u : 0u);
    unsigned int reader_ranks[3] = {0u, rdr, rdr};
    arts_guid_t er = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int k = 0; k < 3; k++) {
      arts_guid_t rd = arts_edt_create(
          rd_edt, 1, &v, 1,
          &(arts_edt_hint_t){.rank = reader_ranks[k], .finish_event = er});
      arts_add_dependence(db, rd, 0, DB_MODE_RO);
    }
    arts_event_wait(er);
  }

  arts_printf("PASS val_snapshot_request (%s) rounds=%u\n", placement, ROUNDS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif
