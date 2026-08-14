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

/// @file guid_keygen_table_coupling.c
/// @brief T038 — the keygen divisor (total_thread_count) must equal the number
///        of per-thread key tables; if non-worker threads collapsed onto one
///        slot the per-thread key blocks would overlap and produce DUPLICATE
///        GUIDs.  Regression for that past bug.
///
/// set_guid_generator_after_parallel_start sets
///   keys_per_thread = global_guid_on / (total_thread_count * rank_count)
/// and the per-thread key encoding is value + keys_per_thread *
/// global_guid_thread_id[thread].  arts_guid_key_generator_init sets
/// num_tables = total_thread_count and global_guid_thread_id per thread.  The
/// two must use the SAME total_thread_count divisor; otherwise two threads'
/// key blocks alias and emit identical GUIDs.
///
/// The test spawns many reserver EDTs per rank (more than workers, so the
/// scheduler spreads them across every worker thread) that each reserve a batch
/// of GUIDs on their own rank (alternating EVENT — the flat per-thread key
/// partition this regression was written for — and DB, whose keys come from
/// the chunk-leased per-creator seq slices), writing them into a per-rank
/// array DB.
/// A checker EDT then asserts global per-rank uniqueness.  Overlapping key
/// blocks (the past bug) would surface as duplicates here.
///
/// Multinode by registration (runs per rank independently; correct single-node
/// too — but registered with the multinode variants per the census).

#include "arts.h"
#include <stdint.h>

#define RESERVERS_PER_RANK 32 /* > workers so all worker threads participate   \
                               */
#define GUIDS_PER_RESERVER 16
#define TOTAL_PER_RANK (RESERVERS_PER_RANK * GUIDS_PER_RESERVER)

/// Reserver EDT: depv[0] is the per-rank collection DB (RW).  paramv[0] = base
/// slot index into the array.  Reserves GUIDS_PER_RESERVER GUIDs and stores
/// them contiguously starting at the base slot.
static void reserver_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int base = (unsigned int)paramv[0];
  uint64_t *slots = (uint64_t *)depv[0].ptr;
  unsigned int me = arts_get_current_rank();
  if (!slots) {
    return;
  }
  for (unsigned int i = 0; i < GUIDS_PER_RESERVER; i++) {
    /* Alternate kinds so BOTH allocators stay pinned: EVENT exercises the
     * flat per-thread key partition (the original divisor-coupling bug
     * class), DB exercises the chunk-leased seq slices.  Kind bits keep the
     * two value sets disjoint, so the uniqueness sweep needs no change. */
    arts_guid_kind_t kind = (i & 1u) ? ARTS_GUID_EVENT : ARTS_GUID_DB;
    arts_guid_t g = arts_guid_reserve(kind, me);
    slots[base + i] = (uint64_t)g;
  }
}

/// Checker EDT: depv[0] is the per-rank collection DB (RO).  Asserts every
/// reserved GUID is non-NULL and globally unique on this rank.
static void checker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint64_t *slots = (const uint64_t *)depv[0].ptr;
  unsigned int me = arts_get_current_rank();
  if (!slots) {
    arts_printf("  FAIL: rank %u checker got NULL collection\n", me);
    return;
  }

  bool ok = true;
  for (unsigned int i = 0; i < TOTAL_PER_RANK && ok; i++) {
    if (slots[i] == (uint64_t)NULL_GUID) {
      arts_printf("  FAIL: rank %u slot %u never filled (NULL_GUID)\n", me, i);
      ok = false;
      break;
    }
    for (unsigned int j = i + 1; j < TOTAL_PER_RANK; j++) {
      if (slots[i] == slots[j]) {
        arts_printf("  FAIL: rank %u duplicate GUID at slots %u,%u "
                    "(key-block aliasing)\n",
                    me, i, j);
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    arts_printf("  PASS: rank %u all %d reserved GUIDs unique (no key-block "
                "aliasing)\n",
                me, TOTAL_PER_RANK);
  }
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_shutdown();
}

/// Per-rank driver: builds this rank's collection DB, fans out reservers under
/// an inner finish scope, and chains a checker after they complete.
static void rank_driver_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t outer = (arts_guid_t)paramv[0];
  unsigned int me = arts_get_current_rank();

  /* Per-rank collection DB, homed on this rank. */
  void *ptr = NULL;
  arts_guid_t coll =
      arts_db_create(&ptr, TOTAL_PER_RANK * sizeof(uint64_t), ARTS_DB,
                     ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = me});
  uint64_t *slots = (uint64_t *)ptr;
  for (unsigned int i = 0; i < TOTAL_PER_RANK; i++) {
    slots[i] = (uint64_t)NULL_GUID;
  }
  arts_db_release(coll, DB_MODE_RW);

  /* Checker runs after all reservers (slot 1 = inner finish), joined to outer
   * so shutdown waits for it. */
  arts_guid_t checker =
      arts_edt_create(checker_edt, 0, NULL, 2,
                      &(arts_edt_hint_t){.rank = me, .finish_event = outer});
  arts_add_dependence(coll, checker, 0, DB_MODE_RO);

  arts_guid_t inner = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(inner, checker, 1, DB_MODE_NULL);

  /* Fan out reservers inside the inner finish scope, all on this rank. */
  for (unsigned int k = 0; k < RESERVERS_PER_RANK; k++) {
    uint64_t base = (uint64_t)(k * GUIDS_PER_RESERVER);
    arts_guid_t r =
        arts_edt_create(reserver_edt, 1, &base, 1,
                        &(arts_edt_hint_t){.rank = me, .finish_event = inner});
    arts_add_dependence(coll, r, 0, DB_MODE_RW);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int nrank = arts_get_total_ranks();
  arts_printf("=== guid_keygen_table_coupling (%u ranks) ===\n", nrank);

  arts_guid_t shut = arts_edt_create(shutdown_edt, 0, NULL, 1, NULL);
  arts_guid_t outer = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(outer, shut, 0, DB_MODE_NULL);

  uint64_t op = (uint64_t)outer;
  for (unsigned int r = 0; r < nrank; r++) {
    arts_edt_create(rank_driver_edt, 1, &op, 0,
                    &(arts_edt_hint_t){.rank = r, .finish_event = outer});
  }
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
