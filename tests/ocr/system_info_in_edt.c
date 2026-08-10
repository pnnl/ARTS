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

/// @file system_info_in_edt.c
/// @brief Single-node smoke of the in-EDT introspection getters in
///        libs/src/core/utils/system_info.c, run from a real EDT context (where
///        current_edt and arts_thread_info are populated):
///
///          - arts_edt_get_current_guid() == the GUID this EDT was created with
///            (current_edt->guid), and non-NULL inside an EDT.
///          - arts_get_current_worker()   <  arts_get_workers_per_rank()
///          - arts_get_current_numa_domain() <  arts_get_total_numa_domains()
///          - arts_get_total_numa_domains() >= 1
///          - arts_get_gpus_per_rank()    is readable (>=0 trivially for an
///            unsigned; on a CPU build it is 0).
///
/// The current-GUID check is stronger than utility_api.c's "non-NULL" smoke:
/// we pre-reserve the EDT's GUID via the hint and confirm the in-EDT getter
/// returns exactly that value.  Protocol-agnostic, no DB coherence involved.

#include <stdint.h>

#include "arts.h"

/// paramv[0] holds the GUID this EDT was created with (passed by reference as a
/// 64-bit value so the body can compare it against the in-EDT getter).
void info_probe(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t expected = (arts_guid_t)paramv[0];
  bool pass = true;

  arts_guid_t guid = arts_edt_get_current_guid();
  if (guid == NULL_GUID) {
    arts_printf("  FAIL: get_current_guid returned NULL_GUID inside EDT\n");
    pass = false;
  } else if (guid != expected) {
    arts_printf("  FAIL: get_current_guid=%lu != reserved guid=%lu\n",
                (uint64_t)guid, (uint64_t)expected);
    pass = false;
  } else {
    arts_printf("  PASS: get_current_guid == reserved guid (%lu)\n",
                (uint64_t)guid);
  }

  unsigned int worker = arts_get_current_worker();
  unsigned int per_rank = arts_get_workers_per_rank();
  if (per_rank >= 1 && worker < per_rank) {
    arts_printf("  PASS: current_worker=%u < workers_per_rank=%u\n", worker,
                per_rank);
  } else {
    arts_printf("  FAIL: current_worker=%u workers_per_rank=%u\n", worker,
                per_rank);
    pass = false;
  }

  unsigned int numa = arts_get_current_numa_domain();
  unsigned int total_numa = arts_get_total_numa_domains();
  if (total_numa >= 1 && numa < total_numa) {
    arts_printf("  PASS: numa_domain=%u < total_numa_domains=%u\n", numa,
                total_numa);
  } else {
    arts_printf("  FAIL: numa_domain=%u total_numa_domains=%u\n", numa,
                total_numa);
    pass = false;
  }

  /* gpus_per_rank: just confirm it is readable from EDT context.  On a CPU
   * build this is 0; we accept any value and only require the call not to
   * misbehave. */
  unsigned int gpus = arts_get_gpus_per_rank();
  arts_printf("  INFO: gpus_per_rank=%u\n", gpus);

  if (pass) {
    arts_printf("PASS system_info_in_edt\n");
  } else {
    arts_printf("FAIL system_info_in_edt\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== system_info_in_edt ===\n");

  /* Pre-reserve the probe EDT's GUID so the body can verify the in-EDT getter
   * returns exactly this value (current_edt->guid). */
  arts_guid_t pre = arts_guid_reserve(ARTS_GUID_EDT, 0);
  uint64_t pv = (uint64_t)pre;
  arts_edt_create(info_probe, 1, &pv, 0,
                  &(arts_edt_hint_t){.rank = 0, .guid = pre});
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
