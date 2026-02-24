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

/// @file utility_api.c
/// @brief Tests utility APIs: arts_get_current_node, arts_get_total_nodes,
///        arts_get_current_worker, arts_get_total_workers, arts_get_time_stamp,
///        arts_thread_safe_random, arts_get_current_guid,
///        arts_get_current_numa_domain, arts_get_total_numa_domains,
///        arts_yield.

#include "arts.h"

/// EDT that checks utility functions from within an EDT context.
void check_utils(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  bool all_pass = true;

  // arts_get_current_guid should return a non-NULL GUID.
  arts_guid_t my_guid = arts_get_current_guid();
  if (my_guid != NULL_GUID) {
    arts_printf("  PASS: get_current_guid = %lu (non-NULL)\n",
                (uint64_t)my_guid);
  } else {
    arts_printf("  FAIL: get_current_guid returned NULL_GUID\n");
    all_pass = false;
  }

  // arts_get_current_worker should be < total workers.
  unsigned int worker = arts_get_current_worker();
  unsigned int total_workers = arts_get_total_workers();
  if (worker < total_workers) {
    arts_printf("  PASS: current_worker=%u < total_workers=%u\n", worker,
                total_workers);
  } else {
    arts_printf("  FAIL: current_worker=%u >= total_workers=%u\n", worker,
                total_workers);
    all_pass = false;
  }

  // NUMA queries.
  unsigned int numa = arts_get_current_numa_domain();
  unsigned int total_numa = arts_get_total_numa_domains();
  if (total_numa >= 1) {
    arts_printf("  PASS: numa_domain=%u / %u\n", numa, total_numa);
  } else {
    arts_printf("  FAIL: total_numa_domains < 1\n");
    all_pass = false;
  }

  // arts_get_time_stamp monotonicity.
  uint64_t t1 = arts_get_time_stamp();
  // Brief busy loop.
  volatile unsigned int x = 0;
  for (unsigned int i = 0; i < 10000; i++) {
    x += i;
  }
  (void)x;
  uint64_t t2 = arts_get_time_stamp();
  if (t2 > t1) {
    arts_printf("  PASS: get_time_stamp monotonic (delta=%lu ns)\n", t2 - t1);
  } else {
    arts_printf("  FAIL: t2 (%lu) <= t1 (%lu)\n", t2, t1);
    all_pass = false;
  }

  // arts_thread_safe_random produces different values.
  uint64_t r1 = arts_thread_safe_random();
  uint64_t r2 = arts_thread_safe_random();
  uint64_t r3 = arts_thread_safe_random();
  if (r1 != r2 || r2 != r3) {
    arts_printf("  PASS: thread_safe_random produces varying values "
                "(%lu, %lu, %lu)\n",
                r1, r2, r3);
  } else {
    arts_printf("  FAIL: thread_safe_random returned same value 3 times\n");
    all_pass = false;
  }

  if (all_pass) {
    arts_printf("  ALL UTILITY TESTS PASSED\n");
  }
}

/// Test arts_yield in a loop.
void yield_waiter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  for (int i = 0; i < 3; i++) {
    arts_yield();
  }
  arts_printf("  PASS: arts_yield 3 times without crash\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== utility_api ===\n");

  // Node/count queries from main EDT.
  unsigned int node = arts_get_current_node();
  unsigned int total = arts_get_total_nodes();
  unsigned int workers = arts_get_total_workers();
  arts_printf("  Node %u / %u, workers=%u\n", node, total, workers);

  if (node == 0 && total >= 1 && workers >= 1) {
    arts_printf("  PASS: basic node/worker queries\n");
  } else {
    arts_printf("  FAIL: unexpected node/worker values\n");
  }

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  arts_edt_create_with_epoch(check_utils, 0, NULL, 0, epoch,
                             &(arts_hint_t){.route = 0});
  arts_edt_create_with_epoch(yield_waiter, 0, NULL, 0, epoch,
                             &(arts_hint_t){.route = 0});

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
