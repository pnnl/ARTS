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

/// @file edt_dep_variants.c
/// @brief Tests EDT creation with has_depv=false variants:
///        arts_edt_create_dep (no depv), arts_edt_create_with_guid_dep,
///        arts_edt_create_with_epoch_dep.

#include "arts.h"

/// EDT with has_depv=false: receives signal_edt_value but depv is NULL.
void no_depv_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depv;
  // depc still counts the signals needed, but depv is NULL.
  arts_printf("  PASS: edt_create_dep (has_depv=false) ran, depc=%u\n", depc);
}

/// EDT with has_depv=true: depv is allocated and filled.
void with_depv_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  uint64_t val = (uint64_t)depv[0].guid;
  bool ok = (depc == 1 && val == 999);
  if (ok) {
    arts_printf("  PASS: edt_create_dep (has_depv=true) depv[0]=%lu\n",
                (unsigned long)val);
  } else {
    arts_printf("  FAIL: edt_create_dep depv[0] mismatch\n");
  }
}

/// Test with_guid_dep.
void guid_dep_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depv;
  arts_printf("  PASS: edt_create_with_guid_dep ran, depc=%u\n", depc);
}

/// Test with_epoch_dep.
void epoch_dep_task(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depv;
  arts_printf("  PASS: edt_create_with_epoch_dep ran, depc=%u\n", depc);
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_dep_variants ===\n");

  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  // Test 1: arts_edt_create_dep with has_depv=false.
  arts_guid_t e1 = arts_edt_create_dep(no_depv_task, 0, NULL, 2, false,
                                       &(arts_hint_t){.route = 0});
  arts_signal_edt_value(e1, 0, 1);
  arts_signal_edt_value(e1, 1, 2);

  // Test 2: arts_edt_create_dep with has_depv=true.
  arts_guid_t e2 = arts_edt_create_dep(with_depv_task, 0, NULL, 1, true,
                                       &(arts_hint_t){.route = 0});
  arts_signal_edt_value(e2, 0, 999);

  // Test 3: arts_edt_create_with_guid_dep.
  arts_guid_t pre_guid = arts_guid_reserve(ARTS_EDT, 0);
  arts_edt_create_with_guid_dep(guid_dep_task, pre_guid, 0, NULL, 1, false);
  arts_signal_edt_value(pre_guid, 0, 1);

  // Test 4: arts_edt_create_with_epoch_dep.
  arts_guid_t e4 = arts_edt_create_with_epoch_dep(
      epoch_dep_task, 0, NULL, 1, epoch, false, &(arts_hint_t){.route = 0});
  arts_signal_edt_value(e4, 0, 1);

  arts_wait_on_handle(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
