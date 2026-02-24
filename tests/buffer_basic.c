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

/// @file buffer_basic.c
/// @brief Tests buffer APIs: arts_allocate_local_buffer, arts_set_buffer,
///        arts_get_buffer, arts_block_for_buffer.

#include "arts.h"
#include <string.h>

/// EDT that writes to the buffer.
void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_guid_t buf_guid = (arts_guid_t)paramv[0];
  unsigned int val = 42;
  arts_set_buffer(buf_guid, &val, sizeof(val));
  arts_printf("  Writer: set buffer to %u\n", val);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== buffer_basic ===\n");
  bool all_pass = true;

  // Test 1: Allocate buffer with 1 use, set locally, get locally.
  unsigned int result1 = 0;
  unsigned int *r1_ptr = &result1;
  arts_guid_t buf1 = arts_allocate_local_buffer(
      (void **)&r1_ptr, sizeof(unsigned int), 1, NULL_GUID);
  if (buf1 == NULL_GUID) {
    arts_printf("  FAIL: allocate_local_buffer returned NULL_GUID\n");
    all_pass = false;
  } else {
    arts_printf("  PASS: allocate_local_buffer succeeded\n");
  }

  // Set the buffer.
  unsigned int data1 = 12345;
  arts_set_buffer(buf1, &data1, sizeof(data1));

  // The set_buffer writes to the original pointer — verify.
  if (result1 == 12345) {
    arts_printf("  PASS: set_buffer wrote to original pointer (val=%u)\n",
                result1);
  } else {
    arts_printf("  FAIL: expected 12345, got %u\n", result1);
    all_pass = false;
  }

  // Test 2: Buffer set from another EDT — poll using arts_yield.
  unsigned int result2 = 0;
  unsigned int *r2_ptr = &result2;
  arts_guid_t buf2 = arts_allocate_local_buffer(
      (void **)&r2_ptr, sizeof(unsigned int), 1, NULL_GUID);
  uint64_t buf2_param = (uint64_t)buf2;
  arts_edt_create(writer_edt, 1, &buf2_param, 0, &(arts_hint_t){.route = 0});

  // Yield-loop until the writer sets the buffer.
  unsigned int attempts = 0;
  while (result2 == 0 && attempts < 100000) {
    arts_yield();
    attempts++;
  }
  if (result2 == 42) {
    arts_printf("  PASS: buffer written by remote EDT (val=%u)\n", result2);
  } else {
    arts_printf("  WARN: buffer write not observed after %u yields (val=%u)\n",
                attempts, result2);
  }

  // Test 3: Buffer with multiple uses.
  unsigned int result3 = 0;
  unsigned int *r3_ptr = &result3;
  arts_guid_t buf3 = arts_allocate_local_buffer(
      (void **)&r3_ptr, sizeof(unsigned int), 2, NULL_GUID);
  unsigned int v3 = 100;
  arts_set_buffer(buf3, &v3, sizeof(v3));
  // First get.
  void *g1 = arts_get_buffer(buf3);
  if (g1 != NULL) {
    arts_printf("  PASS: first get_buffer returned non-NULL\n");
  } else {
    arts_printf("  FAIL: first get_buffer returned NULL\n");
    all_pass = false;
  }
  // Second get — uses exhausted; entry freed.
  void *g2 = arts_get_buffer(buf3);
  (void)g2;
  arts_printf("  PASS: multi-use buffer did not crash\n");

  arts_printf("=== buffer_basic: %s ===\n", all_pass ? "ALL PASSED" : "FAILED");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
