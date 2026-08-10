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

/// @file paramv_memcpy.c
/// @brief Tests passing different data types through paramv using memcpy
///        (the correct way), verifying no truncation occurs.

#include "arts.h"
#include <string.h>

/// Test passing double through paramv via memcpy.
void check_double(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  double val;
  memcpy(&val, &paramv[0], sizeof(double));
  // Check approximate equality (3.14159...).
  bool ok = (val > 3.14 && val < 3.15);
  if (ok) {
    arts_printf("  PASS: double via memcpy: %f\n", val);
  } else {
    arts_printf("  FAIL: double via memcpy: %f\n", val);
  }
}

/// Test passing float (smaller than uint64_t).
void check_float(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  float val;
  memcpy(&val, &paramv[0], sizeof(float));
  bool ok = (val > 2.71f && val < 2.72f);
  if (ok) {
    arts_printf("  PASS: float via memcpy: %f\n", (double)val);
  } else {
    arts_printf("  FAIL: float via memcpy: %f\n", (double)val);
  }
}

/// Test passing int32_t.
void check_int32(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  int32_t val;
  memcpy(&val, &paramv[0], sizeof(int32_t));
  bool ok = (val == -12345);
  if (ok) {
    arts_printf("  PASS: int32 via memcpy: %d\n", val);
  } else {
    arts_printf("  FAIL: int32 via memcpy: %d\n", val);
  }
}

/// Test passing a struct through paramv.
struct test_struct_s {
  uint32_t a;
  uint16_t b;
  uint8_t c;
  uint8_t d;
};

void check_struct(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  struct test_struct_s s;
  memcpy(&s, &paramv[0], sizeof(struct test_struct_s));
  bool ok = (s.a == 0xDEADBEEF && s.b == 0x1234 && s.c == 0xAB && s.d == 0xCD);
  if (ok) {
    arts_printf("  PASS: struct via memcpy\n");
  } else {
    arts_printf("  FAIL: struct via memcpy\n");
  }
}

/// Test passing multiple values.
void check_multi(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  bool ok = (paramc == 3);
  if (ok) {
    ok = (paramv[0] == 100 && paramv[1] == 200 && paramv[2] == 300);
  }
  if (ok) {
    arts_printf("  PASS: multi-param [%lu, %lu, %lu]\n",
                (unsigned long)paramv[0], (unsigned long)paramv[1],
                (unsigned long)paramv[2]);
  } else {
    arts_printf("  FAIL: multi-param\n");
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== paramv_memcpy ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  // Test 1: double.
  double d = 3.14159265358979;
  uint64_t d_param = 0;
  memcpy(&d_param, &d, sizeof(double));
  arts_edt_create(check_double, 1, &d_param, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = fe});

  // Test 2: float.
  float f = 2.71828f;
  uint64_t f_param = 0;
  memcpy(&f_param, &f, sizeof(float));
  arts_edt_create(check_float, 1, &f_param, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = fe});

  // Test 3: int32_t.
  int32_t i = -12345;
  uint64_t i_param = 0;
  memcpy(&i_param, &i, sizeof(int32_t));
  arts_edt_create(check_int32, 1, &i_param, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = fe});

  // Test 4: struct.
  struct test_struct_s s = {.a = 0xDEADBEEF, .b = 0x1234, .c = 0xAB, .d = 0xCD};
  uint64_t s_param = 0;
  memcpy(&s_param, &s, sizeof(struct test_struct_s));
  arts_edt_create(check_struct, 1, &s_param, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = fe});

  // Test 5: Multiple uint64_t values.
  uint64_t multi[3] = {100, 200, 300};
  arts_edt_create(check_multi, 3, multi, 0,
                  &(arts_edt_hint_t){.rank = 0, .finish_event = fe});

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
