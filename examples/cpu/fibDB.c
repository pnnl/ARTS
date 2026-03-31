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
#include <assert.h>
#include <stdlib.h>

#include "arts.h"
#include "arts/memory/db.h"

#define CXL_DB 1

uint64_t start = 0;

void fib_join(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;

  /* Validate DB pointers before accessing them */
  assert(depv[0].ptr != NULL && "First DB pointer is valid");
  assert(depv[1].ptr != NULL && "Second DB pointer is valid");

  /* Extract values from DBs */
  unsigned int x = *(unsigned int *)depv[0].ptr;
  unsigned int y = *(unsigned int *)depv[1].ptr;

  /* Create a DB to store the result */
  unsigned int *result_ptr;
#if CXL_DB
  arts_guid_t result_guid = arts_db_create((void **)&result_ptr,
                                           sizeof(unsigned int),
                                           ARTS_DB_CXL, NULL);
#else
  arts_guid_t result_guid = arts_db_create((void **)&result_ptr,
                                           sizeof(unsigned int),
                                           ARTS_DB_DEFAULT, NULL);
#endif /* CXL_DB */
  assert(result_ptr && "Result ptr not NULL");
  *result_ptr = x + y;

#if CXL_DB
  arts_cxl_producer_flush(result_guid);
#endif /* CXL_DB */
  /* Signal the parent EDT with the result DB */
  arts_signal_edt((arts_guid_t)paramv[0], (uint32_t)paramv[1], result_guid,
                  DB_MODE_RO);
}

void fib_fork(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;

  unsigned int next =
      (arts_get_current_node() + 1) % arts_get_total_nodes();

  arts_guid_t guid = (arts_guid_t)paramv[0];
  unsigned int slot = (unsigned int)paramv[1];

  assert(depv[0].ptr && "depv[0] ptr not null");
  /* Get the input number from DB */
  unsigned int num = *(unsigned int *)depv[0].ptr;

  if (num < 2) {
    arts_signal_edt(guid, slot, depv[0].guid, DB_MODE_RO);
  } else {
    /* Create a join EDT that will combine results */
    arts_guid_t join_guid =
        arts_edt_create(fib_join, paramc, paramv, 2,
                        &(arts_hint_t){.route = arts_get_current_node()});

    /* Create DB for n-1 */
    unsigned int *n1_ptr;
#if CXL_DB
    arts_guid_t n1_guid = arts_db_create((void **)&n1_ptr,
                                         sizeof(unsigned int),
                                         ARTS_DB_CXL, NULL);
#else
    arts_guid_t n1_guid = arts_db_create((void **)&n1_ptr,
                                         sizeof(unsigned int),
                                         ARTS_DB_DEFAULT, NULL);
#endif /* CXL_DB */
    assert(n1_ptr && "n1_ptr not NULL");
    *n1_ptr = num - 1;
#if CXL_DB
    arts_cxl_producer_flush(n1_guid);
#endif /* CXL_DB */

    /* Create DB for n-2 */
    unsigned int *n2_ptr;
#if CXL_DB
    arts_guid_t n2_guid = arts_db_create((void **)&n2_ptr,
                                         sizeof(unsigned int),
                                         ARTS_DB_CXL, NULL);
#else
    arts_guid_t n2_guid = arts_db_create((void **)&n2_ptr,
                                         sizeof(unsigned int),
                                         ARTS_DB_DEFAULT, NULL);
#endif /* CXL_DB */
    assert(n2_ptr && "n2_ptr not NULL");
    *n2_ptr = num - 2;
#if CXL_DB
    arts_cxl_producer_flush(n2_guid);
#endif /* CXL_DB */

    /* Create first child task with n-1 */
    uint64_t args1[2] = {(uint64_t)join_guid, 0};
    arts_guid_t fib1 =
        arts_edt_create(fib_fork, 2, args1, 1, &(arts_hint_t){.route = next});
    arts_signal_edt(fib1, 0, n1_guid, DB_MODE_RO);

    /* Create second child task with n-2 */
    uint64_t args2[2] = {(uint64_t)join_guid, 1};
    arts_guid_t fib2 =
        arts_edt_create(fib_fork, 2, args2, 1, &(arts_hint_t){.route = next});
    arts_signal_edt(fib2, 0, n2_guid, DB_MODE_RO);
  }
}

void fib_done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  uint64_t time = arts_get_time_stamp() - start;

  /* Extract result from DB */
  unsigned int *result_ptr = depv[0].ptr;
  assert(result_ptr && "fib_done result_ptr not NULL");
  unsigned int result = *result_ptr;

  arts_printf("Fib %u: %u time: %lu nodes: %u workers: %u\n",
              (unsigned int)paramv[0], result, time,
              arts_get_total_nodes(), arts_get_total_workers());
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;

  char **argv = (char **)paramv[1];
  unsigned int num = (unsigned int)strtoul(argv[1], NULL, 10);

  /* Create the done EDT that will receive the final result */
  arts_guid_t done_guid =
      arts_edt_create(fib_done, 1, (uint64_t *)&num, 1,
                      &(arts_hint_t){.route = 0});

  /* Create a DB for the input number */
  unsigned int *num_ptr;
#if CXL_DB
  arts_guid_t num_guid = arts_db_create((void **)&num_ptr,
                                        sizeof(unsigned int),
                                        ARTS_DB_CXL, NULL);
#else
  arts_guid_t num_guid = arts_db_create((void **)&num_ptr,
                                        sizeof(unsigned int),
                                        ARTS_DB_DEFAULT, NULL);
#endif /* CXL_DB */
  assert(num_ptr && "main_edt num_ptr not NULL");
  *num_ptr = num;
#if CXL_DB
  arts_cxl_producer_flush(num_guid);
#endif /* CXL_DB */

  /* Start the computation */
  uint64_t args[2] = {(uint64_t)done_guid, 0};
  start = arts_get_time_stamp();
  arts_guid_t fib_guid =
      arts_edt_create(fib_fork, 2, args, 1, &(arts_hint_t){.route = 0});
  arts_signal_edt(fib_guid, 0, num_guid, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}