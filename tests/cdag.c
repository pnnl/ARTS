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
#include <stdlib.h>

#include "arts.h"

unsigned int num_reads = 0;
unsigned int num_writes = 0;
unsigned int num_dynamic_reads = 0;
unsigned int num_dynamic_writes = 0;
arts_guid_t shutdown_guid;
arts_guid_t db_guid;
arts_guid_t *read_guids;
arts_guid_t *write_guids;

void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  unsigned int *array = (unsigned int *)depv[0].ptr;
  for (unsigned int i = 0; i < num_writes; i++) {
    arts_printf("i: %u %u\n", i, array[i]);
  }
  arts_shutdown();
}

void read_test(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  //    arts_printf("READ\n");
  unsigned int *array = (unsigned int *)depv[0].ptr;
  for (unsigned int i = 0; i < num_writes; i++) {
    if (array[i] != 0 && array[i] != i) {
      arts_printf("BAD VALUE i: %u %u\n", i, array[i]);
    }
  }
  arts_signal_edt_value(shutdown_guid, -1, 0);
}

void write_test(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  unsigned int index = paramv[0];
  unsigned int *array = (unsigned int *)depv[0].ptr;
  //    arts_printf("WRITE %u\n", index);
  array[index] += index;

  for (unsigned int i = 0; i < num_dynamic_reads; i++) {
    arts_guid_t guid =
        arts_edt_create(read_test, 0, NULL, 1,
                        &(arts_hint_t){.route = arts_get_current_node()});
    arts_signal_edt(guid, 0, db_guid, DB_MODE_EW);
  }

  uint64_t idx = paramv[0];
  for (unsigned int i = 0; i < num_dynamic_writes; i++) {
    idx = (idx + 1) % num_writes;
    arts_guid_t guid =
        arts_edt_create(read_test, 0, NULL, 1,
                        &(arts_hint_t){.route = arts_get_current_node()});
    arts_signal_edt(guid, 0, db_guid, DB_MODE_EW);
  }

  if (!index) {
    arts_signal_edt(shutdown_guid, 0, db_guid, DB_MODE_EW);
  } else {
    arts_signal_edt_value(shutdown_guid, -1, 0);
  }
}

void node_setup(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  for (uint64_t i = 0; i < num_reads; i++) {
    if (arts_guid_is_local(read_guids[i])) {
      arts_edt_create_with_guid(read_test, read_guids[i], 0, NULL, 1);
      arts_signal_edt(read_guids[i], 0, db_guid, DB_MODE_EW);
    }
  }

  for (uint64_t i = 0; i < num_writes; i++) {
    if (arts_guid_is_local(write_guids[i])) {
      arts_edt_create_with_guid(write_test, write_guids[i], 1, &i, 1);
      arts_signal_edt(write_guids[i], 0, db_guid, DB_MODE_EW);
    }
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **argv = (char **)paramv[1];
  num_reads = strtol(argv[1], NULL, 10);
  num_writes = strtol(argv[2], NULL, 10);
  num_dynamic_reads = strtol(argv[3], NULL, 10);
  num_dynamic_writes = strtol(argv[4], NULL, 10);
  arts_printf("Reads: %u Writes: %u Dynamic Reads: %u Dynamic Writes: %u Final "
              "Deps: %u\n",
              num_reads, num_writes, num_dynamic_reads, num_dynamic_writes,
              (num_dynamic_reads * num_writes) +
                  (num_dynamic_writes * num_writes) + num_reads + num_writes);

  read_guids = (arts_guid_t *)malloc(sizeof(arts_guid_t) * num_reads);
  write_guids = (arts_guid_t *)malloc(sizeof(arts_guid_t) * num_writes);

  db_guid = arts_guid_reserve(ARTS_DB, 0);

  for (unsigned int i = 0; i < num_reads; i++) {
    read_guids[i] = arts_guid_reserve(ARTS_EDT, i % arts_get_total_nodes());
  }
  for (unsigned int i = 0; i < num_writes; i++) {
    write_guids[i] = arts_guid_reserve(ARTS_EDT, i % arts_get_total_nodes());
  }

  shutdown_guid = arts_guid_reserve(ARTS_EDT, 0);

  unsigned int *ptr = (unsigned int *)arts_db_create_with_guid(
      db_guid, sizeof(unsigned int) * num_writes, NULL);
  for (unsigned int i = 0; i < num_writes; i++) {
    ptr[i] = 0;
  }

  arts_edt_create_with_guid(shutdown_edt, shutdown_guid, 0, NULL,
                            (num_dynamic_reads * num_writes) +
                                (num_dynamic_writes * num_writes) + num_reads +
                                num_writes);

  for (unsigned int n = 0; n < arts_get_total_nodes(); n++) {
    arts_edt_create(node_setup, 0, NULL, 0, &(arts_hint_t){.route = n});
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
