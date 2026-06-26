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

/* --- Parametric multi-reader/writer test (argc/argv-driven) --- */

unsigned int num_reads = 0;
unsigned int num_writes = 0;
unsigned int num_dynamic_reads = 0;
unsigned int num_dynamic_writes = 0;
arts_guid_t shutdown_guid;
arts_guid_t db_guid;
arts_guid_t *read_guids;
arts_guid_t *write_guids;

/* --- Chain test: N sequential RW writers then one RO reader --- */

#define CHAIN_LEN 8

void chain_writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int idx = (unsigned int)paramv[0];
  unsigned int *data = (unsigned int *)depv[0].ptr;
  data[0] = idx;
}

void chain_reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data && data[0] == CHAIN_LEN - 1) {
    arts_printf("  PASS: chain reader saw final write %u\n", data[0]);
  } else {
    arts_printf("  FAIL: chain reader expected %u got %d\n", CHAIN_LEN - 1,
                data ? (int)data[0] : -1);
  }
  /* Signal chain completion to the shared counting shutdown EDT. */
  arts_add_dependence((arts_guid_t)(0), (arts_guid_t)paramv[0], -1,
                      DB_MODE_VAL);
}

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
  arts_add_dependence((arts_guid_t)(0), shutdown_guid, -1, DB_MODE_VAL);
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
                        &(arts_edt_hint_t){.rank = arts_get_current_rank()});
    arts_add_dependence(db_guid, guid, 0, DB_MODE_RW);
  }

  uint64_t idx = paramv[0];
  for (unsigned int i = 0; i < num_dynamic_writes; i++) {
    idx = (idx + 1) % num_writes;
    arts_guid_t guid =
        arts_edt_create(read_test, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = arts_get_current_rank()});
    arts_add_dependence(db_guid, guid, 0, DB_MODE_RW);
  }

  if (!index) {
    arts_add_dependence(db_guid, shutdown_guid, 0, DB_MODE_RW);
  } else {
    arts_add_dependence((arts_guid_t)(0), shutdown_guid, -1, DB_MODE_VAL);
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
      arts_edt_create(read_test, 0, NULL, 1,
                      &(arts_edt_hint_t){.guid = read_guids[i]});
      arts_add_dependence(db_guid, read_guids[i], 0, DB_MODE_RW);
    }
  }

  for (uint64_t i = 0; i < num_writes; i++) {
    if (arts_guid_is_local(write_guids[i])) {
      arts_edt_create(write_test, 1, &i, 1,
                      &(arts_edt_hint_t){.guid = write_guids[i]});
      arts_add_dependence(db_guid, write_guids[i], 0, DB_MODE_RW);
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

  db_guid = arts_guid_reserve(ARTS_GUID_DB, 0);

  for (unsigned int i = 0; i < num_reads; i++) {
    read_guids[i] =
        arts_guid_reserve(ARTS_GUID_EDT, i % arts_get_total_ranks());
  }
  for (unsigned int i = 0; i < num_writes; i++) {
    write_guids[i] =
        arts_guid_reserve(ARTS_GUID_EDT, i % arts_get_total_ranks());
  }

  shutdown_guid = arts_guid_reserve(ARTS_GUID_EDT, 0);

  unsigned int *ptr = (unsigned int *)arts_db_create_with_guid(
      db_guid, sizeof(unsigned int) * num_writes, ARTS_DB, ARTS_DB_PROP_NONE,
      NULL);
  for (unsigned int i = 0; i < num_writes; i++) {
    ptr[i] = 0;
  }

  /* Dep count: parametric-test signals + 1 for the chain test reader. */
  arts_edt_create(shutdown_edt, 0, NULL,
                  (num_dynamic_reads * num_writes) +
                      (num_dynamic_writes * num_writes) + num_reads +
                      num_writes + 1,
                  &(arts_edt_hint_t){.guid = shutdown_guid});

  for (unsigned int n = 0; n < arts_get_total_ranks(); n++) {
    arts_edt_create(node_setup, 0, NULL, 0, &(arts_edt_hint_t){.rank = n});
  }

  /* Chain test: CHAIN_LEN sequential RW writers → one RO reader.
   * Each writer stamps data[0] with its index; the reader verifies the
   * final writer's value is visible.
   * Ordering: inner finish scope covers all writers; when it fires it
   * satisfies the reader's slot 1, unblocking the reader after all writers
   * have released their RW slots.  The reader then signals shutdown_guid. */
  {
    void *cptr = NULL;
    arts_guid_t cdb =
        arts_db_create(&cptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)cptr)[0] = 0;
    arts_db_release(cdb, DB_MODE_RW);

    uint64_t shut_param = (uint64_t)shutdown_guid;
    arts_guid_t reader = arts_edt_create(chain_reader_edt, 1, &shut_param, 2,
                                         &(arts_edt_hint_t){.rank = 0});
    arts_add_dependence(cdb, reader, 0, DB_MODE_RO);

    /* inner finish scope: all writers belong to it; when all writers
     * complete and release their DB slots, inner fires and satisfies
     * the reader's slot 1. */
    arts_guid_t inner = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_add_dependence(inner, reader, 1, DB_MODE_NULL);
    /* Chain the writers with output_events so they execute in index order: RW
     * grant order is scheduler-determined, not registration order, so "the
     * final writer's value" is only well-defined once the writers are
     * explicitly serialized (writer i+1 waits on writer i's output_event,
     * which fires after writer i releases the DB). */
    arts_guid_t prev_oe = (arts_guid_t)0; /* first writer has no predecessor */
    for (unsigned int i = 0; i < CHAIN_LEN; i++) {
      uint64_t param = (uint64_t)i;
      arts_guid_t oe = arts_event_create(&ARTS_EVENT_HINT_LATCH(1));
      arts_guid_t w = arts_edt_create(
          chain_writer_edt, 1, &param, (prev_oe ? 2 : 1),
          &(arts_edt_hint_t){
              .rank = 0, .finish_event = inner, .output_event = oe});
      arts_add_dependence(cdb, w, 0, DB_MODE_RW);
      if (prev_oe) {
        arts_add_dependence(prev_oe, w, 1, DB_MODE_NULL);
      }
      prev_oe = oe;
    }
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
