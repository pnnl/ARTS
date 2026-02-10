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
#include "arts/system/tmt_lite.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <unistd.h>

#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/system/arts_print.h"
#include "arts/system/threads.h"
#include "arts/utils/array_list.h"
#include "arts/utils/atomics.h"

__thread unsigned int tmt_lite_alias_id = 0;

size_t page_size = 0; // Got this from andres' TMT... Not sure if we need it?
arts_array_list_t **thread_to_join;
unsigned int *alias_number;

volatile unsigned int to_create_threads = 0;
volatile unsigned int done_create_threads = 0;
volatile unsigned int done_threads = 0;
volatile uint64_t *outstanding;

volatile unsigned int *thread_reader_lock;
volatile unsigned int *thread_writer_lock;
volatile unsigned int *array_list_lock;

// volatile unsigned int threadCreateReader = 0;
// volatile unsigned int threadCreateWriter = 0;

void arts_writer_lock_yield(const volatile unsigned int *read_lock,
                         volatile unsigned int *write_lock) {
  unsigned int to_swap = tmt_lite_alias_id + 1;
  while (arts_atomic_cswap(write_lock, 0U, to_swap) != 0U) {
    INCREMENT_SLEEP_COUNTER_BY(1);
    sched_yield();
  }
  while ((*read_lock)) {
    INCREMENT_SLEEP_COUNTER_BY(1);
    sched_yield();
  }
  }

void arts_init_tmt_lite_per_node(unsigned int num_workers) {
  long temp = sysconf(_SC_PAGESIZE);
  page_size = temp;
  thread_to_join =
      (arts_array_list_t **)arts_calloc(num_workers, sizeof(arts_array_list_t *));
  alias_number = (unsigned int *)arts_calloc(num_workers, sizeof(unsigned int));
  thread_reader_lock =
      (volatile unsigned int *)arts_calloc(num_workers, sizeof(unsigned int));
  thread_writer_lock =
      (volatile unsigned int *)arts_calloc(num_workers, sizeof(unsigned int));
  array_list_lock =
      (volatile unsigned int *)arts_calloc(num_workers, sizeof(unsigned int));
  outstanding = (volatile uint64_t *)arts_calloc(num_workers, sizeof(uint64_t));
}

void arts_init_tmt_lite_per_worker(unsigned int id) {
  arts_writer_lock_yield(&thread_reader_lock[id], &thread_writer_lock[id]);
  thread_to_join[id] = arts_new_array_list(sizeof(pthread_t), 8);
}

void arts_tmt_lite_shutdown() {
  while (to_create_threads != done_create_threads) {
    ;
}
}

void arts_tmt_lite_private_clean_up(unsigned int id) {
  while (to_create_threads != done_threads) {
    INCREMENT_SLEEP_COUNTER_BY(1);
    sched_yield();
  }
  uint64_t outstanding = arts_length_array_list(thread_to_join[id]);
  for (uint64_t i = 0; i < outstanding; i++) {
    pthread_t *thread = (pthread_t *)arts_get_from_array_list(thread_to_join[id], i);
    pthread_join(*thread, NULL);
  }
}

typedef struct {
  uint32_t alias_id; // alias id
  uint32_t source_id;
  struct arts_edt_s *edtToRun;
  volatile uint64_t *to_dec;
  struct arts_runtime_private_s *tlToCopy; // we copy the master thread's TL
} lite_args_t;

void *arts_alias_lite_thread_loop(void *arg) {
  lite_args_t *t_args = (lite_args_t *)arg;
  tmt_lite_alias_id = t_args->alias_id;
  uint32_t source_id = t_args->source_id;
  memcpy(&arts_thread_info, t_args->tlToCopy, sizeof(struct arts_runtime_private_s));

  if (arts_node_info.pin_threads) {
    arts_pthread_affinity(arts_thread_info.core_id, false);
  }

  unsigned int res = arts_atomic_add(&done_create_threads, 1);
  arts_writer_lock_yield(&thread_reader_lock[source_id], &thread_writer_lock[source_id]);
  if (arts_thread_info.alive) {
    arts_node_info.scheduler();
}
  arts_writer_unlock(&thread_writer_lock[source_id]);
  uint64_t temp_res = arts_atomic_sub_u64(t_args->to_dec, 1);
  arts_atomic_add(&done_threads, 1);
  arts_free(t_args);
  return NULL;
}

void arts_create_lite_contexts(volatile uint64_t *to_dec) {
  unsigned int source_id = arts_thread_info.group_id;
  unsigned int res = arts_atomic_add(&to_create_threads, 1);
  volatile unsigned int spin_flag = 1;
  lite_args_t *args = (lite_args_t *)arts_calloc(1, sizeof(lite_args_t));
  args->alias_id = ++alias_number[source_id];
  args->source_id = source_id;
  args->to_dec = to_dec;
  args->tlToCopy = &arts_thread_info;

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, page_size);
  pthread_t *thread = (pthread_t *)arts_next_free_from_array_list(
      thread_to_join[arts_thread_info.group_id]);

  arts_writer_unlock(&thread_writer_lock[source_id]);

  pthread_create(thread, &attr, &arts_alias_lite_thread_loop, args);

  arts_writer_lock_yield(&thread_reader_lock[source_id], &thread_writer_lock[source_id]);
}

void *arts_alias_lite_thread_loop2(void *arg) {
  lite_args_t *t_args = (lite_args_t *)arg;
  tmt_lite_alias_id = t_args->alias_id;
  uint32_t source_id = t_args->source_id;
  memcpy(&arts_thread_info, t_args->tlToCopy, sizeof(struct arts_runtime_private_s));

  if (arts_node_info.pin_threads) {
    arts_pthread_affinity(arts_thread_info.core_id, false);
  }

  arts_atomic_add(&done_create_threads, 1);
  arts_writer_lock_yield(&thread_reader_lock[source_id], &thread_writer_lock[source_id]);

  arts_run_edt(t_args->edtToRun);

  arts_writer_unlock(&thread_writer_lock[source_id]);

  arts_atomic_sub_u64(t_args->to_dec, 1);
  arts_atomic_sub_u64(&outstanding[source_id], 1);
  arts_atomic_add(&done_threads, 1);
  arts_free(t_args);
  return NULL;
}

void arts_create_lite_contexts2(volatile uint64_t *to_dec, struct arts_edt_s *edt) {
  unsigned int source_id = arts_thread_info.group_id;
  unsigned int res = arts_atomic_add(&to_create_threads, 1);
  arts_atomic_add_u64(&outstanding[source_id], 1);
  volatile unsigned int spin_flag = 1;
  lite_args_t *args = (lite_args_t *)arts_calloc(1, sizeof(lite_args_t));
  args->alias_id = ++alias_number[source_id];
  args->source_id = source_id;
  args->edtToRun = edt;
  args->to_dec = to_dec;
  args->tlToCopy = &arts_thread_info;

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, page_size);
  arts_lock(&array_list_lock[arts_thread_info.group_id]);
  pthread_t *thread = (pthread_t *)arts_next_free_from_array_list(
      thread_to_join[arts_thread_info.group_id]);
  arts_unlock(&array_list_lock[arts_thread_info.group_id]);
  pthread_create(thread, &attr, &arts_alias_lite_thread_loop2, args);
}

void arts_yield_lite_context() {
  unsigned int source_id = arts_thread_info.group_id;
  arts_writer_unlock(&thread_writer_lock[source_id]);
  INCREMENT_SLEEP_COUNTER_BY(1);
  sched_yield();
}

void arts_resume_lite_context() {
  unsigned int source_id = arts_thread_info.group_id;
  arts_writer_lock_yield(&thread_reader_lock[source_id], &thread_writer_lock[source_id]);
}

unsigned int arts_tmt_lite_get_alias() { return tmt_lite_alias_id; }

void arts_tmt_scheduler_yield() {
  unsigned int source_id = arts_thread_info.group_id;
  if (outstanding[source_id]) {
    // ARTS_INFO("Scheduler Yield %u", outstanding[source_id]);
    arts_writer_unlock(&thread_writer_lock[source_id]);
    sched_yield();
    arts_writer_lock_yield(&thread_reader_lock[source_id],
                        &thread_writer_lock[source_id]);
  }
}