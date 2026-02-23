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
#include "arts.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdlib.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"

extern ARTS_THREAD_LOCAL struct arts_edt_s *current_edt;
extern unsigned int num_numa_domains;

arts_guid_t arts_get_current_guid() {
  if (current_edt) {
    return current_edt->current_edt;
  }
  return NULL_GUID;
}

unsigned int arts_get_current_node() {
  return arts_global_rank_id;
}

unsigned int arts_get_total_nodes() {
  return arts_global_rank_count;
}

unsigned int arts_get_total_workers() {
  return arts_node_info.worker_thread_count;
}

unsigned int arts_get_current_worker() {
  return arts_thread_info.group_pos;
}

unsigned int arts_get_current_numa_domain() {
  return arts_thread_info.numa_domain_id;
}

unsigned int arts_get_total_numa_domains() {
  return num_numa_domains;
}

void arts_stop_local_worker() {
  arts_thread_info.alive = false;
}

void arts_stop_local_node() {
  arts_runtime_stop();
}

uint64_t arts_thread_safe_random() {
  long int temp = jrand48(arts_thread_info.drand_buf);
  return (uint64_t)temp;
}

unsigned int arts_get_total_gpus() {
  return arts_node_info.gpu;
}
