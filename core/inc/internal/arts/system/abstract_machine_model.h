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

#ifndef ARTS_SYSTEM_ABSTRACTMACHINEMODEL_H
#define ARTS_SYSTEM_ABSTRACTMACHINEMODEL_H
#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <unistd.h>

#include "arts/system/config.h"

#ifdef USE_HWLOC
#include <hwloc.h>
#include <sched.h>
struct arts_core_info {
  hwloc_bitmap_t cpuset;
#ifndef __APPLE__
  cpu_set_t linuxCpuSet;
#endif
};
#else
struct arts_core_info {
  unsigned int cpuId;
};
#endif
struct unit_thread {
  unsigned int id;
  unsigned int group_id;
  unsigned int group_pos;
  bool worker;
  bool network_send;
  bool network_receive;
  bool status_send;
  bool pin;
  struct unit_thread *next;
};

struct thread_mask {
  unsigned int cluster_id;
  unsigned int core_id;
  unsigned int unit_id;
  bool on;
  unsigned int id;
  unsigned int group_id;
  unsigned int group_pos;
  bool worker;
  bool network_send;
  bool network_receive;
  bool status_send;
  bool pin;
  struct arts_core_info core_info;
};

struct unit_mask {
  unsigned int cluster_id;
  unsigned int core_id;
  unsigned int unit_id;
  bool on;
  unsigned int threads;
  struct unit_thread *listHead;
  struct unit_thread *listTail;
  struct arts_core_info core_info;
};

struct core_mask {
  unsigned int num_units;
  struct unit_mask *unit;
};

struct cluster_mask {
  unsigned int num_cores;
  struct core_mask *core;
};

struct node_mask {
  unsigned int num_clusters;
  struct cluster_mask *cluster;
};

struct thread_mask *get_thread_mask(struct arts_config *config);
void print_mask(struct thread_mask *units, unsigned int number_of_units);
void arts_abstract_machine_model_pin_thread(struct arts_core_info *core_info);
void destroy_thread_mask(struct thread_mask *mask);

#ifdef __cplusplus
}
#endif

#endif /* artsABSTRACTMACHINEMODEL_H */
