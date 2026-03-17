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
#ifndef ARTS_SYSTEM_CONFIG_H
#define ARTS_SYSTEM_CONFIG_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#include "arts/transport/launcher.h"

struct arts_config_table_s {
  unsigned int rank;
  char *ip_address;
  unsigned int *ports; // Port list (port_count entries), always populated
};

struct arts_config_variable_s {
  unsigned int size;
  struct arts_config_variable_s *next;
  char variable[255];
  char value[];
};

struct arts_config_s {
  unsigned int my_rank;
  char *master_node;
  char *net_interface;
  char *launcher;
  unsigned int port_count;
  unsigned int *default_ports;
  unsigned int default_ports_count;
  unsigned int worker_thread_count;
  unsigned int sender_thread_count;
  unsigned int receiver_thread_count;
  unsigned int thread_count;
  unsigned int nodes;
  unsigned int master_rank;
  unsigned int kill_mode;
  unsigned int route_table_size;
  unsigned int route_table_entries;
  unsigned int deque_size;
  char *counter_folder;
  unsigned int counter_capture_interval;
  unsigned int scheduler;
  unsigned int deque_type;
  unsigned int auto_shutdown;
  bool master_boot;
  bool core_dump;
  bool pin_threads;
  bool shared_pu_pool; /* true when all nodes share PUs (local multi-node) */
  uint64_t stack_size;
  struct arts_remote_launcher_s *launcher_data;
  unsigned int table_length;
  unsigned int gpu;
  unsigned int gpu_locality;
  unsigned int gpu_fit;
  unsigned int gpu_lc_sync;
  unsigned int gpu_max_edts;
  uint64_t gpu_max_memory;
  bool gpu_p2p;
  bool gpu_buff_on;
  unsigned int gpu_route_table_size;
  unsigned int gpu_route_table_entries;
  bool free_db_after_gpu_run;
  bool run_gpu_gc_pre_edt;
  bool run_gpu_gc_idle;
  bool delete_zeros_gpu_gc;
  struct arts_config_table_s *table;
};

void arts_config_load(struct arts_config_s *config);
void arts_config_destroy(struct arts_config_s *config);

#ifdef __cplusplus
}
#endif

#endif
