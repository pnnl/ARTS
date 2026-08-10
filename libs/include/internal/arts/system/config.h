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

#ifdef ARTS_USE_CXL
/** CXL DB allocation strategy. */
typedef enum {
  ARTS_CXL_DB_ALLOC_STATIC, /**< Always allocate on a fixed device (default). */
  ARTS_CXL_DB_ALLOC_ROUND_ROBIN /**< Distribute allocations across devices. */
} arts_cxl_db_alloc_strategy_t;
#endif /* ARTS_USE_CXL */

/* Window a local run's port block is drawn from: above the registered-service
   crowd, below the customary Linux ephemeral floor (32768), where a listen port
   would collide at random with an outgoing connection's source port. */
#define ARTS_PORT_WINDOW_LO 20000U
#define ARTS_PORT_WINDOW_HI 32000U
/* How the spawning rank tells the ranks it spawns what the run settled on.
   Not a config key: it is an internal handoff, and a local run rejects any
   attempt to name ports by hand. */
#define ARTS_RESOLVED_PORTS_ENV "ARTS_RESOLVED_PORTS"

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
  /* Base list every node's ports are derived from; exactly port_count entries.
     Named by the config for a remote launcher, chosen by the runtime for a
     local one. */
  unsigned int *ports;
  unsigned int ports_count;
  char *provider; /* libfabric provider name (fi_getinfo hints prov_name);
                     NULL/empty = auto-select.  Overrides the ambient
                     FI_PROVIDER env var when set. */
  char *fabric_domain; /* libfabric domain name (fi_getinfo hints
                          domain_attr->name, e.g. an HCA like "mlx5_0");
                          NULL/empty = provider's first domain. */
  unsigned int regpool_slab_mb; /* registered-memory slab pool size, MB */
  unsigned int worker_thread_count;
  unsigned int progress_thread_count;
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
  bool master_boot;
  bool core_dump;
  bool pin_threads;
  bool shared_pu_pool; /* true when all nodes share PUs (local multi-node) */
  uint64_t stack_size;
  struct arts_launcher_s *launcher_data;
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
#ifdef ARTS_USE_CXL
  arts_cxl_db_alloc_strategy_t
      cxl_db_allocation_strategy;        /**< DB allocation strategy (static or
                                            round_robin). */
  unsigned int cxl_db_allocation_device; /**< Device index for static allocation
                                            (default: 0). */
#endif                                   /* ARTS_USE_CXL */
};

void arts_config_load(struct arts_config_s *config);
void arts_config_destroy(struct arts_config_s *config);

#ifdef __cplusplus
}
#endif

#endif
