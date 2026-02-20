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
#include "arts/introspection/Preamble.h"
#define GNU_SOURCE
#define _FILE_OFFSET_BITS 64 // NOLINT(readability-identifier-naming)
#include "arts.h"
#include "arts/introspection/counter.h"
#include "arts/network/remote.h"
#include "arts/network/remote_launcher.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/system/config.h"
#include "arts/system/debug.h"
#include "arts/system/threads.h"
#include <string.h>

int arts_rt(int argc, char **argv) {
  INITIALIZATION_TIME_START();

  struct arts_config_s *config = arts_config_load();

  if (config->core_dump) {
    arts_turn_on_core_dumps();
  }

  arts_global_rank_id = 0;
  arts_global_rank_count = config->table_length;
  if (strncmp(config->launcher, "local", 5) != 0) {
    arts_server_setup(config);
  }
  arts_global_master_rank_id = config->master_rank;
  if (arts_global_rank_id == config->master_rank && config->master_boot) {
    config->launcher_data->launch_processes(config->launcher_data);
  }

  if (arts_global_rank_count > 1) {
    arts_remote_setup_outgoing();
    if (!arts_remote_setup_incoming()) {
      return -1;
    }
  }

  arts_thread_init(config);
  arts_thread_zero_node_start(argc, argv);

  arts_thread_main_join();

  // Aggregate cluster counters before cleanup (workers may still be writing)
  if (arts_global_rank_id == config->master_rank) {
    arts_counter_write_cluster(config->counter_folder, config->nodes);
  }
  if (arts_global_rank_id == config->master_rank && config->master_boot) {
    config->launcher_data->cleanup_processes(config->launcher_data);
  }
  arts_config_destroy(config);
  return 0;
}
