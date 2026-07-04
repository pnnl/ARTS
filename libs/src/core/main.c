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
#include "arts/counter/Preamble.h"
#define _FILE_OFFSET_BITS 64 // NOLINT(readability-identifier-naming)
#include "arts.h"
#include "arts/counter/counter.h"
#include "arts/gas/guid.h"
#include "arts/runtime_state.h"
#include "arts/system/config.h"
#include "arts/system/print.h"
#include "arts/system/signals.h"
#include "arts/system/threads.h"
#include "arts/transport/dispatcher.h"
#include "arts/transport/launcher.h"
#include "arts/transport/socket.h"
#include "arts/transport/stdio_forward.h"

int arts_rt(int argc, char **argv) {
  struct arts_config_s config;
  arts_config_load(&config);

#ifndef ARTS_TRANSPORT_OFI
  /* After the transport cutover all cross-rank traffic rides the libfabric
   * (OFI) core; a build with the OFI transport compiled out is single-node
   * only.  Fail cleanly rather than starting a multinode run with no data
   * transport. */
  if (config.table_length > 1) {
    ARTS_ERROR("multinode run requested (%d nodes) but this build has the OFI "
               "transport disabled (ARTS_TRANSPORT_OFI=OFF) — it is single-node "
               "only.  Rebuild with -DARTS_TRANSPORT_OFI=ON for multinode.",
               config.table_length);
  }
#endif

  arts_install_signal_handlers();
  /* Start the dedicated SIGTERM/SIGINT/SIGHUP/SIGALRM watcher thread BEFORE
   * any worker/sender/receiver thread is spawned, so the SIG_BLOCK mask is
   * inherited by every downstream thread.  Watcher then drives graceful
   * shutdown via arts_enter_shutdown_state on signal arrival. */
  arts_install_signal_watcher_thread();
  if (config.core_dump) {
    arts_turn_on_core_dumps();
  }

  arts_global_rank_id = 0;
  arts_global_rank_count = config.table_length;
  /* GUID layout encodes rank in 14 bits, with the top two values reserved
   * (ARTS_DISTRIBUTED_RANK = 0x3FFE, ARTS_CXL_RANK = 0x3FFF).  Cap the
   * usable range at 2^14 - 2 = 16382 so encoded ranks never collide with
   * the sentinels.  Anything higher than the field width would silently
   * truncate during encoding. */
  if (arts_global_rank_count > (ARTS_GUID_RANK_MASK - 1U)) {
    ARTS_ERROR("Rank count %u exceeds GUID layout limit %u "
               "(14-bit rank field, two values reserved). "
               "Reduce node count or widen ARTS_GUID_RANK_BITS.",
               arts_global_rank_count, ARTS_GUID_RANK_MASK - 1U);
  }
  if (config.table_length > 1) {
    arts_transport_setup(&config);
  }
  arts_global_master_rank_id = config.master_rank;
  if (arts_global_rank_id == config.master_rank && config.master_boot) {
    config.launcher_data->argc = (unsigned int)argc;
    config.launcher_data->argv = argv;
    config.launcher_data->launch_processes(config.launcher_data);
  }

  if (arts_global_rank_count > 1) {
    arts_transport_setup_outgoing();
    if (!arts_transport_setup_incoming()) {
      return -1;
    }
  }

  arts_thread_init(&config);
  arts_thread_zero_node_start(argc, argv);

  arts_thread_main_join();

  // Aggregate cluster counters before cleanup (workers may still be writing)
  if (arts_global_rank_id == config.master_rank) {
    arts_counter_write_cluster(config.counter_folder, config.nodes);
    arts_object_write_cluster(config.counter_folder, config.nodes);
  }
  if (arts_global_rank_id == config.master_rank && config.master_boot) {
    config.launcher_data->cleanup_processes(config.launcher_data);
    arts_stdio_forwarder_shutdown_all();
  }
  arts_stop_signal_watcher_thread();
  arts_config_destroy(&config);
  return 0;
}
