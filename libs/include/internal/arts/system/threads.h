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

#ifndef ARTS_SYSTEM_THREADS_H
#define ARTS_SYSTEM_THREADS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include "arts/system/config.h"

void arts_thread_init(struct arts_config_s *config);
void arts_thread_main_join(void);

/* Shutdown protocol entry points. See libs/src/core/system/threads.c
 * (or shutdown.c if split out later) for the implementation.
 *   initiator = true  → broadcast MSG_SHUTDOWN to all peers,
 *                       wait for local outbox drain, then stop workers.
 *   initiator = false → just stop workers (passive receiver path). */
void arts_enter_shutdown_state(bool initiator);

/* Local lifecycle controls (runtime-state mutators, not introspection).
 *   arts_stop_local_worker — retire only the calling worker thread.
 *   arts_stop_local_node   — stop the whole local runtime. */
void arts_stop_local_worker(void);
void arts_stop_local_node(void);

/* Rank-identity externs live in arts/system/identity.h (kept here for
 * back-compat of threads.h includers). */
#include "arts/system/identity.h"

#ifdef __cplusplus
}
#endif
#endif
