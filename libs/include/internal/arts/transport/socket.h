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

#ifndef ARTS_TRANSPORT_SOCKET_H
#define ARTS_TRANSPORT_SOCKET_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts/system/config.h"
#include <netinet/in.h>
#include <stdbool.h>
#include <stdint.h>

int arts_get_new_socket();
int arts_get_socket_listening(struct sockaddr_in *listening_socket,
                              unsigned int port);
int arts_get_socket_outgoing(struct sockaddr_in *outgoing_socket,
                             unsigned int port, in_addr_t s_addr);

void arts_transport_set_config(struct arts_config_s *config);
/* Slide a local multi-node run's port block past whatever already holds it, by
 * probing the exact set of ports the run is about to claim; rewrites
 * config->ports on success.  Callable only from the process that spawns
 * the other ranks — they inherit the chosen base and must not probe again.
 * Returns false when no free block fits below the ephemeral range, leaving the
 * configured ports in place. */
bool arts_transport_select_local_ports(struct arts_config_s *config);
void arts_transport_setup_outgoing();
bool arts_transport_setup_incoming();
/* Convert the bootstrap TCP mesh into zero-traffic liveness sentinels once the
 * fi-address exchange has finished: every established connection stays open
 * (connection lifetime doubles as peer liveness — orderly close and process
 * death both surface as HUP on the peer's accept-side socket); only the
 * listening sockets are closed.  The launcher / stdio sockets are untouched. */
void arts_socket_sentinel_arm(void);

/* Non-blocking probe of the accept-side sentinels for peer death
 * (POLLHUP/POLLERR/EOF).  Called from a progress thread's idle cycle; on a
 * dead peer it enters the same idempotent passive-shutdown path an inbound
 * shutdown message uses and returns true.  Stands down (false) once shutdown
 * is already in progress, so an orderly peer exit is never mistaken for a
 * death.  Serialized internally to a single prober. */
bool arts_socket_sentinel_check(void);
#ifdef __cplusplus
}
#endif

#endif
