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
#ifndef ARTS_TRANSPORT_OUTBOX_H
#define ARTS_TRANSPORT_OUTBOX_H
#ifdef __cplusplus
extern "C" {
#endif

/* Outbound message queue API.  The wire ABI (packet structs, arts_msg_type
 * enum, arts_fill_packet_header) lives in protocol.h; the queue that buffers
 * those packets for the sender threads lives here.  Depends on protocol.h one
 * way only — protocol.h never includes this header. */
#include "arts/transport/protocol.h"

void arts_outbox_init(unsigned int size);
void arts_outbox_cleanup(void);
void arts_remote_flush_outbound(void);
bool arts_remote_async_send();
void arts_remote_send_request_async(int rank, char *message,
                                    unsigned int length);
void arts_remote_send_request_payload_async(int rank, char *message,
                                            unsigned int length, char *payload,
                                            uint64_t size);
void arts_remote_send_request_payload_async_free(
    int rank, char *message, unsigned int length, char *payload,
    unsigned int offset, uint64_t size, void (*free_method)(void *));
void arts_remote_set_thread_outbound_queues(unsigned int start,
                                            unsigned int stop);
void arts_remote_thread_outbound_queues_cleanup();
#ifdef __cplusplus
}
#endif

#endif
