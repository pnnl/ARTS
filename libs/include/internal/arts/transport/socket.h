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
void arts_server_set_socket_options_sender(unsigned int socket);
void arts_server_set_socket_options_reciever(unsigned int socket);
int arts_get_socket_listening(struct sockaddr_in *listening_socket,
                              unsigned int port);
int arts_get_socket_outgoing(struct sockaddr_in *outgoing_socket,
                             unsigned int port, in_addr_t s_addr);

void arts_remote_set_message_table(struct arts_config_s *table);
void arts_remote_setup_outgoing();
bool arts_remote_setup_incoming();
unsigned int arts_remote_get_my_rank();
bool arts_server_try_to_receive(char **in_buffer, const int *in_packet_size,
                                const volatile unsigned int *remote_steal_lock);
uint64_t arts_remote_send_request(int rank, unsigned int queue, char *message,
                                  uint64_t length);
uint64_t arts_remote_send_payload_request(int rank, unsigned int queue,
                                          char *message, unsigned int length,
                                          char *payload, uint64_t length2);
void arts_server_ping_pong_test_recieve(char *in_buffer, int in_packet_size);
void arts_remote_set_thread_inbound_queues(unsigned int start,
                                           unsigned int stop);
void arts_remote_thread_inbound_queues_cleanup();
#ifdef __cplusplus
}
#endif

#endif
