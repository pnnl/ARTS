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

#include "sys/socket.h"
#include "arpa/inet.h"
#include "string.h"
#include "netinet/tcp.h"
#include "artsConnection.h" 
#include "arts.h"
#include <fcntl.h>
#define DPRINTF(...)

void artsPrintSocketAddr(struct sockaddr_in *sock)
{
    //char crap[255];
    char * addr = inet_ntoa(sock->sin_addr);
    if(addr!=NULL)
        DPRINTF("socket addr %s\n", addr );
}

unsigned int artsGetNewSocket()
{
    unsigned int socketOut = rsocket(PF_INET, SOCK_STREAM, 0);
    return socketOut;
}

unsigned int artsGetSocketListening( struct sockaddr_in * listeningSocket, unsigned int port  )
{
    memset( (char *)listeningSocket, 0, sizeof(*listeningSocket) );
    unsigned int socketOut = rsocket(PF_INET, SOCK_STREAM, 0);
    listeningSocket->sin_family = AF_INET;
    listeningSocket->sin_addr.s_addr = htonl(INADDR_ANY);
    listeningSocket->sin_port = htons(port);
    return socketOut;
}


unsigned int artsGetSocketOutgoing( struct sockaddr_in * outgoingSocket, unsigned int port, in_addr_t saddr  )
{
    memset( (char *)outgoingSocket, 0, sizeof(*outgoingSocket) );
    unsigned int socketOut = rsocket(PF_INET, SOCK_STREAM, 0);
    outgoingSocket->sin_family = AF_INET;
    outgoingSocket->sin_addr.s_addr = saddr;
    outgoingSocket->sin_port = htons(port);
    artsPrintSocketAddr( outgoingSocket );
    return socketOut;
}

