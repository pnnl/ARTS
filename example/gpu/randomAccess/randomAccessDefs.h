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
#ifndef RANDOMACCESSDEFS_H
#define RANDOMACCESSDEFS_H

#ifdef __cplusplus
extern "C" {
#endif

#define PRINTF(...)
//  #define PRINTF(...) PRINTF(__VA_ARGS__)
#define TURNON(...)
// #define TURNON(...) __VA_ARGS__

#define TABLESIZE 32UL * 2UL * 80UL * 1024UL * 32UL
#define TILESIZE 32UL * 2UL * 80UL * 1024UL * 4UL
#define NUPDATE (16 * TABLESIZE)

#define VALIDATE 1

// Configured for Volta
#define MAXTHREADS 32
#define MAXTHREADBLOCKSPERSM 2
#define NUMBEROFSM 80
#define MAXGRID MAXTHREADBLOCKSPERSM *NUMBEROFSM

#define POLY2 0x0000000000000007UL
#define PERIOD2 1317624576693539401L

#define POLY 0x0000000000000007ULL
#define PERIOD 1317624576693539401LL

#define MAX_TOTAL_PENDING_UPDATES 1024 * 16
#define MAX_TOTAL_PENDING_UPDATES_CU 1024ULL * 16ULL

#define LOCAL_BUFFER_SIZE MAX_TOTAL_PENDING_UPDATES
#define MAX_UPDATES_PER_GPU_STEP                                               \
  MAXTHREADS *MAXTHREADBLOCKSPERSM *NUMBEROFSM *LOCAL_BUFFER_SIZE

typedef unsigned long long int uint64_cu_t;
typedef long long int int64_cu_t;

#ifdef __cplusplus
}
#endif

#endif