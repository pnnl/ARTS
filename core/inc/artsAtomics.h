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
#ifndef ARTSATOMICS_H
#define ARTSATOMICS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "artsRT.h"
#define HW_MEMORY_FENCE() __sync_synchronize() 
#define COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT() __asm__ volatile("": : :"memory")

volatile unsigned int artsAtomicSwap(volatile unsigned int *destination, unsigned int swapIn);
volatile uint64_t artsAtomicSwapU64(volatile uint64_t *destination, uint64_t swapIn);
volatile void * artsAtomicSwapPtr(volatile void **destination, void * swapIn);
volatile unsigned int artsAtomicSub(volatile unsigned int *destination, unsigned int subVal);
volatile unsigned int artsAtomicAdd(volatile unsigned int *destination, unsigned int addVal);
volatile unsigned int artsAtomicFetchAdd(volatile unsigned int *destination, unsigned int addVal);
volatile unsigned int artsAtomicCswap(volatile unsigned int *destination, unsigned int oldVal, unsigned int swapIn);
volatile uint64_t artsAtomicCswapU64(volatile uint64_t *destination, uint64_t oldVal, uint64_t swapIn);
volatile void * artsAtomicCswapPtr(volatile void **destination, void * oldVal, void * swapIn);
volatile bool artsAtomicSwapBool(volatile bool *destination, bool value);
volatile uint64_t artsAtomicFetchAddU64(volatile uint64_t *destination, uint64_t addVal);
volatile uint64_t artsAtomicFetchSubU64(volatile uint64_t *destination, uint64_t subVal);
volatile uint64_t artsAtomicAddU64(volatile uint64_t *destination, uint64_t addVal);
volatile uint64_t artsAtomicSubU64(volatile uint64_t *destination, uint64_t subVal);
bool artsLock( volatile unsigned int * lock);
void artsUnlock( volatile unsigned int * lock);
bool artsTryLock( volatile unsigned int * lock);
volatile uint64_t artsAtomicFetchAndU64(volatile uint64_t * destination, uint64_t addVal);
volatile uint64_t artsAtomicFetchOrU64(volatile uint64_t * destination, uint64_t addVal);
volatile uint64_t artsAtomicFetchXOrU64(volatile uint64_t * destination, uint64_t addVal); //@awmm
volatile unsigned int artsAtomicFetchOr(volatile unsigned int * destination, unsigned int addVal);
volatile unsigned int artsAtomicFetchAnd(volatile unsigned int * destination, unsigned int addVal);
#ifdef __cplusplus
}
#endif

#endif
