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

#include "artsAtomics.h"

unsigned int artsAtomicSwap(volatile unsigned int *destination, unsigned int swapIn)
{
    return __sync_lock_test_and_set(destination, swapIn);
}

uint64_t artsAtomicSwapU64(volatile uint64_t *destination, uint64_t swapIn)
{
    return __sync_lock_test_and_set(destination, swapIn);
}

volatile void * artsAtomicSwapPtr(volatile void ** destination, void * swapIn)
{
    return __sync_lock_test_and_set(destination, swapIn);
}

unsigned int artsAtomicAdd(volatile unsigned int *destination, unsigned int addVal)
{
    return __sync_add_and_fetch(destination, addVal);
}

unsigned int artsAtomicFetchAdd(volatile unsigned int *destination, unsigned int addVal)
{
    return __sync_fetch_and_add(destination, addVal);
}

uint64_t artsAtomicFetchAddU64(volatile uint64_t *destination, uint64_t addVal)
{
    return __sync_fetch_and_add(destination, addVal);
}

uint64_t artsAtomicFetchSubU64(volatile uint64_t *destination, uint64_t subVal)
{
    return __sync_fetch_and_sub(destination, subVal);
}

uint64_t artsAtomicAddU64(volatile uint64_t *destination, uint64_t addVal)
{
    return __sync_add_and_fetch(destination, addVal);
}

uint64_t artsAtomicSubU64(volatile uint64_t *destination, uint64_t subVal)
{
    return __sync_sub_and_fetch(destination, subVal);
}

unsigned int artsAtomicSub(volatile unsigned int *destination, unsigned int subVal)
{
    return __sync_sub_and_fetch(destination, subVal);
}

unsigned int artsAtomicCswap(volatile unsigned int *destination, unsigned int oldVal, unsigned int swapIn)
{
    return __sync_val_compare_and_swap(destination, oldVal, swapIn);

}

uint64_t artsAtomicCswapU64(volatile uint64_t *destination, uint64_t oldVal, uint64_t swapIn)
{
    return __sync_val_compare_and_swap(destination, oldVal, swapIn);
}

volatile void * artsAtomicCswapPtr(volatile void **destination, void * oldVal, void * swapIn)
{
    return __sync_val_compare_and_swap(destination, oldVal, swapIn);
}

bool artsAtomicSwapBool(volatile bool *destination, bool value)
{
    return __sync_lock_test_and_set(destination, value);
}

bool artsLock( volatile unsigned int * lock)
{
    while(artsAtomicCswap( lock, 0U, 1U ) == 1U);
    return true;
}

void artsUnlock( volatile unsigned int * lock)
{
    //artsAtomicSwap( lock, 0U );
    *lock=0U;
}

bool artsTryLock( volatile unsigned int * lock)
{
    return (artsAtomicCswap( lock, 0U, 1U ) == 0U);
}

uint64_t artsAtomicFetchAndU64(volatile uint64_t * destination, uint64_t addVal)
{
    return __sync_fetch_and_and(destination, addVal);
}

uint64_t artsAtomicFetchOrU64(volatile uint64_t * destination, uint64_t addVal)
{
    return __sync_fetch_and_or(destination, addVal);
}

uint64_t artsAtomicFetchXOrU64(volatile uint64_t * destination, uint64_t addVal) // @awmm
{
	return __sync_fetch_and_xor(destination, addVal);
}

unsigned int artsAtomicFetchOr(volatile unsigned int * destination, unsigned int addVal)
{
    return __sync_fetch_and_or(destination, addVal);
}

unsigned int artsAtomicFetchAnd(volatile unsigned int * destination, unsigned int addVal)
{
    return __sync_fetch_and_and(destination, addVal);
}

bool artsAtomicCswapSizet(volatile size_t * destination, size_t old, size_t newVal)
{
    return __sync_bool_compare_and_swap(destination, old, newVal);
}

size_t artsAtomicAddSizet(volatile size_t *destination, size_t addVal)
{
    return __sync_add_and_fetch(destination, addVal);
}