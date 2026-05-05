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
#ifndef ARTS_UTILS_ATOMICS_H
#define ARTS_UTILS_ATOMICS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#define HW_MEMORY_FENCE() __sync_synchronize()
#define COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT()                    \
  __asm__ volatile("" : : : "memory")

unsigned int arts_atomic_swap(volatile unsigned int *destination,
                              unsigned int swap_in);
uint64_t arts_atomic_swap_u64(volatile uint64_t *destination, uint64_t swap_in);
volatile void *arts_atomic_swap_ptr(volatile void **destination, void *swap_in);
unsigned int arts_atomic_sub(volatile unsigned int *destination,
                             unsigned int sub_val);
unsigned int arts_atomic_add(volatile unsigned int *destination,
                             unsigned int add_val);
unsigned int arts_atomic_fetch_add(volatile unsigned int *destination,
                                   unsigned int add_val);
unsigned int arts_atomic_cswap(volatile unsigned int *destination,
                               unsigned int old_val, unsigned int swap_in);
uint64_t arts_atomic_cswap_u64(volatile uint64_t *destination, uint64_t old_val,
                               uint64_t swap_in);
volatile void *arts_atomic_cswap_ptr(volatile void **destination, void *old_val,
                                     void *swap_in);
bool arts_atomic_swap_bool(volatile bool *destination, bool value);
uint64_t arts_atomic_fetch_add_u64(volatile uint64_t *destination,
                                   uint64_t add_val);
uint64_t arts_atomic_fetch_sub_u64(volatile uint64_t *destination,
                                   uint64_t sub_val);
uint64_t arts_atomic_add_u64(volatile uint64_t *destination, uint64_t add_val);
uint64_t arts_atomic_sub_u64(volatile uint64_t *destination, uint64_t sub_val);
bool arts_lock(volatile unsigned int *lock);
void arts_unlock(volatile unsigned int *lock);
bool arts_try_lock(volatile unsigned int *lock);
uint64_t arts_atomic_fetch_and_u64(volatile uint64_t *destination,
                                   uint64_t add_val);
uint64_t arts_atomic_fetch_or_u64(volatile uint64_t *destination,
                                  uint64_t add_val);
uint64_t arts_atomic_fetch_x_or_u64(volatile uint64_t *destination,
                                    uint64_t add_val); //@awmm
unsigned int arts_atomic_fetch_or(volatile unsigned int *destination,
                                  unsigned int add_val);
unsigned int arts_atomic_fetch_and(volatile unsigned int *destination,
                                   unsigned int add_val);

/* Acquire-load primitives (Phase 2.2) — used pervasively by the v3 RC
 * coherence layer (cache->destroy_state, cache->writer_count, etc.).
 * Implemented via __atomic_load_n (__ATOMIC_ACQUIRE) so the load pairs
 * with __sync_*-based stores already in arts_atomic_*. */
unsigned int arts_atomic_read(volatile unsigned int *destination);
uint64_t arts_atomic_read_u64(volatile uint64_t *destination);

void arts_reader_lock(volatile unsigned int *read_lock,
                      const volatile unsigned int *write_lock);
void arts_reader_unlock(volatile unsigned int *read_lock);
void arts_writer_lock(const volatile unsigned int *read_lock,
                      volatile unsigned int *write_lock);
bool arts_writer_try_lock(const volatile unsigned int *read_lock,
                          volatile unsigned int *write_lock);
void arts_writer_unlock(volatile unsigned int *write_lock);
#ifdef __cplusplus
}
#endif

#endif
