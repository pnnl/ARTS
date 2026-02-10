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

#include "arts/utils/atomics.h"

unsigned int arts_atomic_swap(volatile unsigned int *destination,
                            unsigned int swap_in) {
  return __sync_lock_test_and_set(destination, swap_in);
}

uint64_t arts_atomic_swap_u64(volatile uint64_t *destination, uint64_t swap_in) {
  return __sync_lock_test_and_set(destination, swap_in);
}

volatile void *arts_atomic_swap_ptr(volatile void **destination, void *swap_in) {
  return __sync_lock_test_and_set(destination, swap_in);
}

unsigned int arts_atomic_add(volatile unsigned int *destination,
                           unsigned int add_val) {
  return __sync_add_and_fetch(destination, add_val);
}

unsigned int arts_atomic_fetch_add(volatile unsigned int *destination,
                                unsigned int add_val) {
  return __sync_fetch_and_add(destination, add_val);
}

uint64_t arts_atomic_fetch_add_u64(volatile uint64_t *destination,
                               uint64_t add_val) {
  return __sync_fetch_and_add(destination, add_val);
}

uint64_t arts_atomic_fetch_sub_u64(volatile uint64_t *destination,
                               uint64_t sub_val) {
  return __sync_fetch_and_sub(destination, sub_val);
}

uint64_t arts_atomic_add_u64(volatile uint64_t *destination, uint64_t add_val) {
  return __sync_add_and_fetch(destination, add_val);
}

uint64_t arts_atomic_sub_u64(volatile uint64_t *destination, uint64_t sub_val) {
  return __sync_sub_and_fetch(destination, sub_val);
}

unsigned int arts_atomic_sub(volatile unsigned int *destination,
                           unsigned int sub_val) {
  return __sync_sub_and_fetch(destination, sub_val);
}

unsigned int arts_atomic_cswap(volatile unsigned int *destination,
                             unsigned int old_val, unsigned int swap_in) {
  return __sync_val_compare_and_swap(destination, old_val, swap_in);
}

uint64_t arts_atomic_cswap_u64(volatile uint64_t *destination, uint64_t old_val,
                            uint64_t swap_in) {
  return __sync_val_compare_and_swap(destination, old_val, swap_in);
}

volatile void *arts_atomic_cswap_ptr(volatile void **destination, void *old_val,
                                  void *swap_in) {
  return __sync_val_compare_and_swap(destination, old_val, swap_in);
}

bool arts_atomic_swap_bool(volatile bool *destination, bool value) {
  return __sync_lock_test_and_set(destination, value);
}

bool arts_lock(volatile unsigned int *lock) {
  while (arts_atomic_cswap(lock, 0U, 1U) == 1U) {
    ;
}
  return true;
}

void arts_unlock(volatile unsigned int *lock) {
  // arts_atomic_swap( lock, 0U );
  *lock = 0U;
}

bool arts_try_lock(volatile unsigned int *lock) {
  return (arts_atomic_cswap(lock, 0U, 1U) == 0U);
}

uint64_t arts_atomic_fetch_and_u64(volatile uint64_t *destination,
                               uint64_t add_val) {
  return __sync_fetch_and_and(destination, add_val);
}

uint64_t arts_atomic_fetch_or_u64(volatile uint64_t *destination, uint64_t add_val) {
  return __sync_fetch_and_or(destination, add_val);
}

uint64_t arts_atomic_fetch_x_or_u64(volatile uint64_t *destination,
                               uint64_t add_val) // @awmm
{
  return __sync_fetch_and_xor(destination, add_val);
}

unsigned int arts_atomic_fetch_or(volatile unsigned int *destination,
                               unsigned int add_val) {
  return __sync_fetch_and_or(destination, add_val);
}

unsigned int arts_atomic_fetch_and(volatile unsigned int *destination,
                                unsigned int add_val) {
  return __sync_fetch_and_and(destination, add_val);
}

void arts_reader_lock(volatile unsigned int *read_lock,
                    const volatile unsigned int *write_lock) {
  while (1) {
    while (*write_lock) {
      ;
}
    arts_atomic_fetch_add(read_lock, 1U);
    if (*write_lock == 0) {
      break;
}
    arts_atomic_sub(read_lock, 1U);
  }
}

void arts_reader_unlock(volatile unsigned int *read_lock) {
  arts_atomic_sub(read_lock, 1U);
}

void arts_writer_lock(const volatile unsigned int *read_lock,
                    volatile unsigned int *write_lock) {
  while (arts_atomic_cswap(write_lock, 0U, 1U) != 0U) {
    ;
}
  while ((*read_lock)) {
    ;
}
  }

bool arts_writer_try_lock(const volatile unsigned int *read_lock,
                       volatile unsigned int *write_lock) {
  if (arts_atomic_cswap(write_lock, 0U, 1U) == 0U) {
    while (*read_lock) {
      ;
}
    return true;
  }
  return false;
}

void arts_writer_unlock(volatile unsigned int *write_lock) {
  arts_atomic_swap(write_lock, 0U);
}