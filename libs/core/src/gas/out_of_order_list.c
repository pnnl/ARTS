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
#include "arts/gas/out_of_order_list.h"

#include <time.h>

#include "arts/system/arts_print.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

#define FIRE_LOCK 1U
#define RESET_LOCK 2U

bool reader_oo_try_lock(struct arts_out_of_order_list_s *list) {
  while (1) {
    if (list->writerLock == FIRE_LOCK) {
      return false;
    }
    while (list->writerLock == RESET_LOCK) {
      ;
    }
    arts_atomic_fetch_add(&list->readerLock, 1U);
    if (list->writerLock == 0) {
      break;
    }
    arts_atomic_sub(&list->readerLock, 1U);
  }
  return true;
}

inline void reader_oo_lock(struct arts_out_of_order_list_s *list) {
  while (1) {
    while (list->writerLock) {
      ;
    }
    arts_atomic_fetch_add(&list->readerLock, 1U);
    if (list->writerLock == 0) {
      break;
    }
    arts_atomic_sub(&list->readerLock, 1U);
  }
}

void reader_oo_unlock(struct arts_out_of_order_list_s *list) {
  arts_atomic_sub(&list->readerLock, 1U);
}

void writer_oo_lock(struct arts_out_of_order_list_s *list,
                    unsigned int lock_type) {
  while (arts_atomic_cswap(&list->writerLock, 0U, lock_type) != 0U) {
    ;
  }
  while (list->readerLock) {
    ;
  }
}

void writer_oo_unlock(struct arts_out_of_order_list_s *list) {
  arts_atomic_swap(&list->writerLock, 0U);
}

bool writer_try_oo_lock(struct arts_out_of_order_list_s *list,
                        unsigned int lock_type) {
  // Attempt to acquire the writer lock atomically
  unsigned int temp = arts_atomic_cswap(&list->writerLock, 0U, lock_type);

  if (temp == 0U) {
    // We got the writer lock - now check for readers
    unsigned int reader_count = list->readerLock;
    if (reader_count) {
      // Readers are present - release lock and fail immediately
      writer_oo_unlock(list);
      return false;
    }
    // No readers - we have exclusive access
    return true;
  }

  if (temp == lock_type) {
    // We already hold this lock type - prevent re-entry
    return false;
  }

  // Lock is held by someone else - fail immediately
  return false;
}

bool arts_o_ois_fired(struct arts_out_of_order_list_s *list) {
  return list->isFired;
}

bool arts_out_of_order_list_add_item(struct arts_out_of_order_list_s *add_to_me,
                                     void *item) {
  if (!reader_oo_try_lock(add_to_me)) {
    return false;
  }

  if (arts_o_ois_fired(add_to_me)) {
    reader_oo_unlock(add_to_me);
    return false;
  }
  unsigned int pos = arts_atomic_fetch_add(&add_to_me->count, 1U);
  unsigned int num_elements = pos / OOPERELEMENT;
  unsigned int element_pos = pos % OOPERELEMENT;

  volatile struct arts_out_of_order_element_s *current = &add_to_me->head;
  for (unsigned int i = 0; i < num_elements; i++) {
    if (!current->next) {
      if (i + 1 == num_elements && element_pos == 0) {
        current->next = (struct arts_out_of_order_element_s *)arts_calloc(
            1, sizeof(struct arts_out_of_order_element_s));
      } else {
        while (!current->next) {
          ;
        }
      }
    }
    current = current->next;
  }

  // Always insert and always release lock
  // The CAS is used to wait for slot availability, but we should still unlock
  while (arts_atomic_cswap_ptr((volatile void **)&current->array[element_pos],
                               (void *)0, item)) {
    // Slot was occupied - this shouldn't happen in normal operation
    // but we need to wait for it to become available
  }

  reader_oo_unlock(add_to_me);
  return true;
}

void arts_out_of_order_list_reset(struct arts_out_of_order_list_s *list) {
  if (writer_try_oo_lock(list, RESET_LOCK)) {
    list->isFired = false;
    writer_oo_unlock(list);
  }
}

void delete_oo_elements(struct arts_out_of_order_element_s *current) {
  struct arts_out_of_order_element_s *trail = NULL;
  while (current) {
    for (unsigned int i = 0; i < OOPERELEMENT; i++) {
      arts_free((void *)current->array[i]);
      current->array[i] = NULL;
    }
    trail = current;
    current = (struct arts_out_of_order_element_s *)current->next;
    arts_free(trail);
  }
}

// Not threadsafe
void arts_out_of_order_list_delete(struct arts_out_of_order_list_s *list) {
  /* Free any unconsumed items in the head element */
  for (unsigned int i = 0; i < OOPERELEMENT; i++) {
    arts_free((void *)list->head.array[i]);
    list->head.array[i] = NULL;
  }
  delete_oo_elements((struct arts_out_of_order_element_s *)list->head.next);
  list->head.next = NULL;
  list->isFired = false;
  list->count = 0;
}

void arts_out_of_order_list_fire_callback(
    struct arts_out_of_order_list_s *fire_me, void *local_guid_address,
    void (*callback_t)(void *, void *)) {
  // Retry mechanism: Try multiple times with brief delays
  // This allows readers to complete and release locks
  // 1000 attempts: first 6 are busy-spin, remaining ~994 × 10μs ≈ 10ms
  const int max_retries = 1000;

  for (int attempt = 0; attempt < max_retries; attempt++) {
    if (writer_try_oo_lock(fire_me, FIRE_LOCK)) {
      fire_me->isFired = true;
      unsigned int pos = fire_me->count;
      unsigned int j = 0;
      for (volatile struct arts_out_of_order_element_s *current =
               &fire_me->head;
           current; current = current->next) {
        for (unsigned int i = 0; i < OOPERELEMENT; i++) {
          if (j < pos) {
            volatile void *item = NULL;
            while (!item) {
              item = arts_atomic_swap_ptr((volatile void **)&current->array[i],
                                          (void *)0);
            }
            callback_t((void *)item, local_guid_address);
            j++;
          }
        }
        if (j == pos) {
          break;
        }
        while (!current->next) {
          ;
        }
      }
      fire_me->count = 0;
      struct arts_out_of_order_element_s *p =
          (struct arts_out_of_order_element_s *)fire_me->head.next;
      fire_me->head.next = NULL;
      writer_oo_unlock(fire_me);
      delete_oo_elements(p);
      return;
    }

    // Failed to get lock - YIELD CPU briefly to let readers finish
    // Only YIELD on attempts after the first few quick tries
    if (attempt > 5) {
      // Use nanosleep for 10 microseconds
      struct timespec ts = {0, 10000};
      nanosleep(&ts, NULL);
    }
  }

  // If we get here, we failed after max_retries attempts
  // This should be very rare, but log it for debugging
  ARTS_WARN(
      "arts_out_of_order_list_fire_callback: failed to acquire lock after %d "
      "attempts",
      max_retries);
}
