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
#include "arts/sync/epoch_pool.h"

#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/runtime_types.h"
#include "arts/sync/epoch.h" /* arts_send_epoch_init_pool */
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

#define EPOCH_BIT 0x8000000000000000

#define DEFAULT_EPOCH_POOL_SIZE 4096
ARTS_THREAD_LOCAL arts_epoch_pool_t *epoch_thread_pool;

arts_epoch_pool_t *arts_epoch_pool_create(arts_guid_t *epoch_pool_guid,
                                          unsigned int pool_size,
                                          arts_guid_t *start_guid) {
  /* the pool_guid still uses ARTS_GUID_EDT since arts_epoch_pool_t is
   * a different struct.  Only individual epoch entries get the
   * ARTS_GUID_EPOCH tag so lookup_epoch + the cb deleter-by-kind route them
   * correctly.  The pool itself is freed explicitly in arts_epoch_delete /
   * arts_epoch_pool_clean. */
  if (*epoch_pool_guid == NULL_GUID) {
    *epoch_pool_guid =
        arts_guid_create_for_rank(arts_global_rank_id, ARTS_GUID_EDT);
  }

  if (*start_guid == NULL_GUID) {
    *start_guid = arts_guid_reserve_range(ARTS_GUID_EPOCH, pool_size,
                                          arts_global_rank_id);
  }

  arts_epoch_pool_t *epoch_pool = (arts_epoch_pool_t *)arts_calloc(
      1, sizeof(arts_epoch_pool_t) + (sizeof(arts_epoch_t) * pool_size));
  epoch_pool->index = 0;
  epoch_pool->outstanding = pool_size;
  epoch_pool->size = pool_size;

  /* Pool wrapper: install with a NULL deleter (deleter-by-kind would pick
   * arts_edt_deleter for the EDT-tagged guid and type-confuse the pool).  The
   * route_table holds it for lookup only; arts_epoch_delete frees the storage.
   */
  arts_route_table_install_with_deleter(epoch_pool, *epoch_pool_guid, NULL);
  for (unsigned int i = 0; i < pool_size; i++) {
    /* Pool entries are NOT individually heap-allocated — arts_epoch_deleter
     * detects this via pool_guid != NULL_GUID and skips the free.  The cb
     * (with deleter-by-kind) is created at install time by add_item_race. */
    epoch_pool->pool[i].phase = PHASE_1;
    epoch_pool->pool[i].pool_guid = *epoch_pool_guid;
    epoch_pool->pool[i].guid = arts_guid_from_index(*start_guid, i);
    epoch_pool->pool[i].queued =
        (arts_guid_is_local(*epoch_pool_guid)) ? 0 : EPOCH_BIT;
    if (!arts_guid_is_local(*epoch_pool_guid)) {
      /* add_item_race fires the OoO list internally on a successful install. */
      arts_route_table_install_if_absent(&epoch_pool->pool[i],
                                     epoch_pool->pool[i].guid,
                                     arts_global_rank_id, false);
    }
  }

  return epoch_pool;
}

void arts_epoch_pool_clean() {
  arts_epoch_pool_t *trail_pool = NULL;
  arts_epoch_pool_t *pool = epoch_thread_pool;

  while (pool) {
    if (pool->index == epoch_thread_pool->size && !pool->outstanding) {
      arts_epoch_pool_t *to_free = pool;

      pool = pool->next;

      if (trail_pool) {
        trail_pool->next = pool;
      } else {
        epoch_thread_pool = pool;
      }

      arts_free(to_free);
    } else {
      trail_pool = pool;
      pool = pool->next;
    }
  }
}

void arts_link_epoch_pool_to_tls(arts_epoch_pool_t *pool) {
  pool->next = epoch_thread_pool;
  epoch_thread_pool = pool;
}

void arts_cleanup_epoch_pools(void) {
  arts_epoch_pool_t *pool = epoch_thread_pool;
  while (pool) {
    arts_epoch_pool_t *next = pool->next;
    arts_free(pool);
    pool = next;
  }
  epoch_thread_pool = NULL;
}

arts_epoch_t *arts_epoch_pool_get(arts_guid_t edt_guid, unsigned int slot) {
  //    arts_epoch_pool_clean();
  arts_epoch_pool_t *trail_pool = NULL;
  arts_epoch_pool_t *pool = epoch_thread_pool;
  arts_epoch_t *epoch = NULL;
  while (!epoch) {
    if (!pool) {
      arts_guid_t pool_guid = NULL_GUID;
      arts_guid_t start_guid = NULL_GUID;
      pool = arts_epoch_pool_create(&pool_guid, DEFAULT_EPOCH_POOL_SIZE,
                                    &start_guid);

      if (trail_pool) {
        trail_pool->next = pool;
      } else {
        epoch_thread_pool = pool;
      }

      for (unsigned int i = 0; i < arts_global_rank_count; i++) {
        if (i != arts_global_rank_id) {
          arts_send_epoch_init_pool(i, DEFAULT_EPOCH_POOL_SIZE, start_guid,
                                    pool_guid);
        }
      }
    }

    if (pool->index < pool->size) {
      epoch = &pool->pool[pool->index++];
    } else {
      trail_pool = pool;
      pool = pool->next;
    }
  }

  epoch->termination_exit_guid = edt_guid;
  epoch->termination_exit_slot = slot;
  /* add_item_race fires the OoO list internally on a successful install. */
  arts_route_table_install_if_absent(epoch, epoch->guid, arts_global_rank_id,
                                 false);
  return epoch;
}
