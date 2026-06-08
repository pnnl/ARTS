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
#include "arts/edt_context.h"

#include "arts.h"
#include "arts/counter/Preamble.h" /* TIME_CONTEXT_SWITCH_START/STOP */
#include "arts/runtime_state.h"    /* arts_thread_info */
#include "arts/utils/array_list.h"

ARTS_THREAD_LOCAL struct arts_edt_s *current_edt = NULL;
ARTS_THREAD_LOCAL arts_array_list_t *created_db_list = NULL;
ARTS_THREAD_LOCAL arts_array_list_t *owned_finish_list = NULL;

void arts_owned_finish_register(arts_guid_t fe_guid) {
  if (!fe_guid) {
    return;
  }
  if (!owned_finish_list) {
    owned_finish_list = arts_new_array_list(sizeof(arts_guid_t), 8);
  }
  arts_push_to_array_list(owned_finish_list, &fe_guid);
}

/* Mark fe_guid consumed (by arts_event_wait) so completion cleanup skips it.
 * Linear scan + zero-out (lists here are tiny — orchestrator EDTs only). */
void arts_owned_finish_consume(arts_guid_t fe_guid) {
  if (!owned_finish_list) {
    return;
  }
  uint64_t n = arts_length_array_list(owned_finish_list);
  for (uint64_t i = 0; i < n; i++) {
    arts_guid_t *g =
        (arts_guid_t *)arts_get_from_array_list(owned_finish_list, i);
    if (*g == fe_guid) {
      *g = NULL_GUID;
      return;
    }
  }
}

/* Completion: DECR creator-token of every finish event not consumed by wait. */
void arts_owned_finish_cleanup(void) {
  if (!owned_finish_list) {
    return;
  }
  uint64_t n = arts_length_array_list(owned_finish_list);
  for (uint64_t i = 0; i < n; i++) {
    arts_guid_t *g =
        (arts_guid_t *)arts_get_from_array_list(owned_finish_list, i);
    if (*g != NULL_GUID) {
      arts_event_satisfy_slot(*g, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
    }
  }
  arts_reset_array_list(owned_finish_list);
}

arts_guid_t arts_current_finish_event(void) {
  return current_edt ? current_edt->finish_event : NULL_GUID;
}

void arts_track_created_db(arts_guid_t guid) {
  if (!created_db_list) {
    created_db_list = arts_new_array_list(sizeof(arts_guid_t), 65536);
  }
  arts_push_to_array_list(created_db_list, &guid);
}

arts_array_list_t *arts_get_created_db_list(void) { return created_db_list; }

void arts_set_thread_local_edt_info(struct arts_edt_s *edt) {
  arts_thread_info.current_edt_guid = edt->guid;
  current_edt = edt;

  if (created_db_list) {
    arts_reset_array_list(created_db_list);
  }
}

void arts_edt_ctx_save(arts_edt_ctx_t *tl) {
  TIME_CONTEXT_SWITCH_START();
  tl->current_edt_guid = arts_thread_info.current_edt_guid;
  tl->current_edt = current_edt;
  tl->created_db_list = (void *)created_db_list;
  tl->owned_finish_list = (void *)owned_finish_list;

  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
  created_db_list = NULL;
  owned_finish_list = NULL;
  TIME_CONTEXT_SWITCH_STOP();
}

void arts_edt_ctx_restore(arts_edt_ctx_t *tl) {
  TIME_CONTEXT_SWITCH_START();
  arts_thread_info.current_edt_guid = tl->current_edt_guid;
  current_edt = tl->current_edt;
  if (created_db_list) {
    arts_delete_array_list(created_db_list);
  }
  created_db_list = (arts_array_list_t *)tl->created_db_list;
  if (owned_finish_list) {
    arts_delete_array_list(owned_finish_list);
  }
  owned_finish_list = (arts_array_list_t *)tl->owned_finish_list;
  TIME_CONTEXT_SWITCH_STOP();
}

void arts_cleanup_edt_tls() {
  if (created_db_list) {
    arts_delete_array_list(created_db_list);
    created_db_list = NULL;
  }
  if (owned_finish_list) {
    arts_delete_array_list(owned_finish_list);
    owned_finish_list = NULL;
  }
}

void arts_unset_thread_local_edt_info() {
  /* finish_event tracking: emit DECR on completion of the current EDT.
   * - Joined/inherited finish_event: balances the INCR emitted at create time.
   * - Cross-node proxy finish_event: counter==0 fires the latch, propagating
   *   DECR to the parent's finish_event via the dep registered at allocation.
   * `current_edt` is still valid here (cleared on the next line). */
  if (current_edt && current_edt->finish_event != NULL_GUID) {
    arts_event_satisfy_slot(current_edt->finish_event, NULL_GUID,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }
  arts_owned_finish_cleanup(); /* DECR creator-token of un-waited finish events
                                */
  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
}
