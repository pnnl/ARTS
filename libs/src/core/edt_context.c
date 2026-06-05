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
#include "arts/epoch.h"            /* arts_epoch_inc_finished, shutdown */
#include "arts/runtime_state.h"    /* arts_thread_info */
#include "arts/system/print.h"
#include "arts/utils/array_list.h"

#define MAX_EPOCH_ARRAY_LIST 32

ARTS_THREAD_LOCAL arts_array_list_t *epoch_list = NULL;
ARTS_THREAD_LOCAL struct arts_edt_s *current_edt = NULL;
ARTS_THREAD_LOCAL arts_array_list_t *created_db_list = NULL;

void arts_set_current_epoch_guid(arts_guid_t epoch_guid) {
  if (epoch_guid) {
    if (!epoch_list) {
      epoch_list = arts_new_array_list(sizeof(arts_guid_t), 8);
    }
    arts_push_to_array_list(epoch_list, &epoch_guid);
    if (current_edt) {
      current_edt->epoch_guid = epoch_guid;
    }
  }
}

arts_guid_t arts_epoch_get_current_guid() {
  if (epoch_list) {
    uint64_t length = arts_length_array_list(epoch_list);
    if (length) {
      arts_guid_t *guid =
          (arts_guid_t *)arts_get_from_array_list(epoch_list, length - 1);
      return *guid;
    }
  }
  return NULL_GUID;
}

arts_guid_t *arts_check_epoch_is_root(arts_guid_t to_check) {
  if (epoch_list) {
    uint64_t length = arts_length_array_list(epoch_list);
    for (uint64_t i = 0; i < length; i++) {
      arts_guid_t *guid =
          (arts_guid_t *)arts_get_from_array_list(epoch_list, i);
      if (*guid == to_check) {
        return guid;
      }
    }
  }
  ARTS_INFO("ERROR %lu is not a valid epoch", to_check);
  return NULL;
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

  if (epoch_list) {
    arts_reset_array_list(epoch_list);
  }

  if (created_db_list) {
    arts_reset_array_list(created_db_list);
  }

  arts_set_current_epoch_guid(current_edt->epoch_guid);
}

void arts_edt_ctx_save(arts_edt_ctx_t *tl) {
  TIME_CONTEXT_SWITCH_START();
  tl->current_edt_guid = arts_thread_info.current_edt_guid;
  tl->current_edt = current_edt;
  tl->epoch_list = (void *)epoch_list;
  tl->created_db_list = (void *)created_db_list;

  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
  epoch_list = NULL;
  created_db_list = NULL;
  TIME_CONTEXT_SWITCH_STOP();
}

void arts_edt_ctx_restore(arts_edt_ctx_t *tl) {
  TIME_CONTEXT_SWITCH_START();
  arts_thread_info.current_edt_guid = tl->current_edt_guid;
  current_edt = tl->current_edt;
  if (epoch_list) {
    arts_delete_array_list(epoch_list);
  }
  epoch_list = (arts_array_list_t *)tl->epoch_list;
  if (created_db_list) {
    arts_delete_array_list(created_db_list);
  }
  created_db_list = (arts_array_list_t *)tl->created_db_list;
  TIME_CONTEXT_SWITCH_STOP();
}

void arts_cleanup_edt_tls() {
  if (epoch_list) {
    arts_delete_array_list(epoch_list);
    epoch_list = NULL;
  }
  if (created_db_list) {
    arts_delete_array_list(created_db_list);
    created_db_list = NULL;
  }
}

void arts_epoch_list_mark_finished() {
  if (epoch_list) {

    unsigned int epoch_array_length = arts_length_array_list(epoch_list);
    for (unsigned int i = 0; i < epoch_array_length; i++) {
      arts_guid_t *guid =
          (arts_guid_t *)arts_get_from_array_list(epoch_list, i);
#if ARTS_LOG_LEVEL >= 2
      uint64_t current_id = current_edt ? current_edt->arts_id : 0;
      ARTS_INFO("Current EDT[Id:%lu, Guid:%lu] - Unsetting Epoch [Guid:%lu]",
                current_id, arts_thread_info.current_edt_guid, *guid);
#endif
      if (*guid) {
        arts_epoch_inc_finished(*guid);
      }
    }

    if (epoch_array_length > MAX_EPOCH_ARRAY_LIST) {
      arts_delete_array_list(epoch_list);
      epoch_list = NULL;
    } else {
      arts_reset_array_list(epoch_list);
    }
  }
  arts_shutdown_epoch_inc_finished();
}

void arts_unset_thread_local_edt_info() {
  arts_epoch_list_mark_finished();
  /* finish_event tracking: emit DECR on completion of the current EDT.
   * - Inherited finish_event: balances the INCR emitted at create time.
   * - Own (ARTS_EDT_FLAG_FINISH or cross-node proxy) finish_event:
   *   counter==0 fires the latch, propagating DECR to the parent's
   *   finish_event via the dep registered at allocation.
   * `current_edt` is still valid here (cleared on the next line). */
  if (current_edt && current_edt->finish_event != NULL_GUID) {
    arts_event_satisfy_slot(current_edt->finish_event, NULL_GUID,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }
  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
}
