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
#include "arts/system/print.h"
#include "arts/utils/vector.h"

ARTS_THREAD_LOCAL struct arts_edt_s *current_edt = NULL;
ARTS_THREAD_LOCAL arts_vector_t created_db_list = {0};
ARTS_THREAD_LOCAL arts_vector_t owned_finish_list = {0};

void arts_owned_finish_register(arts_guid_t fe_guid) {
  if (!fe_guid) {
    return;
  }
  if (owned_finish_list.element_size == 0) {
    arts_vector_init(&owned_finish_list, sizeof(arts_guid_t), 8);
  }
  arts_vector_push(&owned_finish_list, &fe_guid);
}

/* Mark fe_guid consumed (by arts_event_wait) so completion cleanup skips it.
 * Removed, not blanked: cleanup walks what is still OWED, so a consumed
 * entry leaves the list instead of lingering as a slot to skip. */
void arts_owned_finish_consume(arts_guid_t fe_guid) {
  uint64_t n = arts_vector_count(&owned_finish_list);
  for (uint64_t i = 0; i < n; i++) {
    arts_guid_t *g = (arts_guid_t *)arts_vector_at(&owned_finish_list, i);
    if (*g == fe_guid) {
      arts_vector_swap_remove(&owned_finish_list, i);
      return;
    }
  }
}

/* Completion: DECR creator-token of every finish event not consumed by wait. */
void arts_owned_finish_cleanup(void) {
  uint64_t n = arts_vector_count(&owned_finish_list);
  for (uint64_t i = 0; i < n; i++) {
    arts_guid_t g = *(arts_guid_t *)arts_vector_at(&owned_finish_list, i);
    arts_event_satisfy_slot(g, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  }
  arts_vector_clear(&owned_finish_list);
}

arts_guid_t arts_current_finish_event(void) {
  return current_edt ? current_edt->finish_event : NULL_GUID;
}

void arts_track_created_db(arts_guid_t guid) {
  if (created_db_list.element_size == 0) {
    arts_vector_init(&created_db_list, sizeof(arts_guid_t), 8);
  }
  arts_vector_push(&created_db_list, &guid);
}

arts_vector_t *arts_get_created_db_list(void) { return &created_db_list; }

void arts_set_thread_local_edt_info(struct arts_edt_s *edt) {
  /* The created-DB list must already be EMPTY here: the epilogue
   * (arts_release_created_dbs) drains it after every EDT, and a parked outer
   * EDT's entries are not in the TLS at all — arts_edt_ctx_save moved them
   * out by value.  There is deliberately no clear: clearing would silently
   * DROP any hold a broken run path failed to release, and the entries would
   * otherwise be released against the next EDT's epilogue — a stranger's
   * task.  An entry surviving to this point is that bug, made loud. */
#if ARTS_LOG_LEVEL >= 3
  if (arts_vector_count(&created_db_list) != 0) {
    ARTS_WARN("created_db_list carries %lu entries into a new EDT — a hold "
              "leaked across tasks",
              (unsigned long)arts_vector_count(&created_db_list));
  }
#endif
  arts_thread_info.current_edt_guid = edt->guid;
  current_edt = edt;
}

void arts_edt_ctx_save(arts_edt_ctx_t *tl) {
  TIME_CONTEXT_SWITCH_START();
  tl->current_edt_guid = arts_thread_info.current_edt_guid;
  tl->current_edt = current_edt;
  tl->created_db_list = created_db_list; /* struct copy: ownership moves */
  tl->owned_finish_list = owned_finish_list;

  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
  created_db_list = (arts_vector_t){0};
  owned_finish_list = (arts_vector_t){0};
  TIME_CONTEXT_SWITCH_STOP();
}

void arts_edt_ctx_restore(arts_edt_ctx_t *tl) {
  TIME_CONTEXT_SWITCH_START();
  arts_thread_info.current_edt_guid = tl->current_edt_guid;
  current_edt = tl->current_edt;
  arts_vector_free(&created_db_list);
  created_db_list = tl->created_db_list;
  arts_vector_free(&owned_finish_list);
  owned_finish_list = tl->owned_finish_list;
  TIME_CONTEXT_SWITCH_STOP();
}

void arts_cleanup_edt_tls() {
  arts_vector_free(&created_db_list);
  arts_vector_free(&owned_finish_list);
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
