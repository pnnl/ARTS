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
#ifndef ARTS_SYNC_EDT_CONTEXT_H
#define ARTS_SYNC_EDT_CONTEXT_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/defs.h"             /* ARTS_THREAD_LOCAL */
#include "arts/runtime_types.h"    /* arts_guid_t, struct arts_edt_s */
#include "arts/utils/array_list.h" /* arts_array_list_t */

/*
 * Per-worker EDT-execution context.
 *
 * Each worker thread maintains, while it runs an EDT, a small set of
 * thread-local pieces of state: which EDT is currently running and the list
 * of DBs the running EDT has created.  These are owned by the worker and are
 * saved/restored around any nested EDT-like execution (e.g. an event waiter
 * that re-enters the runtime).
 *
 * The running-EDT pointer has external linkage so foreign TUs that consult
 * the current EDT directly (memory/coherence, event satisfy fast-path,
 * system info) can read it without a route-table lookup.
 */
extern ARTS_THREAD_LOCAL struct arts_edt_s *current_edt;

/* Snapshot of the per-worker EDT context, used to save and restore the
 * thread-locals around a nested execution. */
typedef struct {
  arts_guid_t current_edt_guid;
  struct arts_edt_s *current_edt;
  void *created_db_list;
  void *owned_finish_list;
} arts_edt_ctx_t;

/* Run-start / run-end context management (called from the EDT run path). */
void arts_set_thread_local_edt_info(struct arts_edt_s *edt);
void arts_unset_thread_local_edt_info();
void arts_edt_ctx_save(arts_edt_ctx_t *tl);
void arts_edt_ctx_restore(arts_edt_ctx_t *tl);
void arts_cleanup_edt_tls(void);

/* Created-DB tracking on the current worker (auto-acquire / release path). */
void arts_track_created_db(arts_guid_t guid);
arts_array_list_t *arts_get_created_db_list(void);

/* Finish-event owned-list: register creator-token, consume on wait, cleanup. */
void arts_owned_finish_register(arts_guid_t fe_guid);
void arts_owned_finish_consume(arts_guid_t fe_guid);
void arts_owned_finish_cleanup(void);
/* arts_current_finish_event is declared once in the public header (arts.h);
 * callers include that.  Not re-declared here (redundant declaration). */

#ifdef __cplusplus
}
#endif

#endif
