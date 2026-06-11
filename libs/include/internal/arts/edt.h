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
#ifndef ARTS_COMPUTE_EDT_H
#define ARTS_COMPUTE_EDT_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime_types.h"
#include "arts/utils/atomics.h"

extern volatile uint64_t outstanding_edts;
void check_out_edts(uint64_t threshold);

#define INC_OUTSTANDING_EDTS(num_edts)                                         \
  arts_atomic_fetch_add_u64(&outstanding_edts, num_edts)
#define DEC_OUTSTANDING_EDTS(num_edts)                                         \
  arts_atomic_fetch_sub_u64(&outstanding_edts, num_edts)
#define CHECK_OUTSTANDING_EDTS(threshold) check_out_edts(threshold)

bool arts_edt_create_core(struct arts_edt_s *edt, arts_guid_kind_t guid_kind,
                          arts_guid_t *guid, unsigned int rank,
                          unsigned int edt_space, arts_edt_t func_ptr,
                          uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_guid_t hint_finish_event,
                          arts_guid_t hint_output_event, uint64_t arts_id,
                          uint32_t flags);
void arts_edt_delete(struct arts_edt_s *edt);
/* deleter pointer for foreign TUs that allocate arts_edt_s stubs
 * (e.g. remote handler.c arts_handler_edt_create's race-loser cleanup). */
void (*arts_edt_get_deleter(void))(void *);

/* arts_edt_satisfy_slot is the OCR-standard API — declared once in the public
 * header (arts.h); internal TUs that call it include that.  Not re-declared
 * here to avoid a redundant declaration. */

/* OoO replay handlers (g_ooo_table) — operate on the acquired EDT.
 * arts_handler_edt_satisfy_slot is mode-discriminated: DB_MODE_PTR carries an
 * inline payload trailing the args struct, every other mode a reference only.
 */
void arts_handler_edt_satisfy_slot(void *item, void *args);
void arts_handler_edt_destroy(void *item, void *args);

/* Cross-rank wire TX/RX for EDT create + slot satisfy.
 * arts_send_memory_move is the generic create-marshaller (also used by the
 * event-create path), so it lives here and is declared for that caller. */
void arts_send_memory_move(unsigned int rank, arts_guid_t guid, void *ptr,
                           unsigned int mem_size, unsigned message_type,
                           void (*free_method)(void *));
void arts_handler_edt_create(void *ptr);
/* Cross-rank EDT destroy forwarder: forward to the EDT's home rank.  The
 * home-rank RX handler is arts_handler_edt_destroy (declared above as the Cat-B
 * OoO body): the dispatcher decodes the GUID and routes through
 * arts_ooo_dispatch_or_defer_guid(OOO_EDT_DESTROY), so a DESTROY that races
 * ahead of the EDT's CREATE defers and replays on the create handler's drain.
 */
void arts_send_edt_destroy(unsigned int home_rank, arts_guid_t guid);
void arts_send_edt_satisfy_slot(arts_guid_t edt, arts_guid_t db, uint32_t slot,
                                arts_db_access_mode_t mode, void *ptr,
                                unsigned int size);

/* Per-worker EDT-execution context (current_edt, owned-finish-events list,
 * created-DB tracking, ctx save/restore) is declared in arts/edt_context.h. */

void *arts_get_depv(void *edt_ptr);

#ifdef __cplusplus
}
#endif

#endif
