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
#ifndef ARTS_RUNTIME_COMPUTE_EDTFUNCTIONS_H
#define ARTS_RUNTIME_COMPUTE_EDTFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime/rt.h"

bool arts_edt_create_internal(struct arts_edt_s *edt, arts_type_t mode,
                           arts_guid_t *guid, unsigned int route,
                           unsigned int cluster, unsigned int edt_space,
                           arts_guid_t output_buffer, arts_edt_t func_ptr,
                           uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                           bool use_epoch, arts_guid_t epoch_guid, bool has_depv,
                           uint64_t arts_id);
void arts_edt_delete(struct arts_edt_s *edt);
void internal_signal_edt(arts_guid_t edt_packet, uint32_t slot, arts_guid_t data_guid,
                       arts_type_t mode, void *ptr, unsigned int size);
void internal_signal_edt_with_mode(arts_guid_t edt_packet, uint32_t slot,
                               arts_guid_t data_guid, arts_type_t mode,
                               arts_type_t acquire_mode);

typedef struct {
  arts_guid_t current_edt_guid;
  struct arts_edt_s *current_edt;
  void *epoch_list;
} thread_local_t;

void arts_set_thread_local_edt_info(struct arts_edt_s *edt);
void arts_unset_thread_local_edt_info();
void arts_save_thread_local(thread_local_t *tl);
void arts_restore_thread_local(thread_local_t *tl);

bool arts_set_current_epoch_guid(arts_guid_t epoch_guid);
arts_guid_t *arts_check_epoch_is_root(arts_guid_t to_check);
void arts_increment_finished_epoch_list();

void *arts_get_depv(void *edt_ptr);
#ifdef __cplusplus
}
#endif

#endif
