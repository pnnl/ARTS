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
#include "arts/utils/array_list.h"
#include "arts/utils/atomics.h"

extern volatile uint64_t outstanding_edts;
void check_out_edts(uint64_t threshold);

#define INC_OUTSTANDING_EDTS(num_edts)                                         \
  arts_atomic_fetch_add_u64(&outstanding_edts, num_edts)
#define DEC_OUTSTANDING_EDTS(num_edts)                                         \
  arts_atomic_fetch_sub_u64(&outstanding_edts, num_edts)
#define CHECK_OUTSTANDING_EDTS(threshold) check_out_edts(threshold)

bool arts_edt_create_internal(struct arts_edt_s *edt, arts_type_t mode,
                              arts_guid_t *guid, unsigned int route,
                              unsigned int numa_domain, unsigned int edt_space,
                              arts_guid_t output_buffer, arts_edt_t func_ptr,
                              uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, bool use_epoch,
                              arts_guid_t epoch_guid, bool has_depv,
                              uint64_t arts_id);
void arts_edt_free(struct arts_edt_s *edt);
void arts_edt_delete(struct arts_edt_s *edt);
void internal_signal_edt(arts_guid_t edt_packet, uint32_t slot,
                         arts_guid_t data_guid, arts_db_access_mode_t mode,
                         void *ptr, unsigned int size);
void internal_signal_edt_with_mode(arts_guid_t edt_packet, uint32_t slot,
                                   arts_guid_t data_guid,
                                   arts_db_access_mode_t mode);

void arts_set_dep_mode(arts_guid_t edt_guid, uint32_t slot,
                      arts_db_access_mode_t mode);

typedef struct {
  arts_guid_t current_edt_guid;
  struct arts_edt_s *current_edt;
  void *epoch_list;
  void *created_db_list;
} thread_local_t;

void arts_set_thread_local_edt_info(struct arts_edt_s *edt);
void arts_unset_thread_local_edt_info();
void arts_save_thread_local(thread_local_t *tl);
void arts_restore_thread_local(thread_local_t *tl);

bool arts_set_current_epoch_guid(arts_guid_t epoch_guid);
arts_guid_t *arts_check_epoch_is_root(arts_guid_t to_check);
void arts_increment_finished_epoch_list();

void *arts_get_depv(void *edt_ptr);

void arts_track_created_db(arts_guid_t guid);
arts_array_list_t *arts_get_created_db_list(void);
void arts_cleanup_edt_tls(void);
#ifdef __cplusplus
}
#endif

#endif
