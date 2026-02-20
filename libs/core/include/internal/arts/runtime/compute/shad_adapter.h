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
#ifndef ARTS_RUNTIME_COMPUTE_SHADADAPTER_H
#define ARTS_RUNTIME_COMPUTE_SHADADAPTER_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"

arts_guid_t arts_edt_create_shad(arts_edt_t func_ptr, unsigned int route,
                                 uint32_t paramc, const uint64_t *paramv);
arts_guid_t arts_active_message_shad(arts_edt_t func_ptr, unsigned int route,
                                     uint32_t paramc, const uint64_t *paramv,
                                     void *data, unsigned int size,
                                     arts_guid_t epoch_guid);
void arts_synchronous_active_message_shad(arts_edt_t func_ptr,
                                          unsigned int route, uint32_t paramc,
                                          const uint64_t *paramv, void *data,
                                          unsigned int size);

void arts_inc_lock_shad();
void arts_dec_lock_shad();
void arts_check_lock_shad();
void arts_start_intro_shad(unsigned int start);
void arts_stop_intro_shad();
arts_guid_t arts_allocate_local_buffer_shad(void **buffer,
                                            uint32_t *size_to_write,
                                            arts_guid_t epoch_guid);

bool arts_shad_alias_try_lock(volatile uint64_t *lock);
void arts_shad_alias_unlock(volatile uint64_t *lock);

#ifdef __cplusplus
}
#endif

#endif /* SHADADAPTER_H */
