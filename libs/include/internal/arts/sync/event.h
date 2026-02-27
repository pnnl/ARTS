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
#ifndef ARTS_SYNC_EVENT_H
#define ARTS_SYNC_EVENT_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts/runtime_types.h"
#define INITIAL_DEPENDENT_SIZE 4

bool arts_event_create_internal(arts_guid_t *guid, unsigned int route,
                                unsigned int dependent_count,
                                unsigned int latch_count,
                                arts_event_types_t event_type,
                                arts_guid_t event_data);

void arts_event_free(struct arts_event_s *event);

bool arts_event_create_channel_internal(arts_guid_t *guid, unsigned int route,
                                        arts_guid_t data_guid);

/* Internal channel-event dependence helpers (not public API). */
void arts_event_add_dependence_with_mode(arts_guid_t event_source,
                                         arts_guid_t edt_dest,
                                         uint32_t edt_slot,
                                         arts_db_access_mode_t mode);
void arts_event_add_dependence_with_byte_offset(
    arts_guid_t event_source, arts_guid_t edt_dest, uint32_t edt_slot,
    arts_db_access_mode_t mode, uint64_t byte_offset, uint64_t len);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_SYNC_EVENT_H */
