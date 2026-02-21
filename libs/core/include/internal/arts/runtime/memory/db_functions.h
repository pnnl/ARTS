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
#ifndef ARTS_RUNTIME_MEMORY_DBFUNCTIONS_H
#define ARTS_RUNTIME_MEMORY_DBFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime/rt.h"

void arts_db_create_internal(arts_guid_t guid, void *addr, uint64_t len,
                             uint64_t packet_size, arts_type_t mode,
                             uint64_t arts_id);
void acquire_dbs(struct arts_edt_s *edt);
void release_dbs(unsigned int depc, arts_edt_dep_t *depv,
                 const arts_db_mode_t *modes, bool gpu);
void arts_release_created_dbs(void);
bool arts_add_db_duplicate(struct arts_db_s *db, unsigned int rank,
                           struct arts_edt_s *edt, arts_guid_t edt_guid,
                           unsigned int slot, arts_db_mode_t mode,
                           bool *on_head);
void prep_dbs(unsigned int depc, arts_edt_dep_t *depv,
              const arts_db_mode_t *modes, bool gpu);
void internal_put_in_db(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                        unsigned int slot, unsigned int offset,
                        unsigned int size, arts_guid_t epoch_guid,
                        unsigned int rank);

void *arts_db_malloc(arts_type_t mode, unsigned int size);
void arts_db_free(void *ptr);

#ifdef __cplusplus
}
#endif
#endif /* artsDBFUNCTIONS_H */
