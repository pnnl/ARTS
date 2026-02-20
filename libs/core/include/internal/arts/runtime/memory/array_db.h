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

#ifndef ARTS_RUNTIME_MEMORY_ARRAY_DB_H
#define ARTS_RUNTIME_MEMORY_ARRAY_DB_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/gas/out_of_order.h"
#include "arts/runtime/rt.h"

unsigned int arts_get_size_array_db(arts_array_db_t *array);
unsigned int get_offset_from_index(arts_array_db_t *array, unsigned int index);
unsigned int get_rank_from_index(arts_array_db_t *array, unsigned int index);
arts_guid_t get_array_db_guid(arts_array_db_t *array);
void internal_atomic_add_in_array_db(arts_guid_t db_guid, unsigned int index,
                                     unsigned int to_add, arts_guid_t edt_guid,
                                     unsigned int slot, arts_guid_t epoch_guid);
void internal_atomic_compare_and_swap_in_array_db(
    arts_guid_t db_guid, unsigned int index, unsigned int old_value,
    unsigned int new_value, arts_guid_t edt_guid, unsigned int slot,
    arts_guid_t epoch_guid);

// OOO struct definitions for array DB deferred operations
struct oo_atomic_add_in_array_db_s {
  enum arts_out_of_order_type type;
  arts_guid_t db_guid;
  arts_guid_t edt_guid;
  arts_guid_t epoch_guid;
  unsigned int slot;
  unsigned int index;
  unsigned int to_add;
};

struct oo_atomic_compare_and_swap_in_array_db_s {
  enum arts_out_of_order_type type;
  arts_guid_t db_guid;
  arts_guid_t edt_guid;
  arts_guid_t epoch_guid;
  unsigned int slot;
  unsigned int index;
  unsigned int old_value;
  unsigned int new_value;
};

// Remote handler functions (defined in array_db_remote.c)
void arts_remote_atomic_add_in_array_db(unsigned int rank, arts_guid_t db_guid,
                                        unsigned int index, unsigned int to_add,
                                        arts_guid_t edt_guid, unsigned int slot,
                                        arts_guid_t epoch_guid);
void arts_remote_handle_atomic_add_in_array_db(void *pack);
void arts_remote_atomic_compare_and_swap_in_array_db(
    unsigned int rank, arts_guid_t db_guid, unsigned int index,
    unsigned int old_value, unsigned int new_value, arts_guid_t edt_guid,
    unsigned int slot, arts_guid_t epoch_guid);
void arts_remote_handle_atomic_compare_and_swap_in_array_db(void *pack);

// OOO registration functions (defined in array_db_oo.c)
void arts_out_of_order_atomic_add_in_array_db(
    arts_guid_t db_guid, unsigned int index, unsigned int to_add,
    arts_guid_t edt_guid, unsigned int slot, arts_guid_t epoch_guid);
void arts_out_of_order_atomic_compare_and_swap_in_array_db(
    arts_guid_t db_guid, unsigned int index, unsigned int old_value,
    unsigned int new_value, arts_guid_t edt_guid, unsigned int slot,
    arts_guid_t epoch_guid);

#ifdef __cplusplus
}
#endif
#endif /* ARTS_RUNTIME_MEMORY_ARRAY_DB_H */
