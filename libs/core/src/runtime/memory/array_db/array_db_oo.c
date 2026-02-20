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
#include "arts/runtime/memory/array_db.h"

#include "arts/gas/route_table.h"
#include "arts/system/arts_print.h"
#include "arts/utils/malloc.h"

void arts_out_of_order_atomic_add_in_array_db(
    arts_guid_t db_guid, unsigned int index, unsigned int to_add,
    arts_guid_t edt_guid, unsigned int slot, arts_guid_t epoch_guid) {
  struct oo_atomic_add_in_array_db_s *req =
      (struct oo_atomic_add_in_array_db_s *)arts_malloc(
          sizeof(struct oo_atomic_add_in_array_db_s));
  req->type = OO_ATOMIC_ADD_IN_ARRAY_DB;
  req->edt_guid = edt_guid;
  req->db_guid = db_guid;
  req->epoch_guid = epoch_guid;
  req->slot = slot;
  req->index = index;
  req->to_add = to_add;
  bool res = arts_route_table_add_oo(db_guid, req, false);
  if (!res) {
    ARTS_INFO("edt_guid OO2: %lu", req->edt_guid);
    internal_atomic_add_in_array_db(req->db_guid, req->index, req->to_add,
                                    req->edt_guid, req->slot, req->epoch_guid);
    arts_free(req);
  }
}

void arts_out_of_order_atomic_compare_and_swap_in_array_db(
    arts_guid_t db_guid, unsigned int index, unsigned int old_value,
    unsigned int new_value, arts_guid_t edt_guid, unsigned int slot,
    arts_guid_t epoch_guid) {
  struct oo_atomic_compare_and_swap_in_array_db_s *req =
      (struct oo_atomic_compare_and_swap_in_array_db_s *)arts_malloc(
          sizeof(struct oo_atomic_compare_and_swap_in_array_db_s));
  req->type = OO_ATOMIC_COMPARE_AND_SWAP_IN_ARRAY_DB;
  req->edt_guid = edt_guid;
  req->db_guid = db_guid;
  req->epoch_guid = epoch_guid;
  req->slot = slot;
  req->index = index;
  req->old_value = old_value;
  req->new_value = new_value;
  bool res = arts_route_table_add_oo(db_guid, req, false);
  if (!res) {
    ARTS_INFO("edt_guid OO2: %lu", req->edt_guid);
    internal_atomic_compare_and_swap_in_array_db(
        req->db_guid, req->index, req->old_value, req->new_value, req->edt_guid,
        req->slot, req->epoch_guid);
    arts_free(req);
  }
}
