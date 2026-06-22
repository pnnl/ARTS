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

/// @file db_user_ptr_uaf.c
/// @brief Caller-contract coverage for the user-visible DB payload pointer
///        (arts_db_user_ptr): arts_db_create returns buf->data AFTER dropping
///        the transient buffer ref, so the pointer is valid ONLY for the single
///        owner (the creator) until it hands the DB off / releases it.
///
/// arts_db_user_ptr (static in db.c, reached via arts_db_create) takes a buffer
/// ref, reads buf->data, and drops the ref — returning a pointer that outlives
/// the ref drop.  The contract: this is correct for the single owner (the
/// creator EDT that just made the DB and has not yet released or handed it
/// off); a concurrent destroy racing the ref-drop and the caller's first use
/// would be a use-after-free.  This test documents+exercises the SAFE side of
/// the contract — the creator writes through the returned pointer before
/// releasing, then a consumer reads the published bytes — and asserts the
/// pointer is usable for the full single-owner window.  It does NOT manufacture
/// the racy destroy (that is a caller-misuse UAF, not a runtime obligation to
/// defend); the test's value is proving the documented single-owner window
/// holds end-to-end.
///
/// All configs.  Home on rank 0 so the creator is the single local owner.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define ELEMS 16u

/// consumer: RO reader; the bytes the creator wrote through the user pointer
/// must be visible (the single-owner write window published correctly).
void consumer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  bool ok = (d != NULL);
  for (unsigned int i = 0; i < ELEMS && ok; i++) {
    if (d[i] != (uint64_t)(0xF00D + i)) {
      ok = false;
    }
  }
  if (!ok) {
    (void)fprintf(stderr,
                  "FAIL: db_user_ptr_uaf consumer did not see creator bytes\n");
    arts_abort(1);
    return;
  }
  arts_printf("PASS: db_user_ptr_uaf\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_user_ptr_uaf ===\n");

  /* arts_db_create returns the user-visible payload pointer (arts_db_user_ptr).
   * As the single owner we may write through it for the whole window up to the
   * release that hands the DB off. */
  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, ELEMS * sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  if (ptr == NULL) {
    (void)fprintf(stderr,
                  "FAIL: db_user_ptr_uaf create returned NULL user ptr\n");
    arts_abort(1);
    return;
  }
  uint64_t *d = (uint64_t *)ptr;
  /* Use the returned pointer across many writes — the single-owner window must
   * keep buf->data valid for the whole sequence (the ref was already dropped
   * inside arts_db_create, so this is the exact pointer-outlives-ref contract).
   */
  for (unsigned int i = 0; i < ELEMS; i++) {
    d[i] = (uint64_t)(0xF00D + i);
  }
  arts_db_release(db, DB_MODE_RW);

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t c =
      arts_edt_create(consumer_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, c, 0, DB_MODE_RO);
  arts_event_wait(fe);

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
