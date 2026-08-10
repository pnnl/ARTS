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

/// @file db_no_acquire.c
/// @brief ARTS_DB_PROP_NO_ACQUIRE create: the creator does NOT auto-acquire;
///        the home rank is the sole idle owner, and the FIRST consumer EDT
///        acquires ownership normally.
///
/// Properties exercised:
///   - arts_db_create with NO_ACQUIRE returns *addr == NULL (creator must not
///     write the buffer; home is the idle owner).
///   - Local NO_ACQUIRE (home==self) drops the pre-stamped writer_count 2->1
///     (non-EXCL) so the unreleased creator-hold does not block future writers;
///     under EXCL the equivalent is skip_hold + home-arbiter zero-init.  Either
///     way the first writer must succeed.
///   - Remote NO_ACQUIRE (home!=self): no creator-side cache; the home is the
///     idle owner; the first consumer triggers a normal OWNERSHIP_REQUEST.
///
/// The decisive check: a first RW writer EDT must be schedulable and able to
/// publish a value, and an RO reader after it must observe that value.  If the
/// creator-hold were not dropped (the bug NO_ACQUIRE guards against) the first
/// writer would never get ownership -> hang, caught by the ctest TIMEOUT.
///
/// All configs.  Single-node: home==self path.  Multinode: home on rank 0,
/// first writer on a remote rank (when nranks>1) exercises the home-idle-owner
/// + first-consumer-acquires path.  No in-test watchdog (TIMEOUT reaps hangs).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define VAL 0x5151u

/// first_writer: the first consumer of a NO_ACQUIRE DB.  Must obtain ownership
/// (home is the idle owner) and publish VAL.  depv[0].ptr non-NULL once owned.
void first_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d == NULL) {
    (void)fprintf(stderr, "FAIL: db_no_acquire first writer got NULL ptr\n");
    arts_abort(1);
    return;
  }
  d[0] = VAL;
}

/// reader: RO, MUST observe the first writer's value.
void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d == NULL || d[0] != VAL) {
    (void)fprintf(stderr, "FAIL: db_no_acquire reader got 0x%x want 0x%x\n",
                  d ? d[0] : 0u, VAL);
    arts_abort(1);
    return;
  }
  arts_printf("PASS: db_no_acquire\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_no_acquire ===\n");

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W =
      (nranks > 1) ? 1u : 0u; /* remote first writer when possible */

  /* NO_ACQUIRE create, home on rank 0.  *addr MUST be NULL (creator does not
   * own the buffer). */
  void *ptr = (void *)0x1; /* non-NULL sentinel to prove it gets set to NULL */
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB,
                     ARTS_DB_PROP_NO_ACQUIRE, &(arts_db_hint_t){.rank = 0});
  if (ptr != NULL) {
    (void)fprintf(stderr, "FAIL: db_no_acquire *addr not NULL (%p)\n", ptr);
    arts_abort(1);
    return;
  }
  if (db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: db_no_acquire create returned NULL_GUID\n");
    arts_abort(1);
    return;
  }

  /* First consumer: RW writer.  Home (rank 0) is the idle owner; this EDT must
   * acquire ownership and publish VAL.  If the creator-hold were not dropped,
   * this would never be granted -> TIMEOUT. */
  arts_guid_t e_wr = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t w =
      arts_edt_create(first_writer, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = W, .finish_event = e_wr});
  arts_add_dependence(db, w, 0, DB_MODE_RW);
  arts_event_wait(e_wr);

  /* RO reader on home must see the first writer's published value. */
  arts_guid_t e_rd = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t r =
      arts_edt_create(reader_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e_rd});
  arts_add_dependence(db, r, 0, DB_MODE_RO);
  arts_event_wait(e_rd);

  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
