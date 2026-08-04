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

/// @file self_send_vs_dispatcher_parity.c
/// @brief T112 — self-send shortcut must reproduce the RX dispatcher exactly
///        (B023).
///
/// Every `arts_send_db_*` helper has two paths for the same wire message: a
/// self-send shortcut (destination == local rank — `arts_transport_send_async`
/// drops self-sends, so the helper dispatches the handler inline) and the RX
/// dispatcher (cross-rank — the wire packet is decoded and routed).  The two
/// MUST be behaviorally identical: same defer-vs-inline decision (Cat-B routes
/// through `arts_ooo_dispatch_or_defer_guid`), same MISS action (Cat-C
/// lookup-acquire-or-drop / unconditional ACK post).  A divergence shows up as
/// a bug that reproduces ONLY single-node (self-send) or ONLY cross-rank
/// (dispatcher) — never both.
///
/// This driver exercises EVERY coherence message family — DB_CREATE,
/// SNAPSHOT_REQUEST/RESPONSE (RO acquire), PUBLISH/ownership transfer (RW
/// handoff), DESTROY + CACHE_DESTROY — and asserts the SAME correctness
/// invariant in both topologies.  Run single-node it drives the self-send
/// shortcut for all of them (home == self); run multinode the integrator's
/// 2n/3n/4n/2n_io variants drive the dispatcher.  Identical PASS in both
/// topologies is the parity proof; a topology-specific hang is caught by the
/// ctest TIMEOUT, a topology-specific wrong answer by the in-EDT assertion
/// (arts_abort).
///
/// Config-agnostic: runs under every protocol (RW handoff is a no-op transfer
/// under WRF_VAL but the read-after-write chain still holds).

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define ITERS 200u
#define BASE 0x40000000u

/// writer: RW acquire; stamp the per-iter value.
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

/// reader: RO acquire, causally after the writer; MUST observe the new value
/// regardless of whether the snapshot arrived via self-send or the dispatcher.
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr, "FAIL: parity mismatch — expected 0x%x got 0x%x\n",
                  expect, d ? d[0] : 0u);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== self_send_vs_dispatcher_parity ===\n");

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W = (nranks > 1) ? 1u : 0u; /* remote writer when possible */

  for (unsigned int it = 0; it < ITERS; it++) {
    uint64_t vnew = (uint64_t)(BASE + it);

    /* DB_CREATE on home (self-send create on 1n, wire create from a remote-home
     * acquire on Nn).  Recreate per iter so DESTROY + CACHE_DESTROY are also
     * exercised every generation. */
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)ptr)[0] = 0u;
    arts_db_release(db, DB_MODE_RW);

    /* RW handoff: publish / ownership-transfer message family. */
    arts_guid_t e_w = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t wr =
        arts_edt_create(writer_edt, 1, &vnew, 1,
                        &(arts_edt_hint_t){.rank = W, .finish_event = e_w});
    arts_add_dependence(db, wr, 0, DB_MODE_RW);
    arts_event_wait(e_w);

    /* RO acquire on every rank: SNAPSHOT_REQUEST/RESPONSE message family.  Rank
     * 0's reader is the self-send case (home == self); foreign readers are the
     * dispatcher case. */
    arts_guid_t e_r = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int r = 0; r < nranks; r++) {
      arts_guid_t rd =
          arts_edt_create(reader_edt, 1, &vnew, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = e_r});
      arts_add_dependence(db, rd, 0, DB_MODE_RO);
    }
    arts_event_wait(e_r);

    /* DESTROY (self-send vs wire) + CACHE_DESTROY fan-out. */
    arts_db_destroy(db);
  }

  arts_printf("PASS: self_send_vs_dispatcher_parity %u iters x %u ranks\n",
              ITERS, nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
