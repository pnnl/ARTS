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

/// @file mrsw_eq_mrnew_results.c
/// @brief MRSW == MRNEW final-value equivalence pin (config_specific).
///
/// For a RACE-FREE OCR program (all conflicting same-DB accesses are
/// event-ordered) the final value is protocol-independent: MRSW and MRNEW (the
/// two single-owner OCR-model protocols) MUST produce the same answer.  This
/// test pins a DETERMINISTIC result computed under a strict happens-before
/// chain so the SAME PASS value is printed in an MRSW build and an MRNEW build;
/// the cross-config equivalence is the two builds agreeing on that pinned
/// number.
///
/// Program: a single counter DB, an event-ordered RW chain (each link reads the
/// predecessor's value and adds a per-step constant), interleaved with RO reads
/// that are causally pinned (gated behind the producing writer via the DB RW
/// chain ordering).  The closed-form final value is
///   sum_{i=0..N-1} (i+1) = N*(N+1)/2 ,
/// which is independent of the coherence protocol.  Any protocol that violates
/// the OCR single-writer-visibility contract would yield a different number ->
/// arts_abort.
///
/// Runs under all OCR-contract protocols (MRNEW, MRSW, LOCK).  MRMW is the
/// weaker DB-WRF contract where racy programs may differ — but this program is
/// race-free, so MRMW must also match; we still run it everywhere and pin the
/// same number.  No protocol self-skip is needed; the test is its own oracle.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define N_STEPS 100u

/* Chain link i: value += (i+1).  Strict RW chain -> read-modify-write is
 * event-ordered, no race. */
static void chain_step_edt(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + paramv[0];
  }
}

/* RO observer: value must be within the closed interval the chain can produce
 * (monotone non-decreasing, capped at the final sum). */
static void ro_obs_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t hi = paramv[0];
  uint64_t v = (d != NULL) ? d[0] : 0u;
  if (v > hi) {
    (void)fprintf(stderr, "FAIL: RO read %llu > final %llu\n",
                  (unsigned long long)v, (unsigned long long)hi);
    arts_abort(1);
  }
}

static void final_check_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t expect = paramv[0];
  uint64_t v = (d != NULL) ? d[0] : 0u;
  if (v != expect) {
    (void)fprintf(stderr,
                  "FAIL: final value %llu != %llu (protocol "
                  "divergence)\n",
                  (unsigned long long)v, (unsigned long long)expect);
    arts_abort(1);
  }
  arts_printf("PASS: mrsw_eq_mrnew_results final=%llu\n",
              (unsigned long long)expect);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* No protocol self-skip: the RW chain is event-ordered (each step depends on
   * the previous step's output event, which fires after that step's writeback),
   * so every conflicting same-DB access is happens-before ordered.  That makes
   * the program race-free even under the weaker DB-WRF contract (MRMW) — a
   * race-free program has a defined, protocol-independent result, so MRMW must
   * also reach the same pinned sum. */

  arts_printf("=== mrsw_eq_mrnew_results ===\n");

  unsigned int nranks = arts_get_total_ranks();
  uint64_t expect = (uint64_t)N_STEPS * (uint64_t)(N_STEPS + 1u) / 2u;

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  /* Phase 1: a STRICTLY event-ordered RW chain across ranks plus a
   * causally-bounded RO observer after each step.  All steps share one finish
   * event; waiting on it blocks main_edt until every step has run AND written
   * back, so the checker created in phase 2 deterministically observes the sum.
   *
   * A bare DB RW dependency does NOT serialize the writers: the OCR model
   * admits multiple concurrent RW holders per node, so N writers sharing only
   * the DB would race on the read-modify-write and lose updates on ANY
   * protocol.  The happens-before that makes the chain race-free is an explicit
   * output-event edge: each step's output event fires AFTER its release
   * (writeback committed), and the next step depends on it, so step i+1 reads
   * exactly the value step i wrote — a true serialized chain,
   * protocol-independent. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t prev_oe = NULL_GUID;
  for (unsigned int i = 0; i < N_STEPS; i++) {
    uint64_t add = (uint64_t)(i + 1u);
    unsigned int rank = (nranks > 1u) ? (i % nranks) : 0u;
    arts_guid_t oe = arts_event_create(&ARTS_EVENT_HINT_LATCH(1));
    uint32_t depc = (i == 0u) ? 1u : 2u;
    arts_guid_t s = arts_edt_create(chain_step_edt, 1, &add, depc,
                                    &(arts_edt_hint_t){.rank = rank,
                                                       .finish_event = fe,
                                                       .output_event = oe});
    arts_add_dependence(db, s, 0, DB_MODE_RW);
    if (i > 0u) {
      arts_add_dependence(prev_oe, s, 1, DB_MODE_NULL); /* run after prev WB */
    }
    prev_oe = oe;

    uint64_t hi = expect;
    arts_guid_t ro = arts_edt_create(
        ro_obs_edt, 1, &hi, 1,
        &(arts_edt_hint_t){.rank = (rank + 1u) % (nranks ? nranks : 1u),
                           .finish_event = fe});
    arts_add_dependence(db, ro, 0, DB_MODE_RO);
  }
  arts_event_wait(fe); /* blocks until ALL chain steps ran + wrote back */

  /* Phase 2: the final checker, created only now that phase 1 has fully
   * quiesced, so its RO snapshot observes every increment. */
  arts_guid_t fe2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t fin =
      arts_edt_create(final_check_edt, 1, &expect, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe2});
  arts_add_dependence(db, fin, 0, DB_MODE_RO);
  arts_event_wait(fe2);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
