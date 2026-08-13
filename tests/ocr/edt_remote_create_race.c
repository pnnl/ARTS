/* SPDX-License-Identifier: Apache-2.0
 *
 * T136 — remote EDT create race-loser cleanup + reordered remote satisfy
 *        (arts_handler_edt_create install_if_absent loser path; sentinel).
 *
 * Property under test (arts_handler_edt_create)
 * ---------------------------------------------
 * When an EDT is created with a pre-reserved GUID homed on a REMOTE rank, the
 * RX handler arts_handler_edt_create:
 *   - bumps a sentinel (depc_needed += 1) before the EDT becomes visible,
 *   - install_if_absent into the route table,
 *     * WINNER  → removes the sentinel; if 0, fires; any queued/concurrent
 *                 satisfy that observed the 0 transition must NOT also fire
 *                 (single dispatch),
 *     * LOSER   → if a finish-scope proxy LATCH was allocated, DECR it
 * (forwards the DECR to the remote parent, cancelling the source-rank INCR so
 * the parent scope stays balanced), then free via the deleter and return.
 *
 * Two concurrent creates of the SAME migrated GUID therefore must yield exactly
 * one installed EDT (no double fire, no leak) and a balanced parent finish
 * scope.  Additionally a remote satisfy that arrives BEFORE the remote create
 * (reordered ahead) queues OoO on the home rank and is replayed under the
 * sentinel, still firing the EDT exactly once.
 *
 * Scenario (deterministic, single driver rank)
 * --------------------------------------------
 * Reserve an EDT GUID homed on rank W (remote when nranks>1, else rank 0).
 *   1) Pre-satisfy the EDT's one real dep (a VAL) against the still-RESERVED
 *      remote GUID — this is the "satisfy reordered ahead of create": it queues
 *      OoO on W.
 *   2) Create that GUID TWICE (same hint.guid, same rank W, same finish scope).
 *      On W the first install wins; the second is the race-loser → its proxy
 *      DECR forwards and it is freed.  The winner, with its dep already queued,
 *      fires exactly once under the sentinel.
 * A finish event gates a collector; main_edt waits on it.  The member bumps a
 * shared counter exactly once.  Imbalance → the scope never drains → TIMEOUT;
 * a double fire → counter == 2 → FAIL.
 *
 * exposes B (race-loser balance / single-fire). Requires >1 rank to exercise
 * the remote proxy path; SKIPs cleanly to a trivial single-rank check on 1n.
 */

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  _Atomic unsigned int fired; /* member fire count; must be exactly 1 */
} ctr_t;

/* The migrated member EDT.  depv[0] = VAL (reordered-ahead satisfy),
 * depv[1] = counter DB (RW). */
void member(uint32_t pc, const uint64_t *pv, uint32_t dc, arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  ctr_t *c = (ctr_t *)dv[1].ptr;
  if (c) {
    atomic_fetch_add_explicit(&c->fired, 1u, memory_order_relaxed);
  }
}

void collector(uint32_t pc, const uint64_t *pv, uint32_t dc,
               arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  ctr_t *c = (ctr_t *)dv[1].ptr;
  unsigned int f = atomic_load_explicit(&c->fired, memory_order_relaxed);
  arts_printf("edt_remote_create_race: member fired=%u\n", f);
  if (f == 1u) {
    arts_printf("PASS edt_remote_create_race\n");
  } else {
    arts_printf("FAIL edt_remote_create_race: member fired %u times (want 1)\n",
                f);
    arts_abort(1);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_remote_create_race ===\n");

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2) {
    /* The race-loser/proxy path is a REMOTE-create property; with one rank the
     * pre-reserved create is local (unconditional replace, not the
     * install_if_absent loser path).  Nothing to exercise — skip cleanly. */
    arts_printf("SKIP edt_remote_create_race: requires 2+ ranks (got %u)\n",
                nranks);
    arts_shutdown();
    return;
  }
  unsigned int W = 1u;

  void *cp = NULL;
  arts_guid_t cdb =
      arts_db_create(&cp, sizeof(ctr_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ctr_t *c = (ctr_t *)cp;
  atomic_init(&c->fired, 0u);
  arts_db_release(cdb, DB_MODE_RW);

  uint64_t pv[1] = {(uint64_t)cdb};

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* Reserve the member EDT GUID on the (remote) home rank W. */
  arts_guid_t mguid = arts_guid_reserve(ARTS_GUID_EDT, W);

  /* (1) Reordered-ahead remote satisfy: deliver the VAL dep before the create.
   * It queues OoO on W's RESERVED slot. */
  arts_edt_satisfy_slot(mguid, 0, NULL_GUID, DB_MODE_VAL);
  /* Wire the counter dep (slot 1, RW) — delivered when the EDT installs. */
  arts_add_dependence(cdb, mguid, 1, DB_MODE_RW);

  /* (2) Create the SAME migrated GUID twice; both target rank W with the same
   * finish scope.  On W the second install loses → proxy DECR forwards (scope
   * stays balanced) and it is freed; the winner fires once under the sentinel.
   */
  arts_edt_create(member, 1, pv, 2,
                  &(arts_edt_hint_t){.guid = mguid, .finish_event = fe});
  arts_edt_create(member, 1, pv, 2,
                  &(arts_edt_hint_t){.guid = mguid, .finish_event = fe});

  /* Collector gated on the finish scope. */
  arts_guid_t coll =
      arts_edt_create(collector, 1, pv, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(fe, coll, 0, DB_MODE_NULL);
  arts_add_dependence(cdb, coll, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
