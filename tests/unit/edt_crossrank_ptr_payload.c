/* SPDX-License-Identifier: Apache-2.0
 *
 * T137 — cross-rank DB_MODE_PTR inline payload satisfy (multinode).
 *
 * Property under test (arts_send_edt_satisfy_slot, size>0 path)
 * ------------------------------------------------------------
 * arts_edt_satisfy_slot with mode == DB_MODE_PTR and size > 0, when the target
 * EDT is REMOTE, marshals header + `size` inline payload bytes contiguously
 * (arts_send_edt_satisfy_slot's size>0 branch) and ships them in one
 * MSG_EDT_SATISFY_SLOT.  The receiver materializes the payload (malloc+copy)
 * onto the dep slot so the EDT body sees the bytes without a follow-up fetch.
 * The target rank is resolved via route_table_lookup_rank when the GUID claims
 * a local home (migrated-EDT routing) and via the GUID's rank field otherwise.
 *
 * Scenario
 * --------
 * A consumer EDT is created on rank W (remote when nranks>1).  From rank 0,
 * main_edt satisfies slot 0 of that EDT with DB_MODE_PTR carrying a fixed
 * N-byte pattern.  The consumer verifies every byte arrived intact and bumps a
 * tally on the home counter DB; a collector gated on a finish scope checks the
 * tally. The consumer also has a counter dep (RW) so it joins the same DB the
 * collector reads.  A dropped/corrupted payload → FAIL; a stranded EDT → ctest
 * TIMEOUT.
 *
 * Requires >1 rank to exercise the wire path; SKIPs cleanly on 1n (the PTR
 * payload then takes the local edt_defer_satisfy path, also valid, so we still
 * run a single-rank smoke of the same delivery).
 */

#include "arts.h"
#include "../test_failure_status.h"
#include "arts/db.h" /* DB_MODE_PTR */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define PAYLOAD_N 48u

typedef struct {
  _Atomic unsigned int ok; /* consumer saw the intact payload */
} tally_t;

static unsigned char expected_byte(unsigned i) {
  return (unsigned char)(0xA0u + (i * 7u));
}

/* Consumer: depv[0] = DB_MODE_PTR payload (inline-delivered), depv[1] = tally
 * DB (RW). */
void consumer(uint32_t pc, const uint64_t *pv, uint32_t dc,
              arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  tally_t *t = (tally_t *)dv[1].ptr;
  const unsigned char *p = (const unsigned char *)dv[0].ptr;
  bool ok = (p != NULL);
  if (ok) {
    for (unsigned i = 0; i < PAYLOAD_N; i++) {
      if (p[i] != expected_byte(i)) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    atomic_fetch_add_explicit(&t->ok, 1u, memory_order_relaxed);
  } else {
    arts_test_fail();
    arts_printf("FAIL: consumer payload mismatch (ptr=%p)\n", (const void *)p);
  }
}

void collector(uint32_t pc, const uint64_t *pv, uint32_t dc,
               arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  tally_t *t = (tally_t *)dv[1].ptr;
  unsigned int ok = atomic_load_explicit(&t->ok, memory_order_relaxed);
  arts_printf("edt_crossrank_ptr_payload: ok=%u\n", ok);
  if (ok == 1u) {
    arts_printf("PASS edt_crossrank_ptr_payload\n");
  } else {
    arts_printf("FAIL edt_crossrank_ptr_payload\n");
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

  arts_printf("=== edt_crossrank_ptr_payload ===\n");

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W = (nranks > 1) ? 1u : 0u;

  void *tp = NULL;
  arts_guid_t tally =
      arts_db_create(&tp, sizeof(tally_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  tally_t *t = (tally_t *)tp;
  atomic_init(&t->ok, 0u);
  arts_db_release(tally, DB_MODE_RW);

  uint64_t pv[1] = {(uint64_t)tally};

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* Consumer on rank W with 2 deps. */
  arts_guid_t cons = arts_edt_create(
      consumer, 1, pv, 2, &(arts_edt_hint_t){.rank = W, .finish_event = fe});
  arts_add_dependence(tally, cons, 1, DB_MODE_RW);

  /* Build the payload and satisfy slot 0 cross-rank with DB_MODE_PTR. */
  unsigned char payload[PAYLOAD_N];
  for (unsigned i = 0; i < PAYLOAD_N; i++) {
    payload[i] = expected_byte(i);
  }
  arts_edt_satisfy_slot(cons, 0, NULL_GUID, DB_MODE_PTR, payload, PAYLOAD_N);

  /* Collector gated on the finish scope. */
  arts_guid_t coll =
      arts_edt_create(collector, 1, pv, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(fe, coll, 0, DB_MODE_NULL);
  arts_add_dependence(tally, coll, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  /* Two verdicts to merge: what arts_rt saw of the ranks it spawned (their exit
     status reaches nobody else) and what this rank's own checks found. */
  int rc = arts_rt(argc, argv);
  return rc != 0 ? 1 : arts_test_status();
}
