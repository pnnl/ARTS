/* SPDX-License-Identifier: Apache-2.0
 *
 * outbox_payload_free_once — runtime_multinode, all configs.
 *
 * Target: the scattered free sites of a payload sent via
 * arts_transport_send_payload_async_free (libs/src/core/transport/outbox.c;
 * census 21 §arts_transport_send_payload_async_free — the prime double-free /
 * leak surface).
 *
 * The stored free_method must fire EXACTLY ONCE across a node's lifetime,
 * regardless of which path the node takes:
 *   - pump full-send       : free (outbox.c)
 *   - pump hard-error      : free + abort pump
 *   - flush full-send      : free
 *   - flush final-cleanup  : free
 *   - thread cleanup       : free parked partial
 *   - outbox_cleanup       : free queued (payload && free_method)
 * A PARTIAL send must NOT free (re-park).  The combination partial->park->later
 * -full plus abort/cleanup is where a double-free or leak hides.
 *
 * Black-box exercise: drive a high volume of repeated cross-rank RW ownership
 * transfers — each transfer carries a heap payload that the transport frees on
 * full-send.  By cycling ownership of the SAME DBs many times (RW writer on
 * remote rank, then RW re-writer back on home, repeated), every DB's bytes are
 * shipped as a freeable payload dozens of times, exercising the partial->full
 * free path under churn.  Correctness invariants:
 *   - DOUBLE-FREE -> heap corruption / ASan abort (the build under test is the
 *     sanitizer build) -> visible crash, nonzero exit.
 *   - LEAK -> LSan reports at exit (sanitizer build) -> nonzero exit.
 *   - LOST payload (freed instead of re-parked on partial) -> a transferred DB
 *     arrives corrupt/stale -> the value check below fails -> arts_abort(1).
 *   - STRANDED transfer -> finish scope never fires -> ctest TIMEOUT.
 *
 * Config-agnostic.  On 1n the transfers are local/self -> trivial pass.  No
 * in-test watchdog; hang reaped by ctest TIMEOUT.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"

/* DBs cycled, and how many RW ownership round-trips each makes. */
#define NDBS 24
#define ROUNDS 16
/* Modest payload so many transfers fit in memory; still a real heap payload. */
#define NELEMS 256u

/* bump: RW-acquire, increment every element by 1, release.  Each acquire on a
 * remote owner ships the DB bytes as a freeable transport payload. */
static void bump_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  if (d != NULL) {
    for (unsigned i = 0; i < NELEMS; i++) {
      d[i] += 1u;
    }
  }
}

/* check: RO on home after all rounds; every element must equal the total number
 * of bumps (2 per round: one remote, one home). */
static void check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  uint64_t expect = paramv[0];
  if (d == NULL) {
    (void)fprintf(stderr, "FAIL: payload_free_once check got NULL ptr\n");
    arts_abort(1);
  }
  for (unsigned i = 0; i < NELEMS; i++) {
    if (d[i] != expect) {
      (void)fprintf(stderr,
                    "FAIL: payload_free_once value at elem %u — want %llu got "
                    "%llu (lost/corrupt payload)\n",
                    i, (unsigned long long)expect, (unsigned long long)d[i]);
      arts_abort(1);
    }
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W = (nranks > 1) ? 1u : 0u;
  arts_printf(
      "=== outbox_payload_free_once (ranks=%u, dbs=%d, rounds=%d) ===\n",
      nranks, NDBS, ROUNDS);

  arts_guid_t *dbs = (arts_guid_t *)malloc(sizeof(arts_guid_t) * NDBS);
  if (dbs == NULL) {
    (void)fprintf(stderr, "FAIL: alloc db array\n");
    arts_abort(1);
  }

  /* Create all DBs first. */
  for (int k = 0; k < NDBS; k++) {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, NELEMS * sizeof(uint64_t), ARTS_DB,
                       ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
    if (db == NULL_GUID) {
      (void)fprintf(stderr, "FAIL: db create NULL_GUID\n");
      arts_abort(1);
    }
    memset(ptr, 0, NELEMS * sizeof(uint64_t));
    arts_db_release(db, DB_MODE_RW);
    dbs[k] = db;
  }

  /* Sibling RW EDTs sharing a DB are NOT serialized by the dependence wiring
   * alone — two concurrent read-modify-write bumps would lose an update.  To
   * get a strict RW chain, each round is its own finish-scope barrier: a remote
   * bump on every DB (wait), then a home bump on every DB (wait).  The
   * home<->remote ownership ping-pong still ships the heap payload on every
   * transfer (the partial->full free path under churn), while each DB's bumps
   * are now strictly ordered so the final value is exactly 2*ROUNDS. */
  for (int rnd = 0; rnd < ROUNDS; rnd++) {
    arts_guid_t fe_rw = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (int k = 0; k < NDBS; k++) {
      arts_guid_t wr =
          arts_edt_create(bump_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = W, .finish_event = fe_rw});
      arts_add_dependence(dbs[k], wr, 0, DB_MODE_RW);
    }
    arts_event_wait(fe_rw);

    arts_guid_t fe_hm = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (int k = 0; k < NDBS; k++) {
      arts_guid_t hm =
          arts_edt_create(bump_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = 0, .finish_event = fe_hm});
      arts_add_dependence(dbs[k], hm, 0, DB_MODE_RW);
    }
    arts_event_wait(fe_hm);
  }

  /* Phase 2: RO check after every bump has happened-before. */
  arts_guid_t fe_r = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  uint64_t expect = (uint64_t)(2 * ROUNDS);
  for (int k = 0; k < NDBS; k++) {
    arts_guid_t ck =
        arts_edt_create(check_edt, 1, &expect, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe_r});
    arts_add_dependence(dbs[k], ck, 0, DB_MODE_RO);
  }

  arts_event_wait(fe_r);
  free(dbs);

  printf("PASS outbox_payload_free_once: %d DBs x %d RW rounds, payload freed "
         "exactly once per transfer\n",
         NDBS, ROUNDS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
