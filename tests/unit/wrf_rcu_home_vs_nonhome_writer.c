/* SPDX-License-Identifier: Apache-2.0
 *
 * T104 — WRF_RCU home-rank writer concurrent with non-home writer: no buffer
 *        corruption, only a (legal, DB-WRF) lost update.
 *
 * Targets the in-place buf->version bump in release_rw (census 10, hazard #3).
 * A home-rank RW release bumps buf->version on the canonical buffer in place; a
 * non-home RW release bumps its working copy and rides the version on the
 * WRITEBACK packet, which buf_install()s at home under a monotonic CAS.  Under
 * DB-WRF two concurrent writers race and the last install wins (lost update is
 * expected and legal).  The contract this test PINS is the *non-corruption*
 * invariant: the final buffer must be a clean, whole copy of exactly ONE
 * writer's fill — never a torn mix of both.
 *
 * Structure (2+ ranks; self-skips otherwise):
 *   home = rank 0.  Two writers run concurrently inside one finish scope:
 *     - writer rank 0 (home)  fills the whole DB with byte 0xA0
 *     - writer rank 1 (remote) fills the whole DB with byte 0xB1
 *   The finish-scope verifier (RO) then checks every byte equals a single
 *   value in {0xA0, 0xB1} (internally consistent).  A torn buffer (mix) is a
 *   real corruption FAIL; either uniform value is a legal DB-WRF outcome.
 *
 * config_specific: WRF_RCU only — the concurrent cross-rank RW with no ownership
 * serialization is the DB-WRF contract; ownership protocols serialize writers.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define DB_SIZE 4096
#define FILL_HOME 0xA0u
#define FILL_REMOTE 0xB1u
#define ROUNDS 16u

#if !defined(ARTS_PROTOCOL_WRF_RCU)
void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("SKIP wrf_rcu_home_vs_nonhome_writer: WRF_RCU-only\n");
  arts_shutdown();
}
#else

static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint8_t fill = (uint8_t)paramv[0];
  uint8_t *p = (uint8_t *)depv[0].ptr;
  if (p != NULL) {
    memset(p, fill, DB_SIZE);
  }
}

static void verifier_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint8_t *p = (uint8_t *)depv[0].ptr;
  if (p == NULL) {
    arts_printf("wrf_rcu_home_vs_nonhome_writer: FAIL got NULL buffer\n");
    arts_abort(1);
  }
  uint8_t v = p[0];
  for (size_t i = 1; i < DB_SIZE; i++) {
    if (p[i] != v) {
      arts_printf(
          "wrf_rcu_home_vs_nonhome_writer: FAIL torn buffer at %zu (%u vs %u)\n",
          i, (unsigned)p[i], (unsigned)v);
      arts_abort(1); /* corruption: a real bug, never a legal DB-WRF outcome */
    }
  }
  if (v != FILL_HOME && v != FILL_REMOTE) {
    arts_printf("wrf_rcu_home_vs_nonhome_writer: FAIL unexpected value %u\n",
                (unsigned)v);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  if (arts_get_total_ranks() < 2) {
    arts_printf("SKIP wrf_rcu_home_vs_nonhome_writer: requires 2+ ranks\n");
    arts_shutdown();
    return;
  }

  for (unsigned int rnd = 0; rnd < ROUNDS; rnd++) {
    void *addr = NULL;
    arts_guid_t db = arts_db_create(&addr, DB_SIZE, ARTS_DB, ARTS_DB_PROP_NONE,
                                    &(arts_db_hint_t){.rank = 0});
    memset(addr, 0x00, DB_SIZE);
    arts_db_release(db, DB_MODE_RW);

    /* verifier consumes the writers' finish scope (fe) at slot 1 so it runs
     * only after both writers released the DB; it lives in its OWN finish scope
     * (e_ver) so main_edt can wait for its completion before destroy. */
    arts_guid_t e_ver = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t ver =
        arts_edt_create(verifier_edt, 0, NULL, 2,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = e_ver});

    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_add_dependence(fe, ver, 1, DB_MODE_NULL);

    uint64_t fh = FILL_HOME;
    arts_guid_t wh =
        arts_edt_create(writer_edt, 1, &fh, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db, wh, 0, DB_MODE_RW);

    uint64_t fr = FILL_REMOTE;
    arts_guid_t wr =
        arts_edt_create(writer_edt, 1, &fr, 1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
    arts_add_dependence(db, wr, 0, DB_MODE_RW);

    arts_add_dependence(db, ver, 0, DB_MODE_RO);

    /* Release fe's creator token so fe can fire once both writers complete.
     * ver's slot 1 depends on fe firing; without this wait, fe's creator token
     * is never released, fe never fires, ver never runs, and e_ver deadlocks.
     */
    arts_event_wait(fe);
    arts_event_wait(e_ver);
    arts_db_destroy(db);
  }

  arts_printf("PASS wrf_rcu_home_vs_nonhome_writer: %u rounds, no corruption\n",
              ROUNDS);
  arts_shutdown();
}
#endif

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
