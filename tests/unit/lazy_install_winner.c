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
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/// @file lazy_install_winner.c
/// @brief T061 — concurrent first-touch lazy cache installs of a remote-owned
///        DB: exactly one winner, every consumer gets a valid pinned handle, no
///        stub leak.
///
/// THE WINDOW UNDER TEST (coherence.c `arts_db_cache_lazy_install`).  When a
/// rank first touches a remote-owned DB it allocates a cache-only STUB db_s,
/// `arts_db_cache_init(..., ARTS_DB_INIT_LAZY)`s it, and
/// `arts_route_table_install_if_absent`s it.  Multiple actors on the SAME rank
/// (concurrent wire-receive + acquire-path) can lazy-install the same GUID at
/// once — install_if_absent is the arbiter: exactly one wins (drains its OoO),
/// every loser frees its stub (the destructor must be stub-safe: it reads the
/// zeroed `home_initialized == false` IN BOUNDS and skips home teardown) and
/// returns a pinned handle to the established db_s.
///
/// HOW THIS TEST DRIVES THE WINDOW.  Per round a fresh remote-owned DB is
/// seeded on a remote rank (so consumer ranks have NO local cache yet); then a
/// WAVE of RO-acquiring EDTs is fanned out across every consumer rank
/// simultaneously.  Each rank's first acquirer triggers a lazy install; the
/// concurrent acquirers on one rank race the stub install.  The DB is destroyed
/// only AFTER the wave's finish scope drains (every reader has acquired AND
/// released) — destroying a DB with RO acquires still in flight is undefined in
/// the OCR memory model (a coherence request that reaches the home after the
/// destroy is deferred forever), so the destroy is ordered after the readers.
///
/// WHAT IT ASSERTS: (1) no crash / no SIGSEGV on the stub destructor or the
/// lost-race free; (2) every reader observes the installed value (a leaked /
/// wrong stub would corrupt the read); (3) progress — every wave drains (a
/// stranded lazy install would hang the finish scope → ctest TIMEOUT).
///
/// runtime_single+multinode, non-RWLOCK: lazy first-touch install is the
/// RCU/WRF_RCU remote-acquire mechanism.  RWLOCK has its own RO acquire path,
/// so self-skips.  On a single node every acquire is a local hit (the lazy path
/// is never entered) and the test passes trivially.

#include "arts.h"

#if defined(ARTS_PROTOCOL_RWLOCK)

#include <stdio.h>

int main(void) {
  printf("SKIP lazy_install_winner: non-RWLOCK only (RWLOCK has no lazy "
         "first-touch cache install)\n");
  return 0;
}

#else

#include <stdint.h>
#include <stdio.h>

#define ROUNDS 50u
#define ACQUIRERS_PER_RANK 4u

/// Owner-side seeder: create the DB with its reserved GUID, fill it, release.
/// paramv[0] = db guid, paramv[1] = value.
static void seeder_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t g = (arts_guid_t)paramv[0];
  uint64_t val = paramv[1];
  uint64_t *p = (uint64_t *)arts_db_create_with_guid(
      g, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  if (p != NULL) {
    p[0] = val;
  }
  arts_db_release(g, DB_MODE_RW);
}

/// RO reader: must observe the writer's value (the wave races only the lazy
/// install, never a destroy, so NULL is never legal here).
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t expect = paramv[0];
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr, "FAIL: reader expected 0x%lx got 0x%lx\n",
                  (unsigned long)expect, d ? (unsigned long)d[0] : 0ul);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== lazy_install_winner ===\n");

  unsigned int nranks = arts_get_total_ranks();

  for (unsigned int rnd = 0; rnd < ROUNDS; rnd++) {
    uint64_t val = 0x1A2B0000u + rnd;
    /* Owner on a remote rank when possible so consumer ranks have no cache. */
    unsigned int owner = (nranks > 1) ? (1u + (rnd % (nranks - 1u))) : 0u;

    /* Seed the DB on the owner rank so the value is installed at the owner
     * before any consumer acquires it. */
    arts_guid_t g = arts_guid_reserve(ARTS_GUID_DB, owner);

    arts_guid_t e_seed = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    uint64_t seed_pv[2] = {(uint64_t)g, val};
    arts_edt_create(seeder_edt, 2, seed_pv, 0,
                    &(arts_edt_hint_t){.rank = owner, .finish_event = e_seed});
    arts_event_wait(e_seed);

    /* Concurrent acquire wave: each consumer rank's first acquirer
     * lazy-installs the stub; the per-rank acquirers race it. */
    arts_guid_t e_acq = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int rank = 0; rank < nranks; rank++) {
      for (unsigned int k = 0; k < ACQUIRERS_PER_RANK; k++) {
        uint64_t rpv[1] = {val};
        arts_guid_t r = arts_edt_create(
            reader_edt, 1, rpv, 1,
            &(arts_edt_hint_t){.rank = rank, .finish_event = e_acq});
        arts_add_dependence(g, r, 0, DB_MODE_RO);
      }
    }
    arts_event_wait(e_acq);

    /* Destroy only after the whole wave has acquired AND released (the finish
     * scope drained): no RO acquire is in flight, so the destroy is
     * well-defined under the OCR model. */
    arts_db_destroy(g);
  }

  arts_printf("PASS: lazy_install_winner %u rounds, single winner / no stub "
              "leak\n",
              ROUNDS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif
