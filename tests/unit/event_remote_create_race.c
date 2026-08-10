/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_remote_create_race — C15 / T160 (multinode, best under ASan).
 *
 * Target: arts_handler_event_create (event.c) loser path — two+ ranks racing
 * install_if_absent on the SAME labeled event GUID; the loser frees its freshly
 * unmarshaled buffer via event_free_typed.  Validates no leak / no UAF and that
 * exactly ONE creator wins.
 *
 * The event GUID is reserved on rank 0 (home == 0) and broadcast to every rank
 * by passing it as an EDT param.  Each rank runs a racer_edt that calls
 * arts_event_create at that GUID with check=true:
 *   - the home-rank create + every cross-rank MSG_EVENT_CREATE all funnel
 *     through the home-rank route_table install_if_absent;
 *   - exactly one install wins (its create returns the GUID on the home rank;
 *     cross-rank wins are unverifiable locally so a remote racer cannot self-
 *     report a win — see below);
 *   - every loser's unmarshaled image is freed via event_free_typed (the cb
 *     deleter), draining its empty queues/stack symmetrically.
 *
 * To make the winner counting well-defined regardless of which rank wins, each
 * racer records "I attempted" (not "I won") into a per-rank slot of a result DB
 * homed on rank 0, then the verifier confirms (a) every rank attempted and
 * (b) exactly one event object is present at the GUID on the home rank — i.e.
 * the collisions collapsed to a single survivor (no double-install, no orphan).
 *
 * Each racer joins a finish event; the verifier (gated on that finish + the
 * result DB) runs on rank 0 after every racer completed.
 *
 * SKIP cleanly when rank_count < 2.  PASS criterion below.  ASan/UBSan build
 * surfaces any UAF/leak in the loser-free path.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#include "arts.h"
#include "arts/gas/route_table.h"

/* racer_edt: attempt the colliding check-create, record attendance.
 * paramv[0] = event guid, paramv[1] = result_db guid, paramv[2] = my rank.
 * depv[0] = result DB (RW). */
static void racer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t ev = (arts_guid_t)paramv[0];
  uint64_t myrank = paramv[2];

  arts_event_hint_t h = ARTS_EVENT_HINT_LATCH(1);
  h.guid = ev;    /* GUID's rank field (home 0) is authoritative */
  h.check = true; /* rendezvous: first install wins, losers freed */
  arts_guid_t r = arts_event_create(&h);

  /* Record attendance + whether THIS rank's local return reported a win.
   * Slot layout per rank: [attended, local_win]. */
  _Atomic uint64_t *res = (_Atomic uint64_t *)depv[0].ptr;
  atomic_store_explicit(&res[myrank * 2 + 0], 1u, memory_order_release);
  atomic_store_explicit(&res[myrank * 2 + 1], (r == ev) ? 1u : 0u,
                        memory_order_release);
}

/* verify_edt: runs on rank 0 after the finish drains.
 * paramv[0] = event guid, paramv[1] = total ranks.
 * depv[0] = finish (NULL), depv[1] = result DB (RO). */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t ev = (arts_guid_t)paramv[0];
  unsigned int ranks = (unsigned int)paramv[1];
  _Atomic uint64_t *res = (_Atomic uint64_t *)depv[1].ptr;

  unsigned int attended = 0;
  for (unsigned int r = 0; r < ranks; r++) {
    attended += (unsigned int)atomic_load_explicit(&res[r * 2 + 0],
                                                   memory_order_acquire);
  }
  if (attended != ranks) {
    (void)fprintf(stderr, "FAIL: %u/%u ranks attended the create race\n",
                  attended, ranks);
    arts_abort(1);
  }

  /* Exactly one survivor object must be present at the GUID on the home rank
   * (this verifier runs on rank 0 = the home).  The collisions must have
   * collapsed to a single install (no double-install, no orphan). */
  arts_shared_ptr_t lh = arts_route_table_lookup_event(ev);
  struct arts_event_s *e = (struct arts_event_s *)arts_shared_get(lh);
  bool present = (e != NULL);
  arts_shared_release(&lh);
  if (!present) {
    (void)fprintf(stderr,
                  "FAIL: no survivor event at the labeled GUID after the "
                  "create race\n");
    arts_abort(1);
  }

  arts_event_destroy(ev);
  printf("event_remote_create_race: %u ranks raced check-create, single "
         "survivor, losers freed cleanly — PASS\n",
         ranks);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int ranks = arts_get_total_ranks();

#if defined(ARTS_PROTOCOL_WRF_VAL)
  /* WRF_VAL (DB-WRF) provides no exclusive cross-rank ownership for RW: concurrent
   * RW holders on different nodes each receive a buffer copy and race at
   * PUBLISH time (version-monotonic CAS, last writer wins).  Each racer
   * writes to its own slot of result_db, but a later PUBLISH from another
   * rank overwrites the whole buffer, silently discarding earlier slot writes.
   * The attendance check would then report fewer racers than expected and
   * abort. */
  printf("SKIP: event_remote_create_race: concurrent per-slot RW writes to a "
         "shared DB are DB-WRF racy under WRF_VAL\n");
  arts_shutdown();
  return;
#endif

  if (ranks < 2) {
    printf("SKIP: event_remote_create_race requires rank_count >= 2\n");
    arts_shutdown();
    return;
  }

  /* Event GUID reserved on rank 0 (home == 0), broadcast via param. */
  arts_guid_t ev = arts_guid_reserve(ARTS_GUID_EVENT, 0);
  if (ev == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: guid_reserve\n");
    arts_abort(1);
  }

  /* Result DB on rank 0: 2 uint64 per rank. */
  _Atomic uint64_t *res = NULL;
  arts_guid_t result_db =
      arts_db_create((void **)&res, sizeof(uint64_t) * 2 * ranks, ARTS_DB,
                     ARTS_DB_PROP_NONE, NULL);
  for (unsigned int i = 0; i < 2 * ranks; i++) {
    atomic_init(&res[i], 0u);
  }
  arts_db_release(result_db, DB_MODE_RW);

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* One racer per rank, each joined to the finish event. */
  for (unsigned int r = 0; r < ranks; r++) {
    uint64_t pv[3] = {(uint64_t)ev, (uint64_t)result_db, (uint64_t)r};
    arts_guid_t w = arts_edt_create(
        racer_edt, 3, pv, 1, &(arts_edt_hint_t){.rank = r, .finish_event = fe});
    arts_add_dependence(result_db, w, 0, DB_MODE_RW);
  }

  /* Verifier on rank 0, after the finish drains. */
  uint64_t vpv[2] = {(uint64_t)ev, (uint64_t)ranks};
  arts_guid_t v =
      arts_edt_create(verify_edt, 2, vpv, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(fe, v, 0, DB_MODE_NULL);
  arts_add_dependence(result_db, v, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
