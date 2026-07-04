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

/// @file snapshot_drain_case3.c
/// @brief T060 — snapshot reorder-buffer case-3 push + race-recheck + drain.
///
/// THE WINDOW UNDER TEST (coherence.c `arts_db_drain_pending_snapshot` +
/// handlers.c `arts_handler_db_snapshot_response`).  A remote RO acquire fires
/// a SNAPSHOT_REQUEST and parks.  The home replies with the data, but the two
/// wire facts can REORDER under progress_threads>1: a NO_DATA-style response
/// (a->version > buf->version, data_present == false) can be dispatched AHEAD
/// of the with-data install.  When that happens the handler takes case 3:
///   - push a reorder-buffer node onto cache->pending_snapshot, then
///   - re-check the published version; if a concurrent case-2 install already
///     landed, drain (our own node included) so we never park forever.
/// A later case-2 install runs `arts_db_drain_pending_snapshot`, a single
/// atomic_exchange that claims the whole chain and wakes EVERY parked reader
/// exactly once (no lost wakeup, no double-free of a node).
///
/// HOW THIS TEST DRIVES THE WINDOW.  Per round: an RW writer on a remote rank
/// installs a fresh value, then a WAVE of RO readers spread across all ranks
/// all acquire the SAME DB concurrently and each asserts it observes the
/// writer's value.  The remote readers go through the snapshot request/response
/// path; the concurrency of many simultaneous remote RO acquires against a
/// just-transferred owner is exactly what produces NO_DATA-ahead-of-data
/// reorders and case-3 pushes.  Each reader is in a per-round finish scope;
/// main_edt waits for the wave to fully drain before the next round.  A lost
/// wakeup (a parked reader never drained) leaves the finish scope unquiesced
/// and is caught by the ctest TIMEOUT — there is NO in-test watchdog.
///
/// WHAT IT ASSERTS: (1) progress — every reader of every wave is woken and the
/// finish event fires; (2) correctness — every reader observes the writer's
/// installed value (a dropped/duplicated drain would show a stale read).
///
/// runtime_multinode, non-LOCK: the snapshot reorder buffer is the RO-acquire
/// mechanism for MRNEW/MRSW/MRMW.  LOCK has its own RO acquire/release path
/// (no pending_snapshot reorder buffer), so self-skips.  On a single node the
/// readers are local hits (no snapshot path) and the test passes trivially.

#include "arts.h"

#if defined(ARTS_PROTOCOL_LOCK)

#include <stdio.h>

int main(void) {
  printf("SKIP snapshot_drain_case3: non-LOCK only (LOCK has no snapshot "
         "reorder buffer)\n");
  return 0;
}

#else

#include <stdint.h>
#include <stdio.h>

#define ROUNDS 60u
#define READERS_PER_RANK 4u

/// RW writer: stamp the per-round value.
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t v = paramv[0];
  uint64_t *d = (uint64_t *)depv[0].ptr;
  if (d != NULL) {
    d[0] = v;
  }
}

/// RO reader causally after the writer: MUST observe the writer's value.  A
/// lost-wakeup never reaches here (the finish scope hangs → ctest TIMEOUT); a
/// stale drain reaches here with the wrong value and aborts non-zero.
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t expect = paramv[0];
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr, "FAIL: reader expected 0x%lx got 0x%lx\n",
                  (unsigned long)expect, (unsigned long)(d ? d[0] : 0u));
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== snapshot_drain_case3 ===\n");

  unsigned int nranks = arts_get_total_ranks();

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  for (unsigned int rnd = 0; rnd < ROUNDS; rnd++) {
    uint64_t val = 0xC3000000u + rnd;

    /* Writer on a remote rank (round-robin) so the next wave of RO readers
     * must snapshot from a just-transferred owner — the reorder-prone case. */
    unsigned int wrank = (nranks > 1) ? (1u + (rnd % (nranks - 1u))) : 0u;
    arts_guid_t e_w = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w =
        arts_edt_create(writer_edt, 1, &val, 1,
                        &(arts_edt_hint_t){.rank = wrank, .finish_event = e_w});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
    arts_event_wait(e_w);

    /* A concurrent wave of RO readers across ALL ranks, all created only after
     * the writer finished (causally after the install).  Their simultaneous
     * remote snapshot requests are what produce NO_DATA-ahead reorders and
     * case-3 pushes; the case-2 install must drain every one of them. */
    arts_guid_t e_r = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int rank = 0; rank < nranks; rank++) {
      for (unsigned int k = 0; k < READERS_PER_RANK; k++) {
        arts_guid_t r = arts_edt_create(
            reader_edt, 1, &val, 1,
            &(arts_edt_hint_t){.rank = rank, .finish_event = e_r});
        arts_add_dependence(db, r, 0, DB_MODE_RO);
      }
    }
    arts_event_wait(e_r);
  }

  arts_printf("PASS: snapshot_drain_case3 %u rounds, all readers woken\n",
              ROUNDS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif
