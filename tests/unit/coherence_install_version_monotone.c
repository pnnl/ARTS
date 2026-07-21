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

/// @file coherence_install_version_monotone.c
/// @brief T058 — install-version monotonicity across multi-hop RW ownership.
///
/// Each owner along an RW chain installs a strictly higher DB buffer version
/// (EAGER installs at writeback/GRANT, LAZY at TRANSFER_OWNERSHIP).  A reader
/// inserted in causal order after a given writer must observe a buffer whose
/// value is at least as recent as that writer's — never a version regression
/// across a GRANT/TRANSFER install.
///
/// The DB carries a single monotonically-increasing counter.  A chain of
/// `HOPS` RW incrementer EDTs is laid out round-robin across all ranks, each
/// strictly ordered after the previous one via its own finish event (main_edt
/// waits between hops, so hop k+1 is created only after hop k has fully
/// released).  After each writer hop, an RO reader (also strictly ordered after
/// that writer) acquires the DB and asserts value >= the writer's installed
/// value.  Because the published slot version is monotonic, the reader can
/// never read an older payload than the most-recent install it is causally
/// after.
///
/// config_specific: meaningful only where DB-level ownership transfer + version
/// installs happen, i.e. RCU (both timings).  RWLOCK has a distinct
/// state machine (and is registered separately); WRF_RCU is DB-WRF with no version
/// guard contract.  Self-skips cleanly under RWLOCK / WRF_RCU.
///
/// A lost transfer / stranded acquirer is caught by the ctest TIMEOUT.

#include "arts.h"

#if !defined(ARTS_PROTOCOL_RCU)

#include <stdio.h>

int main(void) {
  printf("SKIP coherence_install_version_monotone: RCU-only\n");
  return 0;
}

#else

#include <stdint.h>
#include <stdio.h>

#define HOPS 40u

/// RW incrementer: bump the counter, then re-read it so the EDT body witnesses
/// its own installed value.  paramv[0] = expected pre-value (the hop index);
/// the post-value must equal hop+1.
static void hop_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t hop = paramv[0];
  uint64_t *v = (uint64_t *)depv[0].ptr;
  if (v == NULL) {
    (void)fprintf(stderr, "FAIL: hop %lu writer got NULL\n",
                  (unsigned long)hop);
    arts_abort(1);
  }
  /* Per-node exclusive RW: the value seen here must be exactly the running
   * count so far (no lost or duplicated install across the transfer). */
  if (*v != hop) {
    (void)fprintf(stderr, "FAIL: hop %lu writer saw %lu (want %lu)\n",
                  (unsigned long)hop, (unsigned long)*v, (unsigned long)hop);
    arts_abort(1);
  }
  *v = hop + 1u;
}

/// RO reader causally after hop `paramv[0]`: must observe value >= hop+1 (the
/// install that immediately precedes it in program order), never a regression.
static void hop_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t after_hop = paramv[0];
  const uint64_t *v = (const uint64_t *)depv[0].ptr;
  if (v == NULL) {
    (void)fprintf(stderr, "FAIL: reader after hop %lu got NULL\n",
                  (unsigned long)after_hop);
    arts_abort(1);
  }
  if (*v < after_hop + 1u) {
    (void)fprintf(stderr,
                  "FAIL: version regression — reader after hop %lu saw %lu "
                  "(want >= %lu)\n",
                  (unsigned long)after_hop, (unsigned long)*v,
                  (unsigned long)(after_hop + 1u));
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_install_version_monotone ===\n");

  unsigned int nranks = arts_get_total_ranks();

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  for (unsigned int hop = 0; hop < HOPS; hop++) {
    /* Writer hop: round-robin across ranks so ownership transfers home->W and
     * W->W', forcing a fresh version install each hop. */
    unsigned int wrank = hop % nranks;
    uint64_t hv = (uint64_t)hop;
    arts_guid_t e_w = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w =
        arts_edt_create(hop_writer, 1, &hv, 1,
                        &(arts_edt_hint_t){.rank = wrank, .finish_event = e_w});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
    arts_event_wait(e_w);

    /* Reader strictly after the writer (created only now): must observe the
     * installed value, never an older version. Placed on a different rank than
     * the writer when possible to force a remote-RO snapshot. */
    unsigned int rrank = (nranks > 1) ? ((wrank + 1u) % nranks) : 0u;
    arts_guid_t e_r = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t r =
        arts_edt_create(hop_reader, 1, &hv, 1,
                        &(arts_edt_hint_t){.rank = rrank, .finish_event = e_r});
    arts_add_dependence(db, r, 0, DB_MODE_RO);
    arts_event_wait(e_r);
  }

  arts_printf("PASS: coherence_install_version_monotone %u hops, no version "
              "regression\n",
              HOPS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif
