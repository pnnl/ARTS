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

/// @file coherence_writer_ro_overlap.c
/// @brief Cross-rank RW write write windows overlapped by concurrent RO readers.
///
/// The rest of the coherence suite either keeps a DB RW-only across ranks
/// (coherence_stress_dist) or phase-separates writers from readers
/// (coherence_payload_large, coherence_rwro_phase).  Neither drives the case
/// where a write write window is OPEN on one rank while readers on other ranks are
/// acquiring the same DB read-only — the regime in which a protocol must
/// decide, per request, whether to hand the reader a durable copy, a transient
/// one, or nothing at all.  That is the gap this test closes.
///
/// SHAPE.  N_DBS DBs, homed round-robin.  The run is a sequence of ROUNDS,
/// each a finish scope, so round r begins only after every round-(r-1) EDT has
/// completed and released.  Inside a round, per DB:
///   * WRITERS_PER_DB RW EDTs are issued with no ordering among themselves, in
///     runs of consecutive EDTs pinned to the same rank, each run on the NEXT
///     rank.  A run longer than one keeps that rank's write write window open across
///     several acquire/release pairs; the run boundaries migrate the write window
///     mid-round, so reader redirects race live hand-offs.  The run length
///     varies per DB so both regimes — long write window and hand-off at every
///     release — are live simultaneously.
///   * READERS_PER_DB RO EDTs, interleaved with the writers at create time so
///     read requests arrive throughout the write windows, pinned to a two-rank window
///     that rotates every other round.  A rank inside the window across
///     consecutive rounds re-reads with a copy it already held (invalidated or
///     superseded by the intervening write window); a rank the window has not reached
///     yet acquires that DB for the first time.
///
/// ORACLE.  The payload is a fixed header plus a "churn" region.
///   * Header — MAGIC, DB index, churn length, and a mixed check word — is
///     written exactly once, by main_edt, and released before round 1.  It has
///     a single writer that happens-before every later EDT, so EVERY acquire in
///     EVERY round must observe it verbatim.  Both readers and writers assert
///     it, which catches a serve of the wrong buffer, of uninitialised memory,
///     or at the wrong offset, on any of the paths above.
///   * Churn — every round-r writer memsets the WHOLE churn region to the
///     single byte STAMP(r).  All writers of a round therefore write identical
///     bytes, so their mutual races are value-idempotent and the round's
///     outcome does not depend on which writer lands last.  A round-r acquirer
///     is happens-after every round-(r-1) writer and happens-before every
///     round-(r+1) EDT, so the only bytes it may legally observe are STAMP(r-1)
///     (the write window has not reached this byte) and STAMP(r) (it has).  Every
///     acquirer asserts exactly that, byte by byte.
///
/// Byte granularity is deliberate: a reader served concurrently with an open
/// write window may legally observe a mix of the two stamps — that is the racy read
/// the OCR model permits — but it may never observe a byte from an older round
/// (a copy that survived a write window it should not have), nor a byte outside the
/// stamp alphabet (0x00 from an unwritten buffer, junk from a foreign slab).
/// The assertion is therefore free of any atomicity assumption about how a
/// snapshot is produced, while still being a true oracle rather than a smoke
/// test.
///
/// Runs under every memory model / protocol / placement combination.  The header
/// has one event-ordered writer and the churn writers of a round are
/// value-idempotent, so a whole-DB last-writer-wins publish (the DB-WRF
/// contract's lossy case) preserves both assertions.
///
/// A violation prints FAIL: and aborts; a stranded acquire surfaces as a ctest
/// TIMEOUT (no in-test watchdog).  The final PASS line is gated on every round
/// scope having drained.

#include "arts.h"

#include <stdint.h>
#include <string.h>

#define N_DBS 8
#define N_ROUNDS 16
#define WRITERS_PER_DB 9
#define READERS_PER_DB 8

/* Consecutive same-rank writers per write window, varied per DB so the run spans both
 * regimes at once: run 1 hands the write window on at every release (maximum
 * migration under read load), run 3 holds one write window across three
 * acquire/release pairs. */
#define WRITER_RUN(db) (1 + ((db) % 3))

#define PAYLOAD_BYTES 4096u
#define HDR_BYTES 32u
#define CHURN_BYTES (PAYLOAD_BYTES - HDR_BYTES)

#define HDR_MAGIC 0x5457455255524554ull
#define HDR_MIX 0x9E3779B97F4A7C15ull

/* Stamp alphabet: 0xA1 .. 0xA1+N_ROUNDS.  Disjoint from 0x00 so an unwritten
 * or zero-filled buffer is a detectable violation, and one byte wide so the
 * legality test needs no assumption about copy atomicity. */
#define STAMP_BASE 0xA1u
#define STAMP(round) ((uint8_t)(STAMP_BASE + (unsigned)(round)))

static void header_write(uint8_t *p, uint64_t db_idx) {
  uint64_t *h = (uint64_t *)(void *)p;
  h[0] = HDR_MAGIC;
  h[1] = db_idx;
  h[2] = (uint64_t)CHURN_BYTES;
  h[3] = HDR_MAGIC ^ (db_idx * HDR_MIX);
}

/* Returns the index of the first header word that differs, or -1 when the
 * header is intact. */
static int header_bad_word(const uint8_t *p, uint64_t db_idx) {
  const uint64_t *h = (const uint64_t *)(const void *)p;
  const uint64_t want[4] = {HDR_MAGIC, db_idx, (uint64_t)CHURN_BYTES,
                            HDR_MAGIC ^ (db_idx * HDR_MIX)};
  for (int i = 0; i < 4; i++) {
    if (h[i] != want[i]) {
      return i;
    }
  }
  return -1;
}

/* Returns the offset of the first churn byte outside {STAMP(round-1),
 * STAMP(round)}, or -1 when every byte is legal. */
static long churn_bad_offset(const uint8_t *p, unsigned round) {
  const uint8_t prev = STAMP(round - 1u);
  const uint8_t curr = STAMP(round);
  const uint8_t *c = p + HDR_BYTES;
  for (uint32_t i = 0; i < CHURN_BYTES; i++) {
    if (c[i] != prev && c[i] != curr) {
      return (long)i;
    }
  }
  return -1;
}

/* Shared assertion for both access modes: an acquire of any kind must deliver
 * the invariant header and a churn region drawn from this round's two-stamp
 * alphabet. */
static void verify_view(const uint8_t *p, unsigned round, uint64_t db_idx,
                        const char *who) {
  int bw = header_bad_word(p, db_idx);
  if (bw >= 0) {
    const uint64_t *h = (const uint64_t *)(const void *)p;
    arts_printf("FAIL: %s rank=%u db=%llu round=%u header word %d = %llx\n", who,
                arts_get_current_rank(), (unsigned long long)db_idx, round, bw,
                (unsigned long long)h[bw]);
    arts_abort(1);
  }
  long bo = churn_bad_offset(p, round);
  if (bo >= 0) {
    arts_printf("FAIL: %s rank=%u db=%llu round=%u churn[%ld]=%02x not in "
                "{%02x,%02x}\n",
                who, arts_get_current_rank(), (unsigned long long)db_idx, round,
                bo, (unsigned)p[HDR_BYTES + (uint32_t)bo],
                (unsigned)STAMP(round - 1u), (unsigned)STAMP(round));
    arts_abort(1);
  }
}

static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)depc;
  if (paramc < 2) {
    arts_printf("FAIL: writer missing paramv\n");
    arts_abort(1);
  }
  unsigned round = (unsigned)paramv[0];
  uint64_t db_idx = paramv[1];
  uint8_t *p = (uint8_t *)depv[0].ptr;
  if (p == NULL) {
    arts_printf("FAIL: writer rank=%u db=%llu round=%u got NULL ptr\n",
                arts_get_current_rank(), (unsigned long long)db_idx, round);
    arts_abort(1);
  }
  verify_view(p, round, db_idx, "writer");
  memset(p + HDR_BYTES, (int)STAMP(round), CHURN_BYTES);
}

static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)depc;
  if (paramc < 2) {
    arts_printf("FAIL: reader missing paramv\n");
    arts_abort(1);
  }
  unsigned round = (unsigned)paramv[0];
  uint64_t db_idx = paramv[1];
  const uint8_t *p = (const uint8_t *)depv[0].ptr;
  if (p == NULL) {
    arts_printf("FAIL: reader rank=%u db=%llu round=%u got NULL ptr\n",
                arts_get_current_rank(), (unsigned long long)db_idx, round);
    arts_abort(1);
  }
  verify_view(p, round, db_idx, "reader");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int nranks = arts_get_total_ranks();
  arts_printf("=== coherence_writer_ro_overlap (%d rounds, %d DBs, %d W + %d R "
              "per DB per round, %u ranks) ===\n",
              N_ROUNDS, N_DBS, WRITERS_PER_DB, READERS_PER_DB, nranks);

  /* Round 0: main_edt is the DBs' single header writer and stamps the churn
   * region with STAMP(0).  Releasing before any round-1 EDT is created makes
   * that write happen-before every acquire below. */
  arts_guid_t dbs[N_DBS];
  for (int d = 0; d < N_DBS; d++) {
    void *raw = NULL;
    dbs[d] = arts_db_create(&raw, PAYLOAD_BYTES, ARTS_DB, ARTS_DB_PROP_NONE,
                            &(arts_db_hint_t){.rank =
                                                  (unsigned int)d % nranks});
    uint8_t *p = (uint8_t *)raw;
    header_write(p, (uint64_t)d);
    memset(p + HDR_BYTES, (int)STAMP(0), CHURN_BYTES);
    arts_db_release(dbs[d], DB_MODE_RW);
  }

  for (unsigned r = 1; r <= (unsigned)N_ROUNDS; r++) {
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    for (int d = 0; d < N_DBS; d++) {
      uint64_t pv[2] = {(uint64_t)r, (uint64_t)d};
      int w = 0;
      int k = 0;
      /* Interleave the two families so read requests keep arriving while the
       * write write windows are open, instead of queueing behind them. */
      while (w < WRITERS_PER_DB || k < READERS_PER_DB) {
        if (w < WRITERS_PER_DB) {
          unsigned int wr =
              (unsigned int)((unsigned)d + r + (unsigned)(w / WRITER_RUN(d))) %
              nranks;
          arts_guid_t we = arts_edt_create(
              writer_edt, 2, pv, 1,
              &(arts_edt_hint_t){.rank = wr, .finish_event = fe});
          arts_add_dependence(dbs[d], we, 0, DB_MODE_RW);
          w++;
        }
        if (k < READERS_PER_DB) {
          /* Two-rank reader window that rotates every other round: a rank
           * inside it in consecutive rounds re-reads a copy it already had,
           * a rank the window has not reached yet acquires this DB cold. */
          unsigned int rr =
              (unsigned int)((unsigned)d + (r / 2u) + (unsigned)(k % 2)) %
              nranks;
          arts_guid_t re = arts_edt_create(
              reader_edt, 2, pv, 1,
              &(arts_edt_hint_t){.rank = rr, .finish_event = fe});
          arts_add_dependence(dbs[d], re, 0, DB_MODE_RO);
          k++;
        }
      }
    }

    arts_event_wait(fe);
  }

  arts_printf("PASS: rw-write window/ro-overlap rounds=%d dbs=%d reads=%d writes=%d "
              "ranks=%u\n",
              N_ROUNDS, N_DBS, N_ROUNDS * N_DBS * READERS_PER_DB,
              N_ROUNDS * N_DBS * WRITERS_PER_DB, nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
