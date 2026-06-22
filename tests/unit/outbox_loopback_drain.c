/* SPDX-License-Identifier: Apache-2.0
 *
 * outbox_loopback_drain — config_specific (MRSW / LOCK), runtime_single.
 *
 * Target: arts_transport_loopback_drain single-drainer CAS token
 * (libs/src/core/transport/outbox.c, census 21 §arts_transport_loopback_drain).
 *
 * Why MRSW / LOCK only: on a single node, MRNEW/MRMW take the empty-queue
 * drain arm — a self-addressed coherence message is dropped by
 * self_send_check rather than loopbacked.  Only the protocols that route an
 * acquire round through the GUID *home* (MRSW's home-arbitrated ownership and
 * LOCK's home lock_state) self-send on 1n: an RW acquire whose home is the
 * same rank hops home -> owner -> home as a self-send, copied onto the
 * g_loopback Treiber stack and delivered asynchronously on a worker tick under
 * the g_loopback_draining single-drainer token.  This faithfully reproduces
 * the single per-rank receiver thread that orders ALL inbound coherence on a
 * multi-node run.  MRNEW/MRMW/MRSW+anything-but-1n still self-skip cleanly.
 *
 * The single-drainer serialization is the highest-value correctness property in
 * outbox.c: draining with many workers concurrently would reorder protocol
 * steps (PROCEED vs CONFIRM, INVALIDATE-publish vs release) and strand a
 * transfer.  We stress it as a black box:
 *
 *   1. A long serialized RW ownership-transfer chain on ONE shared DB.  Each
 *      link's RW acquire forces a home-arbitrated ownership round = a self-send
 *      that must be drained, dispatched, and ordered before the next link can
 *      acquire.  Re-entrancy is exercised: the handler dispatched inside a
 *      drain turn posts fresh self-sends (the next hop of the same round),
 *      which land on the now-empty stack and are taken by a later drain.
 *   2. A wide fan of concurrent RO readers + independent RW acquirers on their
 *      own DBs keeps EVERY worker busy, maximizing single-drainer token
 *      contention (workers that lose the CAS must go do other work, never spin)
 *      and the post-release-of-token window (a self-send posted right as the
 *      drainer releases the token must still be picked up by the next tick).
 *
 * Correctness (black-box): the serialized chain establishes a total order in
 * chain[0]; every link must observe its predecessor's stamp (ordering held) and
 * run EXACTLY ONCE (no drain-induced loss/duplication).  A stranded transfer
 * (reordered protocol step) stalls the chain -> the all-done LATCH never fires
 * -> ctest TIMEOUT.  A duplicated dispatch is caught by the per-link run-once
 * bitset.  No in-test watchdog, no spin — hang is reaped by ctest TIMEOUT.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#if !defined(ARTS_PROTOCOL_MRSW) && !defined(ARTS_PROTOCOL_LOCK)

/* MRNEW / MRMW: no home-arbitrated self-send on 1n -> nothing to drain. */
int main(void) {
  printf("SKIP outbox_loopback_drain: self-loopback drain is MRSW/LOCK-only\n");
  return 0;
}

#else

/* Length of the serialized RW ownership-transfer (self-send) chain. */
#define CHAIN_LEN 96
/* Width of the concurrent fan keeping all workers busy (token contention). */
#define FAN_WIDTH 96
#define TOTAL_EDTS (CHAIN_LEN + FAN_WIDTH)

/* State DB layout (uint64_t):
 *   [0]               — global run count (must end == TOTAL_EDTS)
 *   [1]               — global double-run count (must end == 0)
 *   [2]               — ordering-violation count (must end == 0)
 *   [3 .. 3+TOTAL-1]  — per-EDT run bitset cells (each set exactly once) */
#define ST_RUNS 0
#define ST_DOUBLE 1
#define ST_ORDER 2
#define ST_BITS 3
#define ST_NELEMS (ST_BITS + TOTAL_EDTS)

static void mark_run(uint64_t *state, unsigned idx) {
  uint64_t prev = atomic_fetch_add_explicit(
      (_Atomic uint64_t *)&state[ST_BITS + idx], 1u, memory_order_acq_rel);
  if (prev != 0u) {
    atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_DOUBLE], 1u,
                              memory_order_acq_rel);
  }
  atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_RUNS], 1u,
                            memory_order_acq_rel);
}

/* Chain link: RW-acquires the shared chain DB (ownership transferred from the
 * previous link via a home-arbitrated self-send round on 1n), checks that it
 * sees the predecessor's stamp (ordering preserved by the single drainer),
 * stamps its own index, then signals the next link.
 *   paramv: [state_db, my_index, next_guid_or_0, latch]
 *   depv[0] = chain_db (RW), depv[1] = state_db (RW) */
static void chain_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned idx = (unsigned)paramv[1];
  arts_guid_t next = (arts_guid_t)paramv[2];
  arts_guid_t latch = (arts_guid_t)paramv[3];
  uint64_t *chain = (uint64_t *)depv[0].ptr;
  uint64_t *state = (uint64_t *)depv[1].ptr;

  /* The predecessor (index idx-1) must have written chain[0] before ownership
   * transferred to us.  If the single-drainer ordering were violated, a
   * reordered transfer would let us run before our predecessor's stamp landed.
   */
  if (idx > 0 && chain != NULL && chain[0] != (uint64_t)(idx - 1)) {
    atomic_fetch_add_explicit((_Atomic uint64_t *)&state[ST_ORDER], 1u,
                              memory_order_acq_rel);
  }
  if (chain != NULL) {
    chain[0] = (uint64_t)idx;
  }
  mark_run(state, idx);

  /* Hand the chain forward to the next link (ordering), THEN report completion
   * to the all-done LATCH.  The LATCH is sized for every EDT (TOTAL_EDTS), so
   * EACH link must signal it — not only the tail; otherwise the latch is short
   * by CHAIN_LEN-1 and verify_edt never fires. */
  if (next != NULL_GUID) {
    arts_add_dependence((arts_guid_t)0, next, (uint32_t)-1, DB_MODE_VAL);
  }
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* RO reader on the shared chain DB: a concurrent RO acquire round (also a
 * self-send on 1n) interleaved with the RW transfers — keeps a worker busy and
 * exercises the snapshot-serving drain turn.  paramv: [state_db, my_index,
 * latch]; depv[0] = chain_db (RO), depv[1] = state_db (RW). */
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned idx = (unsigned)paramv[1];
  arts_guid_t latch = (arts_guid_t)paramv[2];
  volatile uint64_t *chain = (volatile uint64_t *)depv[0].ptr;
  uint64_t *state = (uint64_t *)depv[1].ptr;
  if (chain != NULL) {
    (void)
        chain[0]; /* touch the snapshot; value unconstrained (concurrent RW) */
  }
  mark_run(state, idx);
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* verify_edt — bound to the all-done LATCH (slot 0) + state DB RO (slot 1). */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *state = (uint64_t *)depv[1].ptr;
  uint64_t runs = atomic_load_explicit((_Atomic uint64_t *)&state[ST_RUNS],
                                       memory_order_acquire);
  uint64_t dbl = atomic_load_explicit((_Atomic uint64_t *)&state[ST_DOUBLE],
                                      memory_order_acquire);
  uint64_t ord = atomic_load_explicit((_Atomic uint64_t *)&state[ST_ORDER],
                                      memory_order_acquire);
  if (ord != 0u) {
    (void)fprintf(stderr,
                  "FAIL: %llu chain link(s) ran out of order — drain reorder\n",
                  (unsigned long long)ord);
    arts_abort(1);
  }
  if (dbl != 0u) {
    (void)fprintf(stderr, "FAIL: %llu EDT(s) ran twice — drain duplication\n",
                  (unsigned long long)dbl);
    arts_abort(1);
  }
  if (runs != (uint64_t)TOTAL_EDTS) {
    (void)fprintf(stderr, "FAIL: %llu EDTs ran (want %d) — drain lost EDT\n",
                  (unsigned long long)runs, TOTAL_EDTS);
    arts_abort(1);
  }
  for (unsigned i = 0; i < TOTAL_EDTS; i++) {
    uint64_t c = atomic_load_explicit((_Atomic uint64_t *)&state[ST_BITS + i],
                                      memory_order_acquire);
    if (c != 1u) {
      (void)fprintf(stderr, "FAIL: EDT %u ran %llu times (want 1)\n", i,
                    (unsigned long long)c);
      arts_abort(1);
    }
  }
  printf("PASS outbox_loopback_drain: %d-link ordered chain + %d fan, "
         "single-drainer order held\n",
         CHAIN_LEN, FAN_WIDTH);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== outbox_loopback_drain (chain=%d, fan=%d) ===\n", CHAIN_LEN,
              FAN_WIDTH);

  void *state_raw = NULL;
  arts_guid_t state_db =
      arts_db_create(&state_raw, ST_NELEMS * sizeof(uint64_t), ARTS_DB,
                     ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
  if (state_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: state_db create NULL_GUID\n");
    arts_abort(1);
  }
  memset(state_raw, 0, ST_NELEMS * sizeof(uint64_t));
  arts_db_release(state_db, DB_MODE_RW);

  /* Shared DB the whole chain acquires RW in turn; home on rank 0 forces the
   * home-arbitrated self-send round on 1n. */
  void *chain_raw = NULL;
  arts_guid_t chain_db =
      arts_db_create(&chain_raw, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  if (chain_db == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: chain_db create NULL_GUID\n");
    arts_abort(1);
  }
  *(uint64_t *)chain_raw = 0u;
  arts_db_release(chain_db, DB_MODE_RW);

  arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(TOTAL_EDTS);
  latch_hint.rank = 0;
  arts_guid_t latch = arts_event_create(&latch_hint);
  if (latch == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: LATCH create NULL_GUID\n");
    arts_abort(1);
  }

  /* Pre-reserve chain link GUIDs so each link can name its successor. */
  arts_guid_t link[CHAIN_LEN];
  for (int i = 0; i < CHAIN_LEN; i++) {
    link[i] = arts_guid_reserve(ARTS_GUID_EDT, 0);
  }
  for (int i = 0; i < CHAIN_LEN; i++) {
    arts_guid_t next = (i + 1 < CHAIN_LEN) ? link[i + 1] : NULL_GUID;
    uint64_t pv[4] = {(uint64_t)state_db, (uint64_t)i, (uint64_t)next,
                      (uint64_t)latch};
    /* Three dependences gate a link: its two DB slots PLUS the predecessor's
     * control kick (a no-slot DB_MODE_VAL satisfy).  The DB deps satisfy
     * eagerly at arts_add_dependence time, so depc MUST count the control kick
     * as well — otherwise the two DB satisfies alone drive depc_needed to 0 and
     * the link fires before all its slots are registered, racing a kick that
     * arrives concurrently from the predecessor (running on another worker)
     * against this loop's own slot writes.  depc = 2 data + 1 control = 3. */
    arts_edt_create(chain_edt, 4, pv, 3, &(arts_edt_hint_t){.guid = link[i]});
    arts_add_dependence(chain_db, link[i], 0, DB_MODE_RW);
    arts_add_dependence(state_db, link[i], 1, DB_MODE_RW);
  }
  /* Kick the head of the chain. */
  arts_add_dependence((arts_guid_t)0, link[0], (uint32_t)-1, DB_MODE_VAL);

  /* Concurrent fan: RO readers on the shared chain DB, indexed CHAIN_LEN.. */
  for (int j = 0; j < FAN_WIDTH; j++) {
    unsigned idx = (unsigned)(CHAIN_LEN + j);
    uint64_t pv[3] = {(uint64_t)state_db, (uint64_t)idx, (uint64_t)latch};
    arts_guid_t e =
        arts_edt_create(reader_edt, 3, pv, 2, &(arts_edt_hint_t){.rank = 0});
    arts_add_dependence(chain_db, e, 0, DB_MODE_RO);
    arts_add_dependence(state_db, e, 1, DB_MODE_RW);
  }

  arts_guid_t v =
      arts_edt_create(verify_edt, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(latch, v, 0, DB_MODE_NULL);
  arts_add_dependence(state_db, v, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRSW || ARTS_PROTOCOL_LOCK */
