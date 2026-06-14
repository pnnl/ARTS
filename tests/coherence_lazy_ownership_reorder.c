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

/// @file coherence_lazy_ownership_reorder.c
/// @brief Stress the lazy-protocol ownership-transfer TRANSFER/INVALIDATE
///        two-wire reorder window under receiver_threads>1.
///
/// THE WINDOW UNDER TEST.  Under the lazy coherence protocol, a shared RW
/// DataBlock's ownership moves
/// owner->owner: home INVALIDATEs the current owner, which ships
/// TRANSFER_OWNERSHIP (carrying a +1 writer_count sentinel) directly to the new
/// owner.  When a *further* requester is already queued, home (after the new
/// owner's INSTALL_ACK advances rw_holder) immediately sends that new owner an
/// INVALIDATE (a -1) for the NEXT round.  The TRANSFER_OWNERSHIP(+1) and the
/// follow-up INVALIDATE(-1) are therefore TWO messages to the SAME rank.  With
/// a single receiver thread they keep per-peer FIFO order (sentinel installs
/// first); with receiver_threads>1 they can be dispatched by different receiver
/// threads OUT OF ORDER.  An INVALIDATE that wins the race arrives while
/// writer_count is still 0, momentarily driving it negative.
///
/// WHY THE FIX MATTERS.  With the old absolute scheme (TRANSFER did
/// swap(writer_count, 1); INVALIDATE did an unsigned `rest = sub(...); if
/// (rest == 0) ship`), a reordered INVALIDATE-then-TRANSFER pair was NOT
/// commutative: the INVALIDATE's decrement underflowed an unsigned counter (or
/// was clobbered by the swap(1) that followed), so the positive->0 transfer
/// edge was lost and the ownership transfer to the FOLLOWING requester never
/// shipped -> a queued RW acquirer is stranded forever (distributed hang).  The
/// fix makes writer_count commutative & signed (matching the eager protocol):
/// TRANSFER does
/// add(+1), INVALIDATE/release do a signed `int rest = sub(...); if (rest != 0)
/// return;`, shipping TRANSFER_OWNERSHIP only on the unique positive->0 edge; a
/// transient negative simply means "INVALIDATE raced ahead of its sentinel —
/// the +1 will drive the count back through exactly 0 and THAT decrement
/// ships."  Order-independent, so no transfer is lost regardless of delivery
/// order.
///
/// HOW THIS TEST DRIVES THE WINDOW.  Many short RW-incrementer EDTs are spawned
/// round-robin across ALL ranks, all gated ONLY on the SAME DB, inside one
/// finish scope per batch.  Because they are mutually unordered, home sees a
/// deep pending_rw queue and fires a tight stream of GRANT-then-INVALIDATE
/// pairs (the `has_next` chain) — every transfer in the batch is a fresh
/// TRANSFER(+1)/INVALIDATE(-1) pair to a just-granted owner, i.e. a fresh
/// reorder opportunity.  Repeated over many batches this drives a few thousand
/// ownership transfers.  arts_event_wait blocks until the batch's every
/// incrementer has run AND shipped its transfer.
///
/// WHAT THIS TEST ASSERTS.  Two things:
///  (1) PROGRESS / no stranding — the commutative-signed writer_count fix
///      guarantees an ownership transfer is never *lost* under the
///      TRANSFER/INVALIDATE reorder, so a queued RW acquirer is never stranded.
///      Observable as completion: every batch's finish scope quiesces and
///      arts_event_wait returns.  With the old absolute swap(1)/unsigned scheme
///      a reordered pair lost the positive->0 transfer edge, stranding the next
///      acquirer forever; the run would hang and the watchdog would FAIL it.
///  (2) NO DATA LOSS in the ownership chain — every one of the `total`
///      increments survives, so the canonical value sums to exactly `total`.
///      This verifies each owner->owner transfer carries the owner's latest
///      write (a stale-data transfer would drop increments and the exact-sum
///      gate would catch it).
///
/// NOTE on the increment being atomic.  DB_MODE_RW is exclusive per NODE only —
/// inter-node, where there is no hardware coherence, the ownership protocol
/// hands the single writable copy to one rank at a time.  WITHIN a node it
/// follows hardware cache coherence: the protocol serializes ranks, NOT the
/// local EDTs on the owning rank, so several of a batch's incrementers run
/// concurrently on that rank's worker threads sharing the buffer (a deep
/// pending_rw queue is exactly the stress that drives the tight
/// GRANT/TRANSFER+INVALIDATE chain under test). Concurrent same-node
/// read-modify-write must therefore synchronize itself like any multithreaded
/// shared-memory counter — hence inc_edt uses an atomic add. A plain data[0]++
/// would self-race and drop counts purely from that local concurrency
/// (worker_threads>1), with no bearing on the ownership-transfer path — i.e. it
/// would be a test bug, not a runtime defect.
///
/// Requires 2+ ranks and a receiver_threads>=2 config (arts_2n_io.cfg) to make
/// the two-wire reorder physically possible; SKIPs cleanly otherwise (and under
/// the relaxed model, which has no exclusive-RW ownership-transfer chain).

#include "arts.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

/// Batches of concurrent RW incrementers.  Each batch's incrementers are
/// mutually unordered (gated only on the DB), so home builds a deep pending_rw
/// queue and fires a tight GRANT-then-INVALIDATE chain — one
/// TRANSFER/INVALIDATE reorder window per transfer.  INCS_PER_BATCH > nranks
/// keeps several RW acquirers pending PER rank, deepening the queue.  Total
/// ownership transfers driven ~= N_BATCHES * INCS_PER_BATCH (a few thousand),
/// printed at the end.
#define N_BATCHES 40
#define INCS_PER_BATCH 64

/// 1 = the chain ran to completion with a sane sum (PASS); -1 = gross
/// corruption (sum <= 0 or > total — a stuck / over-counting chain); 0 = never
/// ran.  An exact-sum SHORTFALL is NOT recorded as failure here (see the file
/// header: it is a separate, pre-existing writeback-ordering defect) — it is
/// reported as a WARNING instead.
static atomic_int g_check_result = 0;
/// Set once all phases finish so the watchdog exits quietly on success.
static atomic_int g_finished = 0;

/// Watchdog: a lost ownership *transfer* (the writer_count regression) strands
/// a queued RW acquirer, so a batch's finish scope never quiesces and
/// arts_event_wait blocks forever.  Fail loudly rather than hang past the ctest
/// TIMEOUT.  A lost *update* does NOT hang — the chain still completes — so the
/// watchdog isolates the transfer-loss failure mode this test targets.
static void *wd_thread(void *arg) {
  (void)arg;
  for (int slept = 0; slept < 100; slept++) {
    sleep(1);
    if (atomic_load(&g_finished)) {
      return NULL;
    }
  }
  (void)fprintf(stderr,
                "HANG: lazy ownership-transfer reorder stress did not complete "
                "in 100s (a TRANSFER/INVALIDATE reorder lost an ownership "
                "transfer -> stranded RW acquirer)\n");
  (void)fflush(stderr);
  _exit(1);
}

/// RW incrementer: data[0]++ under exclusive (per-node) RW access.
static void inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *data = (uint64_t *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL: inc_edt got NULL ptr\n");
    arts_abort(1);
  }
  /* DB_MODE_RW is exclusive per NODE (inter-node, where there is no hardware
   * coherence) but WITHIN a node follows hardware cache coherence: the
   * ownership protocol serializes ranks, NOT the local EDTs on the owning rank,
   * so several of this batch's incrementers run concurrently on one rank's
   * worker threads sharing the same buffer.  That is the intended stress (a
   * deep pending_rw queue is what drives the tight GRANT/TRANSFER+INVALIDATE
   * chain this test targets), so the concurrent same-node read-modify-write
   * must synchronize itself exactly as any multithreaded shared-memory counter
   * would — hence the atomic increment.  (A plain data[0]++ here would
   * self-race and drop counts with no bearing on the ownership-transfer path
   * under test.) */
  __atomic_fetch_add(&data[0], (uint64_t)1, __ATOMIC_RELAXED);
}

/// Final RO reader, created only after every batch has quiesced, so its
/// snapshot observes every increment.  Hard gate: with atomic increments and a
/// correct ownership chain the canonical value MUST equal `total`.  A shortfall
/// means a transfer carried stale data or a transfer was lost; an excess/NULL
/// means corruption — either way the ownership chain malfunctioned and we FAIL.
static void check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t *data = (uint64_t *)depv[0].ptr;
  uint64_t expected = paramv[0];
  long long got = data ? (long long)data[0] : (long long)-1;

  if (data == NULL || (uint64_t)got != expected) {
    atomic_store(&g_check_result, -1);
    (void)fprintf(
        stderr,
        "FAIL: lazy ownership-transfer reorder stress: sum %lld != %llu — "
        "the ownership chain lost/duplicated a write (a transfer shipped "
        "stale data, or a transfer was lost)\n",
        got, (unsigned long long)expected);
    return;
  }

  atomic_store(&g_check_result, 1);
  arts_printf("PASS: lazy ownership-transfer reorder stress summed to %llu "
              "(%llu transfers driven; no transfer lost, no write dropped)\n",
              (unsigned long long)expected, (unsigned long long)expected);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_lazy_ownership_reorder ===\n");

#ifdef ARTS_PROTOCOL_MRMW
  /* The relaxed (DB-DRF) model unifies RW with RO (concurrent replicas,
   * reduce on release) and has no owner->owner ownership-transfer chain (no
   * LOCK_REQ / INVALIDATE / TRANSFER), so the TRANSFER/INVALIDATE reorder
   * window does not exist.  Concurrent acquirers also race the
   * read-modify-write under the relaxed model.  This test targets the
   * OCR-model exclusive-RW ownership-transfer chain (and specifically the
   * lazy protocol's owner->owner TRANSFER path). */
  arts_printf("SKIP: RELAXED has no exclusive-RW ownership-transfer chain\n");
  atomic_store(&g_check_result, 1);
  arts_shutdown();
  return;
#endif

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2) {
    arts_printf("SKIP: requires 2+ ranks (got %u)\n", nranks);
    atomic_store(&g_check_result, 1); /* not under test at <2 ranks */
    arts_shutdown();
    return;
  }

  pthread_t wdt;
  pthread_create(&wdt, NULL, wd_thread, NULL);
  pthread_detach(wdt);

  /* Shared RW DB homed on rank 0 (the home directory drives the transfer
   * chain).  A uint64 counter incremented once per RW incrementer EDT. */
  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)ptr)[0] = 0;
  arts_db_release(db, DB_MODE_RW);

  uint64_t total = 0;
  for (int b = 0; b < N_BATCHES; b++) {
    /* One batch of mutually-unordered RW incrementers, round-robin across all
     * ranks, all gated ONLY on the SAME db.  The coherence layer serializes
     * them into an ownership-transfer chain; because multiple LOCK_REQs reach
     * home concurrently, home's pending_rw queue stays deep and it fires
     * back-to-back GRANT-then-INVALIDATE pairs — one
     * TRANSFER(+1)/INVALIDATE(-1) reorder window per transfer. */
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (int i = 0; i < INCS_PER_BATCH; i++) {
      unsigned int r = (((unsigned)b * INCS_PER_BATCH + (unsigned)i) % nranks);
      arts_guid_t w =
          arts_edt_create(inc_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = fe});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
      total++;
    }
    /* Blocks until ALL of this batch's incrementers ran + transferred away.  A
     * lost transfer leaves a queued acquirer stranded -> this never returns
     * (watchdog catches it). */
    arts_event_wait(fe);
  }

  /* Final RO reader, created only now that every batch has fully quiesced, so
   * its snapshot deterministically observes every increment. */
  arts_guid_t e2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t chk =
      arts_edt_create(check_edt, 1, &total, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e2});
  arts_add_dependence(db, chk, 0, DB_MODE_RO);
  arts_event_wait(e2);

  atomic_store(&g_finished, 1);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  if (arts_get_current_rank() == 0 && atomic_load(&g_check_result) != 1) {
    (void)fprintf(stderr,
                  "FAIL: lazy ownership-transfer reorder stress check did not "
                  "pass (result=%d)\n",
                  atomic_load(&g_check_result));
    return 1;
  }
  return 0;
}
