/* SPDX-License-Identifier: Apache-2.0
 *
 * coherence_rw_chain — ordered RW migration chain + value preservation.
 *
 * A single counter DB is RW-acquired by a CHAIN of writer EDTs spread across
 * ranks (writer i on rank i % nranks).  Writer i+1 is gated on writer i's
 * output_event, so the runtime's release-before-satisfy rule (OCR §1.6.2: an
 * EDT completes the release of all its data blocks before its post-event is
 * satisfied) gives writer i+1 a happens-before edge to writer i: it acquires
 * the DB only after writer i released it, hence observes writer i's increment.
 * A plain (non-atomic) increment is therefore correct — the chain is strictly
 * ordered, not racy.  Across ranks this drives the LOCK-LAZY ownership
 * MIGRATION chain (DB hops rank→rank); single-node it is the owner local-hit
 * path.
 *
 * A final RO reader, gated on the last writer's output_event, asserts the
 * counter equals the number of writers (model invariant I2: the owner's value
 * is preserved across every migration).  Portable (arts.h + arts_rt only): it
 * asserts COHERENCE, so it passes on every protocol/timing build.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define N_WRITERS 12 /* > nranks so the DB migrates around several times */

/* Writer: depv[0] = counter DB (RW).  depv[1] (when present) = previous
 * writer's output_event (NULL/control) — the chain edge.  Increment is
 * HB-ordered after the previous writer, so a plain ++ observes the committed
 * prior value. */
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *counter = (uint64_t *)depv[0].ptr;
  if (counter != NULL) {
    *counter += 1u;
  }
}

/* Verify: depv[0] = last writer's output_event (NULL), depv[1] = counter DB
 * (RO).  Gated on the whole chain completing, so it reads committed data. */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int n = (unsigned int)paramv[0];
  const uint64_t *counter = (const uint64_t *)depv[1].ptr;
  uint64_t got = (counter != NULL) ? *counter : 0u;
  if (got != (uint64_t)n) {
    (void)fprintf(stderr, "FAIL: coherence_rw_chain counter=%llu (want %u)\n",
                  (unsigned long long)got, n);
    arts_abort(1);
  }
  arts_printf("coherence_rw_chain: %u-writer migration chain, counter=%llu "
              "— PASS\n",
              n, (unsigned long long)got);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_rw_chain (%d writers) ===\n", N_WRITERS);

  unsigned int nranks = arts_get_total_ranks();

  /* Counter DB on rank 0, initialized to 0 and RELEASED before any writer is
   * wired (release-before-satisfy: the first writer must not acquire stale). */
  void *cp = NULL;
  arts_guid_t counter =
      arts_db_create(&cp, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0u});
  if (counter == NULL_GUID || cp == NULL) {
    (void)fprintf(stderr, "FAIL: counter DB create\n");
    arts_abort(1);
  }
  *(uint64_t *)cp = 0u;
  arts_db_release(counter, DB_MODE_RW);

  /* One LATCH output_event per writer — fires at that writer's epilogue, after
   * its DB release publishes the increment. */
  arts_guid_t oe[N_WRITERS];
  for (unsigned int i = 0; i < N_WRITERS; i++) {
    oe[i] = arts_event_create(&ARTS_EVENT_HINT_LATCH(1));
  }

  /* Chain: writer i on rank i%nranks, output_event=oe[i]; depends on the
   * counter (RW, slot 0) and, for i>0, on oe[i-1] (NULL/control, slot 1). */
  for (unsigned int i = 0; i < N_WRITERS; i++) {
    unsigned int r = (nranks > 1u) ? (i % nranks) : 0u;
    uint32_t edepc = (i == 0u) ? 1u : 2u;
    arts_guid_t w =
        arts_edt_create(writer_edt, 0, NULL, edepc,
                        &(arts_edt_hint_t){.rank = r, .output_event = oe[i]});
    arts_add_dependence(counter, w, 0, DB_MODE_RW);
    if (i > 0u) {
      arts_add_dependence(oe[i - 1u], w, 1, DB_MODE_NULL);
    }
  }

  /* Verify gated on the last writer's output_event + the counter (RO). */
  uint64_t vp[1] = {(uint64_t)N_WRITERS};
  arts_guid_t v =
      arts_edt_create(verify_edt, 1, vp, 2, &(arts_edt_hint_t){.rank = 0u});
  arts_add_dependence(oe[N_WRITERS - 1], v, 0, DB_MODE_NULL);
  arts_add_dependence(counter, v, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
