/* SPDX-License-Identifier: Apache-2.0
 *
 * coherence_rwro_phase — RW → RO → RW phase alternation with ordered reads.
 *
 * Each phase k: ONE writer RW-acquires a counter DB and increments it to k+1;
 * then M readers RO-acquire it and ASSERT they observe exactly k+1; then the
 * next phase's writer RW-acquires it again.  The phases are wired with strict
 * happens-before so the assertions are deterministic (no data race):
 *
 *   writer[k] --output_event ew[k]--> readers[k] (RO, ×M) --finish lk[k]-->
 * writer[k+1]
 *
 * - writer[k] → readers[k] via ew[k]: the output_event fires only after
 * writer[k] released the DB (OCR §1.6.2 release-before-satisfy), so every
 * reader sees the committed k+1 (RW→RO serve).
 * - readers[k] → writer[k+1] via the finish event lk[k]: each reader's finish
 *   DECR is emitted at completion, AFTER its RO release, so writer[k+1] RW-
 *   acquires only once every reader returned its RO grant (RO→RW flip at
 * rc==0).
 *
 * Across ranks this exercises the EXCL-OWNER RW→RO flip (SERVE_ALL drain) and
 * RO→RW flip (migrate after the last RO_RETURN).  Portable (arts.h + arts_rt):
 * asserts COHERENCE, so it passes on every protocol/placement build.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define N_PHASES 6
#define N_READERS 4

/* Writer: DB (RW) is the LAST dep slot (slot 0 for phase 0, slot 1 otherwise —
 * the earlier slot is the prior phase's finish gate, NULL/control).  Increment
 * is HB-ordered after the prior phase, so a plain ++ is correct. */
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  uint64_t *counter = (uint64_t *)depv[depc - 1u].ptr;
  if (counter != NULL) {
    *counter += 1u;
  }
}

/* Reader: depv[0] = writer's output_event (NULL/control), depv[1] = counter DB
 * (RO).  Must observe exactly the post-write value paramv[0] (= phase+1). */
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t want = paramv[0];
  const uint64_t *counter = (const uint64_t *)depv[1].ptr;
  uint64_t got = (counter != NULL) ? *counter : ~want;
  if (got != want) {
    (void)fprintf(stderr,
                  "FAIL: coherence_rwro_phase reader saw %llu (want %llu)\n",
                  (unsigned long long)got, (unsigned long long)want);
    arts_abort(1);
  }
}

/* Final: gated on the last phase's finish event + counter (RO); the counter
 * must equal N_PHASES (every writer ran exactly once, in order). */
static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint64_t *counter = (const uint64_t *)depv[1].ptr;
  uint64_t got = (counter != NULL) ? *counter : 0u;
  if (got != (uint64_t)N_PHASES) {
    (void)fprintf(stderr,
                  "FAIL: coherence_rwro_phase final counter=%llu "
                  "(want %d)\n",
                  (unsigned long long)got, N_PHASES);
    arts_abort(1);
  }
  arts_printf("coherence_rwro_phase: %d phases × %d readers, counter=%llu "
              "— PASS\n",
              N_PHASES, N_READERS, (unsigned long long)got);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_rwro_phase (%d phases, %d readers) ===\n",
              N_PHASES, N_READERS);

  unsigned int nranks = arts_get_total_ranks();

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

  /* Per-phase writer output_event (ew) + reader-batch finish event (lk). */
  arts_guid_t ew[N_PHASES];
  arts_guid_t lk[N_PHASES];
  for (unsigned int k = 0; k < N_PHASES; k++) {
    ew[k] = arts_event_create(&ARTS_EVENT_HINT_LATCH(1));
    lk[k] = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  }

  for (unsigned int k = 0; k < N_PHASES; k++) {
    /* writer[k] on rank k%nranks; output_event=ew[k]; gated on lk[k-1] (k>0).
     */
    unsigned int wr = (nranks > 1u) ? (k % nranks) : 0u;
    uint32_t wdepc = (k == 0u) ? 1u : 2u;
    arts_guid_t w =
        arts_edt_create(writer_edt, 0, NULL, wdepc,
                        &(arts_edt_hint_t){.rank = wr, .output_event = ew[k]});
    if (k > 0u) {
      arts_add_dependence(lk[k - 1u], w, 0, DB_MODE_NULL);
      arts_add_dependence(counter, w, 1, DB_MODE_RW);
    } else {
      arts_add_dependence(counter, w, 0, DB_MODE_RW);
    }

    /* M readers, gated on ew[k], joined to finish event lk[k]; assert == k+1.
     */
    uint64_t rp[1] = {(uint64_t)(k + 1u)};
    for (unsigned int m = 0; m < N_READERS; m++) {
      unsigned int rr = (nranks > 1u) ? ((k + m + 1u) % nranks) : 0u;
      arts_guid_t rd = arts_edt_create(
          reader_edt, 1, rp, 2,
          &(arts_edt_hint_t){.rank = rr, .finish_event = lk[k]});
      arts_add_dependence(ew[k], rd, 0, DB_MODE_NULL);
      arts_add_dependence(counter, rd, 1, DB_MODE_RO);
    }
  }

  /* Final verify gated on the last phase's finish event + counter (RO). */
  arts_guid_t v =
      arts_edt_create(verify_edt, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0u});
  arts_add_dependence(lk[N_PHASES - 1], v, 0, DB_MODE_NULL);
  arts_add_dependence(counter, v, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
