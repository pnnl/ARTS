/* SPDX-License-Identifier: Apache-2.0
 *
 * The write right actually comes back, and the next turn has to ask for it.
 *
 * Where a holder hands the right back at its own idle edge, a rank taking
 * three consecutive write turns on a remote block must ask the home three
 * times, and the home must accept three hand-backs.  No value oracle can see
 * that: a policy that let the holder KEEP the right computes exactly the same
 * answers with one request instead of three.  So the property is asserted on
 * the home's own accounting, which is also the only way to tell "the return
 * landed" from "no return was ever needed".
 *
 * The accounting is read from the per-rank teardown output rather than in
 * process: the accepts happen on a progress thread, and a thread-local read
 * from main would see none of them.  A build that did not record those
 * counters SKIPs rather than passing vacuously.
 *
 * Registered only for the configurations whose grants go back unasked;
 * elsewhere the property asserted here is deliberately false.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_failure_status.h"

#define TURNS 3

static arts_guid_t g_db;

/* One write turn on the remote block: take it, bump it, give it back. */
static void turn_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *p = (uint64_t *)depv[0].ptr;
  if (p == NULL) {
    (void)fprintf(stderr, "FAIL: grant_purge_return turn got no storage\n");
    arts_test_fail();
    arts_shutdown();
    return;
  }
  *p += 1u;
}

static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint64_t *p = (const uint64_t *)depv[1].ptr;
  uint64_t got = (p != NULL) ? *p : 0u;
  if (got != (uint64_t)TURNS) {
    (void)fprintf(stderr, "FAIL: grant_purge_return counter=%llu want %d\n",
                  (unsigned long long)got, TURNS);
    arts_test_fail();
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== grant_purge_return ===\n");

  if (arts_get_total_ranks() < 2u) {
    arts_printf("SKIP grant_purge_return: needs >= 2 ranks — handing the "
                "right back to one's own rank is not a hand-back\n");
    arts_shutdown();
    return;
  }

  void *p = NULL;
  g_db = arts_db_create(&p, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                        &(arts_db_hint_t){.rank = 0u});
  *(uint64_t *)p = 0u;
  arts_db_release(g_db, DB_MODE_RW);

  /* Three turns, all on the SAME non-home rank, strictly ordered.  Ordering
   * them is what makes the count exact: each turn's request can only be
   * answered by a right the previous turn had already given back. */
  arts_guid_t prev = NULL_GUID;
  for (int i = 0; i < TURNS; i++) {
    arts_guid_t done = arts_event_create(&ARTS_EVENT_HINT_LATCH(1));
    arts_guid_t t = arts_edt_create(
        turn_edt, 0, NULL, (prev == NULL_GUID) ? 1u : 2u,
        &(arts_edt_hint_t){.rank = 1u, .output_event = done});
    arts_add_dependence(g_db, t, 0, DB_MODE_RW);
    if (prev != NULL_GUID) {
      arts_add_dependence(prev, t, 1, DB_MODE_NULL);
    }
    prev = done;
  }

  arts_guid_t v =
      arts_edt_create(verify_edt, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0u});
  arts_add_dependence(prev, v, 0, DB_MODE_NULL);
  arts_add_dependence(g_db, v, 1, DB_MODE_RO);
}

/* Pull one counter's value out of a rank's teardown output.  An absent file
 * and an absent counter both mean "this build did not record it", which is a
 * skip and not a failure — the counter set is a build-time choice. */
static int read_counter(unsigned int rank, const char *name, long long *out) {
  char path[64];
  (void)snprintf(path, sizeof(path), "counters/n%u.json", rank);
  FILE *f = fopen(path, "r");
  if (f == NULL) {
    return 0;
  }
  static char buf[1 << 20];
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  (void)fclose(f);
  buf[n] = '\0';
  char key[128];
  (void)snprintf(key, sizeof(key), "\"%s\"", name);
  const char *at = strstr(buf, key);
  if (at == NULL) {
    return 0;
  }
  const char *v = strstr(at, "\"value\"");
  if (v == NULL) {
    return 0;
  }
  v = strchr(v, ':');
  if (v == NULL) {
    return 0;
  }
  *out = strtoll(v + 1, NULL, 10);
  return 1;
}

int main(int argc, char **argv) {
  /* Start from no accounting at all.  Tests share a working directory, so a
   * previous run's output left in place would be read as this one's and the
   * assertions below would pass without this run having produced anything. */
  (void)remove("counters/n0.json");
  (void)remove("counters/n1.json");
  int rc = arts_rt(argc, argv);
  if (rc != 0) {
    return 1;
  }
  if (arts_test_status() != 0) {
    return 1;
  }

  long long accepts = 0;
  long long standalone = 0;
  long long flagged = 0;
  long long dedup = 0;
  if (!read_counter(0, "NUM_GRANT_PURGE_ACCEPT", &accepts)) {
    printf("SKIP grant_purge_return: the hand-back counters are not in this "
           "build's counter set\n");
    return 0;
  }
  (void)read_counter(1, "NUM_GRANT_PURGE_RETURN", &standalone);
  (void)read_counter(1, "NUM_GRANT_PURGE_FLAG", &flagged);
  (void)read_counter(0, "NUM_GRANT_REGRANT_DEDUP", &dedup);

  /* One hand-back per turn, accepted at the home: the right went back every
   * time instead of being kept, so every later turn had to travel through the
   * home to get it again. */
  if (accepts < TURNS) {
    (void)fprintf(stderr,
                  "FAIL: grant_purge_return home accepted %lld hand-backs, "
                  "want >= %d (one per write turn)\n",
                  accepts, TURNS);
    return 1;
  }
  /* Whichever vehicle carried them, the holder sent as many as the home took;
   * the two are the only vehicles, so a shortfall is a lost hand-back. */
  if (standalone + flagged < TURNS) {
    (void)fprintf(stderr,
                  "FAIL: grant_purge_return holder sent %lld hand-backs "
                  "(%lld on a publish, %lld of their own), want >= %d\n",
                  standalone + flagged, flagged, standalone, TURNS);
    return 1;
  }
  /* Every hand-back sent was accepted, once.  The holder arms the obligation
   * and two sites race to discharge it, so a surplus here would mean one
   * hand-back travelled twice and a shortfall that one was dropped. */
  if (accepts != standalone + flagged) {
    (void)fprintf(stderr,
                  "FAIL: grant_purge_return home accepted %lld hand-backs but "
                  "the holder sent %lld (%lld on a publish, %lld of their "
                  "own)\n",
                  accepts, standalone + flagged, flagged, standalone);
    return 1;
  }
  /* Nobody else writes this block, so from the second turn onward the copy
   * this rank kept is still the current one and the home owes it the
   * permission only.  The first turn is the exception: it starts holding
   * nothing, so its grant must carry the bytes. */
  if (dedup < TURNS - 1) {
    (void)fprintf(stderr,
                  "FAIL: grant_purge_return %lld of %d re-grants moved no "
                  "payload, want >= %d (only the first turn needs bytes)\n",
                  dedup, TURNS, TURNS - 1);
    return 1;
  }
  printf("PASS grant_purge_return: %d turns, %lld accepted at the home "
         "(%lld rode a publish, %lld sent their own message), "
         "%lld re-grants carried no payload\n",
         TURNS, accepts, flagged, standalone, dedup);
  return 0;
}
