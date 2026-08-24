/*
 * coherence_owner_write_after_lend.c — the owner writes after lending a copy.
 *
 * The shape that motivated the retaining read grant, reduced to an oracle:
 * a rank owns a block, lends a read copy to another rank, and then writes it.
 * The reader reads again afterwards, ordered strictly after the write.
 *
 *   rank 0            rank 1
 *   ------            ------
 *   create, write V1
 *                     read  -> expects V1        (may retain the copy)
 *   write V2                                     (owner, resident bytes)
 *                     read  -> MUST see V2
 *
 * Every step is chained through a finish event, so nothing here is a race:
 * the second read happens-after the second write. A rank that keeps a stale
 * copy and serves it locally answers V1 and the program is silently wrong —
 * no crash, no hang, just a bad value. That is why this is a value oracle and
 * needs no counters.
 *
 * Portable: no protocol #if, no internal headers. Under a purging policy the
 * copy dies at its own zero edge and the second read re-fetches, so this
 * passes there too — it is a regression gate for every arm, not a probe of one.
 */

#include "arts.h"
#include <stdio.h>

#define V1 0x11110000u
#define V2 0x22220000u

static void write_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d == NULL) {
    (void)fprintf(stderr, "FAIL owner_write_after_lend: writer got NULL\n");
    arts_abort(1);
  }
  d[0] = (unsigned int)paramv[0];
}

static void read_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int want = (unsigned int)paramv[0];
  unsigned int seq = (unsigned int)paramv[1];
  const unsigned int *d = (const unsigned int *)depv[0].ptr;
  if (d == NULL || d[0] != want) {
    (void)fprintf(stderr,
                  "FAIL owner_write_after_lend: read %u expected 0x%x got "
                  "0x%x — a retained copy was served across a write\n",
                  seq, want, d ? d[0] : 0u);
    arts_abort(1);
  }
}

static void done_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  printf("PASS owner_write_after_lend: the write was visible to the lender\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  if (arts_get_total_ranks() < 2) {
    printf("SKIP owner_write_after_lend: needs a second rank to lend to\n");
    arts_shutdown();
    return;
  }

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     NULL);
  ((unsigned int *)ptr)[0] = V1;
  arts_db_release(db, DB_MODE_RW);

  /* read #1 on the far rank — this is the lend. */
  arts_guid_t e1 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  uint64_t r1pv[2] = {V1, 1};
  arts_edt_hint_t h1 = {.finish_event = e1, .rank = 1};
  arts_guid_t r1 = arts_edt_create(read_edt, 2, r1pv, 1, &h1);
  arts_add_dependence(db, r1, 0, DB_MODE_RO);

  /* write #2 back on the owner, ordered after the lend. */
  arts_guid_t e2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  uint64_t wpv[1] = {V2};
  arts_edt_hint_t h2 = {.finish_event = e2, .rank = 0};
  arts_guid_t w2 = arts_edt_create(write_edt, 1, wpv, 2, &h2);
  arts_add_dependence(db, w2, 0, DB_MODE_RW);
  arts_add_dependence(e1, w2, 1, DB_MODE_NULL);

  /* read #3 on the far rank again, ordered after the write. */
  arts_guid_t e3 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  uint64_t r2pv[2] = {V2, 2};
  arts_edt_hint_t h3 = {.finish_event = e3, .rank = 1};
  arts_guid_t r2 = arts_edt_create(read_edt, 2, r2pv, 2, &h3);
  arts_add_dependence(db, r2, 0, DB_MODE_RO);
  arts_add_dependence(e2, r2, 1, DB_MODE_NULL);

  arts_guid_t fin = arts_edt_create(done_edt, 0, NULL, 1, NULL);
  arts_add_dependence(e3, fin, 0, DB_MODE_NULL);
}

int main(int argc, char **argv) { return arts_rt(argc, argv) != 0 ? 1 : 0; }
