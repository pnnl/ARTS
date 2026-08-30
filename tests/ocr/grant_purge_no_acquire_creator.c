/* SPDX-License-Identifier: Apache-2.0
 *
 * A data block created NO_ACQUIRE from a rank that is not its home must still
 * be writable by a third rank.
 *
 * NO_ACQUIRE says the creator takes no hold: it neither acquires the block nor
 * releases it, so it never reaches an idle edge and can never hand the write
 * right on.  It does not even keep a cache for the block.  The only rank that
 * can hold that right at rest is therefore the block's home, and the home-side
 * create must end saying so on every one of its arms.  Name the creator
 * instead and the first writer's request is addressed to a rank that holds
 * nothing — and has nothing to be asked with: nothing answers, and the write
 * never runs.
 *
 * The home-side create has three arms, and only ONE of them is reached by an
 * ordinary create: a fresh install.  The other two coalesce onto a descriptor
 * some other message already put in the route table, and reaching them takes a
 * second create of the SAME block.  That is what the two creators below are
 * for — the first lands on the fresh arm, the second on a coalesce arm.  A
 * single creator exercises neither, and would pass against a home that names
 * the creator on both.
 *
 * The failure is a WEDGE, not a wrong value, so the ctest timeout is half the
 * oracle here and the value check is the other half.  Three distinct ranks are
 * needed to keep home, creators and writer apart, which is exactly the
 * configuration in which the creator can be neither.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#include "../test_failure_status.h"

#define SENTINEL 0xDEC0DEDu

/* Writer: runs on a rank that is not the home, takes the write turn the home
 * must have kept for it, and stamps the block. */
static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *p = (unsigned int *)depv[0].ptr;
  if (p == NULL) {
    (void)fprintf(stderr, "FAIL: grant_purge_no_acquire_creator writer got "
                          "no storage for its write turn\n");
    arts_test_fail();
    arts_shutdown();
    return;
  }
  *p = SENTINEL;
}

/* Reader: on the home, gated on the writer's epilogue, so the release that
 * published the write has already completed. */
static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const unsigned int *p = (const unsigned int *)depv[1].ptr;
  unsigned int got = (p != NULL) ? *p : 0u;
  if (got != SENTINEL) {
    (void)fprintf(stderr,
                  "FAIL: grant_purge_no_acquire_creator read 0x%x want 0x%x\n",
                  got, SENTINEL);
    arts_test_fail();
  } else {
    arts_printf("PASS: grant_purge_no_acquire_creator\n");
  }
  arts_shutdown();
}

/* Creator: runs on ranks 1 and 2, both naming the SAME labelled block homed on
 * rank 0, and neither taking a hold on it.  One of the two create messages
 * finds the home descriptor already installed and takes a coalesce arm. */
static void creator_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t child = arts_guid_from_index((arts_guid_t)paramv[0], 0);

  /* NO_ACQUIRE hands back no pointer — the creator holds nothing to write,
   * and on a remote home it does not even keep a cache for the block. */
  void *p = arts_db_create_with_guid(child, sizeof(unsigned int), ARTS_DB,
                                     ARTS_DB_PROP_NO_ACQUIRE, NULL);
  if (p != NULL) {
    (void)fprintf(stderr, "FAIL: grant_purge_no_acquire_creator got a "
                          "writable pointer for a block it never acquired\n");
    arts_test_fail();
  }
}

/* Wiring runs only once both creates have been made, so the coalesce arm has
 * certainly run by the time the first write turn is asked for. */
static void wirer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t child = arts_guid_from_index((arts_guid_t)paramv[0], 0);

  arts_guid_t done = arts_event_create(&ARTS_EVENT_HINT_LATCH(1));

  arts_guid_t w = arts_edt_create(
      writer_edt, 0, NULL, 1,
      &(arts_edt_hint_t){.rank = 2u, .output_event = done});
  arts_add_dependence(child, w, 0, DB_MODE_RW);

  arts_guid_t r =
      arts_edt_create(reader_edt, 0, NULL, 2, &(arts_edt_hint_t){.rank = 0u});
  arts_add_dependence(done, r, 0, DB_MODE_NULL);
  arts_add_dependence(child, r, 1, DB_MODE_RO);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== grant_purge_no_acquire_creator ===\n");

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 3u) {
    arts_printf("SKIP grant_purge_no_acquire_creator: needs >= 3 ranks "
                "(have %u) to keep home, creators and writer apart\n",
                nranks);
    arts_shutdown();
    return;
  }

  /* A one-element labelled range homed on rank 0: both creators derive the
   * identical child GUID from it, so the home sees two creates of one block. */
  arts_guid_t range = arts_guid_reserve_range(ARTS_GUID_DB, 1, 0);
  uint64_t rparam = (uint64_t)range;

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  (void)arts_edt_create(creator_edt, 1, &rparam, 0,
                        &(arts_edt_hint_t){.rank = 1u, .finish_event = fe});
  (void)arts_edt_create(creator_edt, 1, &rparam, 0,
                        &(arts_edt_hint_t){.rank = 2u, .finish_event = fe});

  arts_guid_t wire =
      arts_edt_create(wirer_edt, 1, &rparam, 1, &(arts_edt_hint_t){.rank = 0u});
  arts_add_dependence(fe, wire, 0, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  int rc = arts_rt(argc, argv);
  return rc ? 1 : arts_test_status();
}
