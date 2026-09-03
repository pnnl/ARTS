/// @file acquire_rtt.c
/// @brief One message, timed: the round trip of a remote read acquire.
///
/// A message census only means something next to the cost of one message, so
/// each runtime under comparison needs a probe that isolates exactly one.  On
/// this side the unit is a remote read acquire of an unchanged datablock: the
/// requester holds a version already, so the acquire is a header-only
/// validation to the home and its reply — one round trip, no payload.
///
/// A serial chain drives it: EDT i runs on the requesting rank, acquires the
/// block read-only, and creates EDT i+1 on the same rank with the same
/// dependence.  Exactly one acquire is ever in flight, so the elapsed time
/// divided by the iteration count is a round trip and nothing else.  The
/// leading iterations are discarded: the first acquire on a rank pays for the
/// route-table entry and the payload the later ones re-validate.
///
/// The start stamp travels in the closure rather than in a variable, so the
/// probe holds no state between tasks and no rank ever reads another's.
///
/// Not registered as a test: it measures, it does not assert.  To build it
/// against an existing tree: compile with the public and internal include
/// roots plus that tree's generated internal headers and `-fno-pie` (function
/// pointers cross ranks), then link with the same line the tree links one of
/// its own benchmark binaries with — the runtime archive is a single
/// localized object, so borrowing a working link line is more reliable than
/// restating it.  `add_arts_test` in tests/CMakeLists.txt is the in-tree
/// equivalent.  Run it on two ranks; a single-rank run is the local control.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "arts.h"

enum {
  P_INDEX = 0, /* iteration number */
  P_COUNT,     /* iterations in the chain, warm-up included */
  P_WARM,      /* iterations discarded before the clock starts */
  P_DB,        /* the block every iteration acquires */
  P_START,     /* stamp taken at the warm-up boundary, forwarded onward */
  P_COUNTC     /* parameter count */
};

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void rtt_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const uint64_t index = paramv[P_INDEX];
  const uint64_t count = paramv[P_COUNT];
  const uint64_t warm = paramv[P_WARM];
  const arts_guid_t db = (arts_guid_t)paramv[P_DB];

  /* Read the payload: an acquire whose data is never touched is still a
     round trip, but touching it keeps the probe honest about what arrived. */
  volatile uint64_t *const data = (volatile uint64_t *)depv[0].ptr;
  const uint64_t seen = data != NULL ? data[0] : 0;

  uint64_t start = paramv[P_START];
  if (index == warm) {
    start = now_ns();
  }

  if (index + 1 < count) {
    const uint64_t next[P_COUNTC] = {index + 1, count, warm, (uint64_t)db,
                                     start};
    const arts_guid_t g = arts_edt_create(
        rtt_edt, P_COUNTC, next, 1,
        &(arts_edt_hint_t){.rank = arts_get_current_rank()});
    arts_add_dependence(db, g, 0, DB_MODE_RO);
    return;
  }

  /* The start stamp was taken on entry to iteration `warm` and this is the
     entry to the last one, so the span holds one acquire per iteration
     strictly after the boundary, the current one included. */
  const uint64_t timed = count - warm - 1;
  const double us = (double)(now_ns() - start) / 1000.0 / (double)timed;
  arts_printf("acquire round trip: %.3f us over %llu iterations (%llu)\n", us,
              (unsigned long long)timed, (unsigned long long)seen);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  char **const argv = (char **)paramv[1];
  const uint64_t argc = paramv[0];
  const uint64_t count = argc > 1 ? strtoull(argv[1], NULL, 10) : 100000;
  const uint64_t warm = argc > 2 ? strtoull(argv[2], NULL, 10) : 1000;
  /* The requester is another rank whenever there is one; on a single rank the
     same chain measures the local acquire, which is the control the ratio
     wants. */
  const unsigned int requester = arts_get_total_ranks() > 1 ? 1u : 0u;

  void *addr = NULL;
  const arts_guid_t db =
      arts_db_create(&addr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = arts_get_current_rank(),
                                       .access_size = UINT64_MAX});
  ((uint64_t *)addr)[0] = 0x5eedull;
  arts_db_release(db, DB_MODE_RW);

  arts_printf("acquire_rtt: %llu iterations (%llu warm-up), home rank %u, "
              "requesting rank %u\n",
              (unsigned long long)count, (unsigned long long)warm,
              arts_get_current_rank(), requester);

  const uint64_t first[P_COUNTC] = {0, count, warm, (uint64_t)db, 0};
  const arts_guid_t g = arts_edt_create(rtt_edt, P_COUNTC, first, 1,
                                        &(arts_edt_hint_t){.rank = requester});
  arts_add_dependence(db, g, 0, DB_MODE_RO);
}

int main(int argc, char **argv) { return arts_rt(argc, argv) != 0 ? 1 : 0; }
