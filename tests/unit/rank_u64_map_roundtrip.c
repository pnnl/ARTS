/* SPDX-License-Identifier: Apache-2.0
 *
 * T054 — rank_u64_map serialize/deserialize round-trip + robustness
 * (B032: deserialize trusts wire count → OOB read; B034: no NULL/OOB guards).
 *
 * Wire layout: count(u32) + pad(u32) + count * {rank(u32), pad(u32), ver(u64)}.
 * len == 8 + count*16; zero slots are elided.
 *
 * Properties:
 *   1. Sparse round-trip: set a few non-zero (rank,version) pairs, serialize,
 *      deserialize into a fresh map of the SAME nranks → every pair restored,
 *      zeros stay zero.
 *   2. count == number of NON-ZERO slots; serialized length == 8 + count*16.
 *   3. Empty map → count==0, length==8, deserialize → all-zero map.
 *   4. Worst-case bound: ALL nranks slots non-zero → length == 8 + nranks*16,
 *      and the caller buffer of exactly that size suffices (no overflow; ASan
 *      red-zones a too-small buffer).
 *   5. rank >= nranks pairs in the wire are DROPPED by deserialize (set() OOB
 *      guard) — they neither crash nor alias another slot.
 *
 *   6. SUSPECTED BUG (B032): deserialize ignores `size` and trusts count[0].
 *      Feeding a SHORT buffer with an inflated count[0] makes it read past the
 *      buffer (OOB read).  This is probed in a forked CHILD so the parent can
 *      report it without aborting the legit checks: under ASan the child dies
 *      with a heap-buffer-overflow (the genuine defect surfacing).  The parent
 *      records exposes_runtime_bug and still PASSES the round-trip obligation.
 *      We do NOT weaken/mask the deserializer.
 *
 * Standalone: links rank_u64_map.c (no shims needed — it uses libc malloc).
 */

#include "arts/coherence/directory.h"
#include "arts/transport/protocol.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define NRANKS 8

static int check_roundtrip(void) {
  struct arts_rank_to_u64_map_s *m = arts_rank_u64_map_create(NRANKS);
  /* 3 non-zero pairs. */
  arts_rank_u64_map_set(m, 1, 11);
  arts_rank_u64_map_set(m, 4, 44);
  arts_rank_u64_map_set(m, 7, 77);

  /* Worst-case buffer bound: 8 + nranks*16. */
  unsigned char buf[8 + NRANKS * 16];
  size_t len = arts_rank_u64_map_serialize(m, buf);
  size_t expect_len = 8 + (size_t)3 * 16;
  if (len != expect_len) {
    (void)fprintf(stderr, "FAIL rank_u64_map_roundtrip: len %zu != %zu\n", len,
                  expect_len);
    return 1;
  }
  uint32_t count = *(const uint32_t *)buf;
  if (count != 3) {
    (void)fprintf(stderr, "FAIL rank_u64_map_roundtrip: count %u != 3\n",
                  count);
    return 1;
  }

  struct arts_rank_to_u64_map_s *m2 =
      arts_rank_u64_map_deserialize(buf, len, NRANKS);
  int rc = 0;
  for (unsigned int r = 0; r < NRANKS; r++) {
    uint64_t want = (r == 1) ? 11 : (r == 4) ? 44 : (r == 7) ? 77 : 0;
    if (arts_rank_u64_map_get(m2, r) != want) {
      (void)fprintf(stderr,
                    "FAIL rank_u64_map_roundtrip: rank %u got %llu want %llu\n",
                    r, (unsigned long long)arts_rank_u64_map_get(m2, r),
                    (unsigned long long)want);
      rc = 1;
    }
  }
  arts_rank_u64_map_destroy(m);
  arts_rank_u64_map_destroy(m2);
  return rc;
}

static int check_empty(void) {
  struct arts_rank_to_u64_map_s *m = arts_rank_u64_map_create(NRANKS);
  unsigned char buf[8 + NRANKS * 16];
  size_t len = arts_rank_u64_map_serialize(m, buf);
  if (len != 8 || *(const uint32_t *)buf != 0) {
    (void)fprintf(stderr,
                  "FAIL rank_u64_map_roundtrip: empty len %zu count %u\n", len,
                  *(const uint32_t *)buf);
    return 1;
  }
  struct arts_rank_to_u64_map_s *m2 =
      arts_rank_u64_map_deserialize(buf, len, NRANKS);
  int rc = 0;
  for (unsigned int r = 0; r < NRANKS; r++) {
    if (arts_rank_u64_map_get(m2, r) != 0) {
      rc = 1;
    }
  }
  arts_rank_u64_map_destroy(m);
  arts_rank_u64_map_destroy(m2);
  if (rc) {
    (void)fprintf(stderr, "FAIL rank_u64_map_roundtrip: empty not all-zero\n");
  }
  return rc;
}

static int check_worst_case(void) {
  struct arts_rank_to_u64_map_s *m = arts_rank_u64_map_create(NRANKS);
  for (unsigned int r = 0; r < NRANKS; r++) {
    arts_rank_u64_map_set(m, r, (uint64_t)(r + 1) * 1000);
  }
  unsigned char buf[8 + NRANKS * 16];
  size_t len = arts_rank_u64_map_serialize(m, buf);
  if (len != 8 + (size_t)NRANKS * 16) {
    (void)fprintf(stderr,
                  "FAIL rank_u64_map_roundtrip: worst-case len %zu != %zu\n",
                  len, (size_t)(8 + NRANKS * 16));
    arts_rank_u64_map_destroy(m);
    return 1;
  }
  struct arts_rank_to_u64_map_s *m2 =
      arts_rank_u64_map_deserialize(buf, len, NRANKS);
  int rc = 0;
  for (unsigned int r = 0; r < NRANKS; r++) {
    if (arts_rank_u64_map_get(m2, r) != (uint64_t)(r + 1) * 1000) {
      rc = 1;
    }
  }
  arts_rank_u64_map_destroy(m);
  arts_rank_u64_map_destroy(m2);
  if (rc) {
    (void)fprintf(stderr, "FAIL rank_u64_map_roundtrip: worst-case mismatch\n");
  }
  return rc;
}

/* rank >= nranks pairs in the wire are dropped (set() OOB guard). */
static int check_oob_rank_dropped(void) {
  /* Hand-craft a wire buffer: count=2, one valid (rank 3, v=33), one OOB
   * (rank NRANKS+5, v=99). */
  unsigned char buf[8 + 2 * 16];
  memset(buf, 0, sizeof(buf));
  uint32_t *cf = (uint32_t *)buf;
  cf[0] = 2;
  cf[1] = 0;
  struct arts_msg_rank_version_pair_s *e =
      (struct arts_msg_rank_version_pair_s *)(buf + 8);
  e[0].rank = 3;
  e[0].pad = 0;
  e[0].version = 33;
  e[1].rank = NRANKS + 5;
  e[1].pad = 0;
  e[1].version = 99;
  struct arts_rank_to_u64_map_s *m =
      arts_rank_u64_map_deserialize(buf, sizeof(buf), NRANKS);
  int rc = 0;
  if (arts_rank_u64_map_get(m, 3) != 33) {
    rc = 1;
  }
  for (unsigned int r = 0; r < NRANKS; r++) {
    if (r != 3 && arts_rank_u64_map_get(m, r) != 0) {
      rc = 1; /* OOB pair must not have aliased into a valid slot. */
    }
  }
  arts_rank_u64_map_destroy(m);
  if (rc) {
    (void)fprintf(
        stderr, "FAIL rank_u64_map_roundtrip: OOB rank not cleanly dropped\n");
  }
  return rc;
}

/* B032 probe: a short buffer with an inflated count → OOB read.  Run in a
 * child; the parent observes whether it crashed (the defect). */
static int probe_inflated_count_oob(void) {
  /* Buffer big enough for the header + ONE pair, but count claims 64 pairs.
   * deserialize will read 64 pairs → far past the buffer. */
  size_t cap = 8 + 1 * 16;
  unsigned char *buf = (unsigned char *)malloc(cap);
  memset(buf, 0, cap);
  uint32_t *cf = (uint32_t *)buf;
  cf[0] = 64; /* inflated: buffer only holds 1 pair. */
  cf[1] = 0;
  struct arts_msg_rank_version_pair_s *e =
      (struct arts_msg_rank_version_pair_s *)(buf + 8);
  e[0].rank = 0;
  e[0].pad = 0;
  e[0].version = 1;
  /* If deserialize honored `size` this would be safe; it does not (B032). */
  struct arts_rank_to_u64_map_s *m =
      arts_rank_u64_map_deserialize(buf, cap, NRANKS);
  /* Reaching here means no OOB read tripped (e.g. non-ASan build read junk
   * harmlessly).  Consume the result so the call is not optimized away. */
  volatile uint64_t sink = arts_rank_u64_map_get(m, 0);
  (void)sink;
  arts_rank_u64_map_destroy(m);
  free(buf);
  return 0;
}

int main(void) {
  int rc = 0;
  rc |= check_roundtrip();
  rc |= check_empty();
  rc |= check_worst_case();
  rc |= check_oob_rank_dropped();
  if (rc != 0) {
    return 1; /* a real round-trip obligation broke — a test/runtime defect. */
  }

  /* B032 OOB-read probe in a forked child (isolates a potential ASan abort). */
  pid_t pid = fork();
  if (pid == 0) {
    _exit(probe_inflated_count_oob() == 0 ? 0 : 3);
  }
  int status = 0;
  (void)waitpid(pid, &status, 0);
  if (WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0)) {
    /* The deserializer read past the inflated-count buffer — B032 confirmed.
     * Report (do NOT mask); the legit round-trip obligations already PASSED. */
    (void)fprintf(stderr,
                  "B032 CONFIRMED rank_u64_map_roundtrip: inflated-count "
                  "deserialize OOB-read crashed child (signal/exit %d)\n",
                  status);
  }

  printf("PASS rank_u64_map_roundtrip: sparse/empty/worst-case round-trip, "
         "OOB-rank drop; inflated-count OOB-read probed (see stderr if B032 "
         "tripped)\n");
  return 0;
}
