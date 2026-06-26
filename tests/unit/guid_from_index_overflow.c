/* SPDX-License-Identifier: Apache-2.0
 *
 * guid_from_index_overflow — pins the unguarded non-distributed
 * arts_guid_from_index arithmetic (suspected bug B047 / census 18-gas B10).
 *
 * For a non-distributed (rank != ARTS_DISTRIBUTED_RANK) range GUID,
 * arts_guid_from_index(range_guid, idx) computes `range_guid + idx` with NO
 * bound check.  Because the 48-bit key occupies the LSBs, a sufficiently
 * large idx (or a range whose key starts near 2^48) carries out of the key
 * field and into the 14-bit rank field — silently producing a GUID with a
 * DIFFERENT rank than the range's home.  This test constructs exactly that
 * carry and pins the surprising behavior so a future bound-guard fix is a
 * deliberate, reviewed change (the assertion will then flip).
 *
 * No runtime: guid.c is #include'd with libc-backed shims.  Pure arithmetic,
 * single-threaded.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

unsigned int arts_global_rank_id = 0;
unsigned int arts_global_rank_count = 4;
void arts_abort(uint8_t code) { _exit(code ? code : 70); }
void *arts_malloc(size_t s) { return malloc(s); }
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void *arts_calloc_aligned(size_t n, size_t s, size_t a) {
  void *p = NULL;
  if (posix_memalign(&p, a < sizeof(void *) ? sizeof(void *) : a, n * s)) {
    return NULL;
  }
  memset(p, 0, n * s);
  return p;
}
void arts_free(void *p) { free(p); }

#include "/home/whnbaek/pnnl/ARTS/libs/src/core/gas/guid.c"

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

#define FAIL(...)                                                              \
  do {                                                                         \
    (void)fprintf(stderr, "FAIL guid_from_index_overflow: " __VA_ARGS__);      \
    return 1;                                                                  \
  } while (0)

int main(void) {
  /* ---- Sanity: small-idx, non-overflowing range behaves correctly. ---- */
  arts_guid_t base = ARTS_GUID_MAKE(ARTS_GUID_DB, 2, 1000);
  arts_guid_t g5 = arts_guid_from_index(base, 5);
  if (arts_guid_get_rank(g5) != 2) {
    FAIL("small idx changed rank: %u\n", arts_guid_get_rank(g5));
  }
  if (arts_guid_get_key(g5) != 1005) {
    FAIL("small idx key wrong: %lu\n", (unsigned long)arts_guid_get_key(g5));
  }
  if (arts_guid_index_from(base, g5) != 5) {
    FAIL("small idx round-trip broke\n");
  }

  /* ---- Overflow: range key one below the 48-bit max, idx = 1 carries. ---- *
   * key = ARTS_GUID_KEY_MASK (all 48 key bits set) ; +1 overflows the key
   * field into the rank field, bumping rank from its home value. */
  unsigned int home = 2;
  arts_guid_t edge = ARTS_GUID_MAKE(ARTS_GUID_DB, home, ARTS_GUID_KEY_MASK);
  arts_guid_t carried = arts_guid_from_index(edge, 1);

  /* The DEFECT: with no bound guard, the +1 carry lands in the rank field.
   * Pin it: the produced GUID's rank is NOT the range home (home+1 here), and
   * the key wrapped to 0.  If a future fix adds a bound check / saturating
   * arithmetic, this assertion intentionally fails and must be revisited. */
  if (arts_guid_get_key(carried) != 0) {
    FAIL("expected key field to wrap to 0 on carry, got %lu\n",
         (unsigned long)arts_guid_get_key(carried));
  }
  if (arts_guid_get_rank(carried) == home) {
    FAIL("expected rank to be corrupted by carry (no bound guard); "
         "rank stayed %u — has from_index gained a guard?\n",
         home);
  }
  if (arts_guid_get_rank(carried) != home + 1) {
    FAIL("carry produced unexpected rank %u (want %u)\n",
         arts_guid_get_rank(carried), home + 1);
  }

  printf("PASS guid_from_index_overflow: pinned unguarded +idx carry "
         "(key 2^48-1 + 1 -> rank %u, key 0) [exposes B047]\n",
         arts_guid_get_rank(carried));
  return 0;
}
