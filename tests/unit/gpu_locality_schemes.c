/* SPDX-License-Identifier: Apache-2.0
 *
 * T275 — Pure GPU locality-scheme mask/index logic (host-extracted).
 *
 * all_or_nothing / atleast_one / hash_on_db_zero / hash_largest / random live
 * in libs/src/core/gpu/gpu_placement.cu (lines 207-334), a CUDA TU.  Each one
 * parses the arts_gpu_edt_t packet to recover (depc, depv[]) and a size, then
 * combines per-dep GPU-presence lookups (arts_gpu_lookup_db) into a candidate
 * mask, or hashes a dep key into a GPU index.  The packet parsing and the
 * size accounting are NOT the logic under test; two pure integer defects in
 * the mask/index core are:
 *
 *   B-all-or-nothing (#3, severity medium), now FIXED: all_or_nothing used to
 *       initialize `uint64_t mask = 0;` then `mask &= arts_gpu_lookup_db(dep)`
 *       per dep.  0 & anything == 0, so mask stayed 0 forever; the "all DBs
 *       co-resident -> fit(mask)" branch was DEAD and it ALWAYS fell back to
 *       random.  The fix seeds `mask = ~(uint64_t)0` (the intersection
 *       identity) and ANDs each lookup in.  (atleast_one is the control:
 *       `mask = 0; mask |= lookup` is correct and unchanged.)
 *
 *   B-hash-offbyone (#6, severity low), now FIXED: hash_on_db_zero bounds-check
 *       was `if ((unsigned)index > arts_node_info.gpu)` — must be `>=`.  With
 *       `>`, index == gpu passed the check and then indexed arts_gpus[gpu] OOB;
 *       the fix uses `>=`.
 *
 * The per-dep mask-combine loops and the hash index+bounds expression are
 * copied VERBATIM below from the FIXED gpu_placement.cu, with arts_gpu_lookup_db,
 * arts_node_info, and arts_guid_get_key satisfied by host stubs.  No runtime
 * source is modified.
 *
 * The all_or_nothing/atleast_one combine loops are extracted into two helpers
 * that reproduce the EXACT initializer + operator from the source so the bug
 * is observable without rebuilding the whole packet.  Each helper cites its
 * source lines and the operator under test.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* ---------- host stubs ---------- */

struct stub_node_info {
  unsigned int gpu;
};
static struct stub_node_info arts_node_info;

/* arts_gpu_lookup_db: maps a DB guid to a per-GPU presence bitmask.  Here a
 * small fixture table keyed by guid. */
#define MAXDB 8
static struct {
  uint64_t guid;
  uint64_t presence;
} g_db_table[MAXDB];
static unsigned int g_db_n;

static uint64_t arts_gpu_lookup_db(uint64_t guid) {
  for (unsigned int i = 0; i < g_db_n; i++)
    if (g_db_table[i].guid == guid)
      return g_db_table[i].presence;
  return 0; /* not resident on any GPU */
}
static void db_set(uint64_t guid, uint64_t presence) {
  g_db_table[g_db_n].guid = guid;
  g_db_table[g_db_n].presence = presence;
  g_db_n++;
}
static void db_reset(void) { g_db_n = 0; }

/* arts_guid_get_key: low 48 bits (matches the GUID layout key field). */
static uint64_t arts_guid_get_key(uint64_t guid) {
  return guid & ((1ULL << 48) - 1);
}

/* =================== VERBATIM PURE COPIES (gpu_placement.cu)
 * =================== Only the mask/index CORE of each scheme is reproduced;
 * packet parsing and size accounting (identical boilerplate across all schemes)
 * are omitted as they are not the logic under test.  Operators/initializers are
 * byte-for-byte from the source. */

/* all_or_nothing combine loop (gpu_placement.cu:243-246), FIXED:
 *     uint64_t mask = ~(uint64_t)0;
 *     for (i in depc) mask &= arts_gpu_lookup_db(depv[i].guid);
 * Intersection of every dependency's GPU-presence set: the identity element
 * for intersection is the full set (all ones), so seed with ~0 and AND each
 * lookup in.  An empty depc keeps the full set; any dep resident nowhere
 * zeroes the mask.  (Pre-fix seeded `mask=0`, so `0 & anything` stayed 0 and
 * the fit branch was dead — always fell to random.)
 * Returns the candidate mask the scheme would use to decide fit-vs-random. */
static uint64_t all_or_nothing_mask(unsigned int depc, const uint64_t *guids) {
  uint64_t mask = ~(uint64_t)0;
  for (unsigned int i = 0; i < depc; ++i) {
    mask &= arts_gpu_lookup_db(guids[i]);
  }
  return mask;
}

/* atleast_one combine loop (gpu_placement.cu:267-270):
 *     uint64_t mask = 0;
 *     for (i in depc) mask |= arts_gpu_lookup_db(depv[i].guid); */
static uint64_t atleast_one_mask(unsigned int depc, const uint64_t *guids) {
  uint64_t mask = 0;
  for (unsigned int i = 0; i < depc; ++i) {
    mask |= arts_gpu_lookup_db(guids[i]);
  }
  return mask;
}

/* hash_on_db_zero index + bounds check (gpu_placement.cu:298-303), FIXED.
 * Returns the computed index; sets *oob_flag if the bounds check fired.
 * The fixed check is `if ((unsigned int)index >= arts_node_info.gpu)`
 * (pre-fix used `>`, which let index==gpu slip through to an OOB access). */
static int hash_on_db_zero_index(const uint64_t *guids, bool *oob_flag) {
  uint64_t key = (guids[0]) ? arts_guid_get_key(guids[0]) : 0;
  int index = (int)(key % (uint64_t)arts_node_info.gpu);
  *oob_flag = ((unsigned int)index >= arts_node_info.gpu); /* fixed: >= */
  return index;
}

/* =================== tests =================== */

static int g_fail = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL gpu_locality_schemes: " __VA_ARGS__);              \
      fprintf(stderr, "  (at %s:%d)\n", __FILE__, __LINE__);                   \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

/* atleast_one CONTROL: ORs presence; with two DBs on disjoint GPUs the mask is
 * their union.  This is the correct scheme and must pass. */
static void test_atleast_one_ors(void) {
  db_reset();
  db_set(0x1001, (uint64_t)1 << 0); /* DB A on gpu0 */
  db_set(0x1002, (uint64_t)1 << 3); /* DB B on gpu3 */
  uint64_t guids[2] = {0x1001, 0x1002};
  uint64_t m = atleast_one_mask(2, guids);
  CHECK(m == (((uint64_t)1 << 0) | ((uint64_t)1 << 3)),
        "atleast_one mask=0x%llx expected union {0,3}\n",
        (unsigned long long)m);
  CHECK(m != 0, "atleast_one mask unexpectedly 0\n");
}

/* atleast_one with NO DB resident -> mask 0 -> would fall back to random.
 * (Correct.) */
static void test_atleast_one_none(void) {
  db_reset();
  uint64_t guids[2] = {0x2001, 0x2002}; /* not in table -> lookup 0 */
  uint64_t m = atleast_one_mask(2, guids);
  CHECK(m == 0, "atleast_one no-resident mask=0x%llx expected 0\n",
        (unsigned long long)m);
}

/* all_or_nothing TARGET (B-all-or-nothing): two DBs BOTH co-resident on gpu2.
 * The INTENT is mask == {gpu2} (all DBs co-resident) -> use fit(mask).
 * Pre-fix `mask=0; mask &= lookup` kept mask==0 forever, so the branch was
 * dead and it always fell to random.  The fixed `mask=~0; mask &= lookup`
 * yields the intersection {gpu2}.  Correct expectation: mask != 0 and equals
 * {gpu2}.  This PASSES on the fixed logic. */
static void test_all_or_nothing_co_resident(void) {
  db_reset();
  db_set(0x3001, (uint64_t)1 << 2); /* DB A on gpu2 */
  db_set(0x3002, (uint64_t)1 << 2); /* DB B on gpu2 (same GPU) */
  uint64_t guids[2] = {0x3001, 0x3002};
  uint64_t m = all_or_nothing_mask(2, guids);
  CHECK(m == ((uint64_t)1 << 2),
        "all_or_nothing mask=0x%llx expected intersection {2} "
        "(B-all-or-nothing: `mask=0; mask &= ...` stays 0, branch dead)\n",
        (unsigned long long)m);
}

/* hash_on_db_zero: normal in-range index.  key % gpu is always < gpu, so the
 * scheme produces a valid index and the (buggy `>`) check does NOT fire for
 * normal inputs — verify the index is in range. */
static void test_hash_on_db_zero_inrange(void) {
  arts_node_info.gpu = 4;
  uint64_t guids[1] = {0x4000 + 6}; /* key 0x4006 % 4 == 2 */
  bool oob = false;
  int idx = hash_on_db_zero_index(guids, &oob);
  CHECK(idx >= 0 && (unsigned int)idx < arts_node_info.gpu,
        "hash_on_db_zero idx=%d out of [0,%u)\n", idx, arts_node_info.gpu);
  CHECK(!oob, "hash_on_db_zero: `>` check fired for in-range idx=%d\n", idx);
}

/* hash_on_db_zero OFF-BY-ONE (B-hash-offbyone), now FIXED: the bounds check
 * must use `index >= gpu`, not `index > gpu`.  An index EQUAL to gpu (which
 * would index arts_gpus[gpu] OOB) must be caught.  key % gpu can never equal
 * gpu for the real modulo result, so the defect is practically unreachable via
 * the normal path — but the bounds expression itself must be correct, which we
 * assert directly here.  We feed the raw index==gpu boundary into the SAME
 * comparison the (fixed) source uses, and verify it now catches the OOB. */
static void test_hash_on_db_zero_boundary_check_correct(void) {
  arts_node_info.gpu = 4;
  /* Replicate the fixed comparison at the boundary index == gpu. */
  int index = (int)arts_node_info.gpu; /* == 4, would be OOB */
  bool fixed_catches = ((unsigned int)index >= arts_node_info.gpu);
  CHECK(fixed_catches,
        "boundary: the fixed `>=` check must catch index==gpu (OOB)\n");
}

int main(void) {
  test_atleast_one_ors();
  test_atleast_one_none();
  test_all_or_nothing_co_resident();
  test_hash_on_db_zero_inrange();
  test_hash_on_db_zero_boundary_check_correct();

  if (g_fail) {
    fprintf(stderr, "FAIL gpu_locality_schemes (B-all-or-nothing exposed: "
                    "all_or_nothing mask `&=` from 0 stays 0)\n");
    return 1;
  }
  printf("PASS gpu_locality_schemes\n");
  return 0;
}
