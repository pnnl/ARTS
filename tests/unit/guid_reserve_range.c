/* SPDX-License-Identifier: Apache-2.0
 *
 * guid_reserve_range — pure-unit coverage of arts_guid_reserve_range,
 * arts_guid_reserve_range_hash and the labeled-vs-auto collision-avoidance
 * invariant (census 18-gas obligations 11 & 12; suspected bug B046).
 *
 * Properties:
 *
 *   1. ROUND_ROBIN reserve_range clears EVERY per-rank counter of this thread
 *      to base+stride, so a subsequent AUTO arts_guid_reserve (which advances
 *      a single per-(rank,kind) counter) cannot mint a key inside the labeled
 *      span — the documented cure for a real collision bug.  We reserve a
 *      distributed range, expand it via from_index, then auto-reserve on every
 *      rank and assert NO auto GUID equals any labeled GUID.
 *
 *   2. reserve_range_hash returns a hash-ALIGNED start (key % hash_size == 0)
 *      while still leaving `size` usable contiguous GUIDs.
 *
 *   3. reserve_range_hash with hash_size == 0 is SAFE, contrary to suspected
 *      bug B046.  B046 predicted a modulo-by-zero on hash_size == 0, but the
 *      alignment loop is `for (i = 0; i < hash_size; i++)` — its body (the only
 *      site of `% hash_size`) never runs when hash_size == 0, so the modulo is
 *      never reached.  The function simply returns the unaligned start.  This
 *      test pins that the call does NOT trap and yields a valid start GUID,
 *      documenting B046 as a FALSE POSITIVE (loop-guarded).
 *
 * No runtime: guid.c is #include'd with libc-backed shims.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
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

#include "../../libs/src/core/gas/guid.c"

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

#define FAIL(...)                                                              \
  do {                                                                         \
    (void)fprintf(stderr, "FAIL guid_reserve_range: " __VA_ARGS__);            \
    return 1;                                                                  \
  } while (0)

static void generator_reset(uint64_t key_budget) {
  arts_thread_info.thread_id = 0;
  arts_node_info.total_thread_count = 1;
  arts_node_info.gpu = 0;
  if (arts_node_info.keys) {
    free(arts_node_info.keys[0]);
    free(arts_node_info.keys);
  }
  free(arts_node_info.global_guid_thread_id);
  arts_node_info.keys = (uint64_t **)calloc(1, sizeof(uint64_t *));
  arts_node_info.keys[0] = (uint64_t *)calloc(
      (size_t)ARTS_GUID_LAST * arts_global_rank_count, sizeof(uint64_t));
  for (unsigned i = 0; i < ARTS_GUID_LAST * arts_global_rank_count; i++) {
    arts_node_info.keys[0][i] = 1;
  }
  arts_node_info.global_guid_thread_id =
      (uint64_t *)calloc(1, sizeof(uint64_t));
  arts_node_info.global_guid_thread_id[0] = 0;
  num_tables = 1;
  min_global_guid_thread = 0;
  max_global_guid_thread = 1;
  keys_per_thread = key_budget;
  global_guid_on = 0;
}

static void generator_free(void) {
  if (arts_node_info.keys) {
    free(arts_node_info.keys[0]);
    free(arts_node_info.keys);
    arts_node_info.keys = NULL;
  }
  free(arts_node_info.global_guid_thread_id);
  arts_node_info.global_guid_thread_id = NULL;
}

int main(void) {
  /* ---- 1. labeled-vs-auto collision avoidance. ---- */
  generator_reset(1u << 20);
  const unsigned RSIZE = 40; /* spans several keys per rank (stride=10) */
  arts_guid_t dist =
      arts_guid_reserve_range(ARTS_GUID_DB, RSIZE, ARTS_HINT_ROUND_ROBIN);
  if (dist == NULL_GUID) {
    FAIL("distributed reserve_range returned NULL_GUID\n");
  }
  if (arts_guid_get_rank(dist) != ARTS_DISTRIBUTED_RANK) {
    FAIL("distributed range rank tag wrong: %u\n", arts_guid_get_rank(dist));
  }

  /* materialize all labeled GUIDs */
  arts_guid_t labeled[RSIZE];
  for (unsigned i = 0; i < RSIZE; i++) {
    labeled[i] = arts_guid_from_index(dist, i);
  }

  /* now auto-reserve on EVERY rank; none may collide with a labeled GUID */
  for (unsigned r = 0; r < arts_global_rank_count; r++) {
    for (int k = 0; k < 8; k++) {
      arts_guid_t auto_g = arts_guid_create_for_rank(r, ARTS_GUID_DB);
      for (unsigned i = 0; i < RSIZE; i++) {
        if (auto_g == labeled[i]) {
          FAIL("auto GUID on rank %u collided with labeled[%u]\n", r, i);
        }
      }
    }
  }

  /* ---- 2. hash-aligned start. ---- */
  generator_reset(1u << 20);
  const unsigned HSZ = 16;
  const unsigned NEED = 8;
  arts_guid_t hstart = arts_guid_reserve_range_hash(ARTS_GUID_DB, NEED, 0, HSZ);
  if (hstart == NULL_GUID) {
    FAIL("reserve_range_hash returned NULL_GUID\n");
  }
  if (ARTS_GUID_GET_KEY(hstart) % HSZ != 0) {
    FAIL("reserve_range_hash start not aligned: key=%lu hsz=%u\n",
         (unsigned long)ARTS_GUID_GET_KEY(hstart), HSZ);
  }
  /* the `NEED` GUIDs past the aligned start must be distinct & contiguous */
  for (unsigned i = 1; i < NEED; i++) {
    if (arts_guid_get_key(hstart + i) != arts_guid_get_key(hstart) + i) {
      FAIL("hash range not contiguous at %u\n", i);
    }
  }

  /* ---- 3. hash_size == 0 is SAFE (B046 false positive — loop-guarded). ---- *
   * Run in a forked child so that, if a future refactor ever DID introduce a
   * raw `% hash_size`, the resulting SIGFPE wouldn't take the whole test down
   * — instead it would surface as a child-trap and flip this assertion. */
  pid_t pid = fork();
  if (pid == 0) {
    generator_reset(1u << 20);
    arts_guid_t s = arts_guid_reserve_range_hash(ARTS_GUID_DB, 4, 0, 0);
    /* Must return a valid (non-NULL) start without trapping. */
    _exit(s == NULL_GUID ? 2 : 0);
  } else if (pid > 0) {
    int st = 0;
    (void)waitpid(pid, &st, 0);
    if (WIFSIGNALED(st)) {
      FAIL("hash_size==0 TRAPPED (signal %d) — B046 is no longer a false "
           "positive; a raw modulo-by-zero was introduced.\n",
           WTERMSIG(st));
    }
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
      FAIL("hash_size==0 child failed (status %d): expected a valid start, "
           "no trap.\n",
           st);
    }
  } else {
    FAIL("fork failed\n");
  }

  generator_free();
  printf("PASS guid_reserve_range: labeled-vs-auto collision-free, "
         "hash-aligned start, hash_size==0 SAFE (B046 false positive)\n");
  return 0;
}
