/* SPDX-License-Identifier: Apache-2.0
 *
 * guid_from_index_overflow — pins arts_guid_from_index's overflow behavior
 * per GUID kind (originally suspected bug B047 / census 18-gas B10).
 *
 * DB kind: the key is [szhint | seq] and range arithmetic must stay inside
 * the seq field — a carry through the szhint bits would silently change the
 * HOME RANK (a labeled range's sentinel szhint is all-ones).  from_index
 * therefore bound-checks the seq addition and fails LOUDLY on overflow;
 * this test forks the overflowing call and asserts the loud failure.
 *
 * Non-DB kinds keep the flat 48-bit key and the historical plain-add
 * contract with NO bound check: a key near 2^48 plus a carrying idx still
 * silently bumps the rank field.  That surprising behavior stays pinned
 * here (as before) so any future guard on those kinds is a deliberate,
 * reviewed change.
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

/* Wait-free counter primitives for the DB seq allocator (libc-free unit
 * pattern: mirror the atomics.c definitions verbatim). */
uint64_t arts_atomic_fetch_add_u64(volatile uint64_t *d, uint64_t v) {
  return __sync_fetch_and_add(d, v);
}
uint64_t arts_atomic_cswap_u64(volatile uint64_t *d, uint64_t o, uint64_t n) {
  return __sync_val_compare_and_swap(d, o, n);
}
uint64_t arts_atomic_read_u64(const volatile uint64_t *d) {
  return __atomic_load_n(d, __ATOMIC_ACQUIRE);
}

#include "../../libs/src/core/gas/guid.c"

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

#define FAIL(...)                                                              \
  do {                                                                         \
    (void)fprintf(stderr, "FAIL guid_from_index_overflow: " __VA_ARGS__);      \
    return 1;                                                                  \
  } while (0)

int main(void) {
  /* ---- Sanity: small-idx DB range behaves correctly (in-seq add). ---- */
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

  /* ---- DB overflow: seq at its max, idx = 1 must fail LOUDLY. ---- *
   * A silent carry here would flow through the szhint field into the rank
   * bits — with a labeled range's all-ones sentinel that names a different
   * home.  Fork the call: the child must abort, never return a GUID. */
  pid_t pid = fork();
  if (pid == 0) {
    arts_guid_t edge = ARTS_GUID_MAKE(
        ARTS_GUID_DB, 2, ARTS_GUID_DB_KEY(0, ARTS_GUID_DB_SEQ_MASK));
    arts_guid_t carried = arts_guid_from_index(edge, 1);
    (void)carried;
    _exit(0); /* reaching here means the guard did NOT fire */
  } else if (pid > 0) {
    int st = 0;
    (void)waitpid(pid, &st, 0);
    if (WIFEXITED(st) && WEXITSTATUS(st) == 0) {
      FAIL("DB seq overflow returned a GUID instead of failing loudly\n");
    }
  } else {
    FAIL("fork failed\n");
  }

  /* ---- Non-DB kinds: the historical unguarded carry stays pinned. ---- */
  unsigned int home = 2;
  arts_guid_t edge = ARTS_GUID_MAKE(ARTS_GUID_EDT, home, ARTS_GUID_KEY_MASK);
  arts_guid_t carried = arts_guid_from_index(edge, 1);
  if (arts_guid_get_key(carried) != 0) {
    FAIL("expected key field to wrap to 0 on carry, got %lu\n",
         (unsigned long)arts_guid_get_key(carried));
  }
  if (arts_guid_get_rank(carried) == home) {
    FAIL("expected rank to be corrupted by carry (no bound guard); "
         "rank stayed %u — has from_index gained a guard for flat kinds?\n",
         home);
  }
  if (arts_guid_get_rank(carried) != home + 1) {
    FAIL("carry produced unexpected rank %u (want %u)\n",
         arts_guid_get_rank(carried), home + 1);
  }

  printf("PASS guid_from_index_overflow: DB overflow fails loudly; flat-kind "
         "unguarded carry stays pinned (key 2^48-1 + 1 -> rank %u, key 0)\n",
         arts_guid_get_rank(carried));
  return 0;
}
