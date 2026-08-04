/* SPDX-License-Identifier: Apache-2.0
 *
 * Pure-unit test for arts_csr_free's NULL handling (T272; targets B151 /
 * B-csr-free-null).
 *
 * Contract under test:
 *   arts_csr_free(csr) should tolerate a NULL csr without crashing.  Callers
 *   routinely receive NULL from arts_csr_init / arts_csr_from_guid /
 *   arts_csr_from_partition for a REMOTE partition (the block is not
 *   materialized locally), and a "free everything" teardown loop will then call
 *   arts_csr_free(NULL).
 *
 * Actual implementation (libs/src/graph/csr.c):
 *     void arts_csr_free(arts_csr_graph_t *csr) {
 *       arts_db_destroy(csr->partGuid);   // <-- unconditional deref of csr
 *     }
 *   There is NO NULL guard, so arts_csr_free(NULL) dereferences a NULL pointer
 *   (reads csr->partGuid) BEFORE the call -> SIGSEGV (or an ASan/UBSan
 *   null-deref report that aborts the process).
 *
 * Verification strategy: call arts_csr_free(NULL) in a forked CHILD and observe
 * its exit.  A NULL-tolerant implementation lets the child exit 0; the current
 * implementation crashes the child (signal / sanitizer abort).  The parent
 * survives either way and reports.  Because the correct contract is "no crash",
 * this test is CORRECT-AND-FAILING while the bug exists (exposes_runtime_bug).
 *
 * We also pin the happy-path: arts_csr_free on a NON-NULL csr reaches
 * arts_db_destroy(partGuid) (verified via a destroy-recording stub) — proving
 * the test harness exercises the real arts_csr_free body, not a fake.
 *
 * The ARTS runtime is NOT started.  csr.c references the edge_vector and
 * block_distribution helpers and (in arts_csr_from_guid) the complete
 * struct arts_db_s layout, so we compile all three graph TUs via #include and
 * pull the coherence type header (ARTS_PROTOCOL_VAL selected — the graph code
 * is protocol-INDEPENDENT, the macro only fixes the arts_db_s layout for the
 * pointer arithmetic in arts_csr_from_guid, which this test does not call).
 * Every other runtime symbol csr.c references is stubbed below; only
 * arts_db_destroy is actually reached.
 */

#ifndef ARTS_PROTOCOL_VAL
#define ARTS_PROTOCOL_VAL 1
#endif

#define ARTS_SYSTEM_PRINT_H
#define ARTS_INFO(...) ((void)0)
#define ARTS_DEBUG(...) ((void)0)
#define ARTS_WARN(...) ((void)0)
#define ARTS_ERROR(...) ((void)0)

#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "arts/coherence/types.h" /* complete struct arts_db_s (VAL layout) */
#include "arts/gas/route_table.h"
#include "arts/graph.h"
#include "arts/utils/shared.h" /* arts_shared_ptr_t (opaque) */

/* --- allocator contract (mirrors libs/src/core/utils/malloc.c) -------------
 */
void *arts_malloc(size_t size) { return size ? malloc(size) : NULL; }
void *arts_calloc(size_t nmemb, size_t size) {
  return (nmemb && size) ? calloc(nmemb, size) : NULL;
}
void *arts_realloc(void *ptr, size_t size) {
  if (!ptr) {
    return arts_malloc(size);
  }
  if (!size) {
    free(ptr);
    return NULL;
  }
  return realloc(ptr, size);
}
void arts_free(void *ptr) { free(ptr); }

/* --- stubs for the runtime symbols csr.c references -------------------------
 */
static arts_guid_t g_last_destroyed;
static int g_destroy_calls;
void arts_db_destroy(arts_guid_t guid) {
  g_last_destroyed = guid;
  g_destroy_calls++;
}
arts_guid_t arts_db_create(void **ptr, uint64_t len, arts_db_types_t db_type,
                           uint16_t flags, const arts_db_hint_t *hint) {
  (void)len;
  (void)db_type;
  (void)flags;
  (void)hint;
  if (ptr) {
    *ptr = NULL;
  }
  return 0;
}
bool arts_guid_is_local(arts_guid_t guid) {
  (void)guid;
  return true;
}
unsigned int arts_get_current_rank(void) { return 0; }
unsigned int arts_get_total_ranks(void) { return 1; }
arts_guid_t arts_guid_reserve(arts_guid_kind_t kind, unsigned int rank) {
  (void)kind;
  (void)rank;
  return 0;
}
arts_shared_ptr_t arts_route_table_lookup_db(arts_guid_t guid) {
  (void)guid;
  return NULL;
}
void *arts_shared_get(arts_shared_ptr_t p) {
  (void)p;
  return NULL;
}
void arts_shared_release(arts_shared_ptr_t *p) { (void)p; }

/* Compile the three graph TUs into this test (single TU so the stubs above
 * satisfy their references). */
#include "../../libs/src/graph/block_distribution.c"
#include "../../libs/src/graph/csr.c"
#include "../../libs/src/graph/edge_vector.c"

static int g_fail = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL csr_free_null: %s (%s:%d)\n", msg, __FILE__,       \
              __LINE__);                                                       \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

/* Happy path: arts_csr_free on a non-NULL csr reaches arts_db_destroy with the
 * stored partGuid -> confirms we are calling the real body. */
static void test_nonnull_free_reaches_destroy(void) {
  arts_csr_graph_t csr;
  memset(&csr, 0, sizeof(csr));
  csr.partGuid = (arts_guid_t)0x1234567ULL;
  g_destroy_calls = 0;
  g_last_destroyed = 0;
  arts_csr_free(&csr);
  CHECK(g_destroy_calls == 1, "non-NULL free calls arts_db_destroy once");
  CHECK(g_last_destroyed == (arts_guid_t)0x1234567ULL,
        "free passes csr->partGuid to arts_db_destroy");
}

/* B151: arts_csr_free(NULL).  Run in a child; correct behavior is a clean
 * exit (no crash).  The current no-guard implementation derefs NULL -> the
 * child dies by signal / sanitizer abort. */
static void child_free_null(void) {
  arts_csr_free(NULL);
  _exit(0); /* reached only if NULL was tolerated (no deref crash) */
}

static void test_free_null_no_crash(void) {
  pid_t pid = fork();
  if (pid == 0) {
    child_free_null();
    _exit(99);
  }
  CHECK(pid > 0, "fork for B151 child");
  int status = 0;
  waitpid(pid, &status, 0);

  bool clean = WIFEXITED(status) && WEXITSTATUS(status) == 0;
  if (clean) {
    CHECK(1, "B151: arts_csr_free(NULL) tolerated (would be a fix)");
  } else {
    if (WIFSIGNALED(status)) {
      fprintf(stderr,
              "  B151 CONFIRMED: arts_csr_free(NULL) crashed by signal %d "
              "(NULL deref of csr->partGuid; no NULL guard)\n",
              WTERMSIG(status));
    } else {
      fprintf(stderr,
              "  B151 CONFIRMED: arts_csr_free(NULL) aborted (sanitizer "
              "null-deref report; no NULL guard) status=0x%x\n",
              status);
    }
    /* correct contract = no crash -> fail loudly while the bug exists. */
    CHECK(0, "B151: arts_csr_free(NULL) must not crash (missing NULL guard)");
  }
}

int main(void) {
  test_nonnull_free_reaches_destroy();
  test_free_null_no_crash();

  if (g_fail) {
    fprintf(stderr, "FAIL csr_free_null: one or more checks failed "
                    "(see CONFIRMED for the suspected runtime bug)\n");
    return 1;
  }
  printf("PASS csr_free_null: arts_csr_free(NULL) tolerated; non-NULL free "
         "reaches arts_db_destroy(partGuid)\n");
  return 0;
}
