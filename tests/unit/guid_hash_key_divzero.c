/* SPDX-License-Identifier: Apache-2.0
 *
 * guid_hash_key_divzero — pins the unguarded divisor in arts_guid_hash_key
 * (suspected bug B045 / census 18-gas B3).
 *
 * arts_guid_hash_key(guid) returns `key % arts_node_info.gpu` with NO guard
 * against gpu == 0.  On a CPU-only node (gpu == 0) this is an integer
 * division by zero — UB, in practice SIGFPE on x86.  There is no live caller
 * inside libs today, but it is a public-internal symbol and a latent landmine.
 *
 * This test:
 *   1. Confirms the modulo result is correct for gpu > 0.
 *   2. Confirms gpu == 0 traps (forked child dies on SIGFPE / non-zero) —
 *      pinning the missing guard so a future fix (return key, or saturate to
 *      1) is a deliberate, reviewed change.
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
unsigned int arts_global_rank_count = 1;
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
    (void)fprintf(stderr, "FAIL guid_hash_key_divzero: " __VA_ARGS__);         \
    return 1;                                                                  \
  } while (0)

int main(void) {
  /* ---- gpu > 0: correct modulo. ---- */
  arts_node_info.gpu = 4;
  arts_guid_t g = ARTS_GUID_MAKE(ARTS_GUID_DB, 0, 4242);
  uint64_t h = arts_guid_hash_key(g);
  if (h != (4242ULL % 4ULL)) {
    FAIL("gpu>0 modulo wrong: got %lu want %lu\n", (unsigned long)h,
         (unsigned long)(4242ULL % 4ULL));
  }
  arts_node_info.gpu = 1;
  if (arts_guid_hash_key(g) != 0) {
    FAIL("gpu==1 should map everything to 0\n");
  }

  /* ---- gpu == 0: division by zero must trap (pinned defect B045). ---- */
  pid_t pid = fork();
  if (pid == 0) {
    arts_node_info.gpu = 0;
    arts_guid_t cg = ARTS_GUID_MAKE(ARTS_GUID_DB, 0, 7);
    /* volatile sink so the divide is not optimized away. */
    volatile uint64_t sink = arts_guid_hash_key(cg);
    (void)sink;
    _exit(0); /* if we get here, no trap happened — the guard exists now */
  } else if (pid > 0) {
    int st = 0;
    (void)waitpid(pid, &st, 0);
    if (WIFEXITED(st) && WEXITSTATUS(st) == 0) {
      FAIL("gpu==0 did NOT trap — has arts_guid_hash_key gained a zero "
           "guard? Revisit this pin.\n");
    }
    if (WIFSIGNALED(st)) {
      printf("PASS guid_hash_key_divzero: gpu>0 modulo correct; gpu==0 traps "
             "(signal %d) [exposes B045]\n",
             WTERMSIG(st));
      return 0;
    }
    /* exited non-zero without a signal — still a trap of sorts; accept. */
    printf("PASS guid_hash_key_divzero: gpu>0 modulo correct; gpu==0 aborts "
           "(exit %d) [exposes B045]\n",
           WEXITSTATUS(st));
    return 0;
  } else {
    FAIL("fork failed\n");
  }
  return 1;
}
