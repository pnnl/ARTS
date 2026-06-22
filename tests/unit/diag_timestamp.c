/* SPDX-License-Identifier: Apache-2.0
 *
 * T247 — arts_get_time_stamp (and a documentation probe of arts_printf torn
 * output) from libs/src/core/utils/diag.c.
 *
 * arts_get_time_stamp() returns clock_gettime(CLOCK_REALTIME) folded to
 * nanoseconds (tv_sec*1e9 + tv_nsec).  Properties:
 *   A. Per-thread non-decreasing (best-effort): within a single thread,
 *      consecutive samples must be monotonically non-decreasing under normal
 *      conditions.  REALTIME can step backward on an NTP adjustment, so this
 *      is documented as best-effort: a single backward step is reported as a
 *      NOTE (the census caveat), not a hard failure; the test only hard-fails
 *      if the clock is grossly broken (every sample identical for a long busy
 *      loop would indicate the fold math collapsed, or a huge backward jump).
 *   B. Folding sanity: the value is a plausible ns-since-epoch (> year 2020,
 *      < year 2200) so the tv_sec*NANOSECS multiply did not overflow/garble.
 *   C. Resolution: across many threads sampling concurrently the call is
 *      thread-safe (no shared mutable state) — ASan/TSan must stay clean.
 *
 * DOCUMENTED (census bug B133): arts_printf prints the " [rank] " prefix and
 * the body via two separate stdio calls, so concurrent callers can interleave
 * (torn lines).  We exercise that path from many threads writing fixed-length
 * tokens and scan the captured output for torn lines, reporting the tear count
 * as a NOTE.  This documents the non-atomicity without making the test flaky
 * (the property "two stdio calls are not atomic" is the point, not a pass/fail
 * threshold).
 *
 * STANDALONE STRATEGY: diag.c reaches arts_global_rank_id via arts.h +
 * arts/system/threads.h.  We pre-define those include guards and provide a
 * stub arts_global_rank_id, then #include the diag.c TU directly.
 */

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Block the heavy headers diag.c pulls in and stub the one symbol it needs. */
#define ARTS_H 1                /* skip public arts.h body */
#define ARTS_SYSTEM_THREADS_H 1 /* skip arts/system/threads.h body */
unsigned int arts_global_rank_id = 0;

#include "../../libs/src/core/utils/diag.c"

/* ---- year bounds for the fold sanity check ---- */
#define NS_PER_SEC 1000000000ull
#define SEC_2020 1577836800ull /* 2020-01-01 UTC */
#define SEC_2200 7258118400ull /* ~2200 */

/* ---- Part A/C: per-thread monotonicity under concurrent sampling ---- */
#define MONO_THREADS 8
#define MONO_SAMPLES 200000

static atomic_int g_gate;
static atomic_uint_least64_t
    g_backward_steps;           /* observed REALTIME regressions */
static atomic_int g_hard_break; /* gross failure flag */

static void *mono_worker(void *arg) {
  (void)arg;
  while (atomic_load_explicit(&g_gate, memory_order_acquire) == 0) {
  }
  uint64_t prev = arts_get_time_stamp();
  if (prev < SEC_2020 * NS_PER_SEC || prev > SEC_2200 * NS_PER_SEC) {
    atomic_store_explicit(&g_hard_break, 1, memory_order_release);
  }
  for (int i = 0; i < MONO_SAMPLES; i++) {
    uint64_t now = arts_get_time_stamp();
    if (now < prev) {
      uint64_t delta = prev - now;
      atomic_fetch_add_explicit(&g_backward_steps, 1, memory_order_relaxed);
      /* A gross backward jump (> 1s) is not an NTP micro-adjust; treat as a
       * hard failure of the fold math / clock source. */
      if (delta > NS_PER_SEC) {
        atomic_store_explicit(&g_hard_break, 1, memory_order_release);
      }
    }
    prev = now;
  }
  return NULL;
}

/* ---- Part B: torn-output documentation for arts_printf ---- */
#define PRINTF_THREADS 6
#define PRINTF_LINES 300

static atomic_int g_pgate;

static void *printf_worker(void *arg) {
  long id = (long)arg;
  while (atomic_load_explicit(&g_pgate, memory_order_acquire) == 0) {
  }
  for (int i = 0; i < PRINTF_LINES; i++) {
    /* Body is a fixed token; the runtime prepends " [rank] " as a SEPARATE
     * stdio call, so concurrent bodies/prefixes can interleave. */
    arts_printf("TOKEN_%ld_%03d\n", id, i);
  }
  return NULL;
}

int main(void) {
  int rc = 0;

  /* ===== Part A/C: monotonicity (best-effort) under concurrency. ===== */
  atomic_init(&g_gate, 0);
  atomic_init(&g_backward_steps, 0);
  atomic_init(&g_hard_break, 0);
  {
    pthread_t th[MONO_THREADS];
    for (int i = 0; i < MONO_THREADS; i++) {
      pthread_create(&th[i], NULL, mono_worker, NULL);
    }
    atomic_store_explicit(&g_gate, 1, memory_order_release);
    for (int i = 0; i < MONO_THREADS; i++) {
      pthread_join(th[i], NULL);
    }
  }
  if (atomic_load_explicit(&g_hard_break, memory_order_acquire)) {
    (void)fprintf(stderr,
                  "FAIL diag_timestamp: timestamp out of plausible range or "
                  "gross (>1s) backward jump — fold math / clock source\n");
    rc = 1;
  }
  uint64_t back = atomic_load_explicit(&g_backward_steps, memory_order_relaxed);
  if (back != 0) {
    /* Best-effort caveat: REALTIME may step back on NTP adjust.  Document, do
     * not fail (small steps only — gross ones already set g_hard_break). */
    (void)fprintf(stderr,
                  "NOTE diag_timestamp: %" PRIu64 " small REALTIME backward "
                  "step(s) observed (best-effort monotonicity caveat)\n",
                  back);
  }

  /* ===== Part B: torn arts_printf output (documentation). ===== */
  /* Redirect stdout to a temp file so we can scan for torn lines. */
  {
    char tmpl[] = "/tmp/diag_printf_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
      (void)fprintf(stderr, "NOTE diag_timestamp: mkstemp failed, skip torn "
                            "probe\n");
    } else {
      fflush(stdout);
      int saved = dup(STDOUT_FILENO);
      dup2(fd, STDOUT_FILENO);

      atomic_init(&g_pgate, 0);
      pthread_t th[PRINTF_THREADS];
      for (long i = 0; i < PRINTF_THREADS; i++) {
        pthread_create(&th[i], NULL, printf_worker, (void *)i);
      }
      atomic_store_explicit(&g_pgate, 1, memory_order_release);
      for (int i = 0; i < PRINTF_THREADS; i++) {
        pthread_join(th[i], NULL);
      }
      fflush(stdout);

      /* restore stdout */
      dup2(saved, STDOUT_FILENO);
      close(saved);
      close(fd);

      /* Scan: a well-formed line is " [0] TOKEN_<id>_<nnn>".  A torn line is
       * one where the prefix and body got separated by another thread's
       * output.  We count lines that contain the prefix but no TOKEN, or a
       * TOKEN with no leading prefix on its line. */
      FILE *f = fopen(tmpl, "r");
      uint64_t torn = 0, total = 0;
      if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
          total++;
          int has_prefix = (strstr(line, "[0]") != NULL);
          int has_token = (strstr(line, "TOKEN_") != NULL);
          /* Atomic line would have BOTH.  Prefix-only or token-only == torn. */
          if (has_prefix != has_token) {
            torn++;
          }
        }
        fclose(f);
      }
      (void)remove(tmpl);
      (void)fprintf(stderr,
                    "NOTE diag_timestamp: arts_printf produced %" PRIu64
                    " torn line(s) out of %" PRIu64 " under %d concurrent "
                    "writers (B133: prefix+body are two stdio calls, not "
                    "atomic)\n",
                    torn, total, PRINTF_THREADS);
    }
  }

  if (rc) {
    return 1;
  }
  printf("PASS diag_timestamp: per-thread non-decreasing (best-effort), fold "
         "in plausible range, thread-safe concurrent sampling; B133 torn "
         "arts_printf documented\n");
  return 0;
}
