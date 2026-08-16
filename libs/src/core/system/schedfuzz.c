/* SPDX-License-Identifier: Apache-2.0 — see schedfuzz.h for the contract. */
#include "arts/system/schedfuzz.h"

#ifndef NDEBUG

#include <stdlib.h>
#include <time.h>

#include "arts/utils/random.h"

/* -1 = unread, 0 = disabled, 1-100 = per-point probability (percent). */
static int g_fuzz_pct = -1;

void arts_sched_fuzz_point(void) {
  int pct = __atomic_load_n(&g_fuzz_pct, __ATOMIC_RELAXED);
  if (pct < 0) {
    const char *env = getenv("ARTS_SCHED_FUZZ");
    pct = 0;
    if (env != NULL) {
      long v = strtol(env, NULL, 10);
      if (v > 0) {
        pct = (int)(v > 100 ? 100 : v);
      }
    }
    __atomic_store_n(&g_fuzz_pct, pct, __ATOMIC_RELAXED);
  }
  if (pct == 0) {
    return;
  }
  if ((arts_thread_safe_random() % 100u) >= (uint64_t)pct) {
    return;
  }
  /* 0.5-5us: long enough to swallow a cache-miss-scale decision window,
   * short enough that even 100% keeps a stress test's wall-clock usable. */
  struct timespec ts = {0, (long)(500 + (arts_thread_safe_random() % 4500))};
  (void)nanosleep(&ts, NULL);
}

#endif /* !NDEBUG */
