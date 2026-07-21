/*
 * rwmix.c -- racy single-data-block reader/writer contention probe.
 *
 * One data block; N task EDTs all depend on it, W of them in RW mode
 * (writers) and N-W in RO mode (readers).  Roles are spread uniformly
 * through the creation order (Bresenham) so writer arrivals interleave
 * the readers evenly -- the arrival-order control matters for
 * immediate-serve (invalidation) runtimes, whose cost depends on how
 * reads and writes are ordered at the directory.  All tasks are gated
 * on one trigger event and become ready at once (saturation); each
 * spins for E microseconds to model compute.  The elapsed time from
 * trigger to global completion (finish event) is the measurement.
 *
 * Everything except the tasks is pinned to PD 0 so both timestamps are
 * taken on one clock; task i is affinitized to PD (i % PDcount).  The
 * data block's home is pinned to PD 0.  The same source builds against
 * every OCR-API backend in this repo (xsocr, ocr-vx, ARTS shim).
 *
 * args: N W E_us DB_bytes   (defaults 2048 0 50 1024)
 * output: RWMIX PDS=<p> N=<n> W=<w> E_US=<e> BYTES=<b> ELAPSED_US=<t>
 */

#include "ocr.h"
#include "extensions/ocr-affinity.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static inline u64 guid_u64(ocrGuid_t g) { return (u64)g.guid; }
static inline ocrGuid_t u64_guid(u64 v) {
  ocrGuid_t g;
  g.guid = (intptr_t)v;
  return g;
}

static inline u64 now_us(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (u64)tv.tv_sec * 1000000ull + (u64)tv.tv_usec;
}

static void spin_us(u64 us) {
  if (us == 0) return;
  u64 t0 = now_us();
  while (now_us() - t0 < us)
    ;
}

/* paramv = {role(1=writer), e_us, bytes, idx, census}; depv[0]=data block, depv[1]=trigger */
ocrGuid_t task_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 role = paramv[0];
  u64 e_us = paramv[1];
  u64 bytes = paramv[2];
  u64 idx = paramv[3];
  if (paramv[4]) {
    ocrGuid_t cur;
    ocrAffinityGetCurrent(&cur);
    u64 me = 0, cnt = 0;
    ocrAffinityCount(AFFINITY_PD, &cnt);
    for (u64 p = 0; p < cnt; p++) {
      ocrGuid_t a;
      ocrAffinityGetAt(AFFINITY_PD, p, &a);
      if (ocrGuidIsEq(a, cur)) me = p;
    }
    PRINTF("RWMIX_TASK idx=%lu role=%lu pd=%lu\n", (unsigned long)idx,
           (unsigned long)role, (unsigned long)me);
  }

  if (role) {
    u64 *p = (u64 *)depv[0].ptr;
    u64 words = bytes / sizeof(u64);
    p[0] = idx;
    if (words > 1) p[words - 1] = idx;
  } else {
    const volatile u64 *p = (const volatile u64 *)depv[0].ptr;
    u64 words = bytes / sizeof(u64);
    u64 sum = 0;
    for (u64 i = 0; i < words; i++) sum += p[i];
    if (sum == 0xDEADBEEFDEADBEEFull) PRINTF("");
  }
  spin_us(e_us);
  return NULL_GUID;
}

/* paramv = {db, trigger, N, W, e_us, bytes}; spawns the N tasks, then
 * stamps t0 into the timing block, releases it, and fires the trigger.
 * Runs as a FINISH EDT so its output event is global task completion.
 * depv[0] = timing block (RW). */
ocrGuid_t spawner_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  ocrGuid_t db = u64_guid(paramv[0]);
  ocrGuid_t trigger = u64_guid(paramv[1]);
  u64 n = paramv[2];
  u64 w = paramv[3];
  u64 e_us = paramv[4];
  u64 bytes = paramv[5];

  u64 pd_count = 0;
  ocrAffinityCount(AFFINITY_PD, &pd_count);

  ocrGuid_t task_tpl;
  ocrEdtTemplateCreate(&task_tpl, task_edt, 5, 2);

  /* Roles spread by Bresenham over ARRIVAL order; placement round-robins
   * PER ROLE CLASS.  Placing by (i % pd_count) instead correlates role
   * parity with node parity (a 50:50 mix on an even node count puts every
   * writer on the odd nodes), which co-locates all writers and turns the
   * node-granular runtimes' batching into an artifact. */
  u64 acc = 0; /* Bresenham spread of W writers over N slots */
  u64 r_idx = 0, w_idx = 0;
  for (u64 i = 0; i < n; i++) {
    u64 role = 0;
    acc += w;
    if (acc >= n) {
      acc -= n;
      role = 1;
    }
    u64 pd = (role ? w_idx++ : r_idx++) % pd_count;
    ocrHint_t h;
    ocrHintInit(&h, OCR_HINT_EDT_T);
    ocrGuid_t aff;
    ocrAffinityGetAt(AFFINITY_PD, pd, &aff);
    ocrSetHintValue(&h, OCR_HINT_EDT_AFFINITY, ocrAffinityToHintValue(aff));

    u64 params[5] = {role, e_us, bytes, i, paramv[6]};
    ocrGuid_t task;
    ocrEdtCreate(&task, task_tpl, 5, params, 2, NULL, EDT_PROP_NONE, &h, NULL);
    ocrAddDependence(db, task, 0, role ? DB_MODE_RW : DB_MODE_RO);
    ocrAddDependence(trigger, task, 1, DB_MODE_NULL);
  }

  u64 *tp = (u64 *)depv[0].ptr;
  tp[0] = now_us();
  ocrDbRelease(depv[0].guid);
  ocrEventSatisfy(trigger, NULL_GUID);
  return NULL_GUID;
}

/* paramv = {N, W, e_us, bytes}; depv[0]=finish output, depv[1]=timing (RO) */
ocrGuid_t sink_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 t1 = now_us();
  const u64 *tp = (const u64 *)depv[1].ptr;
  u64 pd_count = 0;
  ocrAffinityCount(AFFINITY_PD, &pd_count);
  PRINTF("RWMIX PDS=%lu N=%lu W=%lu E_US=%lu BYTES=%lu ELAPSED_US=%lu\n",
         (unsigned long)pd_count, (unsigned long)paramv[0],
         (unsigned long)paramv[1], (unsigned long)paramv[2],
         (unsigned long)paramv[3], (unsigned long)(t1 - tp[0]));
  ocrShutdown();
  return NULL_GUID;
}

ocrGuid_t mainEdt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 n = 2048, w = 0, e_us = 50, bytes = 1024, census = 0;
  u64 argc = getArgc(depv[0].ptr);
  if (argc > 1) n = (u64)atol(getArgv(depv[0].ptr, 1));
  if (argc > 2) w = (u64)atol(getArgv(depv[0].ptr, 2));
  if (argc > 3) e_us = (u64)atol(getArgv(depv[0].ptr, 3));
  if (argc > 4) bytes = (u64)atol(getArgv(depv[0].ptr, 4));
  if (argc > 5) census = (u64)atol(getArgv(depv[0].ptr, 5));
  if (bytes < sizeof(u64)) bytes = sizeof(u64);
  if (w > n) w = n;

  ocrGuid_t pd0_aff;
  ocrAffinityGetAt(AFFINITY_PD, 0, &pd0_aff);
  ocrHint_t pd0_edt_hint;
  ocrHintInit(&pd0_edt_hint, OCR_HINT_EDT_T);
  ocrSetHintValue(&pd0_edt_hint, OCR_HINT_EDT_AFFINITY,
                  ocrAffinityToHintValue(pd0_aff));
  ocrHint_t pd0_db_hint;
  ocrHintInit(&pd0_db_hint, OCR_HINT_DB_T);
  ocrSetHintValue(&pd0_db_hint, OCR_HINT_DB_AFFINITY,
                  ocrAffinityToHintValue(pd0_aff));

  /* the contended data block, homed at PD 0, initialized then released */
  ocrGuid_t db;
  u64 *dbp;
  ocrDbCreate(&db, (void **)&dbp, bytes, DB_PROP_NONE, &pd0_db_hint, NO_ALLOC);
  for (u64 i = 0; i < bytes / sizeof(u64); i++) dbp[i] = i;
  ocrDbRelease(db);

  /* timing block */
  ocrGuid_t tdb;
  u64 *tdbp;
  ocrDbCreate(&tdb, (void **)&tdbp, sizeof(u64), DB_PROP_NONE, &pd0_db_hint,
              NO_ALLOC);
  tdbp[0] = 0;
  ocrDbRelease(tdb);

  ocrGuid_t trigger;
  ocrEventCreate(&trigger, OCR_EVENT_ONCE_T, EVT_PROP_NONE);

  /* spawner: FINISH EDT pinned to PD 0; its output event = all tasks done */
  ocrGuid_t sp_tpl, spawner, all_done;
  ocrEdtTemplateCreate(&sp_tpl, spawner_edt, 7, 1);
  u64 sp_params[7] = {guid_u64(db), guid_u64(trigger), n, w, e_us, bytes, census};
  ocrEdtCreate(&spawner, sp_tpl, 7, sp_params, 1, NULL, EDT_PROP_FINISH,
               &pd0_edt_hint, &all_done);

  /* sink: pinned to PD 0 so t0/t1 come from one clock */
  ocrGuid_t sink_tpl, sink;
  ocrEdtTemplateCreate(&sink_tpl, sink_edt, 4, 2);
  u64 sink_params[4] = {n, w, e_us, bytes};
  ocrEdtCreate(&sink, sink_tpl, 4, sink_params, 2, NULL, EDT_PROP_NONE,
               &pd0_edt_hint, NULL);
  ocrAddDependence(all_done, sink, 0, DB_MODE_NULL);
  ocrAddDependence(tdb, sink, 1, DB_MODE_RO);

  /* arm the spawner last */
  ocrAddDependence(tdb, spawner, 0, DB_MODE_RW);
  return NULL_GUID;
}
