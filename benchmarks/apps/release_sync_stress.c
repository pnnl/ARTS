/* release_sync_stress.c  (portable OCR: builds on ARTS, xsocr, ocr-vx)
 *
 * Tests whether ocrDbRelease acts as a *synchronizing* operation: once an EDT
 * releases a data block, any subsequent cross-node hand-off it issues in the
 * SAME body (ocrEventSatisfySlot / ocrAddDependence) must publish the released
 * writes to the downstream consumer; the consumer must never read a stale,
 * pre-release copy.
 *
 * Triangle (3 PDs):
 *   home A   = data-block owner/directory node (PD index HOME)
 *   producer B (PD index PROD): RW-acquire g, write `i`, ocrDbRelease(g),
 *              then -- depending on MODE -- hand off to the consumer.
 *   consumer C (PD index CONS): RO-acquire g, assert value == i.
 *
 * Each consumer caches an RO snapshot of g on node C. The next producer must
 * invalidate that snapshot on release. If release does not synchronize, the
 * invalidate can lag behind the hand-off and C re-reads its stale copy.
 *
 *   MODE=0 : producer satisfies a ONCE event MID-BODY (right after release).
 *            (the unsynchronized-release-sensitive pattern)
 *   MODE=1 : consumer is wired to the producer's OUTPUT EVENT, which fires only
 *            on EDT completion (the release-completion-gated, "safe" pattern).
 *
 * Any "STALE ..." line on stdout is a hard failure. Loop many iterations to
 * widen the race window. Iteration count = argv[1] (default 400). MODE = env.
 */

#include <extensions/ocr-affinity.h>
#include <ocr.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DB_DEFAULT_MODE
#define DB_DEFAULT_MODE DB_MODE_RW
#endif

#define PROD_NODE 0
#define CONS_NODE 1
/* home node = PD index 2 when >=3 nodes, else creator's node */

typedef struct {
  u64 i;
  ocrGuid_t db;
  ocrGuid_t readyEvt; /* producer satisfies this (consumer waits on it)   */
  ocrGuid_t nextGo;   /* consumer satisfies this to launch producer[i+1]  */
  u64 iters;
  u64 mode; /* 0 = mid-body satisfy, 1 = output-event gated      */
} chain_params_t;

#define PWORDS ((sizeof(chain_params_t) + sizeof(u64) - 1) / sizeof(u64))

ocrGuid_t producerEdt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  chain_params_t p;
  memcpy(&p, paramv, sizeof(p));
  /* depv[0] = db (RW), depv[1] = goEvt[i] */
  int *d = (int *)depv[0].ptr;
  *d = (int)p.i;              /* write the iteration's value         */
  ocrDbRelease(depv[0].guid); /* release under test                  */
  if (p.mode == 0) {
    /* MID-BODY satisfy, strictly AFTER the release. Must publish the write. */
    ocrEventSatisfySlot(p.readyEvt, NULL_GUID, 0);
  }
  /* mode 1: rely on this EDT's OUTPUT EVENT (fires only on completion, which
   * is deferred until the release's invalidation is acknowledged).           */
  return NULL_GUID;
}

ocrGuid_t consumerEdt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  chain_params_t p;
  memcpy(&p, paramv, sizeof(p));
  /* depv[0] = db (RO), depv[1] = trigger */
  int got = *(int *)depv[0].ptr;
  int expected = (int)p.i;
  if (got != expected) {
    fprintf(stdout, "STALE iter=%llu expected=%d got=%d\n",
            (unsigned long long)p.i, expected, got);
    fflush(stdout);
  }
  ocrDbRelease(depv[0].guid);
  if (p.i + 1 < p.iters) {
    ocrEventSatisfySlot(p.nextGo, NULL_GUID, 0);
  } else {
    fprintf(stdout, "DONE iters=%llu (no stale line above => clean)\n",
            (unsigned long long)p.iters);
    fflush(stdout);
    ocrShutdown();
  }
  return NULL_GUID;
}

ocrGuid_t mainEdt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  u64 iters = 400;
  if (depc >= 1 && depv[0].ptr) {
    u64 argc = getArgc(depv[0].ptr);
    if (argc >= 2) {
      char *a1 = getArgv(depv[0].ptr, 1);
      u64 v = (u64)strtoull(a1, 0, 10);
      if (v > 0)
        iters = v;
    }
  }

  u64 mode = 0;
  {
    const char *m = getenv("MODE");
    if (m)
      mode = (u64)strtoull(m, 0, 10);
  }

  u64 nodes = 0;
  ocrAffinityCount(AFFINITY_PD, &nodes);
  if (nodes == 0)
    nodes = 1;
  ocrGuid_t *aff = (ocrGuid_t *)malloc(sizeof(ocrGuid_t) * nodes);
  u64 got = nodes;
  ocrAffinityGet(AFFINITY_PD, &got, aff);

  u64 prodNode = (nodes > PROD_NODE) ? PROD_NODE : 0;
  u64 consNode = (nodes > CONS_NODE) ? CONS_NODE : (nodes - 1);
  u64 homeNode = (nodes > 2) ? 2 : 0;

  fprintf(stdout,
          "release_sync_stress: nodes=%llu iters=%llu prod=PD%llu cons=PD%llu "
          "home=PD%llu mode=%llu(%s)\n",
          (unsigned long long)nodes, (unsigned long long)iters,
          (unsigned long long)prodNode, (unsigned long long)consNode,
          (unsigned long long)homeNode, (unsigned long long)mode,
          mode == 0 ? "mid-body-satisfy" : "output-event-gated");
  fflush(stdout);

  ocrGuid_t db;
  void *dbp = 0;
  {
    ocrHint_t dh;
    ocrHintInit(&dh, OCR_HINT_DB_T);
    ocrSetHintValue(&dh, OCR_HINT_DB_AFFINITY,
                    ocrAffinityToHintValue(aff[homeNode]));
    ocrDbCreate(&db, &dbp, sizeof(int), 0, &dh, NO_ALLOC);
    *(int *)dbp = -1;
    ocrDbRelease(db);
  }

  ocrGuid_t prodTmpl, consTmpl;
  ocrEdtTemplateCreate(&prodTmpl, producerEdt, (u32)PWORDS, 2);
  ocrEdtTemplateCreate(&consTmpl, consumerEdt, (u32)PWORDS, 2);

  ocrHint_t ph, ch;
  ocrHintInit(&ph, OCR_HINT_EDT_T);
  ocrSetHintValue(&ph, OCR_HINT_EDT_AFFINITY,
                  ocrAffinityToHintValue(aff[prodNode]));
  ocrHintInit(&ch, OCR_HINT_EDT_T);
  ocrSetHintValue(&ch, OCR_HINT_EDT_AFFINITY,
                  ocrAffinityToHintValue(aff[consNode]));

  ocrGuid_t *readyEvt = (ocrGuid_t *)malloc(sizeof(ocrGuid_t) * iters);
  ocrGuid_t *goEvt = (ocrGuid_t *)malloc(sizeof(ocrGuid_t) * iters);
  u64 i;
  for (i = 0; i < iters; ++i) {
    ocrEventCreate(&readyEvt[i], OCR_EVENT_ONCE_T, 0);
    ocrEventCreate(&goEvt[i], OCR_EVENT_ONCE_T, 0);
  }

  for (i = 0; i < iters; ++i) {
    chain_params_t pp, cp;
    pp.i = i;
    pp.db = db;
    pp.readyEvt = readyEvt[i];
    pp.nextGo = NULL_GUID;
    pp.iters = iters;
    pp.mode = mode;
    cp.i = i;
    cp.db = db;
    cp.readyEvt = readyEvt[i];
    cp.nextGo = (i + 1 < iters) ? goEvt[i + 1] : NULL_GUID;
    cp.iters = iters;
    cp.mode = mode;

    ocrGuid_t prod, cons, prodOut = NULL_GUID;
    u64 pbuf[PWORDS];
    memcpy(pbuf, &pp, sizeof(pp));
    u64 cbuf[PWORDS];
    memcpy(cbuf, &cp, sizeof(cp));

    ocrEdtCreate(&prod, prodTmpl, (u32)PWORDS, pbuf, 2, 0, 0, &ph,
                 (mode == 1) ? &prodOut : 0);
    ocrEdtCreate(&cons, consTmpl, (u32)PWORDS, cbuf, 2, 0, 0, &ch, 0);

    ocrGuid_t consTrigger = (mode == 1) ? prodOut : readyEvt[i];

    /* producer[i]: slot0 = db(RW), slot1 = goEvt[i] */
    ocrAddDependence(goEvt[i], prod, 1, DB_MODE_RO);
    ocrAddDependence(db, prod, 0, DB_MODE_RW);
    /* consumer[i]: slot0 = db(RO), slot1 = consTrigger */
    ocrAddDependence(consTrigger, cons, 1, DB_MODE_RO);
    ocrAddDependence(db, cons, 0, DB_MODE_RO);
  }

  ocrEventSatisfySlot(goEvt[0], NULL_GUID, 0);

  free(readyEvt);
  free(goEvt);
  free(aff);
  return NULL_GUID;
}
