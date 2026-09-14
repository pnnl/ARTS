/* SPDX-License-Identifier: Apache-2.0 */
#include "ocr.h"
#include "extensions/ocr-affinity.h"

/* One line per slot on failure, so a runtime's deviation names the slot and
 * the policy it broke rather than a single verdict. */
static ocrGuid_t check_payload(u32 pc, u64 *pv, u32 dc, ocrEdtDep_t dv[]) {
  int bad = (pc != 2 || dc != 7);
  const char *what[7] = {"latch discards payload", "params latch discards payload",
                         "once passes DB, RO acquires", "idem passes DB, RO acquires",
                         "sticky passes DB, NULL keeps GUID without acquire",
                         "sticky passes retired DB, NULL keeps GUID without acquire",
                         "NULL source delivers NULL_GUID"};
  int ok[7];
  ok[0] = ocrGuidIsNull(dv[0].guid) && !dv[0].ptr;
  ok[1] = ocrGuidIsNull(dv[1].guid) && !dv[1].ptr;
  ok[2] = dv[2].guid.guid == pv[0] && dv[2].ptr && *(u64 *)dv[2].ptr == 42;
  ok[3] = dv[3].guid.guid == pv[0] && dv[3].ptr && *(u64 *)dv[3].ptr == 42;
  ok[4] = dv[4].guid.guid == pv[0] && !dv[4].ptr;
  ok[5] = dv[5].guid.guid == pv[1] && !dv[5].ptr;
  ok[6] = ocrGuidIsNull(dv[6].guid) && !dv[6].ptr;
  for (u32 i = 0; i < 7 && i < dc; ++i) {
    if (!ok[i]) {
      PRINTF("FAIL slot %u (%s): guid=%lx ptr=%p expected guid=%lx\n", i, what[i],
             (unsigned long)dv[i].guid.guid, dv[i].ptr,
             (unsigned long)(i == 2 || i == 3 || i == 4 ? pv[0] : i == 5 ? pv[1] : 0));
      bad = 1;
    }
  }
  if (bad) {
    PRINTF("FAIL: OCR event payload or NULL acquisition policy\n");
    ocrAbort(1);
    return NULL_GUID;
  }
  PRINTF("PASS: OCR latch discard, data event preservation, NULL payload\n");
  ocrShutdown();
  return NULL_GUID;
}

ocrGuid_t mainEdt(u32 pc, u64 *pv, u32 dc, ocrEdtDep_t dv[]) {
  (void)pc; (void)pv; (void)dc; (void)dv;
  ocrGuid_t data, retired, latch, params_latch, once, idem, sticky, expired;
  u64 *bytes;
  ocrDbCreate(&data, (void **)&bytes, sizeof(*bytes), DB_PROP_NONE, NULL_HINT, NO_ALLOC);
  *bytes = 42;
  ocrDbRelease(data);
  ocrDbCreate(&retired, (void **)&bytes, sizeof(*bytes), DB_PROP_NONE, NULL_HINT, NO_ALLOC);
  *bytes = 77;
  ocrDbRelease(retired);

  ocrEventCreate(&latch, OCR_EVENT_LATCH_T, EVT_PROP_NONE);
  ocrEventSatisfySlot(latch, NULL_GUID, OCR_EVENT_LATCH_INCR_SLOT);
  ocrEventParams_t params = {0};
  params.EVENT_LATCH.counter = 1;
  ocrEventCreateParams(&params_latch, OCR_EVENT_LATCH_T, EVT_PROP_NONE, &params);
  ocrEventCreate(&once, OCR_EVENT_ONCE_T, EVT_PROP_TAKES_ARG);
  ocrEventCreate(&idem, OCR_EVENT_IDEM_T, EVT_PROP_TAKES_ARG);
  ocrEventCreate(&sticky, OCR_EVENT_STICKY_T, EVT_PROP_TAKES_ARG);
  ocrEventCreate(&expired, OCR_EVENT_STICKY_T, EVT_PROP_TAKES_ARG);

  u64 ranks = 0;
  ocrAffinityCount(AFFINITY_PD, &ranks);
  ocrGuid_t affinity;
  ocrAffinityGetAt(AFFINITY_PD, ranks - 1, &affinity);
  ocrHint_t hint;
  ocrHintInit(&hint, OCR_HINT_EDT_T);
  ocrSetHintValue(&hint, OCR_HINT_EDT_AFFINITY, ocrAffinityToHintValue(affinity));
  ocrGuid_t tpl, task;
  u64 expected[] = {data.guid, retired.guid};
  ocrEdtTemplateCreate(&tpl, check_payload, 2, 7);
  ocrEdtCreate(&task, tpl, 2, expected, 7, NULL, EDT_PROP_NONE, &hint, NULL);
  ocrEdtTemplateDestroy(tpl);
  ocrAddDependence(latch, task, 0, DB_MODE_NULL);
  ocrAddDependence(params_latch, task, 1, DB_MODE_NULL);
  ocrAddDependence(once, task, 2, DB_MODE_RO);
  ocrAddDependence(idem, task, 3, DB_MODE_RO);
  ocrAddDependence(once, latch, OCR_EVENT_LATCH_DECR_SLOT, DB_MODE_NULL);
  ocrEventSatisfy(once, data);
  ocrEventSatisfy(idem, data);
  ocrEventSatisfy(params_latch, data);
  ocrEventSatisfy(sticky, data);
  ocrAddDependence(sticky, task, 4, DB_MODE_NULL);
  ocrEventSatisfy(expired, retired);
  ocrAddDependence(expired, task, 5, DB_MODE_NULL);
  ocrDbDestroy(retired);
  ocrAddDependence(NULL_GUID, task, 6, DB_MODE_NULL);
  return NULL_GUID;
}
