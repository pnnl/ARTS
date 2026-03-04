/*
 * OCR-to-ARTS Compatibility Shim
 *
 * Implements the OCR v1.2.0 API using ARTS v2 primitives. OCR applications
 * call ocrEdtCreate(), ocrDbCreate(), etc., and this shim translates those
 * calls into the corresponding arts_edt_create(), arts_db_create(), etc.
 *
 * This file is part of the ARTS benchmark infrastructure — it is NOT
 * general-purpose user code. It deliberately accesses ARTS internal headers
 * (route table, runtime types) to implement features like artsDbDataFromGuid().
 */

#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Both ARTS and OCR define enum values with identical names but different
 * semantics (DB_MODE_NULL, DB_MODE_RO, DB_MODE_EW, DB_MODE_RW).
 * ARTS: sequential (0,1,2,3...).  OCR: bitmask (0x0,0x1,0x2,0x4,0x8).
 *
 * Strategy: include OCR headers first (get the real OCR names), then redirect
 * ARTS's enum names through the preprocessor before including ARTS headers.
 * OCR's `typedef u8 bool` also conflicts with C11 <stdbool.h>, so we include
 * OCR before any ARTS/system header that pulls in stdbool.
 */

/* --- OCR headers first --- */
/* OCR extension defines are set via CMake target_compile_definitions. */
#include "extensions/ocr-affinity.h"
#include "extensions/ocr-reduction-event.h"
#include "ocr-db.h"
#include "ocr-edt.h"
#include "ocr-std.h"
#include "ocr-types.h"
#include "ocr.h"

/* --- Redirect ARTS enum names so they don't collide with OCR's --- */
#define DB_MODE_NULL ARTS_DB_MODE_NULL_
#define DB_MODE_RO ARTS_DB_MODE_RO_
#define DB_MODE_EW ARTS_DB_MODE_EW_
#define DB_MODE_RW ARTS_DB_MODE_RW_
#define DB_MODE_VALUE ARTS_DB_MODE_VALUE_
#define DB_MODE_PTR ARTS_DB_MODE_PTR_
#define DB_MODE_LC_SYNC ARTS_DB_MODE_LC_SYNC_
#define DB_MODE_LC_NO_COPY ARTS_DB_MODE_LC_NO_COPY_
#define DB_MODE_MEMSET ARTS_DB_MODE_MEMSET_

/* OCR defined NULL_GUID as ocrGuid_t struct; save and undef for ARTS */
#undef NULL_GUID

/* ARTS headers pull in <stdbool.h>; OCR already typedef'd bool as u8.
 * Let ARTS redefine bool back to _Bool via stdbool. */

/* --- ARTS headers second --- */
#include "arts.h"
#include "arts/compute/edt.h"
#include "arts/gas/route_table.h"
#include "arts/system/threads.h"
#include "arts/utils/malloc.h"

/* Clean up the redirects */
#undef DB_MODE_NULL
#undef DB_MODE_RO
#undef DB_MODE_EW
#undef DB_MODE_RW
#undef DB_MODE_VALUE
#undef DB_MODE_PTR
#undef DB_MODE_LC_SYNC
#undef DB_MODE_LC_NO_COPY
#undef DB_MODE_MEMSET

/* ARTS DB_MODE values we need (sequential: NULL=0, RO=1, EW=2) */
#define ARTS_MODE_NULL ((arts_db_access_mode_t)ARTS_DB_MODE_NULL_)
#define ARTS_MODE_RO ((arts_db_access_mode_t)ARTS_DB_MODE_RO_)
#define ARTS_MODE_EW ((arts_db_access_mode_t)ARTS_DB_MODE_EW_)

/* NULL GUID helpers */
#define ARTS_NULL_GUID ((arts_guid_t)0x0)
#define OCR_NULL_GUID ((ocrGuid_t)NULL_GUID_INITIALIZER)

/* =========================================================================
 * Helper: OCR-safe paramv copy.
 *
 * OCR apps commonly cast a smaller struct to (u64*) and pass
 * paramc = ceil(sizeof(struct)/sizeof(u64)).  The last word may
 * extend past the struct allocation by up to 7 bytes.  This is
 * harmless (the source is always on the stack), but ASAN flags the
 * overread.  Copy byte-by-byte for the last partial word to keep
 * ASAN clean while matching original OCR runtime behavior.
 * ========================================================================= */
static inline void ocr_copy_paramv(uint64_t *dst, const u64 *src, u32 paramc) {
  if (paramc > 0 && src != NULL) {
    memcpy(dst, src, paramc * sizeof(uint64_t));
  }
}

#if defined(__SANITIZE_ADDRESS__) ||                                           \
    (defined(__has_feature) && __has_feature(address_sanitizer))
/* Under ASAN, suppress the harmless stack overread from struct-to-u64* casts.
 * This matches the original OCR runtime's behavior exactly.
 * We use volatile byte-by-byte copy to avoid the ASAN-intercepted memcpy
 * which checks source bounds even inside no_sanitize functions. */
__attribute__((no_sanitize("address"), noinline)) static void
ocr_copy_paramv_safe(uint64_t *dst, const u64 *src, u32 paramc) {
  volatile const char *s = (volatile const char *)src;
  volatile char *d = (volatile char *)dst;
  for (u32 i = 0; i < paramc * sizeof(uint64_t); i++) {
    d[i] = s[i];
  }
}
#else
#define ocr_copy_paramv_safe ocr_copy_paramv
#endif

/* =========================================================================
 * Helper: get datablock data pointer from its GUID.
 * The DB structure is: [struct arts_db_s header][actual data]
 * ========================================================================= */
static void *arts_db_data_from_guid(arts_guid_t db_guid) {
  void *ptr = arts_route_table_lookup_db(db_guid, NULL, false);
  if (ptr == NULL) {
    return NULL;
  }
  /* Note: caller uses the data pointer without holding the route table ref.
   * Safe because OCR semantics guarantee the DB is acquired by the calling
   * EDT and will not be destroyed until released. */
  arts_route_table_return_db(db_guid, false);
  return (void *)((struct arts_db_s *)ptr + 1);
}

/* =========================================================================
 * Collective Event Support
 *
 * OCR collective events are implemented using a metadata DB + mutex.
 * When all contributions arrive, the last contributor triggers reduction.
 * ========================================================================= */

#define MAX_COLLECTIVE_DEPENDENTS 256
#define MAX_COLLECTIVE_CONTRIBS 256

typedef struct {
  redOp_t op;
  collectiveType_t type;
  u32 nbContribs;
  u32 nbDatum;
  u32 generation;
  volatile u32 numDependents;
  volatile u32 contribCount;
  volatile double contributions[MAX_COLLECTIVE_CONTRIBS];
  volatile u8 contribFlags[MAX_COLLECTIVE_CONTRIBS];
  arts_guid_t dependents[MAX_COLLECTIVE_DEPENDENTS];
  u32 dependentSlots[MAX_COLLECTIVE_DEPENDENTS];
  arts_guid_t metaDbGuid;
  arts_guid_t edtGuid;
  pthread_mutex_t lock;
} CollectiveMetadata;

static double performReductionOp(double a, double b, redOp_t op) {
  u32 opType = (op >> 7) & 0x7;
  switch (opType) {
  case 0:
    return a + b;
  case 1:
    return a * b;
  case 2:
    return (a < b) ? a : b;
  case 3:
    return (a > b) ? a : b;
  default:
    return a + b;
  }
}

static void performCollectiveReduction(CollectiveMetadata *meta) {
  double result = 0.0;
  u32 firstValid = 1;

  for (u32 i = 0; i < meta->nbContribs && i < MAX_COLLECTIVE_CONTRIBS; i++) {
    if (meta->contribFlags[i]) {
      if (firstValid) {
        result = meta->contributions[i];
        firstValid = 0;
      } else {
        result = performReductionOp(result, meta->contributions[i], meta->op);
      }
    }
  }

  u32 numDeps = meta->numDependents;
  arts_guid_t localDeps[MAX_COLLECTIVE_DEPENDENTS];
  u32 localSlots[MAX_COLLECTIVE_DEPENDENTS];
  for (u32 i = 0; i < numDeps && i < MAX_COLLECTIVE_DEPENDENTS; i++) {
    localDeps[i] = meta->dependents[i];
    localSlots[i] = meta->dependentSlots[i];
    meta->dependents[i] = NULL_GUID;
    meta->dependentSlots[i] = 0;
  }

  meta->generation++;
  meta->contribCount = 0;
  meta->numDependents = 0;
  for (u32 i = 0; i < MAX_COLLECTIVE_CONTRIBS; i++) {
    meta->contribFlags[i] = 0;
  }

  pthread_mutex_unlock(&meta->lock);

  for (u32 i = 0; i < numDeps && i < MAX_COLLECTIVE_DEPENDENTS; i++) {
    if (localDeps[i] != NULL_GUID) {
      void *resultPtr;
      arts_guid_t resultDb =
          arts_db_create(&resultPtr, sizeof(double), ARTS_DB_DEFAULT, NULL);
      *(double *)resultPtr = result;

      arts_type_t dstType = arts_guid_get_type(localDeps[i]);
      if (dstType == ARTS_EDT) {
        arts_add_dependence(resultDb, localDeps[i], localSlots[i],
                            ARTS_MODE_RO);
      } else if (dstType == ARTS_EVENT) {
        if (!arts_is_event_fired(localDeps[i])) {
          arts_event_satisfy_slot(localDeps[i], resultDb,
                                  ARTS_EVENT_LATCH_DECR_SLOT);
        }
      }
    }
  }

  pthread_mutex_lock(&meta->lock);
}

#define COLLECTIVE_HASH_SIZE 4096

typedef struct {
  volatile arts_guid_t edtGuid;
  volatile arts_guid_t metaDbGuid;
} CollectiveMapEntry;

static CollectiveMapEntry collectiveMetaMap[COLLECTIVE_HASH_SIZE] = {{0, 0}};

/* =========================================================================
 * Channel Event Support
 *
 * OCR channel events map directly to ARTS CHANNEL events (latch=1).
 * Both use generation-based re-arming: each generation has its own latch
 * counter and dependent list.  Cross-node deps handled natively by ARTS.
 *
 * Protocol (satisfy-channel, latch=1 per version):
 *   ocrEventSatisfy(ch, data)     → DECR slot (stores per-gen data)
 *   ocrAddDependence(ch, edt, s)  → arts_add_dependence (runtime INCRs)
 *   Consumer-first: INCR 0→1 (inside add_dep), DECR 1→0 → FIRE
 *   Producer-first: DECR 0→-1, INCR -1→0 (inside add_dep) → UPDATE → FIRE
 * ========================================================================= */

static u32 collectiveHash(arts_guid_t guid) {
  uint64_t val = (uint64_t)guid;
  return (u32)(val % COLLECTIVE_HASH_SIZE);
}

static int tryRegisterCollectiveMeta(arts_guid_t key, arts_guid_t metaDbGuid) {
  u32 idx = collectiveHash(key);
  for (u32 i = 0; i < COLLECTIVE_HASH_SIZE; i++) {
    u32 probeIdx = (idx + i) % COLLECTIVE_HASH_SIZE;
    arts_guid_t expected = NULL_GUID;

    if (__sync_bool_compare_and_swap(&collectiveMetaMap[probeIdx].edtGuid,
                                     expected, key)) {
      collectiveMetaMap[probeIdx].metaDbGuid = metaDbGuid;
      return 1;
    }

    if (collectiveMetaMap[probeIdx].edtGuid == key) {
      return 0;
    }
  }
  return 0;
}

static arts_guid_t lookupCollectiveMeta(arts_guid_t edtGuid) {
  u32 idx = collectiveHash(edtGuid);
  for (u32 i = 0; i < COLLECTIVE_HASH_SIZE; i++) {
    u32 probeIdx = (idx + i) % COLLECTIVE_HASH_SIZE;
    if (collectiveMetaMap[probeIdx].edtGuid == edtGuid) {
      while (collectiveMetaMap[probeIdx].metaDbGuid == NULL_GUID) {
        __sync_synchronize();
      }
      return collectiveMetaMap[probeIdx].metaDbGuid;
    }
    if (collectiveMetaMap[probeIdx].edtGuid == NULL_GUID) {
      return NULL_GUID;
    }
  }
  return NULL_GUID;
}

/* =========================================================================
 * EDT Template Management
 * ========================================================================= */

typedef struct {
  ocrEdt_t funcPtr;
  u32 paramc;
  u32 depc;
} OcrEdtTemplate;

u8 ocrEdtTemplateCreate_internal(ocrGuid_t *guid, ocrEdt_t funcPtr, u32 paramc,
                                 u32 depc, const char *funcName) {
  (void)funcName;
  OcrEdtTemplate *templ = malloc(sizeof(OcrEdtTemplate));
  if (!templ) {
    return 1;
  }
  templ->funcPtr = funcPtr;
  templ->paramc = paramc;
  templ->depc = depc;

  guid->guid = (intptr_t)templ;
  return 0;
}

u8 ocrEdtTemplateDestroy(ocrGuid_t guid) {
  OcrEdtTemplate *templ = (OcrEdtTemplate *)guid.guid;
  free(templ);
  return 0;
}

/* =========================================================================
 * EDT_PROP_FINISH Support via ARTS Epochs
 *
 * Layout of ARTS paramv for OCR EDTs:
 *   paramv[0] = function pointer (ocrEdt_t)
 *   paramv[1] = original paramc
 *   paramv[2] = epoch GUID (for finish EDTs) or NULL_GUID
 *   paramv[3] = output event GUID or helper EDT GUID
 *   paramv[4] = flags: bit 0 = isFinishEdt
 *   paramv[5..5+paramc-1] = original paramv values
 * ========================================================================= */

#define FINISH_EDT_FLAG 0x1

static void epoch_termination_edt(uint32_t paramc, const uint64_t *paramv,
                                  uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;

  arts_guid_t outputEventGuid = (arts_guid_t)paramv[0];
  arts_guid_t returnGuid = (depc > 1) ? depv[1].guid : NULL_GUID;

  if (outputEventGuid != NULL_GUID) {
    arts_event_satisfy_slot(outputEventGuid, returnGuid,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }
}

static void ocr_edt_trampoline(uint32_t paramc, const uint64_t *paramv,
                               uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;

  ocrEdt_t func = (ocrEdt_t)paramv[0];
  u32 origParamc = (u32)paramv[1];
  arts_guid_t guidOrEpoch = (arts_guid_t)paramv[2];
  arts_guid_t helperOrOutEvt = (arts_guid_t)paramv[3];
  u64 flags = paramv[4];
  /* ARTS v2 paramv is const; copy original params for OCR's non-const API */
  u64 *origParamv = NULL;
  u64 origParamBuf[origParamc > 0 ? origParamc : 1];
  if (origParamc > 0) {
    memcpy(origParamBuf, &paramv[5], origParamc * sizeof(u64));
    origParamv = origParamBuf;
  }

  bool isFinishEdt = (flags & FINISH_EDT_FLAG) != 0;

  /* Convert arts_edt_dep_t to ocrEdtDep_t */
  ocrEdtDep_t ocrDepv[depc > 0 ? depc : 1];

  for (u32 i = 0; i < depc; i++) {
    ocrDepv[i].guid.guid = depv[i].guid;
    ocrDepv[i].ptr = depv[i].ptr;
    ocrDepv[i].mode = DB_DEFAULT_MODE;
  }

  if (isFinishEdt && guidOrEpoch != NULL_GUID) {
    /* Push the finish epoch onto the TLS stack and increment active_count.
     *
     * arts_edt_create_with_epoch() does NOT assign the finish epoch to
     * edt->epoch_guid — because the epoch was just created in ocrEdtCreate()
     * and is NOT on the caller's TLS stack, arts_check_epoch_is_root() fails,
     * and the EDT gets the caller's epoch instead (often NULL_GUID).
     *
     * arts_start_epoch() must be called here to:
     *   (a) push the finish epoch onto TLS so child EDTs inherit it, and
     *   (b) increment active_count by 1, matching the +1 finished_count
     *       that arts_increment_finished_epoch_list() adds for this entry
     *       when the trampoline EDT completes.
     *
     * Net accounting: active = 1 (this call) + N (children), finished =
     * 1 (this TLS entry) + N (children) → epoch fires when all complete. */
    arts_start_epoch(guidOrEpoch);
  }

  ocrGuid_t returnGuid = func(origParamc, origParamv, depc, ocrDepv);

  if (!isFinishEdt && helperOrOutEvt != NULL_GUID) {
    arts_event_satisfy_slot(helperOrOutEvt, returnGuid.guid,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }

  if (isFinishEdt && helperOrOutEvt != NULL_GUID) {
    if (returnGuid.guid != NULL_GUID) {
      /* arts_add_dependence handles both DB (immediate satisfy) and
       * event (register waiter) sources uniformly. */
      arts_add_dependence(returnGuid.guid, helperOrOutEvt, 1, ARTS_MODE_RO);
    } else {
      arts_signal_edt_value(helperOrOutEvt, 1, 0);
    }
  }
}

/* =========================================================================
 * Hint Helpers
 *
 * Extract ARTS node route from OCR hint (EDT or DB affinity).
 * OCR apps set affinity via:
 *   ocrSetHintValue(&hint, OCR_HINT_EDT_AFFINITY, ocrAffinityToHintValue(aff))
 * where ocrAffinityToHintValue() returns the node rank as u64.
 * ========================================================================= */

static unsigned int extract_edt_route_from_hint(ocrHint_t *hint) {
  if (hint == NULL || hint->type != OCR_HINT_EDT_T) {
    return arts_global_rank_id;
  }
  int idx = OCR_HINT_EDT_AFFINITY - OCR_HINT_EDT_PROP_START - 1;
  if (idx < 0) {
    return arts_global_rank_id;
  }
  if (!(hint->propMask & (1ULL << idx))) {
    return arts_global_rank_id;
  }
  u64 val = hint->args.propEDT[idx];
  return (unsigned int)(val % arts_global_rank_count);
}

static unsigned int extract_db_route_from_hint(ocrHint_t *hint) {
  if (hint == NULL || hint->type != OCR_HINT_DB_T) {
    return arts_global_rank_id;
  }
  int idx = OCR_HINT_DB_AFFINITY - OCR_HINT_DB_PROP_START - 1;
  if (idx < 0) {
    return arts_global_rank_id;
  }
  if (!(hint->propMask & (1ULL << idx))) {
    return arts_global_rank_id;
  }
  u64 val = hint->args.propDB[idx];
  return (unsigned int)(val % arts_global_rank_count);
}

/* =========================================================================
 * EDT Creation and Management
 * ========================================================================= */

u8 ocrEdtCreate(ocrGuid_t *guid, ocrGuid_t templateGuid, u32 paramc,
                u64 *paramv, u32 depc, ocrGuid_t *depv, u16 properties,
                ocrHint_t *hint, ocrGuid_t *outputEvent) {
  OcrEdtTemplate *templ = (OcrEdtTemplate *)templateGuid.guid;
  if (!templ) {
    return 1;
  }

  u32 actualParamc = (paramc == EDT_PARAM_DEF) ? templ->paramc : paramc;
  u32 actualDepc = (depc == EDT_PARAM_DEF) ? templ->depc : depc;

  unsigned int route = extract_edt_route_from_hint(hint);
  arts_guid_t outEvt = NULL_GUID;
  arts_guid_t epochGuid = NULL_GUID;
  bool isFinishEdt = (properties & EDT_PROP_FINISH) != 0;
  bool oevtValid = (properties & EDT_PROP_OEVT_VALID) != 0;

  if (outputEvent != NULL) {
    if (oevtValid) {
      outEvt = outputEvent->guid;
    } else {
      outEvt = arts_event_create(route, ARTS_EVENT_IDEM, 1, NULL_GUID);
      outputEvent->guid = outEvt;
    }
  }

  arts_guid_t helperEdtGuid = NULL_GUID;
  if (isFinishEdt && outEvt != NULL_GUID) {
    uint64_t helperParams[1];
    helperParams[0] = (uint64_t)outEvt;

    arts_hint_t h = {.route = route};
    helperEdtGuid =
        arts_edt_create(epoch_termination_edt, 1, helperParams, 2, &h);

    epochGuid = arts_initialize_epoch(route, helperEdtGuid, 0);
  }
  /* When EDT_PROP_FINISH is set but there is no output event, we
   * intentionally do NOT create a child epoch.  Without a child epoch the
   * sub-task inherits its parent's epoch via the TLS epoch stack, so ALL
   * descendants at every depth are tracked by the root finish-epoch.
   * Creating a per-sub-task epoch here would intercept the TLS current
   * epoch, causing the root epoch to miss deeply-nested descendants and
   * fire prematurely (the nqueens bug: variable solution count + heap
   * corruption on shutdown). */

  u32 artsParamc = 5 + actualParamc;
  /* arts_calloc zero-initializes, so trailing padding bytes beyond
   * the actual struct size are safe (OCR apps pass struct pointers
   * cast to u64* with paramc = ceil(sizeof(struct)/sizeof(u64)),
   * which may overread the source by up to 7 bytes). */
  uint64_t *artsParamv = (uint64_t *)arts_calloc(artsParamc, sizeof(uint64_t));
  artsParamv[0] = (uint64_t)(uintptr_t)templ->funcPtr;
  artsParamv[1] = (uint64_t)actualParamc;
  artsParamv[2] = (uint64_t)epochGuid;
  artsParamv[3] = isFinishEdt ? (uint64_t)helperEdtGuid : (uint64_t)outEvt;
  artsParamv[4] = isFinishEdt ? FINISH_EDT_FLAG : 0;
  ocr_copy_paramv_safe(&artsParamv[5], paramv, actualParamc);

  arts_guid_t edtGuid;
  arts_hint_t edtHint = {.route = route};

  if (isFinishEdt && epochGuid != NULL_GUID) {
    edtGuid =
        arts_edt_create_with_epoch(ocr_edt_trampoline, artsParamc, artsParamv,
                                   actualDepc, epochGuid, &edtHint);
  } else {
    edtGuid = arts_edt_create(ocr_edt_trampoline, artsParamc, artsParamv,
                              actualDepc, &edtHint);
  }

  arts_free(artsParamv);

  if (guid != NULL) {
    guid->guid = edtGuid;
  }

  if (depv != NULL && actualDepc > 0) {
    for (u32 i = 0; i < actualDepc; i++) {
      if (ocrGuidIsNull(depv[i])) {
        /* NULL_GUID = pre-satisfied slot (OCR spec §2.4.3).
         * Signal immediately so the EDT doesn't wait forever. */
        arts_signal_edt_value(edtGuid, i, 0);
      } else if (!ocrGuidIsUninitialized(depv[i])) {
        /* Valid GUID — signal now.  UNINITIALIZED_GUID slots are
         * left open for later ocrAddDependence calls. */
        /* arts_add_dependence handles both DB (immediate satisfy) and
         * event (register waiter) sources uniformly. */
        arts_add_dependence(depv[i].guid, edtGuid, i, ARTS_MODE_RO);
      }
    }
  }

  return 0;
}

u8 ocrEdtDestroy(ocrGuid_t guid) {
  arts_edt_destroy(guid.guid);
  return 0;
}

/* =========================================================================
 * Event Management
 * ========================================================================= */

u8 ocrEventCreate(ocrGuid_t *guid, ocrEventTypes_t eventType, u16 properties) {
  unsigned int latchCount = 1;

  if (eventType == OCR_EVENT_LATCH_T) {
    latchCount = 0;
  }

  arts_event_types_t artsEvtType;
  switch (eventType) {
  case OCR_EVENT_ONCE_T:
    artsEvtType = ARTS_EVENT_IDEM;
    break;
  case OCR_EVENT_STICKY_T:
    artsEvtType = ARTS_EVENT_STICKY;
    break;
  case OCR_EVENT_IDEM_T:
    artsEvtType = ARTS_EVENT_IDEM;
    break;
  case OCR_EVENT_LATCH_T:
    artsEvtType = ARTS_EVENT_LATCH;
    break;
  case OCR_EVENT_CHANNEL_T:
    artsEvtType = ARTS_EVENT_CHANNEL;
    latchCount = 1; /* Satisfy-channel: needs both satisfy + addDep to fire. */
    break;
  default:
    return 1;
  }

  if (properties & GUID_PROP_IS_LABELED) {
    arts_guid_t result = arts_event_create_with_guid(guid->guid, artsEvtType,
                                                     latchCount, NULL_GUID);
    if (result == NULL_GUID && (properties & GUID_PROP_CHECK)) {
      return OCR_EGUIDEXISTS;
    }
    return 0;
  }

  guid->guid = arts_event_create(arts_global_rank_id, artsEvtType, latchCount,
                                 NULL_GUID);
  return 0;
}

u8 ocrEventDestroy(ocrGuid_t guid) {
  arts_event_destroy(guid.guid);
  return 0;
}

u8 ocrEventSatisfy(ocrGuid_t eventGuid, ocrGuid_t dataGuid) {
  /* For non-channel events, guard against re-satisfy of already-fired events.
   * Channel events handle re-fire natively via generations. */
  if (arts_is_event_fired(eventGuid.guid)) {
    return 0;
  }
  arts_event_satisfy_slot(eventGuid.guid, dataGuid.guid,
                          ARTS_EVENT_LATCH_DECR_SLOT);
  return 0;
}

u8 ocrEventSatisfySlot(ocrGuid_t eventGuid, ocrGuid_t dataGuid, u32 slot) {
  if (arts_is_event_fired(eventGuid.guid)) {
    return 0;
  }
  arts_event_satisfy_slot(eventGuid.guid, dataGuid.guid, slot);
  return 0;
}

u8 ocrEventCreateParams(ocrGuid_t *guid, ocrEventTypes_t eventType,
                        u16 properties, ocrEventParams_t *params) {

  if (eventType == OCR_EVENT_COUNTED_T && params != NULL) {
    /*
     * OCR COUNTED events fire on ONE satisfy call (like ONCE).  The
     * params->EVENT_COUNTED.nbDeps value tracks expected downstream
     * registrations for auto-destruction — it is NOT the number of
     * satisfies needed to fire.  Map to ARTS IDEM (latch=1, persist,
     * silent re-satisfy) so late-arriving ocrAddDependence callers
     * get signaled immediately.
     */

    if (properties & GUID_PROP_IS_LABELED) {
      arts_guid_t result = arts_event_create_with_guid(
          guid->guid, ARTS_EVENT_IDEM, 1, NULL_GUID);
      if (result == NULL_GUID && (properties & GUID_PROP_CHECK)) {
        return OCR_EGUIDEXISTS;
      }
      return 0;
    }

    guid->guid =
        arts_event_create(arts_global_rank_id, ARTS_EVENT_IDEM, 1, NULL_GUID);
    return 0;
  }

  if (eventType == OCR_EVENT_COLLECTIVE_T && params != NULL) {
    u32 nbContribs = params->EVENT_COLLECTIVE.nbContribs;
    arts_guid_t labeledGuid = guid->guid;

    if ((properties & GUID_PROP_IS_LABELED) && labeledGuid != NULL_GUID) {
      arts_guid_t existingMeta = lookupCollectiveMeta(labeledGuid);
      if (existingMeta != NULL_GUID) {
        return 0;
      }
    }

    void *metaPtr;
    arts_guid_t metaDb = arts_db_create(&metaPtr, sizeof(CollectiveMetadata),
                                        ARTS_DB_DEFAULT, NULL);
    CollectiveMetadata *meta = (CollectiveMetadata *)metaPtr;

    meta->op = params->EVENT_COLLECTIVE.op;
    meta->type = params->EVENT_COLLECTIVE.type;
    meta->nbContribs = nbContribs;
    meta->nbDatum = params->EVENT_COLLECTIVE.nbDatum;
    meta->generation = 0;
    meta->numDependents = 0;
    meta->contribCount = 0;
    meta->metaDbGuid = metaDb;
    meta->edtGuid = NULL_GUID;
    pthread_mutex_init(&meta->lock, NULL);

    for (u32 i = 0; i < MAX_COLLECTIVE_CONTRIBS; i++) {
      meta->contributions[i] = 0.0;
      meta->contribFlags[i] = 0;
    }
    for (u32 i = 0; i < MAX_COLLECTIVE_DEPENDENTS; i++) {
      meta->dependents[i] = NULL_GUID;
      meta->dependentSlots[i] = 0;
    }

    if ((properties & GUID_PROP_IS_LABELED) && labeledGuid != NULL_GUID) {
      if (!tryRegisterCollectiveMeta(labeledGuid, metaDb)) {
        return 0;
      }
      return 0;
    }

    tryRegisterCollectiveMeta(metaDb, metaDb);
    guid->guid = metaDb;
    return 0;
  }

  if (eventType == OCR_EVENT_CHANNEL_T && params != NULL) {
    /* params->EVENT_CHANNEL.maxGen is informational only — ARTS channel
     * versions grow dynamically via the linked list, so we don't need
     * to pre-allocate a queue.  Just create a native CHANNEL event. */
    (void)params->EVENT_CHANNEL.maxGen;

    if (properties & GUID_PROP_IS_LABELED) {
      arts_guid_t result = arts_event_create_with_guid(
          guid->guid, ARTS_EVENT_CHANNEL, 1, NULL_GUID);
      if (result == NULL_GUID && (properties & GUID_PROP_CHECK)) {
        return OCR_EGUIDEXISTS;
      }
      return 0;
    }

    guid->guid = arts_event_create(arts_global_rank_id, ARTS_EVENT_CHANNEL, 1,
                                   NULL_GUID);
    return 0;
  }

  return ocrEventCreate(guid, eventType, properties);
}

u8 ocrEventCollectiveSatisfySlot(ocrGuid_t eventGuid, void *dataPtr,
                                 u32 islot) {
  arts_guid_t metaDbGuid = lookupCollectiveMeta(eventGuid.guid);

  if (metaDbGuid == NULL_GUID) {
    if (arts_is_event_fired(eventGuid.guid)) {
      return 1;
    }
    arts_guid_t dataGuid =
        (dataPtr != NULL) ? (arts_guid_t)(uintptr_t)dataPtr : NULL_GUID;
    arts_event_satisfy_slot(eventGuid.guid, dataGuid, islot);
    return 0;
  }

  CollectiveMetadata *meta =
      (CollectiveMetadata *)arts_db_data_from_guid(metaDbGuid);
  if (meta == NULL) {
    return 1;
  }

  double value = 0.0;
  if (dataPtr != NULL) {
    value = *(double *)dataPtr;
  }

  pthread_mutex_lock(&meta->lock);

  if (islot < MAX_COLLECTIVE_CONTRIBS) {
    meta->contributions[islot] = value;
    meta->contribFlags[islot] = 1;
  }

  u32 newCount = ++meta->contribCount;
  u32 isLast = (newCount == meta->nbContribs);

  if (isLast) {
    performCollectiveReduction(meta);
  }

  pthread_mutex_unlock(&meta->lock);

  return 0;
}

/* =========================================================================
 * Data Block Management
 * ========================================================================= */

u8 ocrDbCreate(ocrGuid_t *db, void **addr, u64 len, u16 flags, ocrHint_t *hint,
               ocrInDbAllocator_t allocator) {
  (void)allocator;

  if (flags & GUID_PROP_IS_LABELED) {
    arts_guid_t labeledGuid = db->guid;

    void *data =
        arts_db_create_with_guid(labeledGuid, len, ARTS_DB_DEFAULT, NULL, NULL);
    if (data == NULL) {
      data = arts_route_table_lookup_db(labeledGuid, NULL, false);
      if (data != NULL) {
        *addr = (void *)((struct arts_db_s *)data + 1);
        arts_route_table_return_db(labeledGuid, false);
        return 0;
      }
      return 1;
    }
    *addr = data;
    return 0;
  }

  unsigned int route = extract_db_route_from_hint(hint);
  arts_hint_t artsHint = {.route = route};
  db->guid = arts_db_create(addr, len, ARTS_DB_DEFAULT, &artsHint);

  return 0;
}

u8 ocrDbDestroy(ocrGuid_t db) {
  /* OCR spec: ocrDbDestroy marks the DB for destruction.  If the calling
   * EDT has acquired this DB, arts_db_destroy implicitly releases it.
   * The route table's deferred deletion keeps the DB alive while other
   * EDTs still hold route table references. */
  if (!ocrGuidIsNull(db)) {
    arts_db_destroy(db.guid);
  }
  return 0;
}

u8 ocrDbRelease(ocrGuid_t db) {
  if (!ocrGuidIsNull(db)) {
    arts_db_release(db.guid);
  }
  return 0;
}

/* =========================================================================
 * Dependence Management
 * ========================================================================= */

/*
 * Map OCR access mode → ARTS access mode.
 *
 * ARTS CDAG model supports RO (shared read) and EW (exclusive write) for
 * normal datablocks.  OCR RW and EW both imply exclusive access, so both
 * map to ARTS EW.  OCR RO maps to ARTS RO.
 *
 * ocrDbRelease() calls arts_db_release() to release frontier locks early
 * for EW deps, allowing consumer EDTs to proceed before the current EDT
 * completes.  For RO deps, no frontier action is needed.
 *
 * IMPORTANT: After the macro cleanup in the header section, bare
 * DB_MODE_RO/EW/RW names resolve to OCR enum constants (0x8/0x4/0x2),
 * NOT ARTS values.  Always use ARTS_MODE_RO/ARTS_MODE_EW for ARTS values.
 */
static arts_db_access_mode_t ocr_to_arts_mode(ocrDbAccessMode_t ocr_mode) {
  switch (ocr_mode) {
  case DB_MODE_EW: /* OCR 0x4 → ARTS EW (true exclusive write) */
    return ARTS_MODE_EW;
  case DB_MODE_RO: /* OCR 0x8 → ARTS RO */
  case DB_MODE_RW: /* OCR 0x2 → ARTS RO (advisory; OCR doesn't enforce RW
                    * exclusion, and apps routinely use RW as a default even
                    * for shared reads.  Mapping to EW causes frontier
                    * serialization and performance collapse.) */
  default:
    return ARTS_MODE_RO;
  }
}

u8 ocrAddDependence(ocrGuid_t source, ocrGuid_t destination, u32 slot,
                    ocrDbAccessMode_t mode) {

  /* NULL source → signal immediately (slot satisfied with no data). */
  if (ocrGuidIsNull(source)) {
    arts_type_t dstType = arts_guid_get_type(destination.guid);
    if (dstType == ARTS_EDT) {
      arts_signal_edt_value(destination.guid, slot, 0);
    } else if (dstType == ARTS_EVENT) {
      arts_event_satisfy_slot(destination.guid, NULL_GUID,
                              ARTS_EVENT_LATCH_DECR_SLOT);
    }
    return 0;
  }

  arts_type_t srcType = arts_guid_get_type(source.guid);
  arts_type_t dstType = arts_guid_get_type(destination.guid);

  if (srcType == ARTS_DB) {
    /* DB → EDT/Event: arts_add_dependence does immediate satisfy for DB
     * sources (DBs are passive objects — no channel event, no waiting).
     * Map OCR access modes to ARTS: RO→RO, EW/RW→EW.
     * GUID-sorted acquisition in acquire_dbs prevents frontier deadlocks
     * that previously required forcing all deps to RO. */
    if (dstType == ARTS_EDT) {
      arts_add_dependence(source.guid, destination.guid, slot,
                          ocr_to_arts_mode(mode));
    } else if (dstType == ARTS_EVENT) {
      arts_event_satisfy_slot(destination.guid, source.guid,
                              ARTS_EVENT_LATCH_DECR_SLOT);
    }
  } else if (srcType == ARTS_EVENT) {
    /* ARTS channels (latch=1) do INCR internally in add_dependence_with_mode.
     * Non-channel events use direct dependent registration. Both paths
     * are handled by arts_add_dependence. Cross-node: handled natively. */
    if (dstType == ARTS_EDT) {
      arts_add_dependence(source.guid, destination.guid, slot, ARTS_MODE_RO);
    } else if (dstType == ARTS_EVENT) {
      /* Event→event: OCR spec says "satisfy dest when source fires".
       * Always use DECR slot regardless of the incoming slot param. */
      arts_add_dependence(source.guid, destination.guid,
                          ARTS_EVENT_LATCH_DECR_SLOT, ARTS_MODE_NULL);
    }
  }

  return 0;
}

u8 ocrAddDependenceSlot(ocrGuid_t source, u32 sslot, ocrGuid_t destination,
                        u32 dslot, ocrDbAccessMode_t mode) {
  (void)sslot;

  arts_guid_t metaDbGuid = lookupCollectiveMeta(source.guid);

  if (metaDbGuid != NULL_GUID) {
    CollectiveMetadata *meta =
        (CollectiveMetadata *)arts_db_data_from_guid(metaDbGuid);
    if (meta != NULL) {
      u32 idx = __sync_fetch_and_add(&meta->numDependents, 1);
      if (idx < MAX_COLLECTIVE_DEPENDENTS) {
        meta->dependents[idx] = destination.guid;
        meta->dependentSlots[idx] = dslot;
      }
    }
    return 0;
  }

  return ocrAddDependence(source, destination, dslot, mode);
}

/* =========================================================================
 * Runtime Control
 * ========================================================================= */

void ocrShutdown(void) { arts_shutdown(); }

void ocrAbort(u8 errorCode) { arts_abort(errorCode); }

/* =========================================================================
 * Printf Support
 * ========================================================================= */

u32 PRINTF(const char *fmt, ...) {
  printf(" [%u] ", arts_global_rank_id);
  va_list args;
  va_start(args, fmt);
  int written = vprintf(fmt, args);
  va_end(args);
  (void)fflush(stdout);
  return (u32)(written >= 0 ? written : 0);
}

u32 ocrPrintf(const char *fmt, ...) {
  printf(" [%u] ", arts_global_rank_id);
  va_list args;
  va_start(args, fmt);
  int written = vprintf(fmt, args);
  va_end(args);
  (void)fflush(stdout);
  return (u32)(written >= 0 ? written : 0);
}

u32 SNPRINTF(char *buf, u32 size, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int written = vsnprintf(buf, size, fmt, args);
  va_end(args);
  return (u32)(written >= 0 ? written : 0);
}

void _ocrAssert(u8 val, const char *str, const char *file, u32 line) {
  if (!val) {
    (void)fprintf(stderr, "ASSERTION FAILED: %s at %s:%" PRIu32 "\n", str, file,
                  line);
    abort();
  }
}

/* =========================================================================
 * EDT Local Storage (ELS) Extension
 * ========================================================================= */

#include "extensions/ocr-runtime-itf.h"

#define OCR_ELS_SIZE 16

static _Thread_local ocrGuid_t els_storage[OCR_ELS_SIZE] = {{0}};

ocrGuid_t ocrElsUserGet(u8 offset) {
  if (offset >= OCR_ELS_SIZE) {
    ocrGuid_t null_guid = {0};
    return null_guid;
  }
  return els_storage[offset];
}

void ocrElsUserSet(u8 offset, ocrGuid_t data) {
  if (offset < OCR_ELS_SIZE) {
    els_storage[offset] = data;
  }
}

/* =========================================================================
 * GUID Labeling Extension
 * ========================================================================= */

#include "extensions/ocr-labeling.h"

static arts_type_t kindToArtsType(ocrGuidUserKind kind) {
  switch (kind) {
  case GUID_USER_DB:
    return ARTS_DB;
  case GUID_USER_EDT:
  case GUID_USER_EDT_TEMPLATE:
    return ARTS_EDT;
  case GUID_USER_EVENT_ONCE:
  case GUID_USER_EVENT_COUNTED:
  case GUID_USER_EVENT_IDEM:
  case GUID_USER_EVENT_STICKY:
  case GUID_USER_EVENT_LATCH:
  case GUID_USER_EVENT_COLLECTIVE:
    return ARTS_EVENT;
  default:
    return ARTS_NULL;
  }
}

u8 ocrGuidRangeCreate(ocrGuid_t *rangeGuid, u64 numberGuid,
                      ocrGuidUserKind kind) {
  if (!rangeGuid || numberGuid == 0) {
    return 1;
  }
  arts_type_t artsType = kindToArtsType(kind);

  arts_guid_t range = arts_guid_reserve_range(
      artsType, (unsigned int)numberGuid, arts_global_rank_id);
  rangeGuid->guid = range;
  return 0;
}

u8 ocrGuidMapDestroy(ocrGuid_t mapGuid) {
  (void)mapGuid;
  return 0;
}

u8 ocrGuidFromIndex(ocrGuid_t *outGuid, ocrGuid_t rangeGuid, u64 idx) {
  if (!outGuid) {
    return 1;
  }
  outGuid->guid = arts_guid_from_index(rangeGuid.guid, (unsigned int)idx);
  if (ocrGuidIsNull(*outGuid)) {
    return 1;
  }
  return 0;
}

/* =========================================================================
 * Argument Handling
 * ========================================================================= */

u64 getArgc(void *dbPtr) {
  u64 *data = (u64 *)dbPtr;
  return data[0];
}

char *getArgv(void *dbPtr, u64 count) {
  u64 *data = (u64 *)dbPtr;
  u64 offset = data[count + 1];
  return (char *)((u8 *)dbPtr + offset);
}

u64 ocrGetArgc(void *dbPtr) { return getArgc(dbPtr); }

char *ocrGetArgv(void *dbPtr, u64 count) { return getArgv(dbPtr, count); }

/* =========================================================================
 * Affinity Extension
 * ========================================================================= */

u8 ocrAffinityCount(ocrAffinityKind kind, u64 *count) {
  if (!count) {
    return 1;
  }
  switch (kind) {
  case AFFINITY_PD:
    *count = (u64)arts_get_total_nodes();
    break;
  case AFFINITY_CURRENT:
    *count = 1;
    break;
  default:
    *count = 1;
    break;
  }
  return 0;
}

u8 ocrAffinityGet(ocrAffinityKind kind, u64 *count, ocrGuid_t *affinities) {
  if (!count || !affinities) {
    return 1;
  }

  u64 requested = *count;
  u64 available = 0;
  ocrAffinityCount(kind, &available);

  u64 toReturn = (requested < available) ? requested : available;

  switch (kind) {
  case AFFINITY_PD:
    for (u64 i = 0; i < toReturn; i++) {
      affinities[i].guid = (intptr_t)i;
    }
    break;
  case AFFINITY_CURRENT:
    affinities[0].guid = (intptr_t)arts_global_rank_id;
    toReturn = 1;
    break;
  default:
    affinities[0].guid = (intptr_t)arts_global_rank_id;
    toReturn = 1;
    break;
  }

  *count = toReturn;
  return 0;
}

u8 ocrAffinityGetAt(ocrAffinityKind kind, u64 idx, ocrGuid_t *affinity) {
  if (!affinity) {
    return 1;
  }

  u64 count = 0;
  ocrAffinityCount(kind, &count);

  if (idx >= count) {
    return OCR_EINVAL;
  }

  switch (kind) {
  case AFFINITY_PD:
    affinity->guid = (intptr_t)idx;
    break;
  case AFFINITY_CURRENT:
    affinity->guid = (intptr_t)arts_global_rank_id;
    break;
  default:
    affinity->guid = (intptr_t)arts_global_rank_id;
    break;
  }
  return 0;
}

u8 ocrAffinityGetCurrent(ocrGuid_t *affinity) {
  if (!affinity) {
    return 1;
  }
  affinity->guid = (intptr_t)arts_global_rank_id;
  return 0;
}

u8 ocrAffinityQuery(ocrGuid_t guid, u64 *count, ocrGuid_t *affinities) {
  if (!count || !affinities) {
    return 1;
  }
  (void)guid;
  if (*count >= 1) {
    affinities[0].guid = (intptr_t)arts_global_rank_id;
    *count = 1;
  }
  return 0;
}

u64 ocrAffinityToHintValue(ocrGuid_t affinity) { return (u64)affinity.guid; }

/* =========================================================================
 * Hints Extension
 * ========================================================================= */

u8 ocrHintInit(ocrHint_t *hint, ocrHintType_t hintType) {
  if (!hint) {
    return 1;
  }
  hint->type = hintType;
  hint->propMask = 0;
  memset(&hint->args, 0, sizeof(hint->args));
  return 0;
}

static int getHintPropIndex(ocrHintType_t type, ocrHintProp_t prop) {
  switch (type) {
  case OCR_HINT_EDT_T:
    if (prop > OCR_HINT_EDT_PROP_START && prop < OCR_HINT_EDT_PROP_END) {
      return (int)(prop - OCR_HINT_EDT_PROP_START - 1);
    }
    break;
  case OCR_HINT_DB_T:
    if (prop > OCR_HINT_DB_PROP_START && prop < OCR_HINT_DB_PROP_END) {
      return (int)(prop - OCR_HINT_DB_PROP_START - 1);
    }
    break;
  case OCR_HINT_EVT_T:
    // OCR spec: EVT property range is empty (START == END - 1)
    if (prop > OCR_HINT_EVT_PROP_START && // NOLINT
        prop < OCR_HINT_EVT_PROP_END) {
      return (int)(prop - OCR_HINT_EVT_PROP_START - 1);
    }
    break;
  case OCR_HINT_GROUP_T:
    // OCR spec: GROUP property range is empty (START == END - 1)
    if (prop > OCR_HINT_GROUP_PROP_START && // NOLINT
        prop < OCR_HINT_GROUP_PROP_END) {
      return (int)(prop - OCR_HINT_GROUP_PROP_START - 1);
    }
    break;
  default:
    break;
  }
  return -1;
}

u8 ocrSetHintValue(ocrHint_t *hint, ocrHintProp_t hintProp, u64 value) {
  if (!hint) {
    return 1;
  }

  int idx = getHintPropIndex(hint->type, hintProp);
  if (idx < 0) {
    return OCR_EINVAL;
  }

  switch (hint->type) {
  case OCR_HINT_EDT_T:
    hint->args.propEDT[idx] = value;
    break;
  case OCR_HINT_DB_T:
    hint->args.propDB[idx] = value;
    break;
  case OCR_HINT_EVT_T:
    hint->args.propEVT[idx] = value;
    break;
  case OCR_HINT_GROUP_T:
    hint->args.propGROUP[idx] = value;
    break;
  default:
    return OCR_EINVAL;
  }

  hint->propMask |= (1ULL << idx);
  return 0;
}

u8 ocrUnsetHintValue(ocrHint_t *hint, ocrHintProp_t hintProp) {
  if (!hint) {
    return 1;
  }

  int idx = getHintPropIndex(hint->type, hintProp);
  if (idx < 0) {
    return OCR_EINVAL;
  }

  hint->propMask &= ~(1ULL << idx);
  return 0;
}

u8 ocrGetHintValue(ocrHint_t *hint, ocrHintProp_t hintProp, u64 *value) {
  if (!hint || !value) {
    return 1;
  }

  int idx = getHintPropIndex(hint->type, hintProp);
  if (idx < 0) {
    return OCR_EINVAL;
  }

  if (!(hint->propMask & (1ULL << idx))) {
    return OCR_ENOENT;
  }

  switch (hint->type) {
  case OCR_HINT_EDT_T:
    *value = hint->args.propEDT[idx];
    break;
  case OCR_HINT_DB_T:
    *value = hint->args.propDB[idx];
    break;
  case OCR_HINT_EVT_T:
    *value = hint->args.propEVT[idx];
    break;
  case OCR_HINT_GROUP_T:
    *value = hint->args.propGROUP[idx];
    break;
  default:
    return OCR_EINVAL;
  }

  return 0;
}

u8 ocrSetHint(ocrGuid_t guid, ocrHint_t *hint) {
  (void)guid;
  (void)hint;
  return 0;
}

u8 ocrGetHint(ocrGuid_t guid, ocrHint_t *hint) {
  (void)guid;
  (void)hint;
  return 0;
}

/* =========================================================================
 * Main EDT Entry Point
 * ========================================================================= */

extern ocrGuid_t mainEdt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]);

static void mainEdtTrampoline(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;

  ocrEdtDep_t ocrDepv[depc > 0 ? depc : 1];
  for (u32 i = 0; i < depc; i++) {
    ocrDepv[i].guid.guid = depv[i].guid;
    ocrDepv[i].ptr = depv[i].ptr;
    ocrDepv[i].mode = DB_DEFAULT_MODE;
  }

  mainEdt(0, NULL, depc, ocrDepv);
}

/* ARTS v2 main_edt entry point */
void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;

  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];

  /* Build OCR argument datablock:
   * Format: [argc][offset0][offset1]...[offsetN-1][arg0\0][arg1\0]... */
  size_t headerSize = sizeof(u64) * (1 + argc);
  size_t stringsSize = 0;
  for (int i = 0; i < argc; i++) {
    stringsSize += strlen(argv[i]) + 1;
  }
  size_t totalSize = headerSize + stringsSize;

  void *dbPtr;
  arts_guid_t argsDbGuid =
      arts_db_create(&dbPtr, totalSize, ARTS_DB_DEFAULT, NULL);

  u64 *header = (u64 *)dbPtr;
  header[0] = (u64)argc;

  size_t currentOffset = headerSize;
  for (int i = 0; i < argc; i++) {
    header[i + 1] = currentOffset;
    size_t len = strlen(argv[i]) + 1;
    memcpy((u8 *)dbPtr + currentOffset, argv[i], len);
    currentOffset += len;
  }

  arts_hint_t h = {.route = arts_global_rank_id};
  arts_guid_t mainEdtGuid = arts_edt_create(mainEdtTrampoline, 0, NULL, 1, &h);
  arts_add_dependence(argsDbGuid, mainEdtGuid, 0, ARTS_MODE_RO);
}

int main(int argc, char **argv) { return arts_rt(argc, argv); }
