/*
 * OCR-to-ARTS Compatibility Shim
 *
 * Implements the OCR v1.2.0 API using ARTS primitives. OCR applications
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
#include <unistd.h>

/*
 * Both ARTS and OCR define enum values with identical names but different
 * semantics (DB_MODE_NULL, DB_MODE_RO, DB_MODE_RW, DB_MODE_RW).
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
#define DB_MODE_RW ARTS_DB_MODE_RW_
#define DB_MODE_VAL ARTS_DB_MODE_VAL_

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
#undef DB_MODE_RW
#undef DB_MODE_VAL

/* ARTS DB_MODE values used by the shim (sequential: NULL=0, RO=1, RW=2,
 * VAL=3). */
#define ARTS_MODE_NULL ((arts_db_access_mode_t)ARTS_DB_MODE_NULL_)
#define ARTS_MODE_RO ((arts_db_access_mode_t)ARTS_DB_MODE_RO_)
#define ARTS_MODE_RW ((arts_db_access_mode_t)ARTS_DB_MODE_RW_)
#define ARTS_MODE_VAL ((arts_db_access_mode_t)ARTS_DB_MODE_VAL_)

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

#if defined(__SANITIZE_ADDRESS__)
#define _ARTS_OCR_ASAN 1
#elif defined(__clang__)
#if __has_feature(address_sanitizer)
#define _ARTS_OCR_ASAN 1
#endif
#endif
#ifdef _ARTS_OCR_ASAN
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

  /* Fan out under lock so the next generation's add/satisfy cannot race
   * with our reset/fan-out.  External callbacks (arts_db_create,
   * arts_add_dependence, arts_event_satisfy_slot) only schedule work for
   * other threads; they never re-enter this collective's lock on the
   * current thread, so this cannot self-deadlock. */
  for (u32 i = 0; i < numDeps && i < MAX_COLLECTIVE_DEPENDENTS; i++) {
    if (localDeps[i] != NULL_GUID) {
      void *resultPtr;
      arts_guid_t resultDb = arts_db_create(
          &resultPtr, sizeof(double), ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, NULL);
      *(double *)resultPtr = result;

      arts_type_t dstType = arts_guid_get_type(localDeps[i]);
      if (dstType == ARTS_EDT) {
        arts_add_dependence(resultDb, localDeps[i], localSlots[i],
                            ARTS_MODE_RO);
      } else if (dstType == ARTS_EVENT) {
        arts_event_satisfy_slot(localDeps[i], resultDb,
                                ARTS_EVENT_LATCH_DECR_SLOT);
      }
    }
  }
}

#define COLLECTIVE_HASH_SIZE 4096

/*
 * Collective metadata registry — open-addressed linear-probing hash map
 * keyed by event GUID (the OCR-visible identifier) → metadata DB GUID
 * (the per-collective state struct).
 *
 * Concurrency model:
 *   - `edtGuid` is published via __sync_bool_compare_and_swap (full
 *     barrier).  `metaDbGuid` is a plain store BEFORE the CAS so the
 *     CAS itself is the publication point: any thread that observes the
 *     edtGuid is guaranteed (by the CAS's full barrier) to see the
 *     paired metaDbGuid.  This eliminates the previous lookup-side
 *     spin loop on metaDbGuid==NULL.
 *   - The TOMBSTONE marker lets unregister clear an entry without
 *     breaking the probe chain — subsequent lookups skip past
 *     tombstones, and inserts may reuse them.
 *
 * Result codes from tryRegisterCollectiveMeta let the caller distinguish
 * "newly registered", "already exists", and "table full" — instead of
 * collapsing the latter two into a single 0 return.
 */
#define COLLECTIVE_TOMBSTONE ((arts_guid_t) ~(uint64_t)0)

typedef struct {
  volatile arts_guid_t edtGuid;
  arts_guid_t metaDbGuid;
} CollectiveMapEntry;

static CollectiveMapEntry collectiveMetaMap[COLLECTIVE_HASH_SIZE] = {{0, 0}};

enum collective_register_result {
  COLLECTIVE_REGISTER_OK = 0,
  COLLECTIVE_REGISTER_EXISTS = 1,
  COLLECTIVE_REGISTER_FULL = 2,
};

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

static enum collective_register_result
tryRegisterCollectiveMeta(arts_guid_t key, arts_guid_t metaDbGuid) {
  u32 idx = collectiveHash(key);
  for (u32 i = 0; i < COLLECTIVE_HASH_SIZE; i++) {
    u32 probeIdx = (idx + i) % COLLECTIVE_HASH_SIZE;
    arts_guid_t cur = collectiveMetaMap[probeIdx].edtGuid;

    if (cur == NULL_GUID || cur == COLLECTIVE_TOMBSTONE) {
      /* Publish metaDbGuid BEFORE the CAS that publishes edtGuid.
       * The CAS is a full barrier, so any thread observing edtGuid
       * after the CAS is guaranteed to see this metaDbGuid write. */
      collectiveMetaMap[probeIdx].metaDbGuid = metaDbGuid;
      if (__sync_bool_compare_and_swap(&collectiveMetaMap[probeIdx].edtGuid,
                                       cur, key)) {
        return COLLECTIVE_REGISTER_OK;
      }
      /* CAS lost the race; the slot is now occupied by some other key.
       * The metaDbGuid we just wrote is harmless because the next
       * probe iteration will check (and possibly overwrite) it before
       * its own CAS. */
    }

    /* Plain volatile read is fine here because we only check equality
     * against `key`, and the writer published edtGuid via CAS. */
    if (collectiveMetaMap[probeIdx].edtGuid == key) {
      return COLLECTIVE_REGISTER_EXISTS;
    }
  }
  return COLLECTIVE_REGISTER_FULL;
}

static arts_guid_t lookupCollectiveMeta(arts_guid_t edtGuid) {
  u32 idx = collectiveHash(edtGuid);
  for (u32 i = 0; i < COLLECTIVE_HASH_SIZE; i++) {
    u32 probeIdx = (idx + i) % COLLECTIVE_HASH_SIZE;
    arts_guid_t cur = collectiveMetaMap[probeIdx].edtGuid;
    if (cur == edtGuid) {
      /* metaDbGuid was published BEFORE the CAS that set edtGuid; the
       * CAS's full barrier means our read of edtGuid synchronizes with
       * that prior write.  No spin needed. */
      return collectiveMetaMap[probeIdx].metaDbGuid;
    }
    if (cur == NULL_GUID) {
      /* End of probe chain — entry definitively not present. */
      return NULL_GUID;
    }
    /* TOMBSTONE: skip past, the chain continues. */
  }
  return NULL_GUID;
}

/*
 * Mark the entry for `edtGuid` as a tombstone so its slot can be reused
 * by future inserts while preserving the probe chain.  Used by
 * ocrEventDestroy to clean up after a collective event.  Returns the
 * removed metaDbGuid (or NULL_GUID if not found) so the caller can
 * destroy the metadata DB.
 */
static arts_guid_t unregisterCollectiveMeta(arts_guid_t edtGuid) {
  u32 idx = collectiveHash(edtGuid);
  for (u32 i = 0; i < COLLECTIVE_HASH_SIZE; i++) {
    u32 probeIdx = (idx + i) % COLLECTIVE_HASH_SIZE;
    arts_guid_t cur = collectiveMetaMap[probeIdx].edtGuid;
    if (cur == edtGuid) {
      arts_guid_t metaDb = collectiveMetaMap[probeIdx].metaDbGuid;
      collectiveMetaMap[probeIdx].metaDbGuid = NULL_GUID;
      collectiveMetaMap[probeIdx].edtGuid = COLLECTIVE_TOMBSTONE;
      return metaDb;
    }
    if (cur == NULL_GUID) {
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

/* Cross-rank-portable template GUID encoding.
 *
 * The OCR shim previously stored ocrEdtTemplate as a malloc'd struct
 * and used the heap pointer as the "GUID".  That works under
 * single-node OCR but breaks under fork-launched multinode: one
 * rank's heap pointer is meaningless on another, so any cross-rank
 * EDT_MOVE that carries a template GUID dereferences garbage.
 *
 * Benchmarks build non-PIE, so every funcPtr is a fixed absolute
 * address — identical across forked ranks.  Pack the entire template
 * into the GUID itself:
 *
 *   bits 63-48 (16) = depc   (0xFFFF = EDT_PARAM_UNK; max non-UNK 65534)
 *   bits 47-32 (16) = paramc (0xFFFF = EDT_PARAM_UNK; max non-UNK 65534)
 *   bits 31-0  (32) = funcPtr (4 GiB; non-PIE x86-64 .text segments
 *                     are well below this in practice)
 *
 * 16 bits each for paramc/depc gives headroom for templates that
 * dynamically size depc to thousands; a narrower split silently
 * truncates those and strands the consumer.
 *
 * No magic tag is needed: callers always know the GUID came from
 * ocrEdtTemplateCreate when they pass it to ocrEdtCreate or
 * ocrEdtTemplateDestroy, so a runtime distinction from "other"
 * GUIDs is unnecessary. */
#define ARTS_TPL_FUNCPTR_BITS 32
#define ARTS_TPL_FUNCPTR_MASK ((1ULL << ARTS_TPL_FUNCPTR_BITS) - 1)
#define ARTS_TPL_PARAMC_SHIFT ARTS_TPL_FUNCPTR_BITS
#define ARTS_TPL_DEPC_SHIFT (ARTS_TPL_PARAMC_SHIFT + 16)
#define ARTS_TPL_COUNT_UNK 0xFFFFu
#define ARTS_TPL_COUNT_MAX 0xFFFEu

static inline OcrEdtTemplate arts_tpl_decode(ocrGuid_t g) {
  uint64_t v = (uint64_t)g.guid;
  OcrEdtTemplate t;
  t.funcPtr = (ocrEdt_t)(uintptr_t)(v & ARTS_TPL_FUNCPTR_MASK);
  uint32_t enc_paramc = (uint32_t)((v >> ARTS_TPL_PARAMC_SHIFT) & 0xFFFFu);
  uint32_t enc_depc = (uint32_t)((v >> ARTS_TPL_DEPC_SHIFT) & 0xFFFFu);
  /* 0xFFFF sentinel = EDT_PARAM_UNK (caller MUST provide explicit value
   * at ocrEdtCreate; passing EDT_PARAM_DEF here is an OCR-app bug). */
  t.paramc = (enc_paramc == ARTS_TPL_COUNT_UNK) ? EDT_PARAM_UNK : enc_paramc;
  t.depc = (enc_depc == ARTS_TPL_COUNT_UNK) ? EDT_PARAM_UNK : enc_depc;
  return t;
}

u8 ocrEdtTemplateCreate_internal(ocrGuid_t *guid, ocrEdt_t funcPtr, u32 paramc,
                                 u32 depc, const char *funcName) {
  (void)funcName;
  uint64_t fp = (uint64_t)(uintptr_t)funcPtr;
  if ((fp & ~ARTS_TPL_FUNCPTR_MASK) != 0) {
    /* funcPtr beyond 32 bits — non-PIE x86-64 .text never reaches this
     * boundary in practice; treat as a build-time invariant violation. */
    return OCR_EINVAL;
  }
  /* OCR allows EDT_PARAM_UNK ((u32)-1) for paramc/depc when the count is
   * dynamic at template creation but provided explicitly at every
   * ocrEdtCreate call (reductionLaunch is the canonical user).  Encode
   * EDT_PARAM_UNK as the 0xFFFF sentinel; non-UNK values must fit in 16
   * bits (max 65534, i.e. ARTS_TPL_COUNT_MAX). */
  uint16_t enc_paramc;
  if (paramc == EDT_PARAM_UNK) {
    enc_paramc = ARTS_TPL_COUNT_UNK;
  } else if (paramc > ARTS_TPL_COUNT_MAX) {
    return OCR_EINVAL;
  } else {
    enc_paramc = (uint16_t)paramc;
  }
  uint16_t enc_depc;
  if (depc == EDT_PARAM_UNK) {
    enc_depc = ARTS_TPL_COUNT_UNK;
  } else if (depc > ARTS_TPL_COUNT_MAX) {
    return OCR_EINVAL;
  } else {
    enc_depc = (uint16_t)depc;
  }
  uint64_t v = fp | ((uint64_t)enc_paramc << ARTS_TPL_PARAMC_SHIFT) |
               ((uint64_t)enc_depc << ARTS_TPL_DEPC_SHIFT);
  guid->guid = (intptr_t)v;
  return 0;
}

u8 ocrEdtTemplateDestroy(ocrGuid_t guid) {
  /* Encoded GUID has no backing allocation. */
  (void)guid;
  return 0;
}

/* Forward declaration: defined alongside ocr_to_arts_mode further down. */
static u32 arts_to_ocr_mode(arts_db_access_mode_t arts_mode);

/* Forward declaration: defined alongside the ELS storage further down.
 * Called at the start of every trampoline so ELS truly is "EDT-local"
 * (zero-initialized at the start of each EDT) instead of leaking
 * stale values from the previous EDT that ran on the same worker. */
static void ocr_els_reset(void);

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

  /* Restore EDT-local semantics for the OCR ELS array — see ocr_els_reset. */
  ocr_els_reset();

  ocrEdt_t func = (ocrEdt_t)paramv[0];
  u32 origParamc = (u32)paramv[1];
  arts_guid_t guidOrEpoch = (arts_guid_t)paramv[2];
  arts_guid_t helperOrOutEvt = (arts_guid_t)paramv[3];
  u64 flags = paramv[4];
  /* ARTS paramv is const; copy original params for OCR's non-const API */
  u64 *origParamv = NULL;
  u64 origParamBuf[origParamc > 0 ? origParamc : 1];
  if (origParamc > 0) {
    memcpy(origParamBuf, &paramv[5], origParamc * sizeof(u64));
    origParamv = origParamBuf;
  }

  bool isFinishEdt = (flags & FINISH_EDT_FLAG) != 0;

  /* Convert arts_edt_dep_t to ocrEdtDep_t.  Preserve the mode that ARTS
   * resolved during acquire_dbs (RO vs EW vs NULL) — the OCR EDT body
   * may inspect depv[i].mode for assertions or behavior, and surfacing
   * the actual ARTS-resolved mode is more honest than the previous
   * DB_DEFAULT_MODE hardcode. */
  ocrEdtDep_t ocrDepv[depc > 0 ? depc : 1];

  for (u32 i = 0; i < depc; i++) {
    ocrDepv[i].guid.guid = depv[i].guid;
    ocrDepv[i].ptr = depv[i].ptr;
    ocrDepv[i].mode = arts_to_ocr_mode(depv[i].mode);
  }

  if (isFinishEdt && guidOrEpoch != NULL_GUID) {
    /* Push the finish epoch onto the TLS stack and increment active_count.
     *
     * arts_edt_create_with_epoch() does NOT assign the finish epoch to
     * edt->epoch_guid — because the epoch was just created in ocrEdtCreate()
     * and is NOT on the caller's TLS stack, arts_check_epoch_is_root() fails,
     * and the EDT gets the caller's epoch instead (often NULL_GUID).
     *
     * arts_epoch_start() must be called here to:
     *   (a) push the finish epoch onto TLS so child EDTs inherit it, and
     *   (b) increment active_count by 1, matching the +1 finished_count
     *       that arts_increment_finished_epoch_list() adds for this entry
     *       when the trampoline EDT completes.
     *
     * Net accounting: active = 1 (this call) + N (children), finished =
     * 1 (this TLS entry) + N (children) → epoch fires when all complete. */
    arts_epoch_start(guidOrEpoch);
  }

  ocrGuid_t returnGuid = func(origParamc, origParamv, depc, ocrDepv);

  if (!isFinishEdt && helperOrOutEvt != NULL_GUID) {
    arts_event_satisfy_slot(helperOrOutEvt, returnGuid.guid,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }

  if (isFinishEdt && helperOrOutEvt != NULL_GUID) {
    if (returnGuid.guid != NULL_GUID) {
      arts_add_dependence(returnGuid.guid, helperOrOutEvt, 1, ARTS_MODE_RO);
    } else {
      arts_add_dependence((arts_guid_t)(0), helperOrOutEvt, 1, ARTS_MODE_VAL);
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

/* One-shot warning when an affinity hint exceeds the available rank
 * count.  Modulo wrap is preserved for compatibility, but the silent
 * wrap can mask app affinity bugs (a typo'd hint silently maps to a
 * different rank).  Warn once globally so the developer notices. */
static void warn_oversized_affinity_once(const char *what, u64 val) {
  static volatile u32 warned = 0;
  if (__sync_bool_compare_and_swap(&warned, 0, 1)) {
    fprintf(stderr,
            "[ocr_shim] %s affinity hint %lu exceeds rank count %u, "
            "wrapping via modulo.  Subsequent oversized hints suppressed.\n",
            what, (unsigned long)val, arts_global_rank_count);
  }
}

/* Route extraction for OCR EDT/DB creation.
 *
 * The shim is a thin wrapper: distribution decisions belong to ARTS
 * runtime, not here.  These helpers only translate an explicit
 * OCR_HINT_*_AFFINITY value into an ARTS rank.  When no affinity hint
 * is set, callers fall back to ARTS's own defaults:
 *   - arts_edt_create: hint=NULL → self-rank (caller-local EDT)
 *   - arts_db_create:  hint=NULL → round-robin starting at self-rank
 *                                  (atomic counter, see db.c)
 *
 * Both helpers return -1 when no affinity hint is set, signaling to the
 * caller "no override; let ARTS decide". */
static int extract_edt_affinity(ocrHint_t *hint) {
  if (hint == NULL || hint->type != OCR_HINT_EDT_T) {
    return -1;
  }
  int idx = OCR_HINT_EDT_AFFINITY - OCR_HINT_EDT_PROP_START - 1;
  if (idx < 0 || !(hint->propMask & (1ULL << idx))) {
    return -1;
  }
  u64 val = hint->args.propEDT[idx];
  if (val >= arts_global_rank_count) {
    warn_oversized_affinity_once("EDT", val);
  }
  return (int)(val % arts_global_rank_count);
}

static int extract_db_affinity(ocrHint_t *hint) {
  if (hint == NULL || hint->type != OCR_HINT_DB_T) {
    return -1;
  }
  int idx = OCR_HINT_DB_AFFINITY - OCR_HINT_DB_PROP_START - 1;
  if (idx < 0 || !(hint->propMask & (1ULL << idx))) {
    return -1;
  }
  u64 val = hint->args.propDB[idx];
  if (val >= arts_global_rank_count) {
    warn_oversized_affinity_once("DB", val);
  }
  return (int)(val % arts_global_rank_count);
}

/* =========================================================================
 * EDT Creation and Management
 * ========================================================================= */

u8 ocrEdtCreate(ocrGuid_t *guid, ocrGuid_t templateGuid, u32 paramc,
                u64 *paramv, u32 depc, ocrGuid_t *depv, u16 properties,
                ocrHint_t *hint, ocrGuid_t *outputEvent) {
  /* Decode the cross-rank-portable template GUID (see arts_tpl_*
   * helpers above).  templateGuid carries the full (funcPtr, paramc,
   * depc) tuple and is identical on every rank thanks to non-PIE
   * binary loading.  Reject NULL_GUID early so funcPtr=NULL never
   * reaches the EDT trampoline. */
  if (templateGuid.guid == 0) {
    return OCR_EINVAL;
  }
  OcrEdtTemplate templ_local = arts_tpl_decode(templateGuid);
  OcrEdtTemplate *templ = &templ_local;

  u32 actualParamc = (paramc == EDT_PARAM_DEF) ? templ->paramc : paramc;
  u32 actualDepc = (depc == EDT_PARAM_DEF) ? templ->depc : depc;

  /* hint affinity → ARTS rank, else self-rank (matches arts_edt_create's
   * hint=NULL fallback).  ARTS does NOT round-robin EDTs by default — the
   * intended policy is "EDT runs on the calling rank unless the user asks
   * otherwise." */
  int aff = extract_edt_affinity(hint);
  unsigned int rank = (aff < 0) ? arts_global_rank_id : (unsigned int)aff;
  arts_guid_t outEvt = NULL_GUID;
  arts_guid_t epochGuid = NULL_GUID;
  bool isFinishEdt = (properties & EDT_PROP_FINISH) != 0;
  bool oevtValid = (properties & EDT_PROP_OEVT_VALID) != 0;

  if (outputEvent != NULL) {
    if (oevtValid) {
      outEvt = outputEvent->guid;
    } else {
      arts_event_hint_t h = ARTS_EVENT_HINT_DEFAULTS;
      h.rank = rank;
      h.auto_destroy = false; /* IDEM: persist for late ocrAddDependence */
      outEvt = arts_event_create(&h);
      if (outEvt == NULL_GUID) {
        return OCR_ENOMEM;
      }
      outputEvent->guid = outEvt;
    }
  }

  arts_guid_t helperEdtGuid = NULL_GUID;
  if (isFinishEdt && outEvt != NULL_GUID) {
    uint64_t helperParams[1];
    helperParams[0] = (uint64_t)outEvt;

    arts_edt_hint_t h = {.rank = rank};
    helperEdtGuid =
        arts_edt_create(epoch_termination_edt, 1, helperParams, 2, &h);

    epochGuid = arts_epoch_create(rank, helperEdtGuid, 0);
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
  arts_edt_hint_t edtHint = {.rank = rank};

  if (isFinishEdt && epochGuid != NULL_GUID) {
    edtHint.epoch = epochGuid;
  }
  edtGuid = arts_edt_create(ocr_edt_trampoline, artsParamc, artsParamv,
                            actualDepc, &edtHint);

  arts_free(artsParamv);

  if (edtGuid == NULL_GUID) {
    /* ARTS could not create the EDT (route invalid, OOM, etc.).  Surface
     * the failure as OCR_ENOMEM rather than silently returning success
     * with a bogus GUID. */
    return OCR_ENOMEM;
  }

  if (guid != NULL) {
    guid->guid = edtGuid;
  }

  if (depv != NULL && actualDepc > 0) {
    for (u32 i = 0; i < actualDepc; i++) {
      if (ocrGuidIsNull(depv[i])) {
        /* NULL_GUID = pre-satisfied slot (OCR spec §2.4.3).
         * Signal immediately so the EDT doesn't wait forever. */
        arts_add_dependence((arts_guid_t)(0), edtGuid, i, ARTS_MODE_VAL);
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

/*
 * Map an OCR event flavor + property bits onto a hint snapshot for the
 * unified arts_event_create API.  All OCR
 * flavors collapse onto a single ARTS event type; behavior is selected
 * entirely via hint fields (latch / auto_destroy / multiple_fire / etc.).
 *
 * Default hint = OCR ONCE_T (latch=1, auto_destroy=true, single fire).
 * Each case overrides only the fields that diverge.
 */
static arts_event_hint_t ocr_event_kind_to_hint(ocrEventTypes_t kind,
                                                u16 properties) {
  (void)properties;
  arts_event_hint_t h = ARTS_EVENT_HINT_DEFAULTS;
  switch (kind) {
  case OCR_EVENT_ONCE_T: /* defaults */
    break;
  case OCR_EVENT_IDEM_T:
    h.auto_destroy = false;
    break;
  case OCR_EVENT_STICKY_T:
    h.auto_destroy = false;
    h.negative_latch_allowed = false;
    break;
  case OCR_EVENT_LATCH_T:
    h.latch = 0; /* caller may override via params */
    break;
  case OCR_EVENT_COUNTED_T:
    h.auto_destroy = false;
    break;
  case OCR_EVENT_CHANNEL_T:
    h.multiple_fire = true;
    h.latch = 1;
    h.nb_deps_required = 1;
    break;
  default:
    break;
  }
  return h;
}

u8 ocrEventCreate(ocrGuid_t *guid, ocrEventTypes_t eventType, u16 properties) {
  arts_event_hint_t h = ocr_event_kind_to_hint(eventType, properties);
  if (properties & GUID_PROP_IS_LABELED) {
    h.guid = guid->guid;
    arts_guid_t result = arts_event_create(&h);
    if (result == NULL_GUID && (properties & GUID_PROP_CHECK)) {
      return OCR_EGUIDEXISTS;
    }
    return 0;
  }
  arts_guid_t g = arts_event_create(&h);
  if (g == NULL_GUID)
    return OCR_ENOMEM;
  guid->guid = g;
  return 0;
}

u8 ocrEventDestroy(ocrGuid_t guid) {
  /* If this GUID was used as a key for a collective event metadata
   * entry, drop the entry and destroy the metadata DB.  Two cases:
   *   - Labeled collective: the entry key is the labeled GUID, and the
   *     OCR-visible event GUID equals the labeled GUID; arts_event_destroy
   *     handles the user-visible event side.
   *   - Unlabeled collective: the entry key IS the metadata DB GUID,
   *     and the OCR-visible event GUID is also the metadata DB GUID
   *     (it's not a real ARTS event, but arts_event_destroy on a
   *     non-event GUID is harmless / a no-op).
   * In either case, unregisterCollectiveMeta is the safe lookup that
   * also clears the hash entry via tombstone marker. */
  arts_guid_t metaDb = unregisterCollectiveMeta(guid.guid);
  if (metaDb != NULL_GUID) {
    arts_db_destroy(metaDb);
  }
  arts_event_destroy(guid.guid);
  return 0;
}

u8 ocrEventSatisfy(ocrGuid_t eventGuid, ocrGuid_t dataGuid) {
  arts_event_satisfy_slot(eventGuid.guid, dataGuid.guid,
                          ARTS_EVENT_LATCH_DECR_SLOT);
  return 0;
}

u8 ocrEventSatisfySlot(ocrGuid_t eventGuid, ocrGuid_t dataGuid, u32 slot) {
  arts_event_satisfy_slot(eventGuid.guid, dataGuid.guid, slot);
  return 0;
}

u8 ocrEventCreateParams(ocrGuid_t *guid, ocrEventTypes_t eventType,
                        u16 properties, ocrEventParams_t *params) {

  if (eventType == OCR_EVENT_COLLECTIVE_T && params != NULL) {
    u32 nbContribs = params->EVENT_COLLECTIVE.nbContribs;
    arts_guid_t labeledGuid = guid->guid;

    /* Labeled fast-path: if a metadata entry already exists for this
     * GUID, surface OCR_EGUIDEXISTS when the caller asked via
     * GUID_PROP_CHECK; otherwise treat the second create as a no-op. */
    if ((properties & GUID_PROP_IS_LABELED) && labeledGuid != NULL_GUID) {
      arts_guid_t existingMeta = lookupCollectiveMeta(labeledGuid);
      if (existingMeta != NULL_GUID) {
        return (properties & GUID_PROP_CHECK) ? OCR_EGUIDEXISTS : 0;
      }
    }

    void *metaPtr;
    /* PIN: Collective metadata is node-local shared state accessed by all
     * contributing EDTs without going through RC acquire/release cycles.
     * RC type would give each EDT its own working-copy view, so writes
     * from the creator EDT (nbContribs etc.) would not be visible to
     * subsequent satisfy/addDep callers — they would see zeros and the
     * reduction would never fire (count == nbContribs == 0). */
    arts_guid_t metaDb = arts_db_create(&metaPtr, sizeof(CollectiveMetadata),
                                        ARTS_DB_PIN, ARTS_DB_PROP_NONE, NULL);
    if (metaDb == NULL_GUID) {
      return OCR_ENOMEM;
    }
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

    arts_guid_t key =
        ((properties & GUID_PROP_IS_LABELED) && labeledGuid != NULL_GUID)
            ? labeledGuid
            : metaDb;

    enum collective_register_result reg =
        tryRegisterCollectiveMeta(key, metaDb);
    if (reg == COLLECTIVE_REGISTER_FULL) {
      arts_db_destroy(metaDb);
      return OCR_ENOSPC;
    }
    if (reg == COLLECTIVE_REGISTER_EXISTS) {
      arts_db_destroy(metaDb);
      return (properties & GUID_PROP_CHECK) ? OCR_EGUIDEXISTS : 0;
    }

    /* For unlabeled collective events the OCR-visible event GUID is
     * the metadata DB GUID itself.  For labeled events, the user-
     * provided GUID is the public identity. */
    if (!((properties & GUID_PROP_IS_LABELED) && labeledGuid != NULL_GUID)) {
      guid->guid = metaDb;
    }
    return 0;
  }

  arts_event_hint_t h = ocr_event_kind_to_hint(eventType, properties);

  if (eventType == OCR_EVENT_LATCH_T && params != NULL) {
    h.latch = (int32_t)params->EVENT_LATCH.counter;
  }
  if (eventType == OCR_EVENT_COUNTED_T && params != NULL) {
    h.nb_deps_required = 1;
    h.max_nb_deps = (uint32_t)params->EVENT_COUNTED.nbDeps;
  }
  if (eventType == OCR_EVENT_CHANNEL_T && params != NULL) {
    /* OCR 1.2 §B.5.2: nbSat and nbDeps are restricted to 1.  ARTS enforces
     * that constraint at the shim — generalized values would require a
     * non-trivial change to the channel drain loop (currently fires one
     * data-dep pair per generation). */
    if (params->EVENT_CHANNEL.nbSat != 1 || params->EVENT_CHANNEL.nbDeps != 1) {
      fprintf(stderr,
              "[ARTS] CHANNEL nbSat=%u nbDeps=%u: only nbSat=nbDeps=1 "
              "supported (OCR 1.2 §B.5.2)\n",
              params->EVENT_CHANNEL.nbSat, params->EVENT_CHANNEL.nbDeps);
      return OCR_EINVAL;
    }
    /* maxGen is implementation-driven — ARTS scales unbounded via mpsc. */
  }

  if (properties & GUID_PROP_IS_LABELED) {
    h.guid = guid->guid;
    arts_guid_t result = arts_event_create(&h);
    if (result == NULL_GUID && (properties & GUID_PROP_CHECK)) {
      return OCR_EGUIDEXISTS;
    }
    return 0;
  }
  arts_guid_t g = arts_event_create(&h);
  if (g == NULL_GUID)
    return OCR_ENOMEM;
  guid->guid = g;
  return 0;
}

u8 ocrEventCollectiveSatisfySlot(ocrGuid_t eventGuid, void *dataPtr,
                                 u32 islot) {
  arts_guid_t metaDbGuid = lookupCollectiveMeta(eventGuid.guid);

  if (metaDbGuid == NULL_GUID) {
    arts_guid_t dataGuid =
        (dataPtr != NULL) ? (arts_guid_t)(uintptr_t)dataPtr : NULL_GUID;
    arts_event_satisfy_slot(eventGuid.guid, dataGuid, islot);
    return 0;
  }

  /* Hold the route table lookup ref for the entire critical section so
   * the metadata DB cannot be destroyed (e.g., by a concurrent
   * ocrEventDestroy) while we're touching its fields.  paired
   * arts_route_table_release at every exit. */
  struct arts_db_s *raw = arts_route_table_lookup_db_safe(metaDbGuid);
  if (raw == NULL) {
    return OCR_EFAULT;
  }
  CollectiveMetadata *meta = (CollectiveMetadata *)(raw + 1);

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
  arts_route_table_release(metaDbGuid);

  return 0;
}

/* =========================================================================
 * Data Block Management
 * ========================================================================= */

u8 ocrDbCreate(ocrGuid_t *db, void **addr, u64 len, u16 flags, ocrHint_t *hint,
               ocrInDbAllocator_t allocator) {
  (void)allocator;
  int aff = extract_db_affinity(hint);

  if (flags & GUID_PROP_IS_LABELED) {
    arts_guid_t labeledGuid = db->guid;

    void *data = arts_db_create_with_guid(labeledGuid, len, ARTS_DB_DEFAULT,
                                          ARTS_DB_PROP_NONE, NULL);
    if (data == NULL) {
      /* Labeled GUID already taken — fall back to looking it up so the
       * caller still gets a valid pointer.  lookup_db_safe pairs
       * with release immediately (the descriptor lifetime is owned by
       * route_table; the user data pointer remains valid because the DB
       * itself wasn't destroyed). */
      struct arts_db_s *db_existing =
          arts_route_table_lookup_db_safe(labeledGuid);
      if (db_existing != NULL) {
        *addr = (void *)(db_existing + 1);
        arts_route_table_release(labeledGuid);
        /* Match ocrEventCreate's labeling convention: only surface
         * EGUIDEXISTS when the caller asked to be told via GUID_PROP_CHECK. */
        return (flags & GUID_PROP_CHECK) ? OCR_EGUIDEXISTS : 0;
      }
      return OCR_ENOMEM;
    }
    *addr = data;
    return 0;
  }

  /* No affinity hint → pass NULL to arts_db_create so its built-in
   * round-robin (atomic counter starting at self-rank, see db.c) takes
   * effect.  Explicit affinity → wrap in arts_db_hint_t. */
  arts_db_hint_t artsHint;
  const arts_db_hint_t *hintp = NULL;
  if (aff >= 0) {
    artsHint = (arts_db_hint_t){.rank = (unsigned int)aff};
    hintp = &artsHint;
  }
  db->guid =
      arts_db_create(addr, len, ARTS_DB_DEFAULT, ARTS_DB_PROP_NONE, hintp);
  if (db->guid == NULL_GUID) {
    return OCR_ENOMEM;
  }

  return 0;
}

u8 ocrDbDestroy(ocrGuid_t db) {
  /* OCR spec: ocrDbDestroy marks the DB for destruction.
   *
   * Several real-world OCR apps call ocrDbDestroy on intermediate DBs
   * while later sibling EDTs still hold add_dependence wirings to the
   * same GUIDs.  That is use-after-destroy by OCR spec, but the
   * pattern is entrenched in shipped apps.  The underlying runtime
   * propagates the destroyed state correctly (NULL data in the route
   * entry, DB_DESTROYED waking parked waiters), but app bodies that
   * dereference depv[slot].ptr unconditionally would still segfault.
   *
   * For pragmatic shim compatibility, skip the destroy and let the DB
   * live until process-exit cleanup.  This trades a bounded memory
   * leak (sized to the app's working set) for OCR-app correctness.
   * arts_db_destroy is still reachable from the collective reduction
   * path in this file, where the lifecycle is shim-internal and
   * well-formed. */
  (void)db;
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
 * NOT ARTS values.  Always use ARTS_MODE_RO/ARTS_MODE_RW for ARTS values.
 */
static arts_db_access_mode_t ocr_to_arts_mode(ocrDbAccessMode_t ocr_mode) {
  switch (ocr_mode) {
  case DB_MODE_EW: /* OCR 0x4 → ARTS RW */
    return ARTS_MODE_RW;
  case DB_MODE_RW: /* OCR 0x2 → ARTS RW */
    return ARTS_MODE_RW;
  case DB_MODE_NULL: /* OCR 0x0 → ARTS NULL (control-only dependence) */
    return ARTS_MODE_NULL;
  case DB_MODE_RO: /* OCR 0x8 → ARTS RO */
  default:
    return ARTS_MODE_RO;
  }
}

/*
 * Inverse mapping for what the EDT body sees in depv[i].mode.
 *
 * ARTS resolves the actual access mode during acquire_dbs.  We surface
 * that to the OCR EDT body so user code (and OCR helper libraries that
 * read depv[i].mode for assertions or branching) sees the truth.
 *
 * RO is reported as DB_MODE_RO rather than DB_DEFAULT_MODE (RW) because
 * the OCR-RW-mapped-to-ARTS-RO path doesn't survive the round trip and
 * we have no way to distinguish "originally RW" from "originally RO".
 * Reporting RO is conservative and matches what acquire_dbs actually did.
 */
static u32 arts_to_ocr_mode(arts_db_access_mode_t arts_mode) {
  switch (arts_mode) {
  case ARTS_DB_MODE_RW_:
    return DB_MODE_RW;
  case ARTS_DB_MODE_NULL_:
    return DB_MODE_NULL;
  case ARTS_DB_MODE_RO_:
  default:
    return DB_MODE_RO;
  }
}

u8 ocrAddDependence(ocrGuid_t source, ocrGuid_t destination, u32 slot,
                    ocrDbAccessMode_t mode) {

  /* NULL source → signal immediately (slot satisfied with no data). */
  if (ocrGuidIsNull(source)) {
    arts_type_t dstType = arts_guid_get_type(destination.guid);
    if (dstType == ARTS_EDT) {
      arts_add_dependence((arts_guid_t)(0), destination.guid, slot,
                          ARTS_MODE_VAL);
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
     * are handled by arts_add_dependence. Cross-node: handled natively.
     *
     * Pass the user's access mode through ocr_to_arts_mode so the EDT slot
     * is registered with the correct ARTS mode (RO/EW).  Previously this
     * was hardcoded to ARTS_MODE_RO, silently downgrading every Event→EDT
     * dependence to read-only access. */
    if (dstType == ARTS_EDT) {
      arts_add_dependence(source.guid, destination.guid, slot,
                          ocr_to_arts_mode(mode));
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
    /* Hold the route table ref for the duration we touch the metadata.
     * paired arts_route_table_release at every exit. */
    struct arts_db_s *raw = arts_route_table_lookup_db_safe(metaDbGuid);
    if (raw == NULL) {
      return OCR_EFAULT;
    }
    CollectiveMetadata *meta = (CollectiveMetadata *)(raw + 1);
    /* Hold meta->lock so the (numDependents++, dependents[idx]=guid) pair is
     * atomic w.r.t. performCollectiveReduction, which reads numDependents
     * and dependents[] under the same lock.  Without this, the last
     * contributor could observe an incremented count but a not-yet-written
     * dependents[idx]==NULL_GUID slot and skip that rank in the fan-out. */
    pthread_mutex_lock(&meta->lock);
    u32 idx = meta->numDependents;
    if (idx >= MAX_COLLECTIVE_DEPENDENTS) {
      pthread_mutex_unlock(&meta->lock);
      arts_route_table_release(metaDbGuid);
      return OCR_ENOSPC;
    }
    meta->dependents[idx] = destination.guid;
    meta->dependentSlots[idx] = dslot;
    meta->numDependents = idx + 1;
    pthread_mutex_unlock(&meta->lock);
    arts_route_table_release(metaDbGuid);
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
  /* Use a stack buffer + single write() syscall so concurrent threads
   * cannot interleave mid-line and we never touch glibc's internal
   * FILE* locks, which can deadlock when fflush(0) runs concurrently
   * with another thread holding the FILE* lock (observed under heavy
   * multinode I/O in forked children). */
  char buffer[4096];
  int prefix_len =
      snprintf(buffer, sizeof(buffer), " [%u] ", arts_global_rank_id);
  if (prefix_len < 0 || prefix_len >= (int)sizeof(buffer)) {
    return 0;
  }
  va_list args;
  va_start(args, fmt);
  int body_len =
      vsnprintf(buffer + prefix_len, sizeof(buffer) - prefix_len, fmt, args);
  va_end(args);
  if (body_len < 0) {
    return 0;
  }
  int total = prefix_len + body_len;
  if (total > (int)sizeof(buffer)) {
    total = (int)sizeof(buffer);
  }
  (void)write(STDOUT_FILENO, buffer, (size_t)total);
  return (u32)body_len;
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

/* OCR EDT-Local Storage is supposed to be EDT-scoped: each EDT sees a
 * fresh, zeroed array.  We back it with _Thread_local for performance,
 * which is per-worker, so a stale value from a previously-finished EDT
 * on the same worker would otherwise leak into the next EDT.
 * Trampolines call this at entry to restore EDT-local semantics. */
static void ocr_els_reset(void) {
  memset((void *)els_storage, 0, sizeof(els_storage));
}

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
    /* Unknown OCR GUID kind.  Return ARTS_LAST_TYPE (out-of-range
     * sentinel) so the downstream arts_guid_reserve_range() rejects it
     * via its `type >= ARTS_LAST_TYPE` validation rather than silently
     * producing a range with type bits 0 (which is now ARTS_EDT). */
    return ARTS_LAST_TYPE;
  }
}

u8 ocrGuidRangeCreate(ocrGuid_t *rangeGuid, u64 numberGuid,
                      ocrGuidUserKind kind) {
  if (!rangeGuid || numberGuid == 0) {
    return 1;
  }
  arts_type_t artsType = kindToArtsType(kind);

  /* OCR semantics: any EDT calling this with the same input must end up with
   * the same range GUID, and ocrGuidFromIndex(range, idx) must yield the same
   * GUID on every rank.  Use ARTS's distributed-range mode so that (a) homes
   * are spread across ranks (idx % nrank) and (b) ocrGuidFromIndex is
   * deterministic across ranks once the range GUID is broadcast (the typical
   * mainEdt-creates-and-distributes-via-DB pattern). */
  arts_guid_t range = arts_guid_reserve_range(
      artsType, (unsigned int)numberGuid, ARTS_HINT_ROUND_ROBIN);
  rangeGuid->guid = range;
  return 0;
}

/*
 * ocrGuidMapDestroy: OCR releases a previously-reserved GUID range/map.
 * ARTS exposes arts_guid_reserve_range but no matching unreserve API,
 * so this is a no-op.  Long-running apps that repeatedly create+destroy
 * GUID ranges will leak ARTS GUID space.  See plan finding L3.
 *
 * TODO: implement arts_guid_unreserve_range in core ARTS, then plumb
 * it through here.
 */
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
    *count = (u64)arts_get_total_ranks();
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

/*
 * ocrSetHint / ocrGetHint apply hints to existing GUIDs (post-creation).
 * ARTS treats EDTs as immutable after creation: the route/hint passed at
 * arts_edt_create time is final.  DBs in principle could be relocated via
 * arts_db_move, but the current shim doesn't translate ocrSetHint to it.
 *
 * Returning 0 means "success" — apps that depend on dynamic hint changes
 * will silently get suboptimal placement.  See plan finding I3.
 *
 * TODO: implement DB hint changes via arts_db_move when an OCR app
 * actually exercises this path.
 */
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

  ocr_els_reset();

  ocrEdtDep_t ocrDepv[depc > 0 ? depc : 1];
  for (u32 i = 0; i < depc; i++) {
    ocrDepv[i].guid.guid = depv[i].guid;
    ocrDepv[i].ptr = depv[i].ptr;
    ocrDepv[i].mode = arts_to_ocr_mode(depv[i].mode);
  }

  mainEdt(0, NULL, depc, ocrDepv);
}

/* ARTS main_edt entry point */
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
  arts_guid_t argsDbGuid = arts_db_create(&dbPtr, totalSize, ARTS_DB_DEFAULT,
                                          ARTS_DB_PROP_NONE, NULL);

  u64 *header = (u64 *)dbPtr;
  header[0] = (u64)argc;

  size_t currentOffset = headerSize;
  for (int i = 0; i < argc; i++) {
    header[i + 1] = currentOffset;
    size_t len = strlen(argv[i]) + 1;
    memcpy((u8 *)dbPtr + currentOffset, argv[i], len);
    currentOffset += len;
  }

  arts_edt_hint_t h = {.rank = arts_global_rank_id};
  arts_guid_t mainEdtGuid = arts_edt_create(mainEdtTrampoline, 0, NULL, 1, &h);
  arts_add_dependence(argsDbGuid, mainEdtGuid, 0, ARTS_MODE_RO);
}

int main(int argc, char **argv) { return arts_rt(argc, argv); }
