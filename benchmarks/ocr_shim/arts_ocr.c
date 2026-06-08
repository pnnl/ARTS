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
#include "arts/edt.h"
#include "arts/gas/guid.h"
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
 * Collective Event Support — cross-rank ARITY=2 reduction tree
 *
 * OCR collective events (OCR_EVENT_COLLECTIVE_T) emulate an MPI-style
 * Allreduce/Reduce/Broadcast over `nbContribs` contributors, each
 * identified by a contributor index (its `islot` in the satisfy call).
 * The single-event-per-process registry of the previous implementation
 * could not work across nodes: each rank only saw its own contributions,
 * so the reduction never reached the global count.  This implementation
 * builds a binary reduction tree whose edges are cross-rank ARTS events,
 * so contributions flow up to the root and the result broadcasts back
 * down regardless of which node hosts which contributor.
 *
 * Tree topology (ARITY=2, indices 0..nrank-1):
 *   parent(r)   = (r-1)/2
 *   children(r) = { 2r+1, 2r+2 } that are < nrank
 *   root        = 0
 *
 * Up-phase   : each node reduces its own datum with each child's partial
 *              (delivered via that child's up-edge event) and forwards the
 *              partial to its parent's up-edge event.  The root ends with
 *              the full reduction.
 * Down-phase : (ALLREDUCE / BROADCAST) the root seeds its own down-edge
 *              with the result; each node forwards the result to its
 *              children's down-edges and delivers it to the locally-
 *              registered dependent for that generation.
 *
 * Re-arm: hpcg reuses the SAME labeled redEvtGuid across every CG phase of
 * every timestep, so the event must support an unbounded number of
 * generations.  Each contributor maintains a node-local generation counter
 * (its k-th satisfy == generation k); because every contributor takes part
 * in every reduction exactly once, the same generation number denotes the
 * same logical reduction on every rank.  Edge event GUIDs are derived from
 * (coll_guid, generation, contributor index, direction) so they are
 * identical on every rank with no explicit GUID exchange (see G1 below).
 *
 * G1 — cross-rank-consistent edge GUIDs:
 *   coll_guid is cross-rank-consistent (the app derives it via a LABELED
 *   GUID-range index, so all ranks agree).  Edge GUIDs are built directly
 *   from coll_guid with ARTS_GUID_MAKE.  coll_guid carries a concrete
 *   home rank (not the round-robin sentinel), so arts_guid_from_index would
 *   collapse to plain addition and collide with sibling GUIDs; instead we
 *   compute a deterministic key in a far-away region of the key space,
 *   keyed by the coll_guid's own (rank,key) plus a per-edge offset, and
 *   spread homes round-robin across ranks.  This is collision-free with
 *   the app's other GUIDs because the offset jumps well past any range the
 *   app reserves, and disjoint per collective event (the coll_guid rank is
 *   folded into the offset so the reduce event and the timer event never
 *   share edge GUIDs).
 * ========================================================================= */

/* Datum payload decode from the redOp_t bitfield (see
 * extensions/ocr-reduction-event.h):
 *   datum size  = (op>>2)&0x7 → 0:1B 1:2B 3:4B 7:8B
 *   signed      = (op>>5)&0x1
 *   real (FP)   = (op>>6)&0x1
 *   operator    = (op>>7)&0x7 → 0 ADD 1 MUL 2 MIN 3 MAX 4 AND 5 OR 6 XOR
 */
static u32 redop_datum_bytes(redOp_t op) {
  u32 code = (u32)((op >> 2) & 0x7);
  switch (code) {
  case 0:
    return 1;
  case 1:
    return 2;
  case 3:
    return 4;
  case 7:
    return 8;
  default:
    return 8;
  }
}

/* Reduce `b` into `a` element-wise over `nbDatum` elements of the type and
 * operator encoded in `op`.  Both buffers hold nbDatum * datum_bytes bytes. */
static void redop_reduce(void *aBuf, const void *bBuf, u32 nbDatum,
                         redOp_t op) {
  u32 bytes = redop_datum_bytes(op);
  u32 oper = (u32)((op >> 7) & 0x7);
  u32 isReal = (u32)((op >> 6) & 0x1);
  u32 isSigned = (u32)((op >> 5) & 0x1);

  for (u32 i = 0; i < nbDatum; i++) {
    void *ap = (char *)aBuf + ((size_t)i * bytes);
    const void *bp = (const char *)bBuf + ((size_t)i * bytes);

    if (isReal) {
      if (bytes == 8) {
        double a = *(double *)ap;
        double b = *(const double *)bp;
        double r;
        switch (oper) {
        case 1:
          r = a * b;
          break;
        case 2:
          r = (a < b) ? a : b;
          break;
        case 3:
          r = (a > b) ? a : b;
          break;
        default:
          r = a + b;
          break;
        }
        *(double *)ap = r;
      } else { /* 4-byte float */
        float a = *(float *)ap;
        float b = *(const float *)bp;
        float r;
        switch (oper) {
        case 1:
          r = a * b;
          break;
        case 2:
          r = (a < b) ? a : b;
          break;
        case 3:
          r = (a > b) ? a : b;
          break;
        default:
          r = a + b;
          break;
        }
        *(float *)ap = r;
      }
      continue;
    }

    /* Integer path: load both operands widened to 64 bits, reduce, store. */
    int64_t sa = 0;
    int64_t sb = 0;
    uint64_t ua = 0;
    uint64_t ub = 0;
    switch (bytes) {
    case 1:
      ua = *(uint8_t *)ap;
      ub = *(const uint8_t *)bp;
      /* Explicit widening: the signed byte is sign-extended on purpose for the
       * signed min/max path; casting through unsigned char would corrupt it. */
      sa = (int64_t)*(int8_t *)ap;
      sb = (int64_t)*(const int8_t *)bp;
      break;
    case 2:
      ua = *(uint16_t *)ap;
      ub = *(const uint16_t *)bp;
      sa = *(int16_t *)ap;
      sb = *(const int16_t *)bp;
      break;
    case 4:
      ua = *(uint32_t *)ap;
      ub = *(const uint32_t *)bp;
      sa = *(int32_t *)ap;
      sb = *(const int32_t *)bp;
      break;
    default:
      ua = *(uint64_t *)ap;
      ub = *(const uint64_t *)bp;
      sa = *(int64_t *)ap;
      sb = *(const int64_t *)bp;
      break;
    }
    uint64_t ur;
    switch (oper) {
    case 0:
      ur = ua + ub;
      break;
    case 1:
      ur = ua * ub;
      break;
    case 2:
      if (isSigned) {
        ur = (uint64_t)((sa < sb) ? sa : sb);
      } else {
        ur = (ua < ub) ? ua : ub;
      }
      break;
    case 3:
      if (isSigned) {
        ur = (uint64_t)((sa > sb) ? sa : sb);
      } else {
        ur = (ua > ub) ? ua : ub;
      }
      break;
    case 4:
      ur = ua & ub;
      break;
    case 5:
      ur = ua | ub;
      break;
    case 6:
      ur = ua ^ ub;
      break;
    default:
      ur = ua + ub;
      break;
    }
    switch (bytes) {
    case 1:
      *(uint8_t *)ap = (uint8_t)ur;
      break;
    case 2:
      *(uint16_t *)ap = (uint16_t)ur;
      break;
    case 4:
      *(uint32_t *)ap = (uint32_t)ur;
      break;
    default:
      *(uint64_t *)ap = ur;
      break;
    }
  }
}

/* ── Deterministic edge-event GUID derivation (G1) ───────────────────────── */

/* Edge GUIDs are laid out in a high region of the 48-bit key space, far from
 * any range a host app reserves.  The key space is partitioned so that:
 *   - each distinct collective event gets its own large per-event block
 *     (indexed by a hash of the coll_guid's rank+key), so the reduce event
 *     and the timer event — which share a key but differ in rank — never
 *     alias even across unbounded generations;
 *   - within a block, each generation gets a fixed slot stride that holds
 *     all up- and down-edge keys for that generation.
 */
#define COLLECTIVE_EDGE_KEY_BASE ((uint64_t)1 << 40) /* 2^40 .. 2^48 */
/* Per-event block: 2^32 keys ⇒ at the default per-generation stride below,
 * far more generations than any real run consumes. */
#define COLLECTIVE_EDGE_BLOCK_BITS 32
#define COLLECTIVE_EDGE_BLOCK ((uint64_t)1 << COLLECTIVE_EDGE_BLOCK_BITS)
/* Per-generation key stride: must exceed 2 * (max nrank + 1) so up- and
 * down-edge keys for one generation never alias the next. */
#define COLLECTIVE_EDGE_GEN_STRIDE ((uint64_t)1 << 16)

/* Mix the coll_guid into a per-event block index in [0, blocks).  The number
 * of available blocks is large enough that practical collision is impossible
 * for the small number of collective events an app creates. */
static uint64_t collective_event_block(arts_guid_t coll_guid) {
  uint64_t blocks = (ARTS_GUID_KEY_MASK + 1 - COLLECTIVE_EDGE_KEY_BASE) /
                    COLLECTIVE_EDGE_BLOCK;
  uint64_t coll_rank = (uint64_t)ARTS_GUID_GET_RANK(coll_guid);
  uint64_t coll_key = ARTS_GUID_GET_KEY(coll_guid);
  uint64_t h = (coll_key * 0x9E3779B97F4A7C15ULL) + coll_rank;
  h ^= h >> 29;
  return h % blocks;
}

/* direction: 0 = up-edge (carries r's partial to parent),
 *            1 = down-edge (carries the broadcast result to r). */
static arts_guid_t collective_edge_guid(arts_guid_t coll_guid, u32 nrank,
                                        u64 gen, u32 r, u32 direction) {
  unsigned int n = nrank ? nrank : 1u;
  uint64_t coll_rank = (uint64_t)ARTS_GUID_GET_RANK(coll_guid);
  uint64_t blockBase =
      COLLECTIVE_EDGE_KEY_BASE +
      (collective_event_block(coll_guid) * COLLECTIVE_EDGE_BLOCK);
  uint64_t key = (blockBase + gen * COLLECTIVE_EDGE_GEN_STRIDE +
                  (uint64_t)direction * ((uint64_t)n + 1) + (uint64_t)r) &
                 ARTS_GUID_KEY_MASK;
  /* Home the edge round-robin across the actual ARTS ranks (NOT the
   * contributor count, which may exceed or undershoot the node count) so
   * no single rank homes every edge event and every home is a live rank. */
  unsigned int nodes = arts_global_rank_count ? arts_global_rank_count : 1u;
  unsigned int home = (unsigned int)((coll_rank + r + direction) % nodes);
  return ARTS_GUID_MAKE(ARTS_GUID_EVENT, home, key);
}

/* Create (idempotently, first-create-wins) a labeled STICKY ARTS event at
 * the given pre-derived GUID.  Concurrent creators on any rank converge on
 * the same single event; losers are silent no-ops. */
static void collective_edge_event_ensure(arts_guid_t edge) {
  arts_event_hint_t h = ARTS_EVENT_HINT_STICKY;
  h.guid = edge;
  /* install-if-absent: this ensure is idempotent (first-create-wins).  A
   * concurrent or repeat create of the same edge GUID must NOT replace the live
   * event, which would orphan dependents already registered on the displaced
   * instance and strand the reduction.  Without this, the default unconditional
   * install replaces the prior generation. */
  h.check = true;
  (void)arts_event_create(&h);
}

/* =========================================================================
 * Collective metadata registry — open-addressed linear-probing hash map
 * keyed by the collective event GUID (the OCR-visible identifier) → the
 * metadata DB GUID holding this node's per-collective state.
 *
 * Concurrency model:
 *   - `edtGuid` is published via __sync_bool_compare_and_swap (full
 *     barrier).  `metaDbGuid` is a plain store BEFORE the CAS so the CAS
 *     itself is the publication point: any thread observing edtGuid is
 *     guaranteed to see the paired metaDbGuid.
 *   - A TOMBSTONE marker lets unregister clear an entry without breaking
 *     the probe chain.
 * ========================================================================= */

#define COLLECTIVE_HASH_SIZE 4096
#define COLLECTIVE_TOMBSTONE ((arts_guid_t) ~(uint64_t)0)

/* Per-contributor node-local state.  A node only ever touches the entries
 * for the contributor indices whose EDTs run on it, but the array is sized
 * to nbContribs so any index is addressable.  `pendingDep`/`pendingSlot`
 * hold the dependent registered by ocrAddDependenceSlot for the NEXT
 * generation of this contributor; the matching satisfy consumes it. */
typedef struct {
  arts_guid_t pendingDep; /* registered result receiver for next gen */
  u32 pendingSlot;
  u32 hasPending;
  u64 nextGen; /* generation of this contributor's next satisfy */
} CollectiveContribState;

typedef struct {
  arts_guid_t collGuid; /* the cross-rank-consistent OCR event GUID */
  redOp_t op;
  collectiveType_t type;
  u32 nbContribs; /* == nrank, number of tree nodes */
  u32 nbDatum;
  u32 datumBytes;
  arts_guid_t metaDbGuid;
  CollectiveContribState contrib[]; /* nbContribs entries (flexible array) */
} CollectiveMetadata;

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
      collectiveMetaMap[probeIdx].metaDbGuid = metaDbGuid;
      if (__sync_bool_compare_and_swap(&collectiveMetaMap[probeIdx].edtGuid,
                                       cur, key)) {
        return COLLECTIVE_REGISTER_OK;
      }
    }

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
      return collectiveMetaMap[probeIdx].metaDbGuid;
    }
    if (cur == NULL_GUID) {
      return NULL_GUID;
    }
  }
  return NULL_GUID;
}

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

/* ── Tree reducer / forwarder EDTs ───────────────────────────────────────── */

/* paramv layout shared by both collective tree EDTs (one uint64_t per slot,
 * indexed by the CP_* enum below). CP_DEP / CP_DEPSLOT are meaningful only for
 * the down forwarder, which delivers the result to a locally-registered
 * dependent; the up reducer ignores them. */
enum {
  CP_COLL = 0,
  CP_OP,
  CP_TYPE,
  CP_NBDATUM,
  CP_DATUMBYTES,
  CP_NRANK,
  CP_GEN,
  CP_R,
  CP_DEP,
  CP_DEPSLOT,
  CP_COUNT
};

static u32 collective_num_children(u32 r, u32 nrank) {
  u32 n = 0;
  if (2u * r + 1u < nrank) {
    n++;
  }
  if (2u * r + 2u < nrank) {
    n++;
  }
  return n;
}

/* Up-phase reducer for contributor r at one generation.
 * depv slots: [0] own datum, [1..numChildren] children up-edge partials.
 * Produces r's partial; forwards it to parent's up-edge, or — at the root —
 * seeds the down-phase by satisfying the root's own down-edge. */
static void collective_up_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  arts_guid_t coll = (arts_guid_t)paramv[CP_COLL];
  redOp_t op = (redOp_t)paramv[CP_OP];
  u32 nbDatum = (u32)paramv[CP_NBDATUM];
  u32 datumBytes = (u32)paramv[CP_DATUMBYTES];
  u32 nrank = (u32)paramv[CP_NRANK];
  u64 gen = (u64)paramv[CP_GEN];
  u32 r = (u32)paramv[CP_R];

  size_t payload = (size_t)nbDatum * datumBytes;

  /* Accumulate own datum (slot 0) with each child's partial (slots 1..). */
  void *partialPtr;
  arts_guid_t partialDb = arts_db_create(&partialPtr, payload, ARTS_DB_DEFAULT,
                                         ARTS_DB_PROP_NONE, NULL);
  memcpy(partialPtr, depv[0].ptr, payload);
  for (uint32_t i = 1; i < depc; i++) {
    if (depv[i].ptr != NULL) {
      redop_reduce(partialPtr, depv[i].ptr, nbDatum, op);
    }
  }

  /* Release WRITE access on the partial so its bytes are written back to the
   * DB's home and become visible to the cross-rank consumer acquiring it RO.
   * Without this the consumer would acquire before the EDT-epilogue writeback
   * and observe stale / unpopulated data. */
  arts_db_release(partialDb, ARTS_MODE_RW);

  if (r == 0) {
    /* Root: seed the broadcast.  The full reduction is the root's partial. */
    arts_guid_t down = collective_edge_guid(coll, nrank, gen, 0, 1);
    collective_edge_event_ensure(down);
    arts_event_satisfy_slot(down, partialDb, ARTS_EVENT_LATCH_DECR_SLOT);
  } else {
    arts_guid_t up = collective_edge_guid(coll, nrank, gen, r, 0);
    collective_edge_event_ensure(up);
    arts_event_satisfy_slot(up, partialDb, ARTS_EVENT_LATCH_DECR_SLOT);
  }
}

/* Down-phase forwarder for contributor r at one generation.
 * depv slot [0] = the broadcast result (from r's down-edge).
 * Forwards the result to each child's down-edge and delivers it to the
 * locally-registered dependent (an EDT slot or a channel event). */
static void collective_down_edt(uint32_t paramc, const uint64_t *paramv,
                                uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t coll = (arts_guid_t)paramv[CP_COLL];
  u32 nbDatum = (u32)paramv[CP_NBDATUM];
  u32 datumBytes = (u32)paramv[CP_DATUMBYTES];
  u32 nrank = (u32)paramv[CP_NRANK];
  u64 gen = (u64)paramv[CP_GEN];
  u32 r = (u32)paramv[CP_R];
  arts_guid_t dep = (arts_guid_t)paramv[CP_DEP];
  u32 depSlot = (u32)paramv[CP_DEPSLOT];

  size_t payload = (size_t)nbDatum * datumBytes;
  const void *resultPtr = depv[0].ptr;

  /* Forward to children's down-edges. */
  for (u32 c = 0; c < 2; c++) {
    u32 child = (2u * r) + 1u + c;
    if (child < nrank) {
      void *fwdPtr;
      arts_guid_t fwdDb = arts_db_create(&fwdPtr, payload, ARTS_DB_DEFAULT,
                                         ARTS_DB_PROP_NONE, NULL);
      if (resultPtr != NULL) {
        memcpy(fwdPtr, resultPtr, payload);
      }
      arts_db_release(fwdDb, ARTS_MODE_RW);
      arts_guid_t cdown = collective_edge_guid(coll, nrank, gen, child, 1);
      collective_edge_event_ensure(cdown);
      arts_event_satisfy_slot(cdown, fwdDb, ARTS_EVENT_LATCH_DECR_SLOT);
    }
  }

  /* Deliver the result to the locally-registered dependent for this gen. */
  if (dep != NULL_GUID) {
    void *outPtr;
    arts_guid_t outDb = arts_db_create(&outPtr, payload, ARTS_DB_DEFAULT,
                                       ARTS_DB_PROP_NONE, NULL);
    if (resultPtr != NULL) {
      memcpy(outPtr, resultPtr, payload);
    }
    arts_db_release(outDb, ARTS_MODE_RW);
    arts_guid_kind_t dstType = arts_guid_get_kind(dep);
    if (dstType == ARTS_GUID_EDT) {
      arts_add_dependence(outDb, dep, depSlot, ARTS_MODE_RO);
    } else if (dstType == ARTS_GUID_EVENT) {
      arts_event_satisfy_slot(dep, outDb, ARTS_EVENT_LATCH_DECR_SLOT);
    }
  }
}

/* Launch the up-reducer and down-forwarder EDTs for contributor r at one
 * generation, wiring their cross-rank edge dependencies, then contribute
 * r's own datum.  Runs on r's node (the contributor's EDT calls satisfy). */
static void collective_launch_generation(CollectiveMetadata *meta, u32 r,
                                         u64 gen, const void *dataPtr,
                                         arts_guid_t dep, u32 depSlot) {
  arts_guid_t coll = meta->collGuid;
  u32 nrank = meta->nbContribs;
  u32 nbDatum = meta->nbDatum;
  u32 datumBytes = meta->datumBytes;
  size_t payload = (size_t)nbDatum * datumBytes;

  uint64_t pv[CP_COUNT];
  pv[CP_COLL] = (uint64_t)coll;
  pv[CP_OP] = (uint64_t)meta->op;
  pv[CP_TYPE] = (uint64_t)meta->type;
  pv[CP_NBDATUM] = nbDatum;
  pv[CP_DATUMBYTES] = datumBytes;
  pv[CP_NRANK] = nrank;
  pv[CP_GEN] = gen;
  pv[CP_R] = r;
  pv[CP_DEP] = (uint64_t)dep;
  pv[CP_DEPSLOT] = depSlot;

  u32 numChildren = collective_num_children(r, nrank);

  /* --- Up reducer: own datum + each child up-edge. --- */
  arts_edt_hint_t upHint = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t upEdt = arts_edt_create(collective_up_edt, CP_COUNT, pv,
                                      1u + numChildren, &upHint);

  /* slot 0: own datum DB.  Release WRITE access so the (possibly cross-rank)
   * up reducer reads the populated bytes; the up reducer for r runs on r's
   * node, which is this node, but a release keeps the RC state well-formed. */
  void *ownPtr;
  arts_guid_t ownDb = arts_db_create(&ownPtr, payload, ARTS_DB_DEFAULT,
                                     ARTS_DB_PROP_NONE, NULL);
  memcpy(ownPtr, dataPtr, payload);
  arts_db_release(ownDb, ARTS_MODE_RW);
  arts_add_dependence(ownDb, upEdt, 0, ARTS_MODE_RO);

  /* slots 1..: children up-edges. */
  u32 slot = 1;
  for (u32 c = 0; c < 2; c++) {
    u32 child = (2u * r) + 1u + c;
    if (child < nrank) {
      arts_guid_t up = collective_edge_guid(coll, nrank, gen, child, 0);
      collective_edge_event_ensure(up);
      arts_add_dependence(up, upEdt, slot, ARTS_MODE_RO);
      slot++;
    }
  }

  /* --- Down forwarder: waits on r's down-edge, fans out + delivers. --- */
  arts_edt_hint_t downHint = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t downEdt =
      arts_edt_create(collective_down_edt, CP_COUNT, pv, 1u, &downHint);
  arts_guid_t myDown = collective_edge_guid(coll, nrank, gen, r, 1);
  collective_edge_event_ensure(myDown);
  arts_add_dependence(myDown, downEdt, 0, ARTS_MODE_RO);
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
 * OCR EDT Trampoline
 *
 * Layout of ARTS paramv for OCR EDTs:
 *   paramv[0] = function pointer (ocrEdt_t)
 *   paramv[1] = original paramc
 *   paramv[2] = output event GUID (for non-finish EDTs) or NULL_GUID
 *   paramv[3..3+paramc-1] = original paramv values
 *
 * For finish EDTs (EDT_PROP_FINISH), ocrEdtCreate pre-creates a FINISH event
 * (via ARTS_EVENT_HINT_FINISH), passes it to the EDT via hint.finish_event, and
 * chains it to the OCR outputEvent.  The trampoline itself is scope-agnostic.
 * ========================================================================= */

static void ocr_edt_trampoline(uint32_t paramc, const uint64_t *paramv,
                               uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;

  /* Restore EDT-local semantics for the OCR ELS array — see ocr_els_reset. */
  ocr_els_reset();

  ocrEdt_t func = (ocrEdt_t)paramv[0];
  u32 origParamc = (u32)paramv[1];
  arts_guid_t outEvt = (arts_guid_t)paramv[2];
  /* ARTS paramv is const; copy original params for OCR's non-const API */
  u64 *origParamv = NULL;
  u64 origParamBuf[origParamc > 0 ? origParamc : 1];
  if (origParamc > 0) {
    memcpy(origParamBuf, &paramv[3], origParamc * sizeof(u64));
    origParamv = origParamBuf;
  }

  /* Convert arts_edt_dep_t to ocrEdtDep_t.  Preserve the mode that ARTS
   * resolved during arts_db_acquire_all (RO vs EW vs NULL) — the OCR EDT body
   * may inspect depv[i].mode for assertions or behavior, and surfacing
   * the actual ARTS-resolved mode is more honest than the previous
   * DB_DEFAULT_MODE hardcode. */
  ocrEdtDep_t ocrDepv[depc > 0 ? depc : 1];

  for (u32 i = 0; i < depc; i++) {
    ocrDepv[i].guid.guid = depv[i].guid;
    ocrDepv[i].ptr = depv[i].ptr;
    ocrDepv[i].mode = arts_to_ocr_mode(depv[i].mode);
  }

  ocrGuid_t returnGuid = func(origParamc, origParamv, depc, ocrDepv);

  /* Non-finish EDTs satisfy the output event directly with the return value.
   * Finish EDTs have their output event satisfied by the runtime when the
   * finish-scope (this EDT + all descendants) drains via finish_event. */
  if (outEvt != NULL_GUID) {
    arts_event_satisfy_slot(outEvt, returnGuid.guid,
                            ARTS_EVENT_LATCH_DECR_SLOT);
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
    (void)fprintf(
        stderr,
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
  bool isFinishEdt = (properties & EDT_PROP_FINISH) != 0;
  bool oevtValid = (properties & EDT_PROP_OEVT_VALID) != 0;

  if (outputEvent != NULL) {
    if (oevtValid) {
      outEvt = outputEvent->guid;
    } else {
      arts_event_hint_t h = ARTS_EVENT_HINT_IDEMPOTENT;
      h.rank = rank;
      outEvt = arts_event_create(&h);
      if (outEvt == NULL_GUID) {
        return OCR_ENOMEM;
      }
      outputEvent->guid = outEvt;
    }
  }

  /* paramv layout passed to ocr_edt_trampoline:
   *   [0] = function pointer, [1] = original paramc,
   *   [2] = outEvt (non-finish only; finish outEvt is wired via finish_event),
   *   [3..3+N-1] = user paramv.
   * arts_calloc zero-initialises, so trailing padding beyond the last user
   * word is safe when OCR apps cast structs to u64* with overshoot. */
  u32 artsParamc = 3 + actualParamc;
  uint64_t *artsParamv = (uint64_t *)arts_calloc(artsParamc, sizeof(uint64_t));
  artsParamv[0] = (uint64_t)(uintptr_t)templ->funcPtr;
  artsParamv[1] = (uint64_t)actualParamc;
  artsParamv[2] = isFinishEdt ? (uint64_t)NULL_GUID : (uint64_t)outEvt;
  ocr_copy_paramv_safe(&artsParamv[3], paramv, actualParamc);

  /* For finish EDTs, create the finish event explicitly so the shim owns it
   * and can chain it to outEvt immediately — without querying the EDT after
   * creation.  The finish event's latch is pre-decremented by the EDT itself
   * (creator-token); when all descendants complete it drains to zero and fires,
   * satisfying outEvt and signalling the OCR scope boundary.  ARTS_MODE_NULL is
   * the correct dependency mode: the scope-drain signal carries no data
   * payload.
   */
  arts_guid_t fe = NULL_GUID;
  if (isFinishEdt) {
    arts_event_hint_t feh = ARTS_EVENT_HINT_FINISH;
    feh.rank = rank;
    fe = arts_event_create(&feh);
  }

  arts_edt_hint_t edtHint = {.rank = rank};
  if (isFinishEdt) {
    edtHint.finish_event = fe;
  }
  arts_guid_t edtGuid = arts_edt_create(ocr_edt_trampoline, artsParamc,
                                        artsParamv, actualDepc, &edtHint);

  if (isFinishEdt && outEvt != NULL_GUID && edtGuid != NULL_GUID &&
      fe != NULL_GUID) {
    arts_add_dependence(fe, outEvt, 0, ARTS_MODE_NULL);
  }

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
  /* The distinct single-fire OCR flavors (ONCE/IDEM/STICKY/COUNTED) all map
   * to the unified LATCH(1) fire-and-linger event; their old auto-destroy /
   * over-satisfy-error / exact-N-dep semantics are subsumed (silent
   * over-satisfy, linger until explicit destroy).  CHANNEL is preserved. */
  arts_event_hint_t h = ARTS_EVENT_HINT_LATCH(1);
  switch (kind) {
  case OCR_EVENT_ONCE_T:
  case OCR_EVENT_IDEM_T:
  case OCR_EVENT_STICKY_T:
  case OCR_EVENT_COUNTED_T:
    break; /* LATCH(1) */
  case OCR_EVENT_LATCH_T:
    /* Counter event; init 0 (caller may override via params). */
    h.latch = 0;
    break;
  case OCR_EVENT_CHANNEL_T:
    h = ARTS_EVENT_HINT_CHANNEL;
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
    /* GUID_PROP_CHECK → fail-if-exists install so the first creator wins and a
     * later rendezvous create observes the collision (arts_event_create
     * returns NULL_GUID).  Without CHECK the install replaces unconditionally.
     */
    h.check = (properties & GUID_PROP_CHECK) != 0;
    arts_guid_t result = arts_event_create(&h);
    if (result == NULL_GUID && (properties & GUID_PROP_CHECK)) {
      return OCR_EGUIDEXISTS;
    }
    return 0;
  }
  arts_guid_t g = arts_event_create(&h);
  if (g == NULL_GUID) {
    return OCR_ENOMEM;
  }
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

    /* Each node independently creates its own node-local metadata for the
     * (labeled) collective event.  The labeled fast-path check below only
     * matches within this node, so every node ends up with exactly one
     * metadata entry — the cross-rank reduction itself flows through ARTS
     * edge events, not this struct. */
    if ((properties & GUID_PROP_IS_LABELED) && labeledGuid != NULL_GUID) {
      arts_guid_t existingMeta = lookupCollectiveMeta(labeledGuid);
      if (existingMeta != NULL_GUID) {
        return (properties & GUID_PROP_CHECK) ? OCR_EGUIDEXISTS : 0;
      }
    }

    if (nbContribs == 0) {
      return OCR_EINVAL;
    }

    void *metaPtr;
    /* PIN, homed on the current rank: collective metadata is node-local
     * shared state accessed by this node's contributing EDTs without going
     * through RC acquire/release cycles.  A PIN DB is not internode
     * relocatable, so it must be created locally (round-robin placement
     * would try to home it on a remote rank and fail). */
    size_t metaBytes = sizeof(CollectiveMetadata) +
                       ((size_t)nbContribs * sizeof(CollectiveContribState));
    arts_db_hint_t metaHint = {.rank = ARTS_HINT_CURRENT_RANK};
    arts_guid_t metaDb = arts_db_create(&metaPtr, metaBytes, ARTS_DB_PIN,
                                        ARTS_DB_PROP_NONE, &metaHint);
    if (metaDb == NULL_GUID) {
      return OCR_ENOMEM;
    }
    CollectiveMetadata *meta = (CollectiveMetadata *)metaPtr;

    arts_guid_t collGuid =
        ((properties & GUID_PROP_IS_LABELED) && labeledGuid != NULL_GUID)
            ? labeledGuid
            : metaDb;

    meta->collGuid = collGuid;
    meta->op = params->EVENT_COLLECTIVE.op;
    meta->type = params->EVENT_COLLECTIVE.type;
    meta->nbContribs = nbContribs;
    meta->nbDatum = params->EVENT_COLLECTIVE.nbDatum;
    if (meta->nbDatum == 0) {
      meta->nbDatum = 1;
    }
    meta->datumBytes = redop_datum_bytes(meta->op);
    meta->metaDbGuid = metaDb;

    for (u32 i = 0; i < nbContribs; i++) {
      meta->contrib[i].pendingDep = NULL_GUID;
      meta->contrib[i].pendingSlot = 0;
      meta->contrib[i].hasPending = 0;
      meta->contrib[i].nextGen = 0;
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
  /* OCR_EVENT_COUNTED_T params.nbDeps ignored: COUNTED collapses to LATCH(1)
   * (exact-N-dep auto-destroy semantics dropped — fire-and-linger). */
  if (eventType == OCR_EVENT_CHANNEL_T && params != NULL) {
    /* OCR 1.2 §B.5.2: nbSat and nbDeps are restricted to 1.  ARTS enforces
     * that constraint at the shim — generalized values would require a
     * non-trivial change to the channel drain loop (currently fires one
     * data-dep pair per generation). */
    if (params->EVENT_CHANNEL.nbSat != 1 || params->EVENT_CHANNEL.nbDeps != 1) {
      (void)fprintf(stderr,
                    "[ARTS] CHANNEL nbSat=%u nbDeps=%u: only nbSat=nbDeps=1 "
                    "supported (OCR 1.2 §B.5.2)\n",
                    params->EVENT_CHANNEL.nbSat, params->EVENT_CHANNEL.nbDeps);
      return OCR_EINVAL;
    }
    /* maxGen is implementation-driven — ARTS scales unbounded via mpsc. */
  }

  if (properties & GUID_PROP_IS_LABELED) {
    h.guid = guid->guid;
    /* GUID_PROP_CHECK → fail-if-exists (first creator wins); else replace. */
    h.check = (properties & GUID_PROP_CHECK) != 0;
    arts_guid_t result = arts_event_create(&h);
    if (result == NULL_GUID && (properties & GUID_PROP_CHECK)) {
      return OCR_EGUIDEXISTS;
    }
    return 0;
  }
  arts_guid_t g = arts_event_create(&h);
  if (g == NULL_GUID) {
    return OCR_ENOMEM;
  }
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

  /* Hold the route table lookup ref for the entire critical section so the
   * metadata DB cannot be destroyed underneath us.  Paired release at exit. */
  arts_shared_ptr_t meta_h = arts_route_table_lookup_db(metaDbGuid);
  struct arts_db_s *raw = (struct arts_db_s *)arts_shared_get(meta_h);
  if (raw == NULL) {
    return OCR_EFAULT;
  }
  CollectiveMetadata *meta = (CollectiveMetadata *)(raw + 1);

  if (islot >= meta->nbContribs) {
    arts_shared_release(&meta_h);
    return OCR_EINVAL;
  }

  /* contributor `islot`'s k-th satisfy is global generation k.  Pair it with
   * the dependent registered by the preceding ocrAddDependenceSlot for this
   * contributor (recorded as "pending" for the upcoming generation), then
   * launch this generation's tree EDTs.  __sync barriers serialize multiple
   * worker threads contributing different islots on the same node. */
  CollectiveContribState *cs = &meta->contrib[islot];
  arts_guid_t dep = NULL_GUID;
  u32 depSlot = 0;
  if (cs->hasPending) {
    dep = cs->pendingDep;
    depSlot = cs->pendingSlot;
    cs->pendingDep = NULL_GUID;
    cs->pendingSlot = 0;
    cs->hasPending = 0;
  }
  u64 gen = cs->nextGen;
  cs->nextGen = gen + 1;

  collective_launch_generation(meta, islot, gen, dataPtr, dep, depSlot);

  arts_shared_release(&meta_h);
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

    /* GUID_PROP_CHECK → fail-if-exists install (first creator wins; a later
     * one is told via EGUIDEXISTS, returning NULL here).  Without CHECK the
     * install replaces unconditionally. */
    arts_db_hint_t lh = ARTS_DB_HINT_DEFAULTS;
    lh.check = (flags & GUID_PROP_CHECK) != 0;
    void *data = arts_db_create_with_guid(labeledGuid, len, ARTS_DB_DEFAULT,
                                          ARTS_DB_PROP_NONE, &lh);
    if (data == NULL) {
      /* Labeled GUID already taken — fall back to looking it up so the
       * caller still gets a valid pointer.  The lookup handle is released
       * immediately (the descriptor lifetime is owned by route_table; the
       * user data pointer remains valid because the DB itself wasn't
       * destroyed). */
      arts_shared_ptr_t ex_h = arts_route_table_lookup_db(labeledGuid);
      struct arts_db_s *db_existing = (struct arts_db_s *)arts_shared_get(ex_h);
      if (db_existing != NULL) {
        *addr = (void *)(db_existing + 1);
        arts_shared_release(&ex_h);
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
  /* DB_PROP_NO_ACQUIRE: the creating EDT does not acquire the block (it is
   * created for a later consumer).  The runtime leaves the home as the sole
   * idle owner and returns a NULL pointer, so no release is required.  Any
   * other property bit maps to the default acquire-on-create behavior. */
  unsigned int arts_flags = (flags & DB_PROP_NO_ACQUIRE)
                                ? ARTS_DB_PROP_NO_ACQUIRE
                                : ARTS_DB_PROP_NONE;
  db->guid = arts_db_create(addr, len, ARTS_DB_DEFAULT, arts_flags, hintp);
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
    arts_db_release(db.guid, ARTS_MODE_RW);
  }
  return 0;
}

/* =========================================================================
 * Dependence Management
 * ========================================================================= */

/*
 * Map OCR access mode → ARTS access mode.
 *
 * ARTS datablocks support RO (shared read) and RW (per-node exclusive write).
 * OCR RW and EW both imply exclusive access, so both map to ARTS RW.  OCR RO
 * maps to ARTS RO.
 *
 * ocrDbRelease() calls arts_db_release() to release a RW DB early, allowing
 * consumer EDTs to proceed before the current EDT completes.  For RO deps, no
 * early release is needed.
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
 * ARTS resolves the actual access mode during arts_db_acquire_all.  We surface
 * that to the OCR EDT body so user code (and OCR helper libraries that
 * read depv[i].mode for assertions or branching) sees the truth.
 *
 * RO is reported as DB_MODE_RO rather than DB_DEFAULT_MODE (RW) because
 * the OCR-RW-mapped-to-ARTS-RO path doesn't survive the round trip and
 * we have no way to distinguish "originally RW" from "originally RO".
 * Reporting RO is conservative and matches what arts_db_acquire_all actually
 * did.
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
    arts_guid_kind_t dstType = arts_guid_get_kind(destination.guid);
    if (dstType == ARTS_GUID_EDT) {
      arts_add_dependence((arts_guid_t)(0), destination.guid, slot,
                          ARTS_MODE_VAL);
    } else if (dstType == ARTS_GUID_EVENT) {
      arts_event_satisfy_slot(destination.guid, NULL_GUID,
                              ARTS_EVENT_LATCH_DECR_SLOT);
    }
    return 0;
  }

  arts_guid_kind_t srcType = arts_guid_get_kind(source.guid);
  arts_guid_kind_t dstType = arts_guid_get_kind(destination.guid);

  if (srcType == ARTS_GUID_DB) {
    /* DB → EDT/Event: arts_add_dependence does immediate satisfy for DB
     * sources (DBs are passive objects — no channel event, no waiting).
     * Map OCR access modes to ARTS: RO→RO, EW/RW→EW.
     * GUID-sorted acquisition in arts_db_acquire_all prevents acquisition-order
     * deadlocks that previously required forcing all deps to RO. */
    if (dstType == ARTS_GUID_EDT) {
      arts_add_dependence(source.guid, destination.guid, slot,
                          ocr_to_arts_mode(mode));
    } else if (dstType == ARTS_GUID_EVENT) {
      arts_event_satisfy_slot(destination.guid, source.guid,
                              ARTS_EVENT_LATCH_DECR_SLOT);
    }
  } else if (srcType == ARTS_GUID_EVENT) {
    /* ARTS channels (latch=1) do INCR internally in add_dependence_with_mode.
     * Non-channel events use direct dependent registration. Both paths
     * are handled by arts_add_dependence. Cross-node: handled natively.
     *
     * Pass the user's access mode through ocr_to_arts_mode so the EDT slot
     * is registered with the correct ARTS mode (RO/EW).  Previously this
     * was hardcoded to ARTS_MODE_RO, silently downgrading every Event→EDT
     * dependence to read-only access. */
    if (dstType == ARTS_GUID_EDT) {
      arts_add_dependence(source.guid, destination.guid, slot,
                          ocr_to_arts_mode(mode));
    } else if (dstType == ARTS_GUID_EVENT) {
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
  arts_guid_t metaDbGuid = lookupCollectiveMeta(source.guid);

  if (metaDbGuid != NULL_GUID) {
    /* Collective: `sslot` is the contributor index whose reduced result
     * `destination`/`dslot` will receive on this node.  Record it as the
     * pending dependent for that contributor's upcoming generation; the
     * matching ocrEventCollectiveSatisfySlot (called right after, in the
     * same EDT body) consumes it and pins it to a concrete generation.
     * Hold the route table ref for the duration we touch the metadata. */
    arts_shared_ptr_t meta_h = arts_route_table_lookup_db(metaDbGuid);
    struct arts_db_s *raw = (struct arts_db_s *)arts_shared_get(meta_h);
    if (raw == NULL) {
      return OCR_EFAULT;
    }
    CollectiveMetadata *meta = (CollectiveMetadata *)(raw + 1);
    if (sslot >= meta->nbContribs) {
      arts_shared_release(&meta_h);
      return OCR_EINVAL;
    }
    CollectiveContribState *cs = &meta->contrib[sslot];
    cs->pendingDep = destination.guid;
    cs->pendingSlot = dslot;
    cs->hasPending = 1;
    arts_shared_release(&meta_h);
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

static arts_guid_kind_t kindToArtsType(ocrGuidUserKind kind) {
  switch (kind) {
  case GUID_USER_DB:
    return ARTS_GUID_DB;
  case GUID_USER_EDT:
  case GUID_USER_EDT_TEMPLATE:
    return ARTS_GUID_EDT;
  case GUID_USER_EVENT_ONCE:
  case GUID_USER_EVENT_COUNTED:
  case GUID_USER_EVENT_IDEM:
  case GUID_USER_EVENT_STICKY:
  case GUID_USER_EVENT_LATCH:
  case GUID_USER_EVENT_COLLECTIVE:
    return ARTS_GUID_EVENT;
  default:
    /* Unknown OCR GUID kind.  Return ARTS_GUID_LAST (out-of-range
     * sentinel) so the downstream arts_guid_reserve_range() rejects it
     * via its `type >= ARTS_GUID_LAST` validation rather than silently
     * producing a range with type bits 0 (which is now ARTS_GUID_EDT). */
    return ARTS_GUID_LAST;
  }
}

u8 ocrGuidRangeCreate(ocrGuid_t *rangeGuid, u64 numberGuid,
                      ocrGuidUserKind kind) {
  if (!rangeGuid || numberGuid == 0) {
    return 1;
  }
  arts_guid_kind_t artsType = kindToArtsType(kind);

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
