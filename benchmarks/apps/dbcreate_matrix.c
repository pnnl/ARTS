/*
 * dbcreate_matrix.c
 *
 * DB-create capability matrix: exercises all 12 logical cells of the
 * (creator-acquisition A) x (home-determination B) x (locality C) grid
 * through the portable OCR API (ocr.h), so the SAME source compiles and
 * runs against every OCR-API-consuming backend (this repo builds it against
 * xsocr, ocr-vx, and ARTS-via-ocr-shim from one CMake wiring).  Prints a
 * scalar CELLS_OK=<k>/12.
 *
 * Grid:
 *   A: ACQUIRE (default hold) | NO_ACQUIRE (DB_PROP_NO_ACQUIRE)
 *   B: (i) affinity hint  (ii) labeled GUID  (iii) no-hint (round robin)
 *   C: local (home == creator) | remote (home != creator)
 *
 *     1 ACQ/affinity/local     7  NOACQ/affinity/local
 *     2 ACQ/affinity/remote    8  NOACQ/affinity/remote
 *     3 ACQ/labeled/local      9  NOACQ/labeled/local
 *     4 ACQ/labeled/remote     10 NOACQ/labeled/remote
 *     5 ACQ/no-hint (a)        11 NOACQ/no-hint (a)
 *     6 ACQ/no-hint (b)        12 NOACQ/no-hint (b)
 *
 * Each cell is driven once from PD 0 and (when the affinity count allows)
 * once from PD 1, for coverage; a cell's pass bit requires every instance
 * that ran to pass.  No-hint cells have no caller-selectable target home,
 * so only data correctness is asserted for them; affinity/labeled cells
 * assert both data correctness AND, where ocrAffinityQuery is available,
 * that the DB's queried affinity equals the intended target.
 *
 * Labeled-GUID home construction: a plain ocrGuidRangeCreate() call reserves
 * a range whose entire index space is homed whichever PD called it (native
 * OCR's labeled-guid provider bakes the calling PD's own location into every
 * GUID derived from that range -- see labeled-guid.c labeledGuidReserve()).
 * To get a labeled GUID homed at an ARBITRARY target PD R (not just the
 * caller's own PD), a small EDT is affinitized to run AT R and creates the
 * range itself; that range's guid is then shared with the driver PDs via a
 * sticky event.  ocrGuidFromIndex(range, idx) with idx chosen so
 * idx % totalPDs == R keeps the SAME construction correct for backends
 * whose range home is idx-modulo-based instead (ARTS's shim / ocr-vx), so
 * one formula satisfies all three backends simultaneously.
 *
 * Ordering note: a NO_ACQUIRE cell's writer EDT (RW dependence) must itself
 * create and wire the consumer EDT (RO dependence) *after* writing the
 * payload -- wiring the consumer independently, from the same body that
 * wired the writer, would race the writer's own acquire (both dependences on
 * a freshly-created, still-idle DB can resolve at roughly the same time;
 * only program-order sequencing inside one EDT body guarantees the
 * hand-off). ACQUIRE cells avoid this entirely: the driver itself holds the
 * write from creation, writes, releases, and only then wires the consumer --
 * pure program order within one function.
 */

#include "ocr.h"
#include "extensions/ocr-affinity.h"
#include "extensions/ocr-labeling.h"

#include <stdbool.h>
#include <stdlib.h>

#define NUM_CELLS 12
#define PAYLOAD_COUNT 4
#define PAYLOAD_BYTES (PAYLOAD_COUNT * sizeof(u64))
#define RANGE_SIZE 256
#define MAX_RANGES 3 /* driver PDs are always 0/1; a driver's local/remote
                       * target is always in {0,1,2} regardless of the total
                       * PD count, so at most 3 distinct home ranges are ever
                       * needed. */
#define HOME_SKIP ((u64)0xFFFFFFFFu)

typedef enum { AXIS_AFFINITY, AXIS_LABELED, AXIS_NOHINT } axis_t;

static inline u64 guid_u64(ocrGuid_t g) { return (u64)g.guid; }
static inline ocrGuid_t u64_guid(u64 v) {
  ocrGuid_t g;
  g.guid = (intptr_t)v;
  return g;
}

/* ---- consumer_edt: paramv={cellIdx, wantHome, createOk}; depv[0]=target DB
 * (RO), depv[1]=results DB (RW, u8[NUM_CELLS]).  createOk is 1 unless a
 * NO_ACQUIRE cell's create call violated the OCR NO_ACQUIRE contract (addr
 * must come back NULL) -- see run_cell. -----------------------------------
 */
ocrGuid_t consumer_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 cell_idx = paramv[0];
  u64 want_home = paramv[1];
  u64 create_ok = paramv[2];

  const u64 *data = (const u64 *)depv[0].ptr;
  bool data_ok = (data != NULL);
  for (u32 i = 0; data_ok && i < PAYLOAD_COUNT; i++) {
    data_ok = (data[i] == cell_idx * 1000 + i);
  }

  bool home_ok = true;
  if (want_home != HOME_SKIP) {
    ocrGuid_t got;
    u64 cnt = 1;
    ocrAffinityQuery(depv[0].guid, &cnt, &got);
    ocrGuid_t want;
    ocrAffinityGetAt(AFFINITY_PD, want_home, &want);
    home_ok = ocrGuidIsEq(got, want);
  }

  bool ok = data_ok && home_ok && (create_ok != 0);
  u8 *results = (u8 *)depv[1].ptr;
  results[cell_idx - 1] = results[cell_idx - 1] && (ok ? 1 : 0);

  ocrPrintf("  cell %2llu: %s (data=%s home=%s create=%s)\n",
            (unsigned long long)cell_idx, ok ? "PASS" : "FAIL",
            data_ok ? "ok" : "BAD",
            (want_home == HOME_SKIP) ? "n/a" : (home_ok ? "ok" : "BAD"),
            create_ok ? "ok" : "BAD");
  return NULL_GUID;
}

/* Wires the RO-consumer(s) that validate a cell's DB.
 *
 *  - AXIS_AFFINITY: one policy-placed consumer that ALSO checks the DB's
 *    queried affinity (ocrAffinityQuery) == home_idx.  Non-tautological for
 *    affinity because the RUNTIME minted the GUID at the home it chose from
 *    the hint, so the query round-trips through actual placement.
 *  - AXIS_LABELED: TWO consumers -- one pinned (OCR_HINT_EDT_AFFINITY) to the
 *    encoded home PD, one pinned to a DIFFERENT PD (home_idx+1)%n (off-home)
 *    -- both of which must read the correct payload.  This is the STRUCTURAL
 *    home check: the GUID's encoded home is tautological on the labeled axis
 *    (the test minted the GUID), so instead we prove the runtime actually
 *    installed the master at the encoded home by acquiring the DB RO from a
 *    PD that is NOT the encoded home.  A labeled master wrongly registered at
 *    the creator instead would fail the off-home acquire (stale/NULL read,
 *    data=BAD) -- the non-tautological signal.  At single node n==1 so the
 *    two consumers collapse onto the same PD (harmless no-op there; only
 *    bites at multinode).
 *  - AXIS_NOHINT: one policy-placed consumer, no home assert.
 */
static void wire_consumers(ocrGuid_t db, u64 cell_idx, axis_t axis,
                           u64 home_idx, u64 n, u64 create_ok,
                           ocrGuid_t results_db, ocrGuid_t gate_event) {
  /* gate_event (NULL_GUID = none): for NO_ACQUIRE cells the writer's payload
   * is published only when the writer EDT finishes and its RW hold is
   * released (writeback).  A bare DB->consumer dep satisfies immediately, so
   * the consumer's RO acquire would race ahead of that writeback and read the
   * home's v1 zero placeholder -- an UNDEFINED RO/RW overlap per the OCR
   * model, not a coherence bug.  The gate event (the writer's OUTPUT EVENT,
   * satisfied strictly AFTER the writer's DBs are released) is added as an
   * extra consumer pre-slot so the consumer becomes runnable -- and thus does
   * its RO acquire -- only happens-after the writeback.  ACQUIRE cells pass
   * NULL_GUID: they are already release-before-wire in run_cell. */
  bool gated = !ocrGuidIsNull(gate_event);
  u32 depc = gated ? 3 : 2;
  ocrGuid_t ctmpl;
  ocrEdtTemplateCreate(&ctmpl, consumer_edt, 3, depc);
  if (axis == AXIS_LABELED) {
    u64 consumer_pds[2] = {home_idx, (home_idx + 1) % n};
    for (u32 j = 0; j < 2; j++) {
      ocrHint_t h;
      ocrHintInit(&h, OCR_HINT_EDT_T);
      ocrGuid_t aff;
      ocrAffinityGetAt(AFFINITY_PD, consumer_pds[j], &aff);
      ocrSetHintValue(&h, OCR_HINT_EDT_AFFINITY, ocrAffinityToHintValue(aff));
      u64 cparam[3] = {cell_idx, HOME_SKIP, create_ok};
      ocrGuid_t consumer;
      ocrEdtCreate(&consumer, ctmpl, EDT_PARAM_DEF, cparam, EDT_PARAM_DEF, NULL,
                   EDT_PROP_NONE, &h, NULL);
      ocrAddDependence(db, consumer, 0, DB_MODE_RO);
      ocrAddDependence(results_db, consumer, 1, DB_MODE_RW);
      if (gated) {
        ocrAddDependence(gate_event, consumer, 2, DB_MODE_NULL);
      }
    }
  } else {
    u64 want_home = (axis == AXIS_AFFINITY) ? home_idx : HOME_SKIP;
    u64 cparam[3] = {cell_idx, want_home, create_ok};
    ocrGuid_t consumer;
    ocrEdtCreate(&consumer, ctmpl, EDT_PARAM_DEF, cparam, EDT_PARAM_DEF, NULL,
                 EDT_PROP_NONE, NULL_HINT, NULL);
    ocrAddDependence(db, consumer, 0, DB_MODE_RO);
    ocrAddDependence(results_db, consumer, 1, DB_MODE_RW);
    if (gated) {
      ocrAddDependence(gate_event, consumer, 2, DB_MODE_NULL);
    }
  }
}

/* ---- writer_edt (NO_ACQUIRE cells only): paramv={cellIdx}; depv[0]=target DB
 * (RW). Writes the payload and returns; OCR satisfies this EDT's output event
 * (its post-slot, see run_cell) after the EDT completes and its DBs are
 * released, so a consumer gated on that event is happens-after the writeback.
 */
ocrGuid_t writer_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 cell_idx = paramv[0];
  u64 *p = (u64 *)depv[0].ptr;
  for (u32 i = 0; i < PAYLOAD_COUNT; i++) {
    p[i] = cell_idx * 1000 + i;
  }
  return NULL_GUID;
}

/* Runs one grid cell from the calling (driver) EDT's body.  rangeForHome /
 * tag / n are only meaningful for AXIS_LABELED (see the file header for the
 * idx construction). */
static void run_cell(u64 cell_idx, axis_t axis, u64 home_idx, bool no_acquire,
                      ocrGuid_t results_db, ocrGuid_t range_for_home, u64 tag,
                      u64 n) {
  ocrHint_t db_hint;
  ocrHint_t *hintp = NULL_HINT;
  ocrGuid_t db = UNINITIALIZED_GUID;
  u16 flags = no_acquire ? DB_PROP_NO_ACQUIRE : DB_PROP_NONE;

  if (axis == AXIS_LABELED) {
    u64 idx = home_idx + tag * n;
    ocrGuidFromIndex(&db, range_for_home, idx);
    flags |= GUID_PROP_IS_LABELED;
  } else if (axis == AXIS_AFFINITY) {
    ocrHintInit(&db_hint, OCR_HINT_DB_T);
    ocrGuid_t aff;
    ocrAffinityGetAt(AFFINITY_PD, home_idx, &aff);
    ocrSetHintValue(&db_hint, OCR_HINT_DB_AFFINITY, ocrAffinityToHintValue(aff));
    hintp = &db_hint;
  } /* AXIS_NOHINT: hintp stays NULL_HINT -> backend's no-hint (RR) policy. */

  void *addr = NULL;
  ocrDbCreate(&db, &addr, PAYLOAD_BYTES, flags, hintp, NO_ALLOC);

  if (!no_acquire) {
    u64 *p = (u64 *)addr;
    for (u32 i = 0; i < PAYLOAD_COUNT; i++) {
      p[i] = cell_idx * 1000 + i;
    }
    ocrDbRelease(db);
    wire_consumers(db, cell_idx, axis, home_idx, n, 1, results_db, NULL_GUID);
  } else {
    /* OCR contract: DB_PROP_NO_ACQUIRE must come back with addr==NULL (the
     * creator does not hold). A backend that silently drops the flag (e.g.
     * on the labeled path) hands back a non-NULL pointer instead -- catch
     * that here rather than relying on it to also corrupt the payload
     * (a mis-acquired creator hold still gets released, and released,
     * before the real writer's RW acquire proceeds, so the data can end up
     * correct despite the contract violation). */
    u64 create_ok = (addr == NULL) ? 1 : 0;
    ocrGuid_t wtmpl;
    ocrEdtTemplateCreate(&wtmpl, writer_edt, 1, 1);
    u64 wparam[1] = {cell_idx};
    /* Writer's output event (post-slot): OCR satisfies it after the writer
     * completes and its RW hold is released (writeback published).  The
     * consumer(s) gate on it so their RO read is happens-after the write.
     * The post-slot event is a single-fire (auto-destroy) event, so the
     * consumer(s) MUST be wired to it BEFORE the writer can become runnable
     * and fire+destroy it -- hence wire_consumers precedes the writer's
     * DB-dep satisfy below (which is what makes the writer runnable). */
    ocrGuid_t writer, writer_done;
    ocrEdtCreate(&writer, wtmpl, EDT_PARAM_DEF, wparam, EDT_PARAM_DEF, NULL,
                 EDT_PROP_NONE, NULL_HINT, &writer_done);
    wire_consumers(db, cell_idx, axis, home_idx, n, create_ok, results_db,
                   writer_done);
    ocrAddDependence(db, writer, 0, DB_MODE_RW);
  }
}

/* ---- driver_edt: paramv={driverPd, resultsDB, R, range0, range1, range2}.
 * Runs all 12 cells for one creator PD. ------------------------------------
 */
ocrGuid_t driver_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  u64 d = paramv[0];
  ocrGuid_t results_db = u64_guid(paramv[1]);
  ocrGuid_t ranges[MAX_RANGES] = {u64_guid(paramv[3]), u64_guid(paramv[4]),
                                  u64_guid(paramv[5])};
  u64 n = 0;
  ocrAffinityCount(AFFINITY_PD, &n);
  u64 remote = (d + 1) % n;
  /* d and remote are always in {0,1,2} by construction (drivers only ever
   * run at PD 0/1), so both index within MAX_RANGES regardless of n. */
  ocrGuid_t range_local = ranges[d];
  ocrGuid_t range_remote = ranges[remote];

  run_cell(1, AXIS_AFFINITY, d, false, results_db, NULL_GUID, 0, n);
  run_cell(2, AXIS_AFFINITY, remote, false, results_db, NULL_GUID, 0, n);
  run_cell(3, AXIS_LABELED, d, false, results_db, range_local, d * 4 + 0, n);
  run_cell(4, AXIS_LABELED, remote, false, results_db, range_remote,
           d * 4 + 1, n);
  run_cell(5, AXIS_NOHINT, HOME_SKIP, false, results_db, NULL_GUID, 0, n);
  run_cell(6, AXIS_NOHINT, HOME_SKIP, false, results_db, NULL_GUID, 0, n);
  run_cell(7, AXIS_AFFINITY, d, true, results_db, NULL_GUID, 0, n);
  run_cell(8, AXIS_AFFINITY, remote, true, results_db, NULL_GUID, 0, n);
  run_cell(9, AXIS_LABELED, d, true, results_db, range_local, d * 4 + 2, n);
  run_cell(10, AXIS_LABELED, remote, true, results_db, range_remote,
           d * 4 + 3, n);
  run_cell(11, AXIS_NOHINT, HOME_SKIP, true, results_db, NULL_GUID, 0, n);
  run_cell(12, AXIS_NOHINT, HOME_SKIP, true, results_db, NULL_GUID, 0, n);
  return NULL_GUID;
}

/* ---- cellrun_edt (FINISH): paramv={resultsDB, R, n}; depv[0]=setup-done
 * control (fires once every range_maker_edt below has completed),
 * depv[1]=rangesDB (RO, u64[R] -- the R labeled-range guids' raw scalar
 * values, one per home PD). Spawns driver PD0 (and PD1, for coverage, when
 * n>1); its outputEvent fires once every driver and all of its transitive
 * descendant EDTs (writers/consumers) have completed. ---------------------
 */
ocrGuid_t cellrun_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  ocrGuid_t results_db = u64_guid(paramv[0]);
  u64 r_count = paramv[1];
  u64 n = paramv[2];
  const u64 *ranges = (const u64 *)depv[1].ptr;

  ocrGuid_t dtmpl;
  ocrEdtTemplateCreate(&dtmpl, driver_edt, 6, 0);

  u64 num_drivers = (n > 1) ? 2u : 1u;
  for (u64 d = 0; d < num_drivers; d++) {
    ocrHint_t h;
    ocrHintInit(&h, OCR_HINT_EDT_T);
    ocrGuid_t aff;
    ocrAffinityGetAt(AFFINITY_PD, d, &aff);
    ocrSetHintValue(&h, OCR_HINT_EDT_AFFINITY, ocrAffinityToHintValue(aff));

    u64 dp[6] = {d, guid_u64(results_db), r_count, 0, 0, 0};
    for (u64 r = 0; r < r_count; r++) {
      dp[3 + r] = ranges[r];
    }
    ocrGuid_t driver;
    ocrEdtCreate(&driver, dtmpl, EDT_PARAM_DEF, dp, EDT_PARAM_DEF, NULL,
                 EDT_PROP_NONE, &h, NULL);
  }
  return NULL_GUID;
}

/* ---- range_maker_edt: paramv={rIndex}; depv[0]=rangesDB (RW, u64[R]),
 * depv[1]=predecessor gate (NULL for the first maker).
 * Runs affinitized to the target home PD so the range it reserves is homed
 * there (see file header), then stores the range guid's raw scalar into its
 * own slot.  NOTE: the range guid is handed off through a real DB, never
 * through ocrEventSatisfy -- a "range"/map guid has no backing object, and
 * at least one backend's event-satisfy path unconditionally looks up the
 * satisfied guid in its object cache, crashing on a guid that was never
 * registered as a real object. ---------------------------------------------
 */
ocrGuid_t range_maker_edt(u32 paramc, u64 *paramv, u32 depc,
                          ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 r = paramv[0];
  ocrGuid_t range;
  ocrGuidRangeCreate(&range, RANGE_SIZE, GUID_USER_DB);
  ((u64 *)depv[0].ptr)[r] = guid_u64(range);
  return NULL_GUID;
}

/* ---- setup_edt (FINISH): paramv={rangesDB, R}. Creates the R
 * range_maker_edts (each RW on rangesDB, hinted to its target PD); its
 * outputEvent fires once all R have written their slot.
 *
 * The makers are CHAINED (each gated on its predecessor's output event) so
 * their same-DB writes are event-ordered: they write disjoint slots, but
 * unordered sibling writers to one DB have defined results only under
 * whole-DB write-exclusion -- a memory model whose writeback is whole-DB and
 * lossy may keep just one sibling's image, dropping the other slots.  The
 * chain keeps this bootstrap within the portable data-race-free contract on
 * every backend at the cost of serializing a 3-EDT setup step. ------------
 */
ocrGuid_t setup_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  ocrGuid_t ranges_db = u64_guid(paramv[0]);
  u64 r_count = paramv[1];

  ocrGuid_t rmtmpl;
  ocrEdtTemplateCreate(&rmtmpl, range_maker_edt, 1, 2);
  ocrGuid_t rm[MAX_RANGES];
  ocrGuid_t rm_done[MAX_RANGES];
  for (u64 r = 0; r < r_count; r++) {
    ocrHint_t h;
    ocrHintInit(&h, OCR_HINT_EDT_T);
    ocrGuid_t aff;
    ocrAffinityGetAt(AFFINITY_PD, r, &aff);
    ocrSetHintValue(&h, OCR_HINT_EDT_AFFINITY, ocrAffinityToHintValue(aff));
    u64 rp[1] = {r};
    ocrEdtCreate(&rm[r], rmtmpl, EDT_PARAM_DEF, rp, EDT_PARAM_DEF, NULL,
                 EDT_PROP_NONE, &h, &rm_done[r]);
  }
  /* Wire every chain gate BEFORE any DB dep: a single-fire output event must
   * be registered before its producer can run, and the DB satisfy below is
   * what makes a maker runnable. */
  for (u64 r = 0; r < r_count; r++) {
    ocrAddDependence(r == 0 ? NULL_GUID : rm_done[r - 1], rm[r], 1,
                     DB_MODE_NULL);
  }
  for (u64 r = 0; r < r_count; r++) {
    ocrAddDependence(ranges_db, rm[r], 0, DB_MODE_RW);
  }
  return NULL_GUID;
}

/* ---- reduce_edt: depv[0]=cellrun's finish control, depv[1]=results DB
 * (RO). --------------------------------------------------------------- */
static void launch_sweep(u64 rounds_left);

/* paramv={rounds_left}. Sequential sweep repetitions (argv[1], default 1):
 * each round re-runs the full 12-cell matrix with fresh DBs; only the last
 * round reports and shuts down (turns the probe into a create-path stress).
 * The round count chains through paramv -- EDTs may run on any PD. */
ocrGuid_t reduce_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  if (paramv[0] > 1) {
    launch_sweep(paramv[0] - 1);
    return NULL_GUID;
  }
  const u8 *results = (const u8 *)depv[1].ptr;
  u32 k = 0;
  for (u32 i = 0; i < NUM_CELLS; i++) {
    k += results[i] ? 1 : 0;
  }
  ocrPrintf("CELLS_OK=%u/12\n", k);
  ocrShutdown();
  return NULL_GUID;
}

static void launch_sweep(u64 rounds_left) {
  u64 n = 0;
  ocrAffinityCount(AFFINITY_PD, &n);
  u64 r_count = (n >= MAX_RANGES) ? MAX_RANGES : n;
  ocrPrintf("=== dbcreate_matrix (%llu PDs) ===\n", (unsigned long long)n);

  ocrGuid_t results_db;
  void *raddr = NULL;
  ocrDbCreate(&results_db, &raddr, NUM_CELLS, DB_PROP_NONE, NULL_HINT,
              NO_ALLOC);
  u8 *results = (u8 *)raddr;
  for (u32 i = 0; i < NUM_CELLS; i++) {
    results[i] = 1;
  }
  ocrDbRelease(results_db);

  /* rangesDB: u64[MAX_RANGES], slot r filled in by range_maker_edt(r).
   * Content is irrelevant until each range_maker_edt overwrites its own
   * slot under RW exclusion; the initial release just lets setup_edt's
   * children start acquiring immediately. */
  ocrGuid_t ranges_db;
  void *rgaddr = NULL;
  ocrDbCreate(&ranges_db, &rgaddr, MAX_RANGES * sizeof(u64), DB_PROP_NONE,
              NULL_HINT, NO_ALLOC);
  ocrDbRelease(ranges_db);

  ocrGuid_t stmpl;
  ocrEdtTemplateCreate(&stmpl, setup_edt, 2, 0);
  u64 sp[2] = {guid_u64(ranges_db), r_count};
  ocrGuid_t setup, setup_done;
  ocrEdtCreate(&setup, stmpl, EDT_PARAM_DEF, sp, EDT_PARAM_DEF, NULL,
               EDT_PROP_FINISH, NULL_HINT, &setup_done);

  ocrGuid_t crtmpl;
  ocrEdtTemplateCreate(&crtmpl, cellrun_edt, 3, 2);
  u64 crparam[3] = {guid_u64(results_db), r_count, n};
  ocrGuid_t cellrun, cells_done;
  ocrEdtCreate(&cellrun, crtmpl, EDT_PARAM_DEF, crparam, EDT_PARAM_DEF, NULL,
               EDT_PROP_FINISH, NULL_HINT, &cells_done);
  ocrAddDependence(setup_done, cellrun, 0, DB_MODE_NULL);
  ocrAddDependence(ranges_db, cellrun, 1, DB_MODE_RO);

  ocrGuid_t rdtmpl;
  ocrEdtTemplateCreate(&rdtmpl, reduce_edt, 1, 2);
  ocrGuid_t reduce;
  ocrEdtCreate(&reduce, rdtmpl, EDT_PARAM_DEF, &rounds_left, EDT_PARAM_DEF, NULL,
               EDT_PROP_NONE, NULL_HINT, NULL);
  ocrAddDependence(cells_done, reduce, 0, DB_MODE_NULL);
  ocrAddDependence(results_db, reduce, 1, DB_MODE_RO);
}

ocrGuid_t mainEdt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;

  u64 rounds = 1;
  if (ocrGetArgc(depv[0].ptr) > 1) {
    u64 r = strtoull(ocrGetArgv(depv[0].ptr, 1), NULL, 10);
    if (r >= 1) {
      rounds = r;
    }
  }
  launch_sweep(rounds);
  return NULL_GUID;
}
