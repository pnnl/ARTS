/*
 * rwhandoff.c -- phase-alternating producer/consumer handoff probe.
 *
 * L independent lattices advance through generations on one data block
 * each.  Generation g's producer writes region (g % NREG), releases, and
 * satisfies one event that wakes BOTH the generation's consumer (reading
 * the region just written) and the next producer (writing the NEXT
 * region).  The two successors run concurrently on byte-disjoint regions,
 * so the program is race-free at byte granularity by event order alone --
 * yet a block-granular exclusive arm must serialize them, and every
 * generation forces a fresh read/write phase change with exactly ONE
 * arrival per side: there is never a second reader to batch into a turn,
 * never a same-rank successor to amortize a resting grant, because
 * producers and consumers each rotate placement every generation.  What
 * this measures is each configuration's cost of an interleaved handoff
 * with zero batching fodder.
 *
 * The lattice rate (generations/s in the window) is the primary figure;
 * the producer release bracket rides along.  Entry-to-entry generation
 * latency spans two ranks' clocks, so it is recorded only when the
 * CROSSCLOCK arg says the ranks share a clock domain (colocated
 * launcher); termination compares are coarse enough to tolerate ordinary
 * clock skew either way.
 *
 * Correctness: a producer is chain-ordered with every earlier producer,
 * so the region it overwrites must hold exactly the stamp from NREG
 * generations ago.  A consumer's snapshot is concurrent with LATER
 * producers only; its region stamp must be its own generation's or a
 * later overwrite of the same region slot (stamp ≡ own (mod NREG),
 * stamp >= own) -- anything else is a lost or misdelivered write.  A
 * consumer that fails withholds its latch decrement, so the collector
 * never runs and no completion marker is printed.
 *
 * args: L NREG BYTES E_HOLD_W_us E_HOLD_R_us E_THINK_us T_MS WARM_MS
 *       CENSUS CROSSCLOCK
 * output: one "RWHANDOFF OK ..." line (the completion marker) or an
 *         RWHANDOFF-ORACLE-FAIL line and no marker.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ocr.h"
#include "extensions/ocr-affinity.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RH_MAX_LAT 256u
#define RH_NB 16u /* coarse log2 buckets, 256ns .. >8ms */

/* result-block layout, in u64 words */
#define RH_F_GENS_TOTAL 0
#define RH_F_GENS_MEAS 1
#define RH_F_VIOL 2
#define RH_F_SUM_REL 3
#define RH_F_SUM_GEN 4
#define RH_F_HREL 5
#define RH_F_HGEN (RH_F_HREL + RH_NB)
#define RH_RESULT_WORDS (RH_F_HGEN + RH_NB)

enum {
  P_LAT,
  P_GEN,
  P_TRIG_NS,
  P_PREV_ENTRY,
  P_PREV_SPIN,
  P_PREV_EVT, /* the event that woke this EDT; its creator's successor
               * destroys it (fire-and-linger events need an explicit
               * destroy; 0 = the shared start trigger, never destroyed) */
  P_PROD_TPL,
  P_CONS_TPL,
  P_FIN_TPL,
  P_COLLECTOR,
  P_SLOT,
  P_RESULT_DB,
  P_LATCH,
  P_DB,
  P_CENSUS,
  P_NREG,
  P_BYTES,
  P_EHW,
  P_EHR,
  P_ETH,
  P_TMS,
  P_WMS,
  P_XCLK,
  P_PDS,
  M_GENS_MEAS,
  M_VIOL,
  M_SUM_REL,
  M_SUM_GEN,
  M_HREL, /* 16 words */
  M_HGEN = M_HREL + RH_NB,
  P_COUNT = M_HGEN + RH_NB
};

static inline u64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

static inline u64 spin_ns(u64 us) {
  if (us == 0) return 0;
  u64 t0 = now_ns(), tgt = us * 1000ull;
  while (now_ns() - t0 < tgt)
    ;
  return now_ns() - t0;
}

static inline unsigned int hbucket(u64 ns) {
  u64 v = ns >> 8;
  if (v == 0) return 0;
  unsigned int b = 63u - (unsigned int)__builtin_clzll(v);
  return b >= RH_NB ? RH_NB - 1 : b;
}

static double hbucket_mid_us(unsigned int b) {
  double lo = (double)(256ull << b);
  return (b == 0 ? lo * 0.5 : lo * 1.5) / 1000.0;
}

static double hist_quantile_us(const uint64_t *h, double q) {
  u64 total = 0;
  for (unsigned int b = 0; b < RH_NB; b++) total += h[b];
  if (total == 0) return -1.0;
  double target = q * (double)total;
  u64 cum = 0;
  for (unsigned int b = 0; b < RH_NB; b++) {
    cum += h[b];
    if ((double)cum >= target) return hbucket_mid_us(b);
  }
  return hbucket_mid_us(RH_NB - 1);
}

static inline u64 guid_u64(ocrGuid_t g) { return (u64)g.guid; }
static inline ocrGuid_t u64_guid(u64 v) {
  ocrGuid_t g;
  g.guid = (intptr_t)v;
  return g;
}

static void pd_hint(ocrHint_t *h, u64 pd, ocrHintType_t type) {
  ocrHintInit(h, type);
  ocrGuid_t aff;
  ocrAffinityGetAt(AFFINITY_PD, pd, &aff);
  ocrSetHintValue(h,
                  type == OCR_HINT_EDT_T ? OCR_HINT_EDT_AFFINITY
                                         : OCR_HINT_DB_AFFINITY,
                  ocrAffinityToHintValue(aff));
}

static inline u64 region_words(u64 bytes, u64 nreg) {
  return ((bytes / nreg) & ~63ull) / sizeof(u64);
}

/* --------------------------------------------------------------- producer */
ocrGuid_t prod_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 t0 = now_ns();
  u64 lat = paramv[P_LAT], g = paramv[P_GEN];
  u64 nreg = paramv[P_NREG], pds = paramv[P_PDS];
  if (paramv[P_PREV_EVT]) ocrEventDestroy(u64_guid(paramv[P_PREV_EVT]));
  if (paramv[P_TRIG_NS] == 0) {
    paramv[P_TRIG_NS] = t0;
    if (paramv[P_CENSUS])
      PRINTF("RWHANDOFF_TASK lat=%lu pd=%lu\n", (unsigned long)lat,
             (unsigned long)((lat + g) % pds));
  }
  u64 warm_ns = paramv[P_WMS] * 1000000ull;
  u64 deadline = paramv[P_TRIG_NS] + warm_ns + paramv[P_TMS] * 1000000ull;

  u64 rw = region_words(paramv[P_BYTES], nreg);
  u64 *base = (u64 *)depv[0].ptr + (g % nreg) * rw;
  /* chain-ordered exact check: this region slot was last written NREG
   * generations ago */
  u64 old = __atomic_load_n((const uint64_t *)&base[0], __ATOMIC_RELAXED);
  u64 want_old = g >= nreg ? g + 1 - nreg : 0;
  if (old != want_old) paramv[M_VIOL]++;
  __atomic_store_n((uint64_t *)&base[0], g + 1, __ATOMIC_RELAXED);
  u64 fill = rw < 512 ? rw : 512;
  for (u64 i = 8; i < fill; i++) base[i] = g + 1;

  u64 spun = spin_ns(paramv[P_EHW]);
  u64 t1 = now_ns();
  ocrDbRelease(depv[0].guid);
  u64 t2 = now_ns();

  int in_window = t0 >= paramv[P_TRIG_NS] + warm_ns && t0 < deadline;
  if (in_window) {
    u64 rel = t2 - t1;
    paramv[M_SUM_REL] += rel;
    paramv[M_HREL + hbucket(rel)]++;
    if (paramv[P_XCLK] && paramv[P_PREV_ENTRY]) {
      u64 cyc = t0 - paramv[P_PREV_ENTRY];
      u64 gen = cyc > paramv[P_PREV_SPIN] ? cyc - paramv[P_PREV_SPIN] : 0;
      paramv[M_SUM_GEN] += gen;
      paramv[M_HGEN + hbucket(gen)]++;
    }
    paramv[M_GENS_MEAS]++;
  }

  if (now_ns() >= deadline) {
    ocrHint_t h;
    pd_hint(&h, (lat + g) % pds, OCR_HINT_EDT_T);
    paramv[P_GEN] = g + 1; /* gens produced in total */
    ocrGuid_t fin;
    ocrEdtCreate(&fin, u64_guid(paramv[P_FIN_TPL]), P_COUNT, paramv, 2, NULL,
                 EDT_PROP_NONE, &h, NULL);
    ocrAddDependence(u64_guid(paramv[P_RESULT_DB]), fin, 0, DB_MODE_RW);
    ocrAddDependence(u64_guid(paramv[P_LATCH]), fin, 1, DB_MODE_NULL);
    ocrEventSatisfySlot(u64_guid(paramv[P_LATCH]), NULL_GUID,
                        OCR_EVENT_LATCH_DECR_SLOT); /* the guard */
    return NULL_GUID;
  }

  /* one consumer for THIS generation, one successor for the next; both
   * woken by one event, placed on rotating, mutually distant ranks.  The
   * think spin runs BEFORE the successors are sealed so the realized
   * spin can ride to the next producer's latency figure; the handoff
   * itself is gated by the satisfy, which stays last. */
  spun += spin_ns(paramv[P_ETH]);
  ocrEventSatisfySlot(u64_guid(paramv[P_LATCH]), NULL_GUID,
                      OCR_EVENT_LATCH_INCR_SLOT);
  ocrGuid_t ev;
  ocrEventCreate(&ev, OCR_EVENT_ONCE_T, EVT_PROP_NONE);

  u64 wpd_next = (lat + g + 1) % pds;
  u64 rpd = ((lat + g) + (pds > 1 ? pds / 2 : 0)) % pds;

  u64 cparams[P_COUNT];
  memcpy(cparams, paramv, sizeof(cparams));
  cparams[P_PREV_EVT] = 0; /* the consumer never destroys the event */
  ocrHint_t ch;
  pd_hint(&ch, rpd, OCR_HINT_EDT_T);
  ocrGuid_t cons;
  ocrEdtCreate(&cons, u64_guid(paramv[P_CONS_TPL]), P_COUNT, cparams, 2, NULL,
               EDT_PROP_NONE, &ch, NULL);
  ocrAddDependence(u64_guid(paramv[P_DB]), cons, 0, DB_MODE_RO);
  ocrAddDependence(ev, cons, 1, DB_MODE_NULL);

  u64 nparams[P_COUNT];
  memcpy(nparams, paramv, sizeof(nparams));
  nparams[P_GEN] = g + 1;
  nparams[P_PREV_ENTRY] = t0;
  nparams[P_PREV_EVT] = guid_u64(ev);
  ocrHint_t nh;
  pd_hint(&nh, wpd_next, OCR_HINT_EDT_T);
  ocrGuid_t succ;
  ocrEdtCreate(&succ, u64_guid(paramv[P_PROD_TPL]), P_COUNT, nparams, 2, NULL,
               EDT_PROP_NONE, &nh, NULL);
  ocrAddDependence(u64_guid(paramv[P_DB]), succ, 0, DB_MODE_RW);
  ocrAddDependence(ev, succ, 1, DB_MODE_NULL);

  spun += spin_ns(paramv[P_ETH]);
  /* nparams were sealed before the think spin; patch the realized value in
   * would reorder creation, so the spin is charged where it was spent */
  (void)spun;
  ocrEventSatisfy(ev, NULL_GUID);
  return NULL_GUID;
}

/* --------------------------------------------------------------- consumer */
ocrGuid_t cons_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 g = paramv[P_GEN], nreg = paramv[P_NREG];
  u64 rw = region_words(paramv[P_BYTES], nreg);
  const u64 *base = (const u64 *)depv[0].ptr + (g % nreg) * rw;
  u64 stamp = __atomic_load_n((const uint64_t *)&base[0], __ATOMIC_RELAXED);
  /* concurrent-with-later-writers legality: own stamp or a later overwrite
   * of the same region slot */
  if (stamp < g + 1 || (stamp - (g + 1)) % nreg != 0) {
    PRINTF("RWHANDOFF-ORACLE-FAIL lat=%lu gen=%lu stamp=%lu\n",
           (unsigned long)paramv[P_LAT], (unsigned long)g,
           (unsigned long)stamp);
    return NULL_GUID; /* withhold the decrement: no marker, cell fails */
  }
  u64 acc = 0;
  u64 fill = rw < 512 ? rw : 512;
  for (u64 i = 8; i < fill; i++) acc += base[i];
  if (acc == 0xDEADBEEFDEADBEEFull) PRINTF("");
  spin_ns(paramv[P_EHR]);
  ocrDbRelease(depv[0].guid);
  ocrEventSatisfySlot(u64_guid(paramv[P_LATCH]), NULL_GUID,
                      OCR_EVENT_LATCH_DECR_SLOT);
  return NULL_GUID;
}

/* --------------------------------------------------------------- finisher */
ocrGuid_t fin_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 *r = (u64 *)depv[0].ptr;
  r[RH_F_GENS_TOTAL] = paramv[P_GEN];
  r[RH_F_GENS_MEAS] = paramv[M_GENS_MEAS];
  r[RH_F_VIOL] = paramv[M_VIOL];
  r[RH_F_SUM_REL] = paramv[M_SUM_REL];
  r[RH_F_SUM_GEN] = paramv[M_SUM_GEN];
  for (unsigned int b = 0; b < RH_NB; b++) {
    r[RH_F_HREL + b] = paramv[M_HREL + b];
    r[RH_F_HGEN + b] = paramv[M_HGEN + b];
  }
  ocrEventDestroy(u64_guid(paramv[P_LATCH]));
  ocrGuid_t rdb = depv[0].guid;
  ocrDbRelease(rdb);
  ocrAddDependence(rdb, u64_guid(paramv[P_COLLECTOR]), (u32)paramv[P_SLOT],
                   DB_MODE_RO);
  return NULL_GUID;
}

/* --------------------------------------------------------------- collector */
/* paramv: {L, NREG, BYTES, EHW, EHR, ETH, T_MS, WARM_MS, PDS, XCLK};
 * depv: L result blocks (RO). */
ocrGuid_t collector_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 L = paramv[0], t_ms = paramv[6];
  static uint64_t prel[RH_NB], pgen[RH_NB];
  memset(prel, 0, sizeof(prel));
  memset(pgen, 0, sizeof(pgen));
  u64 gens_meas = 0, gens_total = 0, viol = 0;
  double sum_rel = 0, sum_gen = 0;
  for (u64 l = 0; l < L; l++) {
    const u64 *r = (const u64 *)depv[l].ptr;
    gens_total += r[RH_F_GENS_TOTAL];
    gens_meas += r[RH_F_GENS_MEAS];
    viol += r[RH_F_VIOL];
    sum_rel += (double)r[RH_F_SUM_REL];
    sum_gen += (double)r[RH_F_SUM_GEN];
    for (unsigned int b = 0; b < RH_NB; b++) {
      prel[b] += r[RH_F_HREL + b];
      pgen[b] += r[RH_F_HGEN + b];
    }
  }
  if (viol) {
    PRINTF("RWHANDOFF-ORACLE-FAIL kind=producer viol=%lu\n",
           (unsigned long)viol);
    ocrShutdown();
    return NULL_GUID;
  }
  double span_s = (double)t_ms / 1000.0;
  PRINTF("RWHANDOFF OK PDS=%lu L=%lu NREG=%lu BYTES=%lu EHW=%lu EHR=%lu "
         "ETH=%lu T_MS=%lu WARM_MS=%lu GENS=%lu GXPUT=%.1f "
         "WREL_P50=%.2f WREL_P99=%.2f WREL_MEAN=%.2f "
         "GTOT_P50=%.2f GTOT_P99=%.2f GTOT_MEAN=%.2f\n",
         (unsigned long)paramv[8], (unsigned long)L, (unsigned long)paramv[1],
         (unsigned long)paramv[2], (unsigned long)paramv[3],
         (unsigned long)paramv[4], (unsigned long)paramv[5],
         (unsigned long)t_ms, (unsigned long)paramv[7],
         (unsigned long)gens_meas, (double)gens_meas / span_s,
         hist_quantile_us(prel, 0.50), hist_quantile_us(prel, 0.99),
         gens_meas ? sum_rel / 1000.0 / (double)gens_meas : -1.0,
         hist_quantile_us(pgen, 0.50), hist_quantile_us(pgen, 0.99),
         gens_meas ? sum_gen / 1000.0 / (double)gens_meas : -1.0);
  ocrShutdown();
  return NULL_GUID;
}

/* ---------------------------------------------------------------- mainEdt */
ocrGuid_t mainEdt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 L = 16, nreg = 8, bytes = 65536, ehw = 1, ehr = 1, eth = 20;
  u64 t_ms = 4000, warm_ms = 1500, census = 0, xclk = 1;
  u64 argc = getArgc(depv[0].ptr);
  u64 *args[] = {&L, &nreg, &bytes, &ehw, &ehr, &eth, &t_ms, &warm_ms,
                 &census, &xclk};
  for (u64 i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    if (argc > i + 1) *args[i] = (u64)atol(getArgv(depv[0].ptr, i + 1));

  if (L == 0 || L > RH_MAX_LAT || nreg == 0) {
    PRINTF("RWHANDOFF-ORACLE-FAIL kind=args L=%lu NREG=%lu\n",
           (unsigned long)L, (unsigned long)nreg);
    ocrShutdown();
    return NULL_GUID;
  }
  if (region_words(bytes, nreg) * sizeof(u64) < 4160) {
    bytes = nreg * 4224; /* keep every region a full stamp + fill span */
  }

  u64 pd_count = 0;
  ocrAffinityCount(AFFINITY_PD, &pd_count);
  if (pd_count == 0) pd_count = 1;

  ocrGuid_t trigger;
  ocrEventCreate(&trigger, OCR_EVENT_ONCE_T, EVT_PROP_NONE);

  ocrGuid_t prod_tpl, start_tpl, cons_tpl, fin_tpl, coll_tpl;
  ocrEdtTemplateCreate(&prod_tpl, prod_edt, P_COUNT, 2);
  ocrEdtTemplateCreate(&start_tpl, prod_edt, P_COUNT, 2);
  ocrEdtTemplateCreate(&cons_tpl, cons_edt, P_COUNT, 2);
  ocrEdtTemplateCreate(&fin_tpl, fin_edt, P_COUNT, 2);
  ocrEdtTemplateCreate(&coll_tpl, collector_edt, 10, (u32)L);

  u64 cparams[10] = {L,    nreg,    bytes, ehw,      ehr,
                     eth,  t_ms,    warm_ms, pd_count, xclk};
  ocrHint_t h0;
  pd_hint(&h0, 0, OCR_HINT_EDT_T);
  ocrGuid_t collector;
  ocrEdtCreate(&collector, coll_tpl, 10, cparams, (u32)L, NULL, EDT_PROP_NONE,
               &h0, NULL);

  for (u64 l = 0; l < L; l++) {
    u64 home = l % pd_count;
    ocrHint_t dh;
    pd_hint(&dh, home, OCR_HINT_DB_T);
    ocrGuid_t db;
    u64 *p;
    ocrDbCreate(&db, (void **)&p, bytes, DB_PROP_NONE, &dh, NO_ALLOC);
    memset(p, 0, bytes);
    ocrDbRelease(db);

    ocrGuid_t rdb;
    u64 *rp;
    ocrDbCreate(&rdb, (void **)&rp, RH_RESULT_WORDS * sizeof(u64),
                DB_PROP_NONE, &dh, NO_ALLOC);
    memset(rp, 0, RH_RESULT_WORDS * sizeof(u64));
    ocrDbRelease(rdb);

    ocrGuid_t latch;
    ocrEventCreate(&latch, OCR_EVENT_LATCH_T, EVT_PROP_NONE);
    ocrEventSatisfySlot(latch, NULL_GUID,
                        OCR_EVENT_LATCH_INCR_SLOT); /* the guard */

    u64 params[P_COUNT];
    memset(params, 0, sizeof(params));
    params[P_LAT] = l;
    params[P_PROD_TPL] = guid_u64(prod_tpl);
    params[P_CONS_TPL] = guid_u64(cons_tpl);
    params[P_FIN_TPL] = guid_u64(fin_tpl);
    params[P_COLLECTOR] = guid_u64(collector);
    params[P_SLOT] = l;
    params[P_RESULT_DB] = guid_u64(rdb);
    params[P_LATCH] = guid_u64(latch);
    params[P_DB] = guid_u64(db);
    params[P_CENSUS] = census;
    params[P_NREG] = nreg;
    params[P_BYTES] = bytes;
    params[P_EHW] = ehw;
    params[P_EHR] = ehr;
    params[P_ETH] = eth;
    params[P_TMS] = t_ms;
    params[P_WMS] = warm_ms;
    params[P_XCLK] = xclk;
    params[P_PDS] = pd_count;

    ocrHint_t eh;
    pd_hint(&eh, home, OCR_HINT_EDT_T);
    ocrGuid_t starter;
    ocrEdtCreate(&starter, start_tpl, P_COUNT, params, 2, NULL, EDT_PROP_NONE,
                 &eh, NULL);
    ocrAddDependence(db, starter, 0, DB_MODE_RW);
    ocrAddDependence(trigger, starter, 1, DB_MODE_NULL);
  }

  ocrEventSatisfy(trigger, NULL_GUID);
  return NULL_GUID;
}
