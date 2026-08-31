/*
 * rwpriv.c -- private-working-set release-axis probe.
 *
 * Every chain re-touches its OWN data block, homed one PD away, so the
 * workload's required steady-state communication is ZERO: no other chain
 * ever touches the block, there is no contention, no queueing, and no
 * fairness term.  Whatever wire cost remains per op is purely what the
 * coherence configuration chooses to pay for the privilege of forgetting
 * or remembering local state across a release edge:
 *   - a voluntary-return (purge) arm re-pays permission and/or payload
 *     every window,
 *   - a validation arm pays its per-acquire check,
 *   - a retain/invalidate arm pays nothing at all.
 * R chains re-read (RO) and W chains re-write (RW); the classes coexist
 * in one run because private blocks cannot interfere.
 *
 * A chain is a self-perpetuating EDT sequence pinned to one PD: acquire
 * own block, touch, hold, explicit release (the zero edge), think, create
 * successor.  think > 0 guarantees the window closes between consecutive
 * ops even for the runtime's acquire pipelining, so every op is its own
 * window.  Timing identical to the shared-block steady-state probe:
 * acquire = entry - predecessor post-think stamp, release = the bracket
 * around the explicit release, total = entry-to-entry minus realized
 * spins; per-chain octave+mantissa histograms, window anchored at each
 * chain's trigger arrival.
 *
 * Correctness rides along, stronger than the shared probe can afford:
 * a writer is its block's ONLY writer, so its counter must advance by
 * exactly 1 per op (any other value is a lost or duplicated write); a
 * reader's block must read 0 forever.  The collector re-checks the final
 * counter of every block against its chain's own op count.
 *
 * WARMRW makes every chain's FIRST op a write regardless of role: the
 * chain touches its block once in RW before settling into its class, so
 * a configuration that lets ownership rest where it was last exercised
 * starts each chain as its own block's ex-writer — the private-reuse
 * shape a first-touch application actually has.  A reader's block then
 * carries exactly one write forever.
 *
 * Fixed-work termination (OPS_R/OPS_W nonzero): each chain runs exactly
 * its class's op count; the deadline degrades to a budget guard that
 * fails the cell loudly (CAP-HIT) if hit short of the count, and every
 * op is measured (WARM_MS ignored).
 *
 * args: R W E_HOLD_R_us E_HOLD_W_us E_THINK_us BYTES T_MS WARM_MS
 *       SEED CENSUS JITTER_PCT WARMRW OPS_R OPS_W
 * output: one "RWPRIV OK ..." line (the completion marker) or a
 *         RWPRIV-ORACLE-FAIL / RWPRIV-CAP-HIT line and no marker.
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

/* ---------------------------------------------------------------- limits */
#define RP_MAX_CHAINS 512u
#define RP_OCT 28u
#define RP_MANT 8u
#define RP_NB (RP_OCT * RP_MANT)
#define RP_SHIFT_MIN 8u
#define RP_G_CAP 5000000ull

/* result-block layout, in u64 words */
#define RP_F_ROLE 0
#define RP_F_OPS_TOTAL 1
#define RP_F_OPS_MEAS 2
#define RP_F_VIOL 3
#define RP_F_CAP 4
#define RP_F_SUM_ACQ 5
#define RP_F_SUM_REL 6
#define RP_F_SUM_TOT 7
#define RP_F_HIST 8
#define RP_RESULT_WORDS (RP_F_HIST + 3u * RP_NB)

enum {
  P_ROLE,
  P_CHAIN,
  P_PD,
  P_TRIG_NS,
  P_PREV_STAMP,
  P_PREV_ENTRY,
  P_PREV_SPIN,
  P_STEP_TPL,
  P_FIN_TPL,
  P_COLLECTOR,
  P_SLOT,
  P_RESULT_DB,
  P_CENSUS,
  P_EHOLD,
  P_ETHINK,
  P_BYTES,
  P_TMS,
  P_WMS,
  P_JITTER,
  P_WARM, /* 1 = this EDT is the chain's untimed first-touch write */
  P_KOPS, /* fixed-work mode: this chain's exact op count (0 = timed mode) */
  P_DB,
  P_COUNT
};

struct rp_row_s {
  uint64_t hist[3][RP_NB]; /* 0=acquire 1=release 2=total */
  uint64_t sum[3];
  uint64_t ops_total;
  uint64_t ops_measured;
  uint64_t viol;
  uint64_t cap_hit;
  uint64_t last_stamp;
};
static struct rp_row_s g_rows[RP_MAX_CHAINS];

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

static inline void hist_add(uint64_t *h, u64 ns) {
  u64 v = ns >> RP_SHIFT_MIN;
  unsigned int b;
  if (v < RP_MANT) {
    b = (unsigned int)v;
  } else {
    unsigned int hb = 63u - (unsigned int)__builtin_clzll(v);
    unsigned int mant = (unsigned int)((v >> (hb - 3)) & (RP_MANT - 1));
    b = (hb - 2) * RP_MANT + mant;
    if (b >= RP_NB) b = RP_NB - 1;
  }
  __atomic_fetch_add(&h[b], 1, __ATOMIC_RELAXED);
}

static double bucket_mid_ns(unsigned int b) {
  double unit = (double)(1ull << RP_SHIFT_MIN);
  if (b < RP_MANT) return ((double)b + 0.5) * unit;
  unsigned int oct = b / RP_MANT, mant = b % RP_MANT;
  double base = (double)(1ull << (oct + 2)) * unit;
  double lo = base * (1.0 + (double)mant / RP_MANT);
  double hi = base * (1.0 + (double)(mant + 1) / RP_MANT);
  return (lo + hi) * 0.5;
}

static double hist_quantile_us(const uint64_t *h, double q) {
  u64 total = 0;
  for (unsigned int b = 0; b < RP_NB; b++) total += h[b];
  if (total == 0) return -1.0;
  double target = q * (double)total;
  u64 cum = 0;
  for (unsigned int b = 0; b < RP_NB; b++) {
    cum += h[b];
    if ((double)cum >= target) return bucket_mid_ns(b) / 1000.0;
  }
  return bucket_mid_ns(RP_NB - 1) / 1000.0;
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

static u64 touch(void *ptr, u64 bytes, int is_writer, u64 *stamp_out) {
  volatile u64 *p = (volatile u64 *)ptr;
  u64 words = bytes / sizeof(u64);
  u64 v;
  if (is_writer) {
    v = __atomic_fetch_add((uint64_t *)&p[0], 1, __ATOMIC_RELAXED) + 1;
  } else {
    v = __atomic_load_n((const uint64_t *)&p[0], __ATOMIC_RELAXED);
  }
  u64 acc = 0;
  u64 first = bytes < 4096 ? words : 4096 / sizeof(u64);
  for (u64 i = 8; i < first; i++) {
    if (is_writer)
      p[i] = v;
    else
      acc += p[i];
  }
  for (u64 off = 65536 / sizeof(u64); off < words; off += 65536 / sizeof(u64)) {
    if (is_writer)
      p[off] = v;
    else
      acc += p[off];
  }
  if (acc == 0xDEADBEEFDEADBEEFull) PRINTF("");
  *stamp_out = v;
  return acc;
}

/* ------------------------------------------------------------ chain step */
ocrGuid_t step_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 t0 = now_ns();
  u64 role = paramv[P_ROLE], chain = paramv[P_CHAIN];
  struct rp_row_s *row = &g_rows[chain];
  if (paramv[P_WARM]) {
    /* Untimed first-touch write: exercise ownership once from this chain's
     * own rank, then hand off to the role.  Runs before the window anchor,
     * counts as no op; the successor's entry anchors the window. */
    u64 s = 0;
    touch(depv[0].ptr, paramv[P_BYTES], 1, &s);
    __atomic_store_n(&row->last_stamp, s, __ATOMIC_RELAXED);
    ocrDbRelease(depv[0].guid);
    u64 wp[P_COUNT];
    memcpy(wp, paramv, sizeof(wp));
    wp[P_WARM] = 0;
    ocrHint_t wh;
    pd_hint(&wh, paramv[P_PD], OCR_HINT_EDT_T);
    ocrGuid_t wsucc;
    ocrEdtCreate(&wsucc, u64_guid(paramv[P_STEP_TPL]), P_COUNT, wp, 1, NULL,
                 EDT_PROP_NONE, &wh, NULL);
    ocrAddDependence(u64_guid(paramv[P_DB]), wsucc, 0,
                     role ? DB_MODE_RW : DB_MODE_RO);
    return NULL_GUID;
  }
  if (paramv[P_TRIG_NS] == 0) {
    paramv[P_TRIG_NS] = t0;
    if (paramv[P_CENSUS]) {
      PRINTF("RWPRIV_TASK chain=%lu role=%lu pd=%lu\n", (unsigned long)chain,
             (unsigned long)role, (unsigned long)paramv[P_PD]);
    }
  }
  u64 warm_ns = paramv[P_WMS] * 1000000ull;
  u64 span_ns = paramv[P_TMS] * 1000000ull;
  u64 deadline = paramv[P_TRIG_NS] + warm_ns + span_ns;
  u64 kops = paramv[P_KOPS];

  u64 stamp = 0;
  touch(depv[0].ptr, paramv[P_BYTES], role != 0, &stamp);
  /* sole-toucher oracle: a writer's counter advances by exactly 1 per op,
   * a reader's block stays 0 forever */
  if (role) {
    u64 last = __atomic_load_n(&row->last_stamp, __ATOMIC_RELAXED);
    if (stamp != last + 1)
      __atomic_fetch_add(&row->viol, 1, __ATOMIC_RELAXED);
    __atomic_store_n(&row->last_stamp, stamp, __ATOMIC_RELAXED);
  } else if (stamp != __atomic_load_n(&row->last_stamp, __ATOMIC_RELAXED)) {
    __atomic_fetch_add(&row->viol, 1, __ATOMIC_RELAXED);
  }
  u64 spun = spin_ns(paramv[P_EHOLD]);
  u64 t1 = now_ns();
  ocrDbRelease(depv[0].guid);
  u64 t2 = now_ns();
  u64 think = paramv[P_ETHINK];
  if (paramv[P_JITTER]) {
    int64_t u2 = (int64_t)(((chain * 2654435761ull) >> 6) & 1023ull);
    think = (u64)((int64_t)think +
                  ((int64_t)think * (int64_t)paramv[P_JITTER] *
                   (2 * u2 - 1023)) / (1023 * 100));
  }
  spun += spin_ns(think);

  int in_window =
      kops ? 1 : (t0 >= paramv[P_TRIG_NS] + warm_ns && t0 < deadline);
  if (in_window) {
    if (paramv[P_PREV_STAMP]) {
      u64 acq = t0 - paramv[P_PREV_STAMP];
      hist_add(row->hist[0], acq);
      __atomic_fetch_add(&row->sum[0], acq, __ATOMIC_RELAXED);
    }
    u64 rel = t2 - t1;
    hist_add(row->hist[1], rel);
    __atomic_fetch_add(&row->sum[1], rel, __ATOMIC_RELAXED);
    if (paramv[P_PREV_ENTRY]) {
      u64 cyc = t0 - paramv[P_PREV_ENTRY];
      u64 tot = cyc > paramv[P_PREV_SPIN] ? cyc - paramv[P_PREV_SPIN] : 0;
      hist_add(row->hist[2], tot);
      __atomic_fetch_add(&row->sum[2], tot, __ATOMIC_RELAXED);
    }
    __atomic_fetch_add(&row->ops_measured, 1, __ATOMIC_RELAXED);
  }
  u64 total = __atomic_fetch_add(&row->ops_total, 1, __ATOMIC_RELAXED) + 1;

  int done;
  if (kops) {
    done = total >= kops;
    if (!done && now_ns() >= deadline) {
      __atomic_store_n(&row->cap_hit, 1, __ATOMIC_RELAXED);
      done = 1;
    }
  } else {
    done = now_ns() >= deadline;
  }
  if (total >= RP_G_CAP) {
    __atomic_store_n(&row->cap_hit, 1, __ATOMIC_RELAXED);
    done = 1;
  }

  ocrHint_t h;
  pd_hint(&h, paramv[P_PD], OCR_HINT_EDT_T);
  if (done) {
    ocrGuid_t fin;
    ocrEdtCreate(&fin, u64_guid(paramv[P_FIN_TPL]), P_COUNT, paramv, 1, NULL,
                 EDT_PROP_NONE, &h, NULL);
    ocrAddDependence(u64_guid(paramv[P_RESULT_DB]), fin, 0, DB_MODE_RW);
    return NULL_GUID;
  }

  u64 t3 = now_ns();
  u64 next_params[P_COUNT];
  memcpy(next_params, paramv, sizeof(next_params));
  next_params[P_PREV_STAMP] = t3;
  next_params[P_PREV_ENTRY] = t0;
  next_params[P_PREV_SPIN] = spun;
  ocrGuid_t succ;
  ocrEdtCreate(&succ, u64_guid(paramv[P_STEP_TPL]), P_COUNT, next_params, 1,
               NULL, EDT_PROP_NONE, &h, NULL);
  ocrAddDependence(u64_guid(paramv[P_DB]), succ, 0,
                   role ? DB_MODE_RW : DB_MODE_RO);
  return NULL_GUID;
}

/* ------------------------------------------------------------- finisher */
ocrGuid_t fin_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  struct rp_row_s *row = &g_rows[paramv[P_CHAIN]];
  u64 *r = (u64 *)depv[0].ptr;
  r[RP_F_ROLE] = paramv[P_ROLE];
  r[RP_F_OPS_TOTAL] = __atomic_load_n(&row->ops_total, __ATOMIC_RELAXED);
  r[RP_F_OPS_MEAS] = __atomic_load_n(&row->ops_measured, __ATOMIC_RELAXED);
  r[RP_F_VIOL] = __atomic_load_n(&row->viol, __ATOMIC_RELAXED);
  r[RP_F_CAP] = __atomic_load_n(&row->cap_hit, __ATOMIC_RELAXED);
  for (unsigned int k = 0; k < 3; k++) {
    r[RP_F_SUM_ACQ + k] = __atomic_load_n(&row->sum[k], __ATOMIC_RELAXED);
    for (unsigned int b = 0; b < RP_NB; b++)
      r[RP_F_HIST + k * RP_NB + b] =
          __atomic_load_n(&row->hist[k][b], __ATOMIC_RELAXED);
  }
  ocrGuid_t rdb = depv[0].guid;
  ocrDbRelease(rdb);
  ocrAddDependence(rdb, u64_guid(paramv[P_COLLECTOR]), (u32)paramv[P_SLOT],
                   DB_MODE_RO);
  return NULL_GUID;
}

/* ------------------------------------------------------------ collector */
/* paramv: {R, W, T_MS, WARM_MS, EHR, EHW, ETH, BYTES, PDS, RHO_MILLI,
 *          JITTER, WARMRW}; depv: A result blocks then A data blocks (RO). */
ocrGuid_t collector_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 R = paramv[0], W = paramv[1], t_ms = paramv[2];
  u64 A = R + W;
  static uint64_t pooled[2][3][RP_NB];
  double sum[2][3] = {{0}};
  u64 ops_meas[2] = {0, 0}, viol = 0, cap = 0, final_bad = 0;
  double cn[2] = {0, 0}, cs[2] = {0, 0}, cq[2] = {0, 0};
  memset(pooled, 0, sizeof(pooled));
  for (u64 i = 0; i < A; i++) {
    const u64 *r = (const u64 *)depv[i].ptr;
    const u64 *blk = (const u64 *)depv[A + i].ptr;
    unsigned int c = r[RP_F_ROLE] ? 1 : 0;
    u64 want = (c ? r[RP_F_OPS_TOTAL] : 0) + paramv[11];
    u64 got = __atomic_load_n((const uint64_t *)&blk[0], __ATOMIC_RELAXED);
    if (got != want) final_bad++;
    ops_meas[c] += r[RP_F_OPS_MEAS];
    double co = (double)r[RP_F_OPS_MEAS];
    cn[c] += 1.0;
    cs[c] += co;
    cq[c] += co * co;
    viol += r[RP_F_VIOL];
    cap += r[RP_F_CAP];
    for (unsigned int k = 0; k < 3; k++) {
      sum[c][k] += (double)r[RP_F_SUM_ACQ + k];
      for (unsigned int b = 0; b < RP_NB; b++)
        pooled[c][k][b] += r[RP_F_HIST + k * RP_NB + b];
    }
  }
  double cv[2] = {-1.0, -1.0};
  for (unsigned int c = 0; c < 2; c++) {
    if (cn[c] >= 2.0 && cs[c] > 0.0) {
      double mean = cs[c] / cn[c];
      double var = cq[c] / cn[c] - mean * mean;
      cv[c] = var > 0.0 ? sqrt(var) / mean : 0.0;
    }
  }

  if (cap) {
    PRINTF("RWPRIV-CAP-HIT chains=%lu\n", (unsigned long)cap);
    ocrShutdown();
    return NULL_GUID;
  }
  if (viol || final_bad) {
    PRINTF("RWPRIV-ORACLE-FAIL viol=%lu final_bad=%lu\n", (unsigned long)viol,
           (unsigned long)final_bad);
    ocrShutdown();
    return NULL_GUID;
  }

  double span_s = (double)t_ms / 1000.0;
  double rx = (double)ops_meas[0] / span_s, wx = (double)ops_meas[1] / span_s;
#define QQ(c, k, q) hist_quantile_us(pooled[c][k], q)
#define MEAN(c, k) \
  (ops_meas[c] ? sum[c][k] / 1000.0 / (double)ops_meas[c] : -1.0)
  PRINTF(
      "RWPRIV OK PDS=%lu R=%lu W=%lu EHR=%lu EHW=%lu ETH=%lu BYTES=%lu "
      "T_MS=%lu WARM_MS=%lu RHO=%.4f ROPS=%lu WOPS=%lu RXPUT=%.1f WXPUT=%.1f "
      "RTOT_P50=%.2f RTOT_P90=%.2f RTOT_P99=%.2f "
      "WTOT_P50=%.2f WTOT_P90=%.2f WTOT_P99=%.2f "
      "RACQ_P99=%.2f RREL_P99=%.2f WACQ_P99=%.2f WREL_P99=%.2f "
      "RACQ_MEAN=%.2f RREL_MEAN=%.2f RTOT_MEAN=%.2f "
      "WACQ_MEAN=%.2f WREL_MEAN=%.2f WTOT_MEAN=%.2f "
      "RCV=%.3f WCV=%.3f JIT=%lu\n",
      (unsigned long)paramv[8], (unsigned long)R, (unsigned long)W,
      (unsigned long)paramv[4], (unsigned long)paramv[5],
      (unsigned long)paramv[6], (unsigned long)paramv[7],
      (unsigned long)t_ms, (unsigned long)paramv[3],
      (double)paramv[9] / 10000.0, (unsigned long)ops_meas[0],
      (unsigned long)ops_meas[1], rx, wx, QQ(0, 2, 0.50), QQ(0, 2, 0.90),
      QQ(0, 2, 0.99), QQ(1, 2, 0.50), QQ(1, 2, 0.90), QQ(1, 2, 0.99),
      QQ(0, 0, 0.99), QQ(0, 1, 0.99), QQ(1, 0, 0.99), QQ(1, 1, 0.99),
      MEAN(0, 0), MEAN(0, 1), MEAN(0, 2), MEAN(1, 0), MEAN(1, 1), MEAN(1, 2),
      cv[0], cv[1], (unsigned long)paramv[10]);
  (void)0;
#undef QQ
#undef MEAN
  ocrShutdown();
  return NULL_GUID;
}

/* -------------------------------------------------------------- mainEdt */
ocrGuid_t mainEdt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 R = 24, W = 24, ehr = 1, ehw = 1, eth = 20, bytes = 65536;
  u64 t_ms = 4000, warm_ms = 1500, seed = 1, census = 0, jitter = 0;
  u64 warmrw = 0, ops_r = 0, ops_w = 0;
  u64 argc = getArgc(depv[0].ptr);
  u64 *args[] = {&R, &W, &ehr, &ehw, &eth, &bytes, &t_ms, &warm_ms,
                 &seed, &census, &jitter, &warmrw, &ops_r, &ops_w};
  for (u64 i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    if (argc > i + 1) *args[i] = (u64)atol(getArgv(depv[0].ptr, i + 1));
  (void)seed;

  if (bytes < 4160) bytes = 4160;
  u64 A = R + W;
  if (A == 0 || A > RP_MAX_CHAINS) {
    PRINTF("RWPRIV-ORACLE-FAIL kind=args A=%lu\n", (unsigned long)A);
    ocrShutdown();
    return NULL_GUID;
  }
  memset(g_rows, 0, sizeof(g_rows));

  u64 pd_count = 0;
  ocrAffinityCount(AFFINITY_PD, &pd_count);
  if (pd_count == 0) pd_count = 1;

  ocrGuid_t trigger;
  ocrEventCreate(&trigger, OCR_EVENT_ONCE_T, EVT_PROP_NONE);

  ocrGuid_t step_tpl, start_tpl, fin_tpl, coll_tpl;
  ocrEdtTemplateCreate(&step_tpl, step_edt, P_COUNT, 1);
  ocrEdtTemplateCreate(&start_tpl, step_edt, P_COUNT, 2);
  ocrEdtTemplateCreate(&fin_tpl, fin_edt, P_COUNT, 1);
  ocrEdtTemplateCreate(&coll_tpl, collector_edt, 12, (u32)(2 * A));

  u64 rho_milli = (u64)((double)W / (double)A * 10000.0 + 0.5);
  u64 cparams[12] = {R,   W,     t_ms,     warm_ms,   ehr,    ehw,
                     eth, bytes, pd_count, rho_milli, jitter, warmrw};
  ocrHint_t h0;
  pd_hint(&h0, 0, OCR_HINT_EDT_T);
  ocrGuid_t collector;
  ocrEdtCreate(&collector, coll_tpl, 12, cparams, (u32)(2 * A), NULL,
               EDT_PROP_NONE, &h0, NULL);

  for (u64 c = 0; c < A; c++) {
    int is_writer = c < W;
    u64 pd = c % pd_count;
    u64 home_pd = (pd + 1) % pd_count; /* always one PD away: the required
                                        * communication is zero, the wire
                                        * distance is not */

    ocrHint_t dh;
    pd_hint(&dh, home_pd, OCR_HINT_DB_T);
    ocrGuid_t db;
    u64 *p;
    ocrDbCreate(&db, (void **)&p, bytes, DB_PROP_NONE, &dh, NO_ALLOC);
    memset(p, 0, bytes);
    ocrDbRelease(db);
    ocrAddDependence(db, collector, (u32)(A + c), DB_MODE_RO);

    ocrHint_t rh;
    pd_hint(&rh, pd, OCR_HINT_DB_T);
    ocrGuid_t rdb;
    u64 *rp;
    ocrDbCreate(&rdb, (void **)&rp, RP_RESULT_WORDS * sizeof(u64),
                DB_PROP_NONE, &rh, NO_ALLOC);
    memset(rp, 0, RP_RESULT_WORDS * sizeof(u64));
    ocrDbRelease(rdb);

    u64 params[P_COUNT];
    memset(params, 0, sizeof(params));
    params[P_ROLE] = is_writer ? 1 : 0;
    params[P_CHAIN] = c;
    params[P_PD] = pd;
    params[P_STEP_TPL] = guid_u64(step_tpl);
    params[P_FIN_TPL] = guid_u64(fin_tpl);
    params[P_COLLECTOR] = guid_u64(collector);
    params[P_SLOT] = c;
    params[P_RESULT_DB] = guid_u64(rdb);
    params[P_CENSUS] = census;
    params[P_EHOLD] = is_writer ? ehw : ehr;
    params[P_ETHINK] = eth;
    params[P_BYTES] = bytes;
    params[P_TMS] = t_ms;
    params[P_WMS] = warm_ms;
    params[P_JITTER] = jitter > 90 ? 90 : jitter;
    params[P_WARM] = warmrw;
    params[P_KOPS] = is_writer ? ops_w : ops_r;
    params[P_DB] = guid_u64(db);

    ocrHint_t eh;
    pd_hint(&eh, pd, OCR_HINT_EDT_T);
    ocrGuid_t starter;
    ocrEdtCreate(&starter, start_tpl, P_COUNT, params, 2, NULL, EDT_PROP_NONE,
                 &eh, NULL);
    ocrAddDependence(db, starter, 0,
                     (is_writer || warmrw) ? DB_MODE_RW : DB_MODE_RO);
    ocrAddDependence(trigger, starter, 1, DB_MODE_NULL);
  }

  ocrEventSatisfy(trigger, NULL_GUID);
  return NULL_GUID;
}
