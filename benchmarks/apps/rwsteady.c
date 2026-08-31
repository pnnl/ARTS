/*
 * rwsteady.c -- closed-loop steady-state reader/writer coherence probe.
 *
 * R reader chains and W writer chains free-run against D shared data blocks
 * for a fixed time window.  A chain is a self-perpetuating EDT sequence
 * pinned to one PD: each EDT acquires the target block in its chain's mode
 * (RO for readers, RW for writers), touches a bounded slice of the payload,
 * optionally spins while holding, releases EXPLICITLY, spins its think time,
 * and creates its successor.  The explicit release forces a coherence zero
 * edge between consecutive ops of one chain; overlap ACROSS chains on one
 * node is deliberate (per-node batching is a protocol property, controlled
 * by chains-per-PD).
 *
 * Timing: a chain never leaves its PD, so all its timestamps come from one
 * monotonic clock; nothing compares clocks across PDs.  Each op yields
 *   acquire  = this EDT's entry - predecessor's post-think stamp
 *              (create + dispatch + acquire, incl. any protocol wait),
 *   release  = the bracket around the explicit release
 *              (blocking arms pay their round/publish here),
 *   total    = entry-to-entry minus the realized spins
 *              (the class's whole per-op overhead; quantiles come from this).
 * Samples land in per-PD process globals (one process per rank; a chain's
 * rows are written only by that chain), in octave+mantissa buckets.
 *
 * The measurement window is anchored at each chain's trigger arrival:
 * ops whose entry falls in [t_trig + warm, t_trig + warm + span) count.
 * Chains run to their deadline regardless of progress, so starvation is
 * reported as a class throughput collapse, never as a hang.  A chain that
 * hits the op cap is a failed cell (RWSTEADY-CAP-HIT), not a short one.
 *
 * Correctness rides along: writers bump the block's op counter with an
 * atomic add (same-PD RW concurrency is legal under per-node-exclusive
 * semantics), readers assert the counter never goes backwards per
 * (chain, block), and the collector -- which runs after every chain slot is
 * satisfied, hence after every release -- checks the summed counters equal
 * the writers' own op counts exactly.  Any violation suppresses the
 * completion marker.
 *
 * Placement: block homes occupy PDs [0, HSPREAD); actors occupy the
 * remaining top PDs -- readers filling from the top downward, writers from
 * the bottom of the actor zone upward, each class round-robin over its own
 * PD set (a shared modulus would correlate role with node parity).  The
 * home PDs run no actors by default so a home-resident reader's free local
 * hit (a write-through-only artifact) cannot skew a write-policy
 * comparison; pass spreads covering the whole machine to include them
 * deliberately.  D=0 is the null cut: every chain loops on a private block
 * homed at its own PD, measuring the create/dispatch/local-acquire floor
 * that sits inside every acquire figure.
 *
 * Fixed-work termination (OPS_R/OPS_W nonzero): each chain runs exactly
 * its class's op count and the figure of record is the runtime's own
 * end-to-end stamp — class interference then lands in e2e as the honest
 * serialization it is, instead of vanishing from a timed window.  The
 * deadline degrades to a budget guard: a chain that hits it short of its
 * count reports CAP-HIT and the marker is withheld.  WARM_MS is ignored
 * (every op is measured).
 *
 * args: R W E_HOLD_R_us E_HOLD_W_us E_THINK_us BYTES D T_MS WARM_MS
 *       RSPREAD WSPREAD HSPREAD SEED CENSUS JITTER_PCT PIPE OPS_R OPS_W
 *       E_THINK_W_us
 * E_THINK_W_us (0 = writers share E_THINK) sets the writer class's think
 * alone: the knob that turns re-use rate and copy lifetime independently
 * of the reader cadence — the churn dial's axis.
 * output: one "RWSTEADY OK ..." line (the completion marker) or a
 *         RWSTEADY-ORACLE-FAIL / RWSTEADY-CAP-HIT line and no marker.
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
#define RS_MAX_CHAINS 1024u
#define RS_MAX_DBS 64u
#define RS_OCT 28u  /* octaves: [2^8, 2^36) ns */
#define RS_MANT 8u  /* linear sub-buckets per octave (~12.5% width) */
#define RS_NB (RS_OCT * RS_MANT)
#define RS_SHIFT_MIN 8u
#define RS_G_CAP 5000000ull

/* result-block layout, in u64 words */
#define RS_F_ROLE 0
#define RS_F_OPS_TOTAL 1
#define RS_F_OPS_MEAS 2
#define RS_F_VIOL 3
#define RS_F_CAP 4
#define RS_F_SUM_ACQ 5
#define RS_F_SUM_REL 6
#define RS_F_SUM_TOT 7
#define RS_F_USEFUL 8
#define RS_F_HIST 9
#define RS_RESULT_WORDS (RS_F_HIST + 3u * RS_NB)

/* paramv layout (all chain EDTs, one shape) */
enum {
  P_ROLE,
  P_CHAIN,
  P_PD,
  P_DBIDX,
  P_TRIG_NS,
  P_PREV_STAMP,
  P_PREV_ENTRY,
  P_PREV_SPIN,
  P_LCG,
  P_STEP_TPL,
  P_FIN_TPL,
  P_COLLECTOR,
  P_SLOT,
  P_RESULT_DB,
  P_CENSUS,
  P_D, /* 0 = null cut */
  P_EHOLD,
  P_ETHINK,
  P_BYTES,
  P_TMS,
  P_WMS,
  P_JITTER, /* percent; per-chain deterministic think spread, mean-preserving */
  P_KOPS, /* fixed-work mode: this chain's exact op count (0 = timed mode) */
  P_ETHINK_W, /* writer-class think override; 0 = share P_ETHINK */
  P_PIPE,   /* 1 = pipeline mode: every chain pinned to block (chain %% D);
             * readers additionally count USEFUL reads (counter advanced
             * since their last visit) — the consumed-value rate that a
             * dependency-coupled application's e2e is actually made of */
  P_DB0, /* P_DB0 .. P_DB0+7: data-block guids (null cut: [0] = private) */
  P_COUNT = P_DB0 + RS_MAX_DBS
};

/* per-rank sample store: written only by the owning chain's EDTs (a chain
 * never leaves its PD, and its EDTs are ordered by the chain itself), read
 * by that chain's finisher on the same rank.  Relaxed atomics keep the
 * cross-worker handoff independent of the scheduler's internal ordering. */
struct rs_row_s {
  uint64_t hist[3][RS_NB]; /* 0=acquire 1=release 2=total */
  uint64_t sum[3];
  uint64_t ops_total;
  uint64_t ops_measured;
  uint64_t viol;
  uint64_t cap_hit;
  uint64_t useful;
  uint64_t last_seen[RS_MAX_DBS];
};
static struct rs_row_s g_rows[RS_MAX_CHAINS];

static inline u64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

/* busy spin; returns realized duration so the total-overhead figure can
 * subtract what was actually spun, not what was asked for */
static inline u64 spin_ns(u64 us) {
  if (us == 0) return 0;
  u64 t0 = now_ns(), tgt = us * 1000ull;
  while (now_ns() - t0 < tgt)
    ;
  return now_ns() - t0;
}

/* octave + 8-way linear mantissa over v = ns/256: buckets 0-7 are linear
 * (v < 8), after that bucket = (log2(v)-2)*8 + top-3-bits-below-leading.
 * Bucket width is ~12.5%, which is what lets the winner map's margins be
 * judged against bucket resolution rather than a factor of sqrt(2). */
static inline void hist_add(uint64_t *h, u64 ns) {
  u64 v = ns >> RS_SHIFT_MIN;
  unsigned int b;
  if (v < RS_MANT) {
    b = (unsigned int)v;
  } else {
    unsigned int hb = 63u - (unsigned int)__builtin_clzll(v);
    unsigned int mant = (unsigned int)((v >> (hb - 3)) & (RS_MANT - 1));
    b = (hb - 2) * RS_MANT + mant;
    if (b >= RS_NB) b = RS_NB - 1;
  }
  __atomic_fetch_add(&h[b], 1, __ATOMIC_RELAXED);
}

/* mid of one bucket, in ns (inverse of hist_add's binning) */
static double bucket_mid_ns(unsigned int b) {
  double unit = (double)(1ull << RS_SHIFT_MIN);
  if (b < RS_MANT) return ((double)b + 0.5) * unit;
  unsigned int oct = b / RS_MANT, mant = b % RS_MANT;
  double base = (double)(1ull << (oct + 2)) * unit;
  double lo = base * (1.0 + (double)mant / RS_MANT);
  double hi = base * (1.0 + (double)(mant + 1) / RS_MANT);
  return (lo + hi) * 0.5;
}

static double hist_quantile_us(const uint64_t *h, double q) {
  u64 total = 0;
  for (unsigned int b = 0; b < RS_NB; b++) total += h[b];
  if (total == 0) return -1.0;
  double target = q * (double)total;
  u64 cum = 0;
  for (unsigned int b = 0; b < RS_NB; b++) {
    cum += h[b];
    if ((double)cum >= target) return bucket_mid_ns(b) / 1000.0;
  }
  return bucket_mid_ns(RS_NB - 1) / 1000.0;
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

/* touch a bounded slice: the counter's own line, the first 4 KB, and one
 * word per 64 KB beyond -- app-side memory traffic stays near-constant
 * while the protocol still moves whole-block payloads.  The counter word
 * is atomic (same-PD RW writers are concurrent by contract; a write-through
 * home publishes into the served buffer in place, so a reader load can
 * otherwise tear). */
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
  for (u64 i = 8; i < first; i++) { /* skip the counter's 64-byte line */
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
/* one template shape; the starter instance carries depc=2 (block+trigger),
 * successors depc=1 (block).  depv[1] is never read. */
ocrGuid_t step_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 t0 = now_ns();
  u64 role = paramv[P_ROLE], chain = paramv[P_CHAIN];
  struct rs_row_s *row = &g_rows[chain];
  u64 d = paramv[P_D];       /* 0 = null cut */
  u64 dbi = paramv[P_DBIDX]; /* index of the block THIS op acquired */
  if (paramv[P_TRIG_NS] == 0) {
    paramv[P_TRIG_NS] = t0; /* starter: the window anchors here */
    if (paramv[P_CENSUS]) {
      PRINTF("RWSTEADY_TASK chain=%lu role=%lu pd=%lu\n",
             (unsigned long)chain, (unsigned long)role,
             (unsigned long)paramv[P_PD]);
    }
  }
  u64 warm_ns = paramv[P_WMS] * 1000000ull;
  u64 span_ns = paramv[P_TMS] * 1000000ull;
  u64 deadline = paramv[P_TRIG_NS] + warm_ns + span_ns;
  u64 kops = paramv[P_KOPS];

  /* the op: touch, hold, explicit release (the zero edge), think */
  u64 stamp = 0;
  touch(depv[0].ptr, paramv[P_BYTES], role != 0, &stamp);
  if (role == 0 && d > 0) {
    u64 last =
        __atomic_load_n(&row->last_seen[dbi], __ATOMIC_RELAXED);
    if (stamp < last)
      __atomic_fetch_add(&row->viol, 1, __ATOMIC_RELAXED);
    else {
      if (stamp > last && t0 >= paramv[P_TRIG_NS] + warm_ns && t0 < deadline)
        __atomic_fetch_add(&row->useful, 1, __ATOMIC_RELAXED);
      __atomic_store_n(&row->last_seen[dbi], stamp, __ATOMIC_RELAXED);
    }
  }
  u64 spun = spin_ns(paramv[P_EHOLD]);
  u64 t1 = now_ns();
  ocrDbRelease(depv[0].guid);
  u64 t2 = now_ns();
  /* Deterministic per-chain think spread (mean-preserving): identical think
   * times phase-lock free-running chains into convoys, and the DB-queue
   * admission pattern then differs run to run — variance that is order, not
   * protocol.  The spread is a pure function of the chain id, so every arm
   * sees the same arrival texture. */
  u64 think = (role && paramv[P_ETHINK_W]) ? paramv[P_ETHINK_W]
                                           : paramv[P_ETHINK];
  if (paramv[P_JITTER]) {
    int64_t u2 = (int64_t)(((chain * 2654435761ull) >> 6) & 1023ull);
    think = (u64)((int64_t)think +
                  ((int64_t)think * (int64_t)paramv[P_JITTER] *
                   (2 * u2 - 1023)) / (1023 * 100));
  }
  spun += spin_ns(think);

  /* record: acquire and release belong to THIS op; the total closes the
   * PREVIOUS op (entry-to-entry minus its realized spins) */
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
    /* A pipeline consumer's work is the produced STREAM, not raw reads:
     * it terminates on having SEEN the stream's final counter value (the
     * producers' exact-count writes make that value known a priori).
     * Terminating on raw reads would dissolve the coupling; terminating
     * on observed-advance counts can strand a consumer that missed
     * overwritten versions after production ends — the final value alone
     * is durable, so awaiting it is both coupled and stall-free. */
    u64 progress = (!role && paramv[P_PIPE])
                       ? __atomic_load_n(&row->last_seen[dbi], __ATOMIC_RELAXED)
                       : total;
    done = progress >= kops;
    if (!done && now_ns() >= deadline) {
      /* fixed-work budget guard: ending short of the count is a loud
       * failure, never a quiet small sample */
      __atomic_store_n(&row->cap_hit, 1, __ATOMIC_RELAXED);
      done = 1;
    }
  } else {
    done = now_ns() >= deadline;
  }
  if (total >= RS_G_CAP) {
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

  /* next block: seeded walk over D (identical across arms); null cut and
   * D==1 stay put */
  u64 next_dbi = dbi;
  if (d > 1 && !paramv[P_PIPE]) {
    paramv[P_LCG] = paramv[P_LCG] * 6364136223846793005ull +
                    1442695040888963407ull;
    next_dbi = (paramv[P_LCG] >> 33) % d;
  }
  u64 t3 = now_ns();
  u64 next_params[P_COUNT];
  memcpy(next_params, paramv, sizeof(next_params));
  next_params[P_DBIDX] = next_dbi;
  next_params[P_PREV_STAMP] = t3;
  next_params[P_PREV_ENTRY] = t0;
  next_params[P_PREV_SPIN] = spun;
  ocrGuid_t succ;
  ocrEdtCreate(&succ, u64_guid(paramv[P_STEP_TPL]), P_COUNT, next_params, 1,
               NULL, EDT_PROP_NONE, &h, NULL);
  ocrAddDependence(u64_guid(paramv[P_DB0 + next_dbi]), succ, 0,
                   role ? DB_MODE_RW : DB_MODE_RO);
  return NULL_GUID;
}

/* ------------------------------------------------------------- finisher */
ocrGuid_t fin_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  struct rs_row_s *row = &g_rows[paramv[P_CHAIN]];
  u64 *r = (u64 *)depv[0].ptr;
  r[RS_F_ROLE] = paramv[P_ROLE];
  r[RS_F_OPS_TOTAL] = __atomic_load_n(&row->ops_total, __ATOMIC_RELAXED);
  r[RS_F_OPS_MEAS] = __atomic_load_n(&row->ops_measured, __ATOMIC_RELAXED);
  r[RS_F_VIOL] = __atomic_load_n(&row->viol, __ATOMIC_RELAXED);
  r[RS_F_CAP] = __atomic_load_n(&row->cap_hit, __ATOMIC_RELAXED);
  r[RS_F_USEFUL] = __atomic_load_n(&row->useful, __ATOMIC_RELAXED);
  for (unsigned int k = 0; k < 3; k++) {
    r[RS_F_SUM_ACQ + k] = __atomic_load_n(&row->sum[k], __ATOMIC_RELAXED);
    for (unsigned int b = 0; b < RS_NB; b++)
      r[RS_F_HIST + k * RS_NB + b] =
          __atomic_load_n(&row->hist[k][b], __ATOMIC_RELAXED);
  }
  /* release BEFORE wiring: the satisfy travels to the collector's rank and
   * its acquire runs there asynchronously -- nothing else orders that read
   * behind this EDT's epilogue */
  ocrGuid_t rdb = depv[0].guid;
  ocrDbRelease(rdb);
  ocrAddDependence(rdb, u64_guid(paramv[P_COLLECTOR]), (u32)paramv[P_SLOT],
                   DB_MODE_RO);
  return NULL_GUID;
}

/* ------------------------------------------------------------ collector */
/* paramv: {R, W, D, T_MS, WARM_MS, EHR, EHW, ETH, BYTES, RSPREAD, WSPREAD,
 *          HSPREAD, PDS, RHO_MILLI};  depv: A result blocks then D data
 *          blocks (RO).  Runs only when every chain slot landed, i.e. after
 *          every chain's last release. */
ocrGuid_t collector_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  u64 R = paramv[0], W = paramv[1], D = paramv[2], t_ms = paramv[3];
  u64 A = R + W;
  static uint64_t pooled[2][3][RS_NB]; /* class x figure x bucket */
  double sum[2][3] = {{0}};
  u64 ops_meas[2] = {0, 0}, ops_total_w = 0, viol = 0, cap = 0;
  u64 useful_sum = 0;
  double cn[2] = {0, 0}, cs[2] = {0, 0}, cq[2] = {0, 0}; /* chain-ops CV */
  memset(pooled, 0, sizeof(pooled));
  for (u64 i = 0; i < A; i++) {
    const u64 *r = (const u64 *)depv[i].ptr;
    unsigned int c = r[RS_F_ROLE] ? 1 : 0;
    ops_meas[c] += r[RS_F_OPS_MEAS];
    if (!c) useful_sum += r[RS_F_USEFUL];
    double co = (double)r[RS_F_OPS_MEAS];
    cn[c] += 1.0;
    cs[c] += co;
    cq[c] += co * co;
    if (c) ops_total_w += r[RS_F_OPS_TOTAL];
    viol += r[RS_F_VIOL];
    cap += r[RS_F_CAP];
    for (unsigned int k = 0; k < 3; k++) {
      sum[c][k] += (double)r[RS_F_SUM_ACQ + k];
      for (unsigned int b = 0; b < RS_NB; b++)
        pooled[c][k][b] += r[RS_F_HIST + k * RS_NB + b];
    }
  }
  /* dispersion of per-chain throughput inside one cell: the convoy /
   * fairness signal (order effects show here before they show in rep-to-rep
   * variance) */
  double cv[2] = {-1.0, -1.0};
  for (unsigned int c = 0; c < 2; c++) {
    if (cn[c] >= 2.0 && cs[c] > 0.0) {
      double mean = cs[c] / cn[c];
      double var = cq[c] / cn[c] - mean * mean;
      cv[c] = var > 0.0 ? sqrt(var) / mean : 0.0;
    }
  }
  u64 final_sum = 0;
  for (u64 d = 0; d < D; d++) {
    const u64 *p = (const u64 *)depv[A + d].ptr;
    final_sum += __atomic_load_n((const uint64_t *)&p[0], __ATOMIC_RELAXED);
  }

  if (cap) {
    PRINTF("RWSTEADY-CAP-HIT chains=%lu\n", (unsigned long)cap);
    ocrShutdown();
    return NULL_GUID;
  }
  if (viol) {
    PRINTF("RWSTEADY-ORACLE-FAIL kind=monotone viol=%lu\n",
           (unsigned long)viol);
    ocrShutdown();
    return NULL_GUID;
  }
  if (D > 0 && final_sum != ops_total_w) {
    PRINTF("RWSTEADY-ORACLE-FAIL kind=final want=%lu got=%lu\n",
           (unsigned long)ops_total_w, (unsigned long)final_sum);
    ocrShutdown();
    return NULL_GUID;
  }

  double span_s = (double)t_ms / 1000.0;
  double rx = (double)ops_meas[0] / span_s, wx = (double)ops_meas[1] / span_s;
#define QQ(c, k, q) hist_quantile_us(pooled[c][k], q)
#define MEAN(c, k) \
  (ops_meas[c] ? sum[c][k] / 1000.0 / (double)ops_meas[c] : -1.0)
  PRINTF(
      "RWSTEADY OK PDS=%lu R=%lu W=%lu EHR=%lu EHW=%lu ETH=%lu BYTES=%lu "
      "D=%lu T_MS=%lu WARM_MS=%lu RSPREAD=%lu WSPREAD=%lu HSPREAD=%lu "
      "RHO=%.4f ROPS=%lu WOPS=%lu RXPUT=%.1f WXPUT=%.1f "
      "RTOT_P50=%.2f RTOT_P90=%.2f RTOT_P99=%.2f "
      "WTOT_P50=%.2f WTOT_P90=%.2f WTOT_P99=%.2f "
      "RACQ_P99=%.2f RREL_P99=%.2f WACQ_P99=%.2f WREL_P99=%.2f "
      "RACQ_MEAN=%.2f RREL_MEAN=%.2f RTOT_MEAN=%.2f "
      "WACQ_MEAN=%.2f WREL_MEAN=%.2f WTOT_MEAN=%.2f "
      "RCV=%.3f WCV=%.3f JIT=%lu RUSE=%.1f FINAL=%lu\n",
      (unsigned long)paramv[12], (unsigned long)R, (unsigned long)W,
      (unsigned long)paramv[5], (unsigned long)paramv[6],
      (unsigned long)paramv[7], (unsigned long)paramv[8], (unsigned long)D,
      (unsigned long)t_ms, (unsigned long)paramv[4],
      (unsigned long)paramv[9], (unsigned long)paramv[10],
      (unsigned long)paramv[11], (double)paramv[13] / 10000.0,
      (unsigned long)ops_meas[0], (unsigned long)ops_meas[1], rx, wx,
      QQ(0, 2, 0.50), QQ(0, 2, 0.90), QQ(0, 2, 0.99), QQ(1, 2, 0.50),
      QQ(1, 2, 0.90), QQ(1, 2, 0.99), QQ(0, 0, 0.99), QQ(0, 1, 0.99),
      QQ(1, 0, 0.99), QQ(1, 1, 0.99), MEAN(0, 0), MEAN(0, 1), MEAN(0, 2),
      MEAN(1, 0), MEAN(1, 1), MEAN(1, 2), cv[0], cv[1],
      (unsigned long)paramv[14], (double)useful_sum / span_s,
      (unsigned long)final_sum);
#undef QQ
#undef MEAN
  ocrShutdown();
  return NULL_GUID;
}

/* -------------------------------------------------------------- mainEdt */
ocrGuid_t mainEdt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 R = 54, W = 9, ehr = 1, ehw = 1, eth = 20, bytes = 65536, D = 1;
  u64 t_ms = 8000, warm_ms = 2000, rspread = 0, wspread = 0, hspread = 1;
  u64 seed = 1, census = 0, jitter = 0, pipe = 0, ops_r = 0, ops_w = 0;
  u64 eth_w = 0;
  u64 argc = getArgc(depv[0].ptr);
  u64 *args[] = {&R, &W, &ehr, &ehw, &eth, &bytes, &D, &t_ms, &warm_ms,
                 &rspread, &wspread, &hspread, &seed, &census, &jitter,
                 &pipe, &ops_r, &ops_w, &eth_w};
  for (u64 i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    if (argc > i + 1) *args[i] = (u64)atol(getArgv(depv[0].ptr, i + 1));

  if (bytes < 4160) bytes = 4160; /* counter line + first-4KB touch region */
  u64 A = R + W;
  int nullcut = (D == 0);
  if (A == 0 || A > RS_MAX_CHAINS || D > RS_MAX_DBS) {
    PRINTF("RWSTEADY-ORACLE-FAIL kind=args A=%lu D=%lu\n", (unsigned long)A,
           (unsigned long)D);
    ocrShutdown();
    return NULL_GUID;
  }
  memset(g_rows, 0, sizeof(g_rows)); /* rank 0's rows; other ranks are bss */

  u64 pd_count = 0;
  ocrAffinityCount(AFFINITY_PD, &pd_count);
  if (pd_count == 0) pd_count = 1;
  u64 homes = hspread ? (hspread > pd_count ? pd_count : hspread) : 1;
  u64 zone_lo = pd_count > homes ? homes : 0; /* actor zone [zone_lo, pd_count) */
  u64 zone = pd_count - zone_lo;
  u64 eff_r = (rspread == 0 || rspread > zone) ? zone : rspread;
  u64 eff_w = (wspread == 0 || wspread > zone) ? zone : wspread;

  /* data blocks: all homes in [0, homes), round-robin (default: all PD 0) */
  u64 db_guids[RS_MAX_DBS] = {0};
  u64 n_data = nullcut ? 0 : D;
  for (u64 d = 0; d < n_data; d++) {
    ocrHint_t dh;
    pd_hint(&dh, d % homes, OCR_HINT_DB_T);
    ocrGuid_t g;
    u64 *p;
    ocrDbCreate(&g, (void **)&p, bytes, DB_PROP_NONE, &dh, NO_ALLOC);
    memset(p, 0, bytes);
    ocrDbRelease(g); /* create takes an implicit hold; under EXCL every
                      * first op would otherwise park behind this EDT */
    db_guids[d] = guid_u64(g);
  }

  ocrGuid_t trigger;
  ocrEventCreate(&trigger, OCR_EVENT_ONCE_T, EVT_PROP_NONE);

  ocrGuid_t step_tpl, start_tpl, fin_tpl, coll_tpl;
  ocrEdtTemplateCreate(&step_tpl, step_edt, P_COUNT, 1);
  ocrEdtTemplateCreate(&start_tpl, step_edt, P_COUNT, 2);
  ocrEdtTemplateCreate(&fin_tpl, fin_edt, P_COUNT, 1);
  ocrEdtTemplateCreate(&coll_tpl, collector_edt, 15, (u32)(A + n_data));

  u64 rho_milli =
      (u64)((double)W / (double)A * 10000.0 + 0.5); /* 4 decimals */
  u64 cparams[15] = {R,       W,       n_data,  t_ms,   warm_ms,
                     ehr,     ehw,     eth,     bytes,  rspread,
                     wspread, hspread, pd_count, rho_milli, jitter};
  ocrHint_t h0;
  pd_hint(&h0, 0, OCR_HINT_EDT_T);
  ocrGuid_t collector;
  ocrEdtCreate(&collector, coll_tpl, 15, cparams, (u32)(A + n_data), NULL,
               EDT_PROP_NONE, &h0, NULL);
  for (u64 d = 0; d < n_data; d++)
    ocrAddDependence(u64_guid(db_guids[d]), collector, (u32)(A + d),
                     DB_MODE_RO);

  u64 r_idx = 0, w_idx = 0;
  for (u64 c = 0; c < A; c++) {
    int is_writer = c < W; /* ids: writers first, then readers */
    u64 pd;
    if (zone == 0) {
      pd = 0;
    } else if (is_writer) {
      pd = zone_lo + (w_idx++ % eff_w); /* bottom of the zone upward */
    } else {
      pd = pd_count - 1 - (r_idx++ % eff_r); /* top downward */
    }

    ocrHint_t dh;
    pd_hint(&dh, pd, OCR_HINT_DB_T);
    ocrGuid_t rdb;
    u64 *rp;
    ocrDbCreate(&rdb, (void **)&rp, RS_RESULT_WORDS * sizeof(u64),
                DB_PROP_NONE, &dh, NO_ALLOC);
    memset(rp, 0, RS_RESULT_WORDS * sizeof(u64));
    ocrDbRelease(rdb);

    u64 priv_db = 0;
    if (nullcut) {
      ocrGuid_t pg;
      u64 *pp;
      ocrDbCreate(&pg, (void **)&pp, bytes, DB_PROP_NONE, &dh, NO_ALLOC);
      memset(pp, 0, bytes);
      ocrDbRelease(pg);
      priv_db = guid_u64(pg);
    }

    u64 lcg = seed * 0x9E3779B97F4A7C15ull + c * 0xBF58476D1CE4E5B9ull + 1;
    u64 first_db = nullcut ? 0
                   : pipe && D > 0 ? c % D
                   : (D > 1 ? (lcg >> 33) % D : 0);
    u64 params[P_COUNT];
    memset(params, 0, sizeof(params));
    params[P_ROLE] = is_writer ? 1 : 0;
    params[P_CHAIN] = c;
    params[P_PD] = pd;
    params[P_DBIDX] = first_db;
    params[P_LCG] = lcg;
    params[P_STEP_TPL] = guid_u64(step_tpl);
    params[P_FIN_TPL] = guid_u64(fin_tpl);
    params[P_COLLECTOR] = guid_u64(collector);
    params[P_SLOT] = c;
    params[P_RESULT_DB] = guid_u64(rdb);
    params[P_CENSUS] = census;
    params[P_D] = nullcut ? 0 : D;
    params[P_EHOLD] = is_writer ? ehw : ehr;
    params[P_ETHINK] = eth;
    params[P_BYTES] = bytes;
    params[P_TMS] = t_ms;
    params[P_WMS] = warm_ms;
    params[P_JITTER] = jitter > 90 ? 90 : jitter;
    params[P_KOPS] = is_writer ? ops_w : ops_r;
    params[P_ETHINK_W] = eth_w;
    params[P_PIPE] = pipe;
    if (nullcut) {
      params[P_DB0] = priv_db;
    } else {
      for (u64 d = 0; d < D; d++) params[P_DB0 + d] = db_guids[d];
    }

    ocrHint_t eh;
    pd_hint(&eh, pd, OCR_HINT_EDT_T);
    ocrGuid_t starter;
    ocrEdtCreate(&starter, start_tpl, P_COUNT, params, 2, NULL, EDT_PROP_NONE,
                 &eh, NULL);
    ocrAddDependence(u64_guid(params[P_DB0 + first_db]), starter, 0,
                     is_writer ? DB_MODE_RW : DB_MODE_RO);
    ocrAddDependence(trigger, starter, 1, DB_MODE_NULL);
  }

  ocrEventSatisfy(trigger, NULL_GUID);
  return NULL_GUID;
}
