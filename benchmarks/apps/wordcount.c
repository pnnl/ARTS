/*
 * wordcount.c -- canonical MapReduce word count, native OCR, weak-scaling
 * form.
 *
 * The corpus is synthesized, never read: every map tile generates its own
 * text from a PRNG seeded by the tile's GLOBAL id, so the input exists
 * only as bytes streaming through the tokenizer -- no ingest I/O at any
 * node count, and the corpus for a given (node count, seed) is identical
 * across runs and runtime configurations.  The vocabulary is CLOSED: each
 * token is drawn from exactly V words (word k spelled as variable-length
 * base-26), so the histogram width is V by construction -- the same
 * closed-list scheme as Hadoop's RandomTextWriter, parameterized.  Token
 * ids are drawn Zipf-like via the continuous inverse CDF (O(1) per token,
 * no per-EDT tables); the draw skew shapes count VALUES only -- with
 * dense V-slot partials the coherence traffic is distribution-invariant.
 *
 * Structure: a spawner tree unfolds a K-ary reduction over M = TPN x N
 * map tiles (rank-contiguous), creating each combine on its FIRST child's
 * rank -- lower tree levels therefore stay inside a node (the map-side
 * combiner) and only the upper levels cross ranks.  Every partial
 * histogram is created LOCALLY by the EDT that fills it and handed
 * upward by wiring the released block straight into the parent's slot;
 * there are no events and no pre-created data blocks.  Setup work is
 * O(log) deep and node-parallel: nothing serializes on the main rank.
 *
 * Fixed work by construction: ITER trees of M tiles each, then the
 * verifier checks the total token count EXACTLY (M x WORDS x ITER-th
 * tree's own M x WORDS; a single lost update anywhere breaks it) and
 * prints an FNV checksum of the final histogram for cross-configuration
 * consensus voting.  The figure of record is the runtime's end-to-end
 * stamp.
 *
 * args: V WORDS TPN K ZIPF GRAIN ITER SEED
 *   V      vocabulary size = histogram width = reduce payload / 8 bytes
 *   WORDS  tokens per map tile
 *   TPN    map tiles per node (M = TPN x nodes)
 *   K      reduction fan-in
 *   ZIPF   token-draw skew x100 (0 = uniform, 100 = s=1)
 *   GRAIN  extra spin per tile, us (heavier map emulation)
 *   ITER   number of map+reduce rounds
 *   SEED   corpus seed
 * output: one "WORDCOUNT OK ..." line (completion marker; TOKENS= and
 *         CHECKSUM= ride on it) or WORDCOUNT-ORACLE-FAIL and no marker.
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

#define WC_MAX_FANIN 64u
#define WC_CHUNK 65536u /* streaming text buffer: generate, tokenize, reuse */

enum {
  P_LO,
  P_HI,
  P_PARENT,
  P_SLOT,
  P_ITER_NOW,
  P_V,
  P_WORDS,
  P_TPN,
  P_K,
  P_ZIPF,
  P_GRAIN,
  P_ITER,
  P_SEED,
  P_SPAWN_TPL,
  P_MAP_TPL,
  P_COMB_TPL,
  P_VERIFY_TPL,
  P_NRANKS,
  P_COUNT
};

static inline u64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

static inline u64 spin_us(u64 us) {
  if (us == 0) return 0;
  u64 t0 = now_ns(), tgt = us * 1000ull;
  while (now_ns() - t0 < tgt)
    ;
  return 0;
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

static inline u64 splitmix64(u64 *s) {
  u64 z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/* Zipf-like draw over [0, V) by the continuous power-law inverse CDF:
 * O(1) per token, no tables.  s == 0 degrades to uniform. */
static inline u64 draw_word(u64 *rng, u64 v, double s, double pre) {
  u64 r = splitmix64(rng);
  if (s == 0.0) return r % v;
  double u = (double)(r >> 11) * (1.0 / 9007199254740992.0);
  double k;
  if (s == 1.0) {
    k = exp(u * pre); /* pre = ln(V) */
  } else {
    k = pow(u * pre + 1.0, 1.0 / (1.0 - s)); /* pre = V^(1-s) - 1 */
  }
  u64 id = (u64)k - 1u;
  return id < v ? id : v - 1u;
}

/* Spell word id as variable-length base-26 (A..Z), most significant first;
 * returns byte count written.  The tokenizer inverts this exactly. */
static inline unsigned int spell(u64 id, char *out) {
  char tmp[16];
  unsigned int n = 0;
  do {
    tmp[n++] = (char)('A' + (id % 26u));
    id /= 26u;
  } while (id != 0);
  for (unsigned int i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
  return n;
}

/* ------------------------------------------------------------------ map */
ocrGuid_t map_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  u64 tile = paramv[P_LO];
  u64 v = paramv[P_V], words = paramv[P_WORDS];
  double s = (double)paramv[P_ZIPF] / 100.0;
  double pre = (s == 1.0) ? log((double)v)
               : (s == 0.0) ? 0.0
                            : (pow((double)v, 1.0 - s) - 1.0);
  u64 rng = paramv[P_SEED] * 0x9E3779B97F4A7C15ull + tile * 0xD1B54A32D192ED03ull;

  ocrHint_t dh;
  pd_hint(&dh, paramv[P_LO] / paramv[P_TPN], OCR_HINT_DB_T);
  ocrGuid_t db;
  u64 *count;
  ocrDbCreate(&db, (void **)&count, v * sizeof(u64), DB_PROP_NONE, &dh,
              NO_ALLOC);
  memset(count, 0, v * sizeof(u64));

  /* Generate and tokenize in streaming chunks: the text exists only as
   * bytes passing through this buffer, and the byte work (spell, scan,
   * parse) is the map phase's honest labor. */
  char buf[WC_CHUNK + 24];
  u64 done = 0;
  while (done < words) {
    unsigned int fill = 0;
    u64 batch = 0;
    while (fill < WC_CHUNK && done + batch < words) {
      u64 id = draw_word(&rng, v, s, pre);
      fill += spell(id, buf + fill);
      buf[fill++] = ' ';
      batch++;
    }
    /* tokenize the chunk back: accumulate base-26 until each space */
    u64 acc = 0;
    for (unsigned int i = 0; i < fill; i++) {
      char c = buf[i];
      if (c == ' ') {
        count[acc]++;
        acc = 0;
      } else {
        acc = acc * 26u + (u64)(c - 'A');
      }
    }
    done += batch;
  }
  spin_us(paramv[P_GRAIN]);

  ocrDbRelease(db);
  ocrAddDependence(db, u64_guid(paramv[P_PARENT]), (u32)paramv[P_SLOT],
                   DB_MODE_RO);
  return NULL_GUID;
}

/* -------------------------------------------------------------- combine */
ocrGuid_t combine_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  u64 v = paramv[P_V];
  ocrHint_t dh;
  pd_hint(&dh, paramv[P_LO] / paramv[P_TPN], OCR_HINT_DB_T);
  ocrGuid_t db;
  u64 *count;
  ocrDbCreate(&db, (void **)&count, v * sizeof(u64), DB_PROP_NONE, &dh,
              NO_ALLOC);
  memset(count, 0, v * sizeof(u64));
  for (u32 i = 0; i < depc; i++) {
    const u64 *child = (const u64 *)depv[i].ptr;
    if (child == NULL) continue; /* padded slot */
    for (u64 w = 0; w < v; w++) count[w] += child[w];
  }
  ocrDbRelease(db);
  ocrAddDependence(db, u64_guid(paramv[P_PARENT]), (u32)paramv[P_SLOT],
                   DB_MODE_RO);
  return NULL_GUID;
}

/* -------------------------------------------------------------- spawner */
static void spawn_subtree(u64 *proto, u64 lo, u64 hi, u64 parent, u64 slot);

ocrGuid_t spawn_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  spawn_subtree(paramv, paramv[P_LO], paramv[P_HI], paramv[P_PARENT],
                paramv[P_SLOT]);
  return NULL_GUID;
}

/* Unfold one tree node: a single tile becomes a map EDT; a range becomes
 * a combine on the range's FIRST rank plus one child (spawner or map) per
 * chunk.  Slots the last combine cannot fill are satisfied empty. */
static void spawn_subtree(u64 *proto, u64 lo, u64 hi, u64 parent, u64 slot) {
  u64 tpn = proto[P_TPN], k = proto[P_K];
  u64 params[P_COUNT];
  memcpy(params, proto, sizeof(params));
  params[P_PARENT] = parent;
  params[P_SLOT] = slot;
  if (hi - lo == 1) {
    params[P_LO] = lo;
    params[P_HI] = hi;
    ocrHint_t eh;
    pd_hint(&eh, lo / tpn, OCR_HINT_EDT_T);
    ocrGuid_t m;
    ocrEdtCreate(&m, u64_guid(proto[P_MAP_TPL]), P_COUNT, params, 0, NULL,
                 EDT_PROP_NONE, &eh, NULL);
    return;
  }
  u64 n = hi - lo;
  u64 parts = n < k ? n : k;
  u64 q = n / parts, r = n % parts;

  params[P_LO] = lo;
  params[P_HI] = hi;
  ocrHint_t ch;
  pd_hint(&ch, lo / tpn, OCR_HINT_EDT_T);
  ocrGuid_t comb;
  ocrEdtCreate(&comb, u64_guid(proto[P_COMB_TPL]), P_COUNT, params, (u32)k,
               NULL, EDT_PROP_NONE, &ch, NULL);
  for (u64 i = parts; i < k; i++) {
    ocrAddDependence(NULL_GUID, comb, (u32)i, DB_MODE_NULL);
  }
  u64 at = lo;
  for (u64 i = 0; i < parts; i++) {
    u64 len = q + (i < r ? 1 : 0);
    if (len == 1) {
      spawn_subtree(proto, at, at + 1, guid_u64(comb), i);
    } else {
      u64 sp[P_COUNT];
      memcpy(sp, proto, sizeof(sp));
      sp[P_LO] = at;
      sp[P_HI] = at + len;
      sp[P_PARENT] = guid_u64(comb);
      sp[P_SLOT] = i;
      ocrHint_t sh;
      pd_hint(&sh, at / tpn, OCR_HINT_EDT_T);
      ocrGuid_t s;
      ocrEdtCreate(&s, u64_guid(proto[P_SPAWN_TPL]), P_COUNT, sp, 0, NULL,
                   EDT_PROP_NONE, &sh, NULL);
    }
    at += len;
  }
}

/* --------------------------------------------------------------- verify */
ocrGuid_t verify_edt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 v = paramv[P_V];
  u64 m = paramv[P_TPN] * paramv[P_NRANKS];
  const u64 *count = (const u64 *)depv[0].ptr;
  u64 total = 0, fnv = 0xCBF29CE484222325ull;
  for (u64 w = 0; w < v; w++) {
    total += count[w];
    fnv = (fnv ^ count[w]) * 0x100000001B3ull;
  }
  u64 want = m * paramv[P_WORDS];
  if (total != want) {
    PRINTF("WORDCOUNT-ORACLE-FAIL iter=%lu want=%lu got=%lu\n",
           (unsigned long)paramv[P_ITER_NOW], (unsigned long)want,
           (unsigned long)total);
    ocrShutdown();
    return NULL_GUID;
  }
  u64 next = paramv[P_ITER_NOW] + 1;
  if (next < paramv[P_ITER]) {
    u64 params[P_COUNT];
    memcpy(params, paramv, sizeof(params));
    params[P_ITER_NOW] = next;
    ocrHint_t vh;
    pd_hint(&vh, 0, OCR_HINT_EDT_T);
    ocrGuid_t ver;
    ocrEdtCreate(&ver, u64_guid(paramv[P_VERIFY_TPL]), P_COUNT, params, 1,
                 NULL, EDT_PROP_NONE, &vh, NULL);
    params[P_PARENT] = guid_u64(ver);
    params[P_SLOT] = 0;
    spawn_subtree(params, 0, m, guid_u64(ver), 0);
    return NULL_GUID;
  }
  PRINTF("WORDCOUNT OK V=%lu WORDS=%lu TPN=%lu K=%lu ZIPF=%lu GRAIN=%lu "
         "ITER=%lu NODES=%lu TOKENS=%lu CHECKSUM=%lu\n",
         (unsigned long)v, (unsigned long)paramv[P_WORDS],
         (unsigned long)paramv[P_TPN], (unsigned long)paramv[P_K],
         (unsigned long)paramv[P_ZIPF], (unsigned long)paramv[P_GRAIN],
         (unsigned long)paramv[P_ITER], (unsigned long)paramv[P_NRANKS],
         (unsigned long)total, (unsigned long)fnv);
  ocrShutdown();
  return NULL_GUID;
}

/* -------------------------------------------------------------- mainEdt */
ocrGuid_t mainEdt(u32 paramc, u64 *paramv, u32 depc, ocrEdtDep_t depv[]) {
  (void)paramc;
  (void)depc;
  u64 V = 16384, words = 1000000, tpn = 128, k = 8, zipf = 100;
  u64 grain = 0, iter = 4, seed = 1;
  u64 argc = getArgc(depv[0].ptr);
  u64 *args[] = {&V, &words, &tpn, &k, &zipf, &grain, &iter, &seed};
  for (u64 i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    if (argc > i + 1) *args[i] = (u64)atol(getArgv(depv[0].ptr, i + 1));
  if (k < 2) k = 2;
  if (k > WC_MAX_FANIN) k = WC_MAX_FANIN;
  if (V == 0 || words == 0 || tpn == 0 || iter == 0 || zipf > 300) {
    PRINTF("WORDCOUNT-ORACLE-FAIL kind=args\n");
    ocrShutdown();
    return NULL_GUID;
  }

  u64 nranks = 0;
  ocrAffinityCount(AFFINITY_PD, &nranks);
  if (nranks == 0) nranks = 1;
  u64 m = tpn * nranks;

  ocrGuid_t spawn_tpl, map_tpl, comb_tpl, verify_tpl;
  ocrEdtTemplateCreate(&spawn_tpl, spawn_edt, P_COUNT, 0);
  ocrEdtTemplateCreate(&map_tpl, map_edt, P_COUNT, 0);
  ocrEdtTemplateCreate(&comb_tpl, combine_edt, P_COUNT, (u32)k);
  ocrEdtTemplateCreate(&verify_tpl, verify_edt, P_COUNT, 1);

  u64 proto[P_COUNT];
  memset(proto, 0, sizeof(proto));
  proto[P_V] = V;
  proto[P_WORDS] = words;
  proto[P_TPN] = tpn;
  proto[P_K] = k;
  proto[P_ZIPF] = zipf;
  proto[P_GRAIN] = grain;
  proto[P_ITER] = iter;
  proto[P_SEED] = seed;
  proto[P_SPAWN_TPL] = guid_u64(spawn_tpl);
  proto[P_MAP_TPL] = guid_u64(map_tpl);
  proto[P_COMB_TPL] = guid_u64(comb_tpl);
  proto[P_VERIFY_TPL] = guid_u64(verify_tpl);
  proto[P_NRANKS] = nranks;
  proto[P_ITER_NOW] = 0;

  ocrHint_t vh;
  pd_hint(&vh, 0, OCR_HINT_EDT_T);
  ocrGuid_t ver;
  ocrEdtCreate(&ver, verify_tpl, P_COUNT, proto, 1, NULL, EDT_PROP_NONE, &vh,
               NULL);
  proto[P_PARENT] = guid_u64(ver);
  proto[P_SLOT] = 0;
  spawn_subtree(proto, 0, m, guid_u64(ver), 0);
  return NULL_GUID;
}
