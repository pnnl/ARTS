/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_placement_compute invariants over synthetic topologies.
 *
 * Sweeps packages x NUMA x cores x SMT x os-index numbering styles x
 * (workers, progress, rank count), asserting on every combination:
 *
 *   I1 exact order    — out[t] is slice[t] of the base order (SMT round 0
 *                       first, NUMA-compact, os ascending), workers on the
 *                       front, progress clustered on the tail.
 *   I2 id contract    — out[0..W) are workers, out[W..W+P) progress, each
 *                       role's group_pos sequential.
 *   I3 determinism    — shuffling the input PU array leaves output unchanged.
 *   I4 rank disjoint  — equal-sized rank slices partition the base-order
 *                       prefix with no overlap.
 *
 * The base order is recomputed here with an independent sort (not the code
 * under test) so I1 is non-circular.  The (pkg, core) composite core key is
 * pinned by the per-package-core-numbering topologies: a core os_index that
 * repeats across sockets must not merge two physical cores' SMT ranks.
 */

#include "arts/system/placement.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL placement_invariants: " __VA_ARGS__);                       \
      printf("\n");                                                            \
      fails++;                                                                 \
    }                                                                          \
  } while (0)

/* ---- synthetic topology ------------------------------------------------- */

struct topo_s {
  struct arts_pu_desc_s pu[512];
  unsigned int n;
  char name[128];
};

/* os_mode: 0 = sequential PU ids, 1 = SMT-interleaved PU ids (siblings offset
 * by the machine core count, Intel style), 2 = per-package core numbering
 * (both sockets carry core 0..k-1). */
static void topo_build(struct topo_s *t, unsigned int npkg,
                       unsigned int numa_per_pkg, unsigned int cores_per_numa,
                       unsigned int extra_cores_pkg0, unsigned int smt,
                       int os_mode) {
  t->n = 0;
  unsigned int total_cores = 0;
  for (unsigned int p = 0; p < npkg; p++) {
    total_cores +=
        numa_per_pkg * cores_per_numa + (p == 0 ? extra_cores_pkg0 : 0);
  }
  unsigned int core_global = 0;
  for (unsigned int p = 0; p < npkg; p++) {
    unsigned int cores_here =
        numa_per_pkg * cores_per_numa + (p == 0 ? extra_cores_pkg0 : 0);
    unsigned int core_in_pkg = 0;
    for (unsigned int c = 0; c < cores_here; c++) {
      unsigned int numa = p * numa_per_pkg + c / cores_per_numa;
      if (numa >= p * numa_per_pkg + numa_per_pkg) {
        numa = p * numa_per_pkg + numa_per_pkg - 1; /* extra cores: last numa */
      }
      for (unsigned int s = 0; s < smt; s++) {
        struct arts_pu_desc_s *d = &t->pu[t->n];
        d->pkg_os_index = p;
        d->core_os_index = (os_mode == 2) ? core_in_pkg : core_global;
        d->numa_id = numa;
        d->pu_os_index =
            (os_mode == 1) ? (s * total_cores + core_global) : t->n;
        t->n++;
      }
      core_in_pkg++;
      core_global++;
    }
  }
  snprintf(t->name, sizeof(t->name), "pkg%u.numa%u.core%u+%u.smt%u.os%d", npkg,
           numa_per_pkg, cores_per_numa, extra_cores_pkg0, smt, os_mode);
}

/* ---- independent base-order oracle -------------------------------------- */

struct opu_s {
  struct arts_pu_desc_s d;
  unsigned int rank;
};

static int ocmp_core(const void *a, const void *b) {
  const struct opu_s *x = a, *y = b;
  if (x->d.pkg_os_index != y->d.pkg_os_index)
    return x->d.pkg_os_index < y->d.pkg_os_index ? -1 : 1;
  if (x->d.core_os_index != y->d.core_os_index)
    return x->d.core_os_index < y->d.core_os_index ? -1 : 1;
  return x->d.pu_os_index < y->d.pu_os_index ? -1 : 1;
}
static int ocmp_base(const void *a, const void *b) {
  const struct opu_s *x = a, *y = b;
  if (x->rank != y->rank)
    return x->rank < y->rank ? -1 : 1;
  if (x->d.numa_id != y->d.numa_id)
    return x->d.numa_id < y->d.numa_id ? -1 : 1;
  return x->d.pu_os_index < y->d.pu_os_index ? -1 : 1;
}

static void oracle_base_order(const struct topo_s *t, struct opu_s *o) {
  for (unsigned int i = 0; i < t->n; i++) {
    o[i].d = t->pu[i];
    o[i].rank = 0;
  }
  qsort(o, t->n, sizeof(*o), ocmp_core);
  unsigned int r = 0;
  for (unsigned int i = 0; i < t->n; i++) {
    if (i > 0 && (o[i].d.pkg_os_index != o[i - 1].d.pkg_os_index ||
                  o[i].d.core_os_index != o[i - 1].d.core_os_index))
      r = 0;
    o[i].rank = r++;
  }
  qsort(o, t->n, sizeof(*o), ocmp_base);
}

/* ---- per-case verification ----------------------------------------------- */

static unsigned int rng_state = 0x5eed;
static unsigned int rng(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state >> 8;
}

static void verify(const struct topo_s *t, unsigned int offset, unsigned int W,
                   unsigned int P) {
  unsigned int n = W + P;
  struct opu_s base[512];
  struct arts_placement_s out[512], out2[512];
  struct arts_pu_desc_s shuffled[512];

  oracle_base_order(t, base);
  const struct opu_s *slice = base + offset;

  if (!arts_placement_compute(t->pu, t->n, offset, W, P, out)) {
    CHECK(0, "[%s] compute failed (off=%u W=%u P=%u)", t->name, offset, W, P);
    return;
  }

  /* I1: exact base-order slice, front workers / tail progress. */
  for (unsigned int k = 0; k < n; k++) {
    CHECK(out[k].pu_os_index == slice[k].d.pu_os_index &&
              out[k].core_os_index == slice[k].d.core_os_index &&
              out[k].pkg_os_index == slice[k].d.pkg_os_index &&
              out[k].numa_id == slice[k].d.numa_id,
          "[%s] I1 mismatch at %u: pu %u want %u (off=%u W=%u P=%u)", t->name,
          k, out[k].pu_os_index, slice[k].d.pu_os_index, offset, W, P);
  }

  /* I2: id contract. */
  for (unsigned int k = 0; k < n; k++) {
    if (k < W) {
      CHECK(out[k].role == ARTS_ROLE_WORKER && out[k].group_pos == k,
            "[%s] I2 id %u not worker/pos (off=%u W=%u P=%u)", t->name, k,
            offset, W, P);
    } else {
      CHECK(out[k].role == ARTS_ROLE_PROGRESS && out[k].group_pos == k - W,
            "[%s] I2 id %u not progress/pos", t->name, k);
    }
  }

  /* I3: input order must not matter. */
  memcpy(shuffled, t->pu, t->n * sizeof(*shuffled));
  for (unsigned int i = t->n - 1; i > 0; i--) {
    unsigned int j = rng() % (i + 1);
    struct arts_pu_desc_s tmp = shuffled[i];
    shuffled[i] = shuffled[j];
    shuffled[j] = tmp;
  }
  if (!arts_placement_compute(shuffled, t->n, offset, W, P, out2)) {
    CHECK(0, "[%s] shuffled compute failed", t->name);
    return;
  }
  CHECK(memcmp(out, out2, n * sizeof(*out)) == 0,
        "[%s] I3 shuffled input changed output (off=%u W=%u P=%u)", t->name,
        offset, W, P);
}

/* I4: equal rank slices partition the base-order prefix disjointly. */
static void verify_multirank(const struct topo_s *t, unsigned int ranks,
                             unsigned int per, unsigned int P) {
  struct arts_placement_s out[512];
  unsigned char used[512] = {0};
  struct opu_s base[512];
  oracle_base_order(t, base);

  for (unsigned int r = 0; r < ranks; r++) {
    if (!arts_placement_compute(t->pu, t->n, r * per, per - P, P, out)) {
      CHECK(0, "[%s] I4 rank %u compute failed", t->name, r);
      return;
    }
    for (unsigned int k = 0; k < per; k++) {
      unsigned int pu = out[k].pu_os_index;
      unsigned int pos = 512;
      for (unsigned int i = 0; i < t->n; i++) {
        if (base[i].d.pu_os_index == pu) {
          pos = i;
          break;
        }
      }
      CHECK(pos < ranks * per, "[%s] I4 rank %u pu %u outside prefix", t->name,
            r, pu);
      CHECK(pos < 512 && !used[pos], "[%s] I4 rank %u pu %u overlaps", t->name,
            r, pu);
      if (pos < 512) {
        used[pos] = 1;
      }
    }
  }
}

int main(void) {
  struct topo_s t;
  /* Fixed fleet: junction (2x32 no SMT), two-socket multi-NUMA, laptop-ish,
   * SMT2 interleaved os ids, asymmetric + per-package core ids, 4 sockets. */
  struct {
    unsigned int npkg, numa, cores, extra, smt;
    int os;
  } fleet[] = {
      {2, 1, 32, 0, 1, 0}, {2, 2, 6, 0, 1, 0},  {1, 1, 14, 0, 1, 0},
      {1, 1, 8, 0, 2, 1},  {2, 1, 8, 4, 2, 2},  {4, 2, 4, 0, 2, 2},
  };
  for (unsigned int f = 0; f < sizeof(fleet) / sizeof(fleet[0]); f++) {
    topo_build(&t, fleet[f].npkg, fleet[f].numa, fleet[f].cores,
               fleet[f].extra, fleet[f].smt, fleet[f].os);
    for (unsigned int P = 0; P <= 4 && P < t.n; P++) {
      verify(&t, 0, t.n - P, P);
    }
    for (unsigned int ranks = 2; ranks <= 4; ranks++) {
      unsigned int per = t.n / ranks;
      if (per >= 2) {
        verify_multirank(&t, ranks, per, 1);
      }
    }
  }

  /* Junction pinned expectations on 2x32 no-SMT: the tail clusters progress
   * on the last package next to each other. */
  topo_build(&t, 2, 1, 32, 0, 1, 0);
  struct arts_placement_s out[512];
  if (arts_placement_compute(t.pu, t.n, 0, 63, 1, out)) {
    CHECK(out[63].role == ARTS_ROLE_PROGRESS && out[63].pu_os_index == 63 &&
              out[63].pkg_os_index == 1,
          "junction 63w+1p: progress pu %u pkg %u want 63/1",
          out[63].pu_os_index, out[63].pkg_os_index);
  } else {
    CHECK(0, "junction 63w+1p compute failed");
  }
  if (arts_placement_compute(t.pu, t.n, 0, 62, 2, out)) {
    CHECK(out[62].pu_os_index == 62 && out[63].pu_os_index == 63 &&
              out[62].pkg_os_index == 1 && out[63].pkg_os_index == 1,
          "junction 62w+2p: progress pus %u,%u want 62,63 clustered on pkg 1",
          out[62].pu_os_index, out[63].pu_os_index);
  } else {
    CHECK(0, "junction 62w+2p compute failed");
  }

  /* Random sweep: 300 seeded cases across random topologies and splits. */
  for (unsigned int i = 0; i < 300; i++) {
    unsigned int npkg = 1 + rng() % 4;
    unsigned int numa = 1 + rng() % 2;
    unsigned int cores = 2 + rng() % 6;
    unsigned int smt = 1 + rng() % 2;
    topo_build(&t, npkg, numa, cores, rng() % 3, smt, (int)(rng() % 3));
    unsigned int n = 2 + rng() % (t.n - 1);
    unsigned int off = rng() % (t.n - n + 1);
    unsigned int P = rng() % (n < 5 ? n : 5);
    verify(&t, off, n - P, P);
  }

  if (fails) {
    return 1;
  }
  printf("PASS placement_invariants: fleet + junction pins + 300 random\n");
  return 0;
}
