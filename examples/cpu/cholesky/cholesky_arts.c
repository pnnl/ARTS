/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

/*
 * Tiled Cholesky (LLT) decomposition — native ARTS port.
 *
 * Ports third_party/ocr-apps/apps/examples/cholesky/cholesky.c to native
 * ARTS.  Distribution: tile (i,j) owner = round-robin chunk-8 on the
 * flattened lower-triangular tile index.
 *
 * Algorithm: standard k-loop tiled Cholesky.  For k = 0..numTiles-1:
 *   Phase 1: sequential_cholesky on A[k][k] -> L[k][k]
 *   Phase 2: trisolve on (A[j][k], L[k][k]) -> L[j][k], j > k
 *   Phase 3a: update_diagonal on (A[j][j], L[j][k]) -> A[j][j] for j > k
 *   Phase 3b: update_nondiagonal on (A[j][i], L[j][k], L[i][k]) -> A[j][i]
 *             for j > i > k
 *
 * DB granularity:
 *   One DB per tile per generation.  Events lkji[i][j][k] carry the DB
 *   for tile (i,j) at generation k.  k=0 is the input matrix.  All
 *   events are home-rooted on rank 0; cross-node forwarding by ARTS.
 *
 * Release markers:
 *   Pattern A (release-before-satisfy) on every DB before its event
 *   satisfy.
 *
 * paramv layout (raw uint64_t, avoids struct-padding portability issues):
 *   seq_chol[3]:      k, tileSize, out_event
 *   trisolve[4]:      k, j, tileSize, out_event
 *   update_diag[4]:   k, j, tileSize, out_event
 *   update_nondiag[5]:k, j, i, tileSize, out_event
 *   wrap_up[3]:       numTiles, tileSize, expect_trace (raw uint64 bits)
 */

#include "arts.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK_SIZE 8

static inline unsigned tile_owner(int i, int j) {
  int linear = i * (i + 1) / 2 + j;
  return (unsigned)((linear / CHUNK_SIZE) % arts_get_total_ranks());
}

/* ========================================================================= */
/*  EDT bodies                                                                */
/* ========================================================================= */

void sequential_cholesky_edt(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int k = (int)paramv[0];
  int tileSize = (int)paramv[1];
  arts_guid_t out_event = (arts_guid_t)paramv[2];
  (void)k;

  double *aBlock = (double *)depv[0].ptr;
  double *lBlock;
#if ARTS_USE_CXL
  arts_guid_t l_guid =
      arts_db_create((void **)&lBlock, sizeof(double) * tileSize * tileSize,
                     ARTS_DB_CXL, 0, NULL);
#else
  arts_guid_t l_guid = arts_db_create(
      (void **)&lBlock, sizeof(double) * tileSize * tileSize, ARTS_DB_DEFAULT,
      0, &(arts_db_hint_t){.rank = arts_get_current_rank()});
#endif
  memset(lBlock, 0, sizeof(double) * (size_t)tileSize * (size_t)tileSize);

  for (int kB = 0; kB < tileSize; ++kB) {
    if (aBlock[kB * tileSize + kB] <= 0.0) {
      arts_printf("Cholesky: non-SPD at k=%d kB=%d val=%f\n", k, kB,
                  aBlock[kB * tileSize + kB]);
      arts_shutdown();
      return;
    }
    lBlock[kB * tileSize + kB] = sqrt(aBlock[kB * tileSize + kB]);
    for (int jB = kB + 1; jB < tileSize; ++jB) {
      lBlock[jB * tileSize + kB] =
          aBlock[jB * tileSize + kB] / lBlock[kB * tileSize + kB];
    }
    for (int jB = kB + 1; jB < tileSize; ++jB) {
      for (int iB = jB; iB < tileSize; ++iB) {
        aBlock[iB * tileSize + jB] -=
            lBlock[iB * tileSize + kB] * lBlock[jB * tileSize + kB];
      }
    }
  }

  arts_db_release(l_guid, DB_MODE_RW); // Pattern A
  arts_event_satisfy_slot(out_event, l_guid, ARTS_EVENT_LATCH_DECR_SLOT);
}

void trisolve_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int k = (int)paramv[0];
  int j = (int)paramv[1];
  int tileSize = (int)paramv[2];
  arts_guid_t out_event = (arts_guid_t)paramv[3];
  (void)k;
  (void)j;

  double *aBlock = (double *)depv[0].ptr;
  const double *liBlock = (const double *)depv[1].ptr;
  double *loBlock;
#if ARTS_USE_CXL
  arts_guid_t lo_guid =
      arts_db_create((void **)&loBlock, sizeof(double) * tileSize * tileSize,
                     ARTS_DB_CXL, 0, NULL);
#else
  arts_guid_t lo_guid = arts_db_create(
      (void **)&loBlock, sizeof(double) * tileSize * tileSize, ARTS_DB_DEFAULT,
      0, &(arts_db_hint_t){.rank = arts_get_current_rank()});
#endif
  memset(loBlock, 0, sizeof(double) * (size_t)tileSize * (size_t)tileSize);

  for (int kB = 0; kB < tileSize; ++kB) {
    for (int iB = 0; iB < tileSize; ++iB) {
      loBlock[iB * tileSize + kB] =
          aBlock[iB * tileSize + kB] / liBlock[kB * tileSize + kB];
    }
    for (int jB = kB + 1; jB < tileSize; ++jB) {
      for (int iB = 0; iB < tileSize; ++iB) {
        aBlock[iB * tileSize + jB] -=
            liBlock[jB * tileSize + kB] * loBlock[iB * tileSize + kB];
      }
    }
  }

  arts_db_release(lo_guid, DB_MODE_RW); // Pattern A
  arts_event_satisfy_slot(out_event, lo_guid, ARTS_EVENT_LATCH_DECR_SLOT);
}

void update_diagonal_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int k = (int)paramv[0];
  int j = (int)paramv[1];
  int tileSize = (int)paramv[2];
  arts_guid_t out_event = (arts_guid_t)paramv[3];
  (void)k;
  (void)j;

  double *aBlock = (double *)depv[0].ptr;
  const double *l2 = (const double *)depv[1].ptr;

  for (int jB = 0; jB < tileSize; ++jB) {
    for (int kB = 0; kB < tileSize; ++kB) {
      double temp = -l2[jB * tileSize + kB];
      for (int iB = jB; iB < tileSize; ++iB) {
        aBlock[iB * tileSize + jB] += temp * l2[iB * tileSize + kB];
      }
    }
  }

  /* In-place modify + forward depv[0].guid (same as OCR cholesky.c).
   * Save the guid before arts_db_release, which nullifies the depv slot. */
  arts_guid_t out_db = depv[0].guid;
  arts_db_release(out_db, DB_MODE_RW); // Pattern A
  arts_event_satisfy_slot(out_event, out_db, ARTS_EVENT_LATCH_DECR_SLOT);
}

void update_nondiagonal_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int k = (int)paramv[0];
  int j = (int)paramv[1];
  int i = (int)paramv[2];
  int tileSize = (int)paramv[3];
  arts_guid_t out_event = (arts_guid_t)paramv[4];
  (void)k;
  (void)j;
  (void)i;

  double *aBlock = (double *)depv[0].ptr;
  const double *l1 = (const double *)depv[1].ptr;
  const double *l2 = (const double *)depv[2].ptr;

  for (int jB = 0; jB < tileSize; ++jB) {
    for (int kB = 0; kB < tileSize; ++kB) {
      double temp = -l2[jB * tileSize + kB];
      for (int iB = 0; iB < tileSize; ++iB) {
        aBlock[iB * tileSize + jB] += temp * l1[iB * tileSize + kB];
      }
    }
  }

  /* In-place modify + forward depv[0].guid (same as OCR cholesky.c).
   * Save the guid before arts_db_release, which nullifies the depv slot. */
  arts_guid_t out_db = depv[0].guid;
  arts_db_release(out_db, DB_MODE_RW); // Pattern A
  arts_event_satisfy_slot(out_event, out_db, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* wrap_up_edt paramv: [numTiles, tileSize, expect_trace_bits] */
void wrap_up_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int numTiles = (int)paramv[0];
  int tileSize = (int)paramv[1];
  double expect_trace;
  memcpy(&expect_trace, &paramv[2], sizeof(double));

  /* trace(A) = trace(L L^T) = sum_{i,j} L[i,j]^2, summed over all
   * nonzero entries of L (full off-diagonal tiles + lower-triangular
   * part of diagonal tiles).                                           */
  double trace = 0.0;
  for (int i = 0; i < numTiles; ++i) {
    for (int j = 0; j <= i; ++j) {
      int idx = i * (i + 1) / 2 + j;
      const double *L = (const double *)depv[idx].ptr;
      if (i == j) {
        for (int a = 0; a < tileSize; ++a)
          for (int b = 0; b <= a; ++b) {
            double d = L[a * tileSize + b];
            trace += d * d;
          }
      } else {
        for (int a = 0; a < tileSize; ++a)
          for (int b = 0; b < tileSize; ++b) {
            double d = L[a * tileSize + b];
            trace += d * d;
          }
      }
    }
  }
  double rel = (expect_trace != 0.0)
                   ? fabs(trace - expect_trace) / fabs(expect_trace)
                   : 0.0;
  const char *verdict = (rel < 1e-6) ? "PASS" : "FAIL";
  arts_printf("Cholesky-ARTS: N=%d tile=%d numTiles=%d trace(L L^T)=%.6e "
              "expect=%.6e rel_err=%.3e %s\n",
              numTiles * tileSize, tileSize, numTiles, trace, expect_trace, rel,
              verdict);
  arts_shutdown();
}

/* ========================================================================= */
/*  main_edt                                                                 */
/* ========================================================================= */

static double *read_matrix_file(const char *path, int N) {
  FILE *f = fopen(path, "r");
  if (!f) {
    arts_printf("Cholesky: cannot open %s\n", path);
    return NULL;
  }
  double *m = (double *)malloc((size_t)N * N * sizeof(double));
  size_t got = fread(m, sizeof(double), (size_t)N * N, f);
  fclose(f);
  if ((int)got != N * N) {
    free(m);
    arts_printf("Cholesky: short read %zu / %d\n", got, N * N);
    return NULL;
  }
  return m;
}

static double *generate_spd(int N) {
  srand(42);
  double *A = (double *)malloc((size_t)N * N * sizeof(double));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double v = (i == j) ? (double)(2 * N) : (double)((i + j) % 7) * 0.1;
      A[i * N + j] = v;
    }
  }
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j)
      A[j * N + i] = A[i * N + j];
  return A;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];

  int matrixSize = 100;
  int tileSize = 50;
  const char *fi = NULL;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--ds") && i + 1 < argc)
      matrixSize = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--ts") && i + 1 < argc)
      tileSize = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--fi") && i + 1 < argc)
      fi = argv[++i];
  }
  if (matrixSize % tileSize != 0) {
    arts_printf("Cholesky: matrixSize %d not divisible by tileSize %d\n",
                matrixSize, tileSize);
    arts_shutdown();
    return;
  }
  int numTiles = matrixSize / tileSize;

  double *matrix =
      fi ? read_matrix_file(fi, matrixSize) : generate_spd(matrixSize);
  if (!matrix) {
    arts_shutdown();
    return;
  }
  double expect_trace = 0.0;
  for (int d = 0; d < matrixSize; ++d)
    expect_trace += matrix[d * matrixSize + d];

  arts_printf("Cholesky-ARTS: N=%d tile=%d numTiles=%d nodes=%u\n", matrixSize,
              tileSize, numTiles, arts_get_total_ranks());

  int lowerN = numTiles * (numTiles + 1) / 2;
  arts_guid_t *lkji = (arts_guid_t *)malloc(
      sizeof(arts_guid_t) * (size_t)lowerN * (size_t)(numTiles + 1));
  for (int i = 0; i < numTiles; ++i)
    for (int j = 0; j <= i; ++j)
      for (int k = 0; k <= numTiles; ++k) {
        int idx = (i * (i + 1) / 2 + j) * (numTiles + 1) + k;
        lkji[idx] = arts_event_create(&ARTS_EVENT_HINT_ONCE);
      }

#define EV(ii, jj, kk)                                                         \
  (lkji[(((ii) * ((ii) + 1) / 2) + (jj)) * (numTiles + 1) + (kk)])

  /* Prescribe all phase EDTs FIRST (before satisfying k=0 events).
   * This avoids races where events fire with no consumers registered
   * yet. */

  for (int k = 0; k < numTiles; ++k) {
    /* Phase 1: seq_chol at (k, k) */
    {
      uint64_t p[3] = {(uint64_t)k, (uint64_t)tileSize,
                       (uint64_t)EV(k, k, k + 1)};
      arts_guid_t e =
          arts_edt_create(sequential_cholesky_edt, 3, p, 1,
                          &(arts_edt_hint_t){.rank = tile_owner(k, k)});
      arts_add_dependence(EV(k, k, k), e, 0, DB_MODE_RW);
    }

    /* Phase 2: trisolve at (j, k) for j > k */
    for (int j = k + 1; j < numTiles; ++j) {
      uint64_t p[4] = {(uint64_t)k, (uint64_t)j, (uint64_t)tileSize,
                       (uint64_t)EV(j, k, k + 1)};
      arts_guid_t e = arts_edt_create(
          trisolve_edt, 4, p, 2, &(arts_edt_hint_t){.rank = tile_owner(j, k)});
      arts_add_dependence(EV(j, k, k), e, 0, DB_MODE_RW);
      arts_add_dependence(EV(k, k, k + 1), e, 1, DB_MODE_RO);
    }

    /* Phase 3a: update_diagonal at (j, j) for j > k */
    for (int j = k + 1; j < numTiles; ++j) {
      uint64_t p[4] = {(uint64_t)k, (uint64_t)j, (uint64_t)tileSize,
                       (uint64_t)EV(j, j, k + 1)};
      arts_guid_t e =
          arts_edt_create(update_diagonal_edt, 4, p, 2,
                          &(arts_edt_hint_t){.rank = tile_owner(j, j)});
      arts_add_dependence(EV(j, j, k), e, 0, DB_MODE_RW);
      arts_add_dependence(EV(j, k, k + 1), e, 1, DB_MODE_RO);
    }

    /* Phase 3b: update_nondiagonal at (j, i) for j > i > k */
    for (int j = k + 1; j < numTiles; ++j) {
      for (int i = k + 1; i < j; ++i) {
        uint64_t p[5] = {(uint64_t)k, (uint64_t)j, (uint64_t)i,
                         (uint64_t)tileSize, (uint64_t)EV(j, i, k + 1)};
        arts_guid_t e =
            arts_edt_create(update_nondiagonal_edt, 5, p, 3,
                            &(arts_edt_hint_t){.rank = tile_owner(j, i)});
        arts_add_dependence(EV(j, i, k), e, 0, DB_MODE_RW);
        arts_add_dependence(EV(j, k, k + 1), e, 1, DB_MODE_RO);
        arts_add_dependence(EV(i, k, k + 1), e, 2, DB_MODE_RO);
      }
    }
  }

  /* wrap_up at the end: reads all final-generation L tiles (lower-tri). */
  {
    uint64_t p[3];
    p[0] = (uint64_t)numTiles;
    p[1] = (uint64_t)tileSize;
    memcpy(&p[2], &expect_trace, sizeof(double));
    arts_guid_t e = arts_edt_create(wrap_up_edt, 3, p, (uint32_t)lowerN,
                                    &(arts_edt_hint_t){.rank = 0});
    int idx = 0;
    for (int i = 0; i < numTiles; ++i) {
      for (int j = 0; j <= i; ++j) {
        /* Last generation for tile (i,j):
         *   diagonal (i==j): seq_chol at k=i writes EV(i,i,i+1)  -> final_k = i+1
         *   off-diagonal (i>j): trisolve at k=j writes EV(i,j,j+1) -> final_k = j+1
         *     (update_nondiagonal runs at k=0..j-1, BEFORE trisolve at k=j,
         *      so trisolve output is the final L tile for off-diagonal tiles) */
        int final_k = j + 1;
        arts_add_dependence(EV(i, j, final_k), e, (uint32_t)idx++, DB_MODE_RO);
      }
    }
  }

  for (int i = 0; i < numTiles; ++i) {
    for (int j = 0; j <= i; ++j) {
      double *tile;
#if ARTS_USE_CXL
      arts_guid_t g =
          arts_db_create((void **)&tile, sizeof(double) * tileSize * tileSize,
                         ARTS_DB_CXL, 0, NULL);
#else
      arts_guid_t g = arts_db_create(
          (void **)&tile, sizeof(double) * tileSize * tileSize, ARTS_DB_DEFAULT,
          0, NULL);
#endif
      for (int ti = 0; ti < tileSize; ++ti)
        for (int tj = 0; tj < tileSize; ++tj) {
          int A_i = i * tileSize + ti;
          int A_j = j * tileSize + tj;
          tile[ti * tileSize + tj] = matrix[A_i * matrixSize + A_j];
        }
      arts_db_release(g, DB_MODE_RW); // Pattern A
      arts_event_satisfy_slot(EV(i, j, 0), g, ARTS_EVENT_LATCH_DECR_SLOT);
    }
  }
  free(matrix);

  free(lkji);
#undef EV
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  (void)node_id;
  (void)argc;
  (void)argv;
}

int main(int argc, char *argv[]) {
  arts_rt(argc, argv);
  return 0;
}

