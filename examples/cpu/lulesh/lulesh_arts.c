/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/

/*
 * LULESH 2.0 -- 6-DB-partition ARTS implementation.
 *
 * 6 DataBlocks per tile: const_db, pos_vel_db, elem_state_db, grad_db,
 *                        force_db, dt_result_db
 *
 * 4-Phase DAG per iteration (ordering via LATCH events, data via depv[] DB
 * deps): Phase 1: forces_edt(t)          MAP      -- stress + hourglass forces
 *   Phase 2: reduce_kin_edt(t)      REDUCE   -- force gather +
 * velocity/position update Phase 3: elem_props_edt(t)      MAP      -- volume,
 * gradients Phase 4: visc_eos_time_edt(t)   MAP      -- monotonic Q, EOS, time
 * constraints Reduction: reduce_dt_edt        REDUCE   -- global dt computation
 *
 * All synchronization via explicit LATCH(N)+DECR-only events
 * (distributed-safe). CDAG is NOT used for ordering -- it is a safety net only.
 * g_init is global but used ONLY by launch_iteration (rank 0).
 * EDT bodies access data exclusively through depv[].
 */

#include "arts.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Simple in-place profiling of serial launch_iteration work ----
 * Enable at compile time with -DLULESH_PROFILE. When on, every call to
 * launch_iteration accumulates its wall time into g_prof_launch_ns and
 * the launch count into g_prof_launch_count. reduce_dt_edt prints the
 * totals at shutdown. */
#ifdef LULESH_PROFILE
static uint64_t g_prof_launch_ns = 0;
static uint64_t g_prof_launch_count = 0;
static uint64_t g_prof_first_ns = 0;
static uint64_t prof_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

/* ========================================================================= */
/*  DB header structures                                                     */
/* ========================================================================= */

#define TILE_PTR(h, offset, type) ((type *)((char *)(h) + (offset)))

static inline size_t align8(size_t s) { return (s + 7u) & ~(size_t)7u; }
static inline int nidx(int T, int x, int y, int z) {
  int n = T + 1;
  return z * n * n + y * n + x;
}
static inline int eidx(int T, int x, int y, int z) {
  return z * T * T + y * T + x;
}

typedef struct {
  int tile_id, tile_ix, tile_iy, tile_iz;
  int tile_elems, num_elems, num_nodes, global_edge_elems, tiles_per_dim,
      max_iterations;
  int num_face_nbrs, num_all_nbrs;
  int face_nbr_dir[6];
  double hgcoef, ss4o3, qstop, monoq_max_slope, monoq_limiter_mult;
  double qlc_monoq, qqc_monoq, qqc;
  double eosvmax, eosvmin, pmin, emin, dvovmax, refdens;
  double e_cut, p_cut, q_cut, v_cut, u_cut;
  double stop_time, max_delta_time, deltatimemultlb, deltatimemultub;
  size_t off_elem_node, off_elem_elem, off_node_elem;
  size_t off_node_mass, off_elem_mass, off_elem_init_vol;
  size_t off_sym_x, off_sym_y, off_sym_z;
  size_t total_size;
} const_db_header_t;

typedef struct {
  size_t off_pos_x, off_pos_y, off_pos_z;
  size_t off_vel_x, off_vel_y, off_vel_z;
  size_t total_size;
} pos_vel_db_header_t;

typedef struct {
  size_t off_pressure, off_energy, off_viscosity, off_sound_speed;
  size_t off_volume, off_prev_volume;
  size_t total_size;
} elem_state_db_header_t;

typedef struct {
  size_t off_vdov, off_arealg;
  size_t off_delv_xi, off_delv_eta, off_delv_zeta;
  size_t off_delx_xi, off_delx_eta, off_delx_zeta;
  size_t off_ql, off_qq, off_new_volume;
  size_t total_size;
} grad_db_header_t;

typedef struct {
  size_t off_elem_fx, off_elem_fy, off_elem_fz;
  size_t off_pq_cache, off_ss_cache, off_vol_cache;
  size_t total_size;
} force_db_header_t;

typedef struct {
  double dt_courant;
  double dt_hydro;
  double origin_energy; /* tile 0 only: energy[0] for final output */
} dt_result_db_t;

/* ========================================================================= */
/*  Offset setup                                                             */
/* ========================================================================= */

#define DECL_A size_t o_
#define A(f, c, t)                                                             \
  do {                                                                         \
    h->f = o_;                                                                 \
    o_ += align8((size_t)(c) * sizeof(t));                                     \
  } while (0)

static void set_const_offsets(const_db_header_t *h, int T) {
  int ne = T * T * T, nn = (T + 1) * (T + 1) * (T + 1);
  DECL_A;
  o_ = align8(sizeof(const_db_header_t));
  A(off_elem_node, ne * 8, int);
  A(off_elem_elem, ne * 6, int);
  A(off_node_elem, nn * 8, int);
  A(off_node_mass, nn, double);
  A(off_elem_mass, ne, double);
  A(off_elem_init_vol, ne, double);
  A(off_sym_x, nn, int);
  A(off_sym_y, nn, int);
  A(off_sym_z, nn, int);
  h->total_size = o_;
}
static void set_pos_vel_offsets(pos_vel_db_header_t *h, int T) {
  int nn = (T + 1) * (T + 1) * (T + 1);
  DECL_A;
  o_ = align8(sizeof(pos_vel_db_header_t));
  A(off_pos_x, nn, double);
  A(off_pos_y, nn, double);
  A(off_pos_z, nn, double);
  A(off_vel_x, nn, double);
  A(off_vel_y, nn, double);
  A(off_vel_z, nn, double);
  h->total_size = o_;
}
static void set_elem_state_offsets(elem_state_db_header_t *h, int T) {
  int ne = T * T * T;
  DECL_A;
  o_ = align8(sizeof(elem_state_db_header_t));
  A(off_pressure, ne, double);
  A(off_energy, ne, double);
  A(off_viscosity, ne, double);
  A(off_sound_speed, ne, double);
  A(off_volume, ne, double);
  A(off_prev_volume, ne, double);
  h->total_size = o_;
}
static void set_grad_offsets(grad_db_header_t *h, int T) {
  int ne = T * T * T;
  DECL_A;
  o_ = align8(sizeof(grad_db_header_t));
  A(off_vdov, ne, double);
  A(off_arealg, ne, double);
  A(off_delv_xi, ne, double);
  A(off_delv_eta, ne, double);
  A(off_delv_zeta, ne, double);
  A(off_delx_xi, ne, double);
  A(off_delx_eta, ne, double);
  A(off_delx_zeta, ne, double);
  A(off_ql, ne, double);
  A(off_qq, ne, double);
  A(off_new_volume, ne, double);
  h->total_size = o_;
}
static void set_force_offsets(force_db_header_t *h, int T) {
  int ne = T * T * T;
  DECL_A;
  o_ = align8(sizeof(force_db_header_t));
  A(off_elem_fx, ne * 8, double);
  A(off_elem_fy, ne * 8, double);
  A(off_elem_fz, ne * 8, double);
  A(off_pq_cache, ne, double);
  A(off_ss_cache, ne, double);
  A(off_vol_cache, ne, double);
  h->total_size = o_;
}
#undef A
#undef DECL_A

static size_t compute_const_db_size(int T) {
  const_db_header_t h;
  memset(&h, 0, sizeof(h));
  set_const_offsets(&h, T);
  return h.total_size;
}
static size_t compute_pos_vel_db_size(int T) {
  pos_vel_db_header_t h;
  memset(&h, 0, sizeof(h));
  set_pos_vel_offsets(&h, T);
  return h.total_size;
}
static size_t compute_elem_state_db_size(int T) {
  elem_state_db_header_t h;
  memset(&h, 0, sizeof(h));
  set_elem_state_offsets(&h, T);
  return h.total_size;
}
static size_t compute_grad_db_size(int T) {
  grad_db_header_t h;
  memset(&h, 0, sizeof(h));
  set_grad_offsets(&h, T);
  return h.total_size;
}
static size_t compute_force_db_size(int T) {
  force_db_header_t h;
  memset(&h, 0, sizeof(h));
  set_force_offsets(&h, T);
  return h.total_size;
}

/* ========================================================================= */
/*  Global immutable topology + init state                                   */
/* ========================================================================= */

#define MAX_TILES 1000

static struct {
  int num_tiles, tiles_per_dim, tile_elems, global_edge_elems;
  int face_nbrs[MAX_TILES][6];
  int all_nbrs[MAX_TILES][26];
  int num_face_nbrs[MAX_TILES];
  int num_all_nbrs[MAX_TILES];
  int face_nbr_dir[MAX_TILES][6];
  unsigned int tile_owner[MAX_TILES];
  arts_guid_t const_guids[MAX_TILES];
  arts_guid_t pos_vel_guids[MAX_TILES];
  arts_guid_t elem_state_guids[MAX_TILES];
  arts_guid_t force_guids[MAX_TILES];
  arts_guid_t grad_guids[MAX_TILES];
  arts_guid_t dt_result_guids[MAX_TILES];
} g_init;

/* ========================================================================= */
/*  LULESH v2.0 physics kernels                                              */
/* ========================================================================= */

static double calc_elem_volume(const double x[8], const double y[8],
                               const double z[8]) {
  double twelfth = 1.0 / 12.0;
  double dx61 = x[6] - x[1], dy61 = y[6] - y[1], dz61 = z[6] - z[1];
  double dx70 = x[7] - x[0], dy70 = y[7] - y[0], dz70 = z[7] - z[0];
  double dx63 = x[6] - x[3], dy63 = y[6] - y[3], dz63 = z[6] - z[3];
  double dx20 = x[2] - x[0], dy20 = y[2] - y[0], dz20 = z[2] - z[0];
  double dx50 = x[5] - x[0], dy50 = y[5] - y[0], dz50 = z[5] - z[0];
  double dx64 = x[6] - x[4], dy64 = y[6] - y[4], dz64 = z[6] - z[4];
  double dx31 = x[3] - x[1], dy31 = y[3] - y[1], dz31 = z[3] - z[1];
  double dx72 = x[7] - x[2], dy72 = y[7] - y[2], dz72 = z[7] - z[2];
  double dx43 = x[4] - x[3], dy43 = y[4] - y[3], dz43 = z[4] - z[3];
  double dx57 = x[5] - x[7], dy57 = y[5] - y[7], dz57 = z[5] - z[7];
  double dx14 = x[1] - x[4], dy14 = y[1] - y[4], dz14 = z[1] - z[4];
  double dx25 = x[2] - x[5], dy25 = y[2] - y[5], dz25 = z[2] - z[5];
#define TP(ax, ay, az, bx, by, bz, cx, cy, cz)                                 \
  ((ax) * ((by) * (cz) - (bz) * (cy)) + (bx) * ((cy) * (az) - (ay) * (cz)) +   \
   (cx) * ((ay) * (bz) - (az) * (by)))
  return twelfth * (TP(dx31 + dx72, dy31 + dy72, dz31 + dz72, dx63, dy63, dz63,
                       dx20, dy20, dz20) +
                    TP(dx43 + dx57, dy43 + dy57, dz43 + dz57, dx64, dy64, dz64,
                       dx70, dy70, dz70) +
                    TP(dx14 + dx25, dy14 + dy25, dz14 + dz25, dx61, dy61, dz61,
                       dx50, dy50, dz50));
#undef TP
}

static void calc_elem_sfd(const double x[8], const double y[8],
                          const double z[8], double b[3][8], double *volume) {
  double fjxxi =
      .125 * ((x[6] - x[0]) + (x[5] - x[3]) - (x[7] - x[1]) - (x[4] - x[2]));
  double fjxet =
      .125 * ((x[6] - x[0]) - (x[5] - x[3]) + (x[7] - x[1]) - (x[4] - x[2]));
  double fjxze =
      .125 * ((x[6] - x[0]) + (x[5] - x[3]) + (x[7] - x[1]) + (x[4] - x[2]));
  double fjyxi =
      .125 * ((y[6] - y[0]) + (y[5] - y[3]) - (y[7] - y[1]) - (y[4] - y[2]));
  double fjyet =
      .125 * ((y[6] - y[0]) - (y[5] - y[3]) + (y[7] - y[1]) - (y[4] - y[2]));
  double fjyze =
      .125 * ((y[6] - y[0]) + (y[5] - y[3]) + (y[7] - y[1]) + (y[4] - y[2]));
  double fjzxi =
      .125 * ((z[6] - z[0]) + (z[5] - z[3]) - (z[7] - z[1]) - (z[4] - z[2]));
  double fjzet =
      .125 * ((z[6] - z[0]) - (z[5] - z[3]) + (z[7] - z[1]) - (z[4] - z[2]));
  double fjzze =
      .125 * ((z[6] - z[0]) + (z[5] - z[3]) + (z[7] - z[1]) + (z[4] - z[2]));
  double cjxxi = fjyet * fjzze - fjzet * fjyze;
  double cjxet = -fjyxi * fjzze + fjzxi * fjyze;
  double cjxze = fjyxi * fjzet - fjzxi * fjyet;
  double cjyxi = -fjxet * fjzze + fjzet * fjxze;
  double cjyet = fjxxi * fjzze - fjzxi * fjxze;
  double cjyze = -fjxxi * fjzet + fjzxi * fjxet;
  double cjzxi = fjxet * fjyze - fjyet * fjxze;
  double cjzet = -fjxxi * fjyze + fjyxi * fjxze;
  double cjzze = fjxxi * fjyet - fjyxi * fjxet;
  b[0][0] = -cjxxi - cjxet - cjxze;
  b[0][1] = cjxxi - cjxet - cjxze;
  b[0][2] = cjxxi + cjxet - cjxze;
  b[0][3] = -cjxxi + cjxet - cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];
  b[1][0] = -cjyxi - cjyet - cjyze;
  b[1][1] = cjyxi - cjyet - cjyze;
  b[1][2] = cjyxi + cjyet - cjyze;
  b[1][3] = -cjyxi + cjyet - cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];
  b[2][0] = -cjzxi - cjzet - cjzze;
  b[2][1] = cjzxi - cjzet - cjzze;
  b[2][2] = cjzxi + cjzet - cjzze;
  b[2][3] = -cjzxi + cjzet - cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];
  *volume = 8.0 * (fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

static void calc_elem_vel_grad(const double xv[8], const double yv[8],
                               const double zv[8], const double b[3][8],
                               double detJ, double d[6]) {
  double inv = 1.0 / detJ;
  const double *pfx = b[0], *pfy = b[1], *pfz = b[2];
  d[0] = inv * (pfx[0] * (xv[0] - xv[6]) + pfx[1] * (xv[1] - xv[7]) +
                pfx[2] * (xv[2] - xv[4]) + pfx[3] * (xv[3] - xv[5]));
  d[1] = inv * (pfy[0] * (yv[0] - yv[6]) + pfy[1] * (yv[1] - yv[7]) +
                pfy[2] * (yv[2] - yv[4]) + pfy[3] * (yv[3] - yv[5]));
  d[2] = inv * (pfz[0] * (zv[0] - zv[6]) + pfz[1] * (zv[1] - zv[7]) +
                pfz[2] * (zv[2] - zv[4]) + pfz[3] * (zv[3] - zv[5]));
  double dyddx = inv * (pfx[0] * (yv[0] - yv[6]) + pfx[1] * (yv[1] - yv[7]) +
                        pfx[2] * (yv[2] - yv[4]) + pfx[3] * (yv[3] - yv[5]));
  double dxddy = inv * (pfy[0] * (xv[0] - xv[6]) + pfy[1] * (xv[1] - xv[7]) +
                        pfy[2] * (xv[2] - xv[4]) + pfy[3] * (xv[3] - xv[5]));
  double dzddx = inv * (pfx[0] * (zv[0] - zv[6]) + pfx[1] * (zv[1] - zv[7]) +
                        pfx[2] * (zv[2] - zv[4]) + pfx[3] * (zv[3] - zv[5]));
  double dxddz = inv * (pfz[0] * (xv[0] - xv[6]) + pfz[1] * (xv[1] - xv[7]) +
                        pfz[2] * (xv[2] - xv[4]) + pfz[3] * (xv[3] - xv[5]));
  double dzddy = inv * (pfy[0] * (zv[0] - zv[6]) + pfy[1] * (zv[1] - zv[7]) +
                        pfy[2] * (zv[2] - zv[4]) + pfy[3] * (zv[3] - zv[5]));
  double dyddz = inv * (pfz[0] * (yv[0] - yv[6]) + pfz[1] * (yv[1] - yv[7]) +
                        pfz[2] * (yv[2] - yv[4]) + pfz[3] * (yv[3] - yv[5]));
  d[5] = .5 * (dxddy + dyddx);
  d[4] = .5 * (dxddz + dzddx);
  d[3] = .5 * (dzddy + dyddz);
}

static void sum_face_normal(double *px0, double *py0, double *pz0, double *px1,
                            double *py1, double *pz1, double *px2, double *py2,
                            double *pz2, double *px3, double *py3, double *pz3,
                            double x0, double y0, double z0, double x1,
                            double y1, double z1, double x2, double y2,
                            double z2, double x3, double y3, double z3) {
  double bx0 = .5 * (x3 + x2 - x1 - x0), by0 = .5 * (y3 + y2 - y1 - y0),
         bz0 = .5 * (z3 + z2 - z1 - z0);
  double bx1 = .5 * (x2 + x1 - x3 - x0), by1 = .5 * (y2 + y1 - y3 - y0),
         bz1 = .5 * (z2 + z1 - z3 - z0);
  double ax = .25 * (by0 * bz1 - bz0 * by1), ay = .25 * (bz0 * bx1 - bx0 * bz1),
         az = .25 * (bx0 * by1 - by0 * bx1);
  *px0 += ax;
  *px1 += ax;
  *px2 += ax;
  *px3 += ax;
  *py0 += ay;
  *py1 += ay;
  *py2 += ay;
  *py3 += ay;
  *pz0 += az;
  *pz1 += az;
  *pz2 += az;
  *pz3 += az;
}

static void calc_node_normals(double pfx[8], double pfy[8], double pfz[8],
                              const double x[8], const double y[8],
                              const double z[8]) {
  for (int i = 0; i < 8; i++) {
    pfx[i] = 0;
    pfy[i] = 0;
    pfz[i] = 0;
  }
  sum_face_normal(&pfx[0], &pfy[0], &pfz[0], &pfx[1], &pfy[1], &pfz[1], &pfx[2],
                  &pfy[2], &pfz[2], &pfx[3], &pfy[3], &pfz[3], x[0], y[0], z[0],
                  x[1], y[1], z[1], x[2], y[2], z[2], x[3], y[3], z[3]);
  sum_face_normal(&pfx[0], &pfy[0], &pfz[0], &pfx[4], &pfy[4], &pfz[4], &pfx[5],
                  &pfy[5], &pfz[5], &pfx[1], &pfy[1], &pfz[1], x[0], y[0], z[0],
                  x[4], y[4], z[4], x[5], y[5], z[5], x[1], y[1], z[1]);
  sum_face_normal(&pfx[1], &pfy[1], &pfz[1], &pfx[5], &pfy[5], &pfz[5], &pfx[6],
                  &pfy[6], &pfz[6], &pfx[2], &pfy[2], &pfz[2], x[1], y[1], z[1],
                  x[5], y[5], z[5], x[6], y[6], z[6], x[2], y[2], z[2]);
  sum_face_normal(&pfx[2], &pfy[2], &pfz[2], &pfx[6], &pfy[6], &pfz[6], &pfx[7],
                  &pfy[7], &pfz[7], &pfx[3], &pfy[3], &pfz[3], x[2], y[2], z[2],
                  x[6], y[6], z[6], x[7], y[7], z[7], x[3], y[3], z[3]);
  sum_face_normal(&pfx[3], &pfy[3], &pfz[3], &pfx[7], &pfy[7], &pfz[7], &pfx[4],
                  &pfy[4], &pfz[4], &pfx[0], &pfy[0], &pfz[0], x[3], y[3], z[3],
                  x[7], y[7], z[7], x[4], y[4], z[4], x[0], y[0], z[0]);
  sum_face_normal(&pfx[4], &pfy[4], &pfz[4], &pfx[7], &pfy[7], &pfz[7], &pfx[6],
                  &pfy[6], &pfz[6], &pfx[5], &pfy[5], &pfz[5], x[4], y[4], z[4],
                  x[7], y[7], z[7], x[6], y[6], z[6], x[5], y[5], z[5]);
}

static void volu_der(double x0, double x1, double x2, double x3, double x4,
                     double x5, double y0, double y1, double y2, double y3,
                     double y4, double y5, double z0, double z1, double z2,
                     double z3, double z4, double z5, double *dvdx,
                     double *dvdy, double *dvdz) {
  double t = 1.0 / 12.0;
  *dvdx =
      ((y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) + (y0 + y4) * (z3 + z4) -
       (y3 + y4) * (z0 + z4) - (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5)) *
      t;
  *dvdy =
      (-(x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) - (x0 + x4) * (z3 + z4) +
       (x3 + x4) * (z0 + z4) + (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5)) *
      t;
  *dvdz =
      (-(x1 + x2) * (y0 + y1) + (x0 + x1) * (y1 + y2) - (x0 + x4) * (y3 + y4) +
       (x3 + x4) * (y0 + y4) + (x2 + x5) * (y3 + y5) - (x3 + x5) * (y2 + y5)) *
      t;
}

static void calc_elem_vol_deriv(double dvdx[8], double dvdy[8], double dvdz[8],
                                const double x[8], const double y[8],
                                const double z[8]) {
  volu_der(x[1], x[2], x[3], x[4], x[5], x[7], y[1], y[2], y[3], y[4], y[5],
           y[7], z[1], z[2], z[3], z[4], z[5], z[7], &dvdx[0], &dvdy[0],
           &dvdz[0]);
  volu_der(x[0], x[1], x[2], x[7], x[4], x[6], y[0], y[1], y[2], y[7], y[4],
           y[6], z[0], z[1], z[2], z[7], z[4], z[6], &dvdx[3], &dvdy[3],
           &dvdz[3]);
  volu_der(x[3], x[0], x[1], x[6], x[7], x[5], y[3], y[0], y[1], y[6], y[7],
           y[5], z[3], z[0], z[1], z[6], z[7], z[5], &dvdx[2], &dvdy[2],
           &dvdz[2]);
  volu_der(x[2], x[3], x[0], x[5], x[6], x[4], y[2], y[3], y[0], y[5], y[6],
           y[4], z[2], z[3], z[0], z[5], z[6], z[4], &dvdx[1], &dvdy[1],
           &dvdz[1]);
  volu_der(x[7], x[6], x[5], x[0], x[3], x[1], y[7], y[6], y[5], y[0], y[3],
           y[1], z[7], z[6], z[5], z[0], z[3], z[1], &dvdx[4], &dvdy[4],
           &dvdz[4]);
  volu_der(x[4], x[7], x[6], x[1], x[0], x[2], y[4], y[7], y[6], y[1], y[0],
           y[2], z[4], z[7], z[6], z[1], z[0], z[2], &dvdx[5], &dvdy[5],
           &dvdz[5]);
  volu_der(x[5], x[4], x[7], x[2], x[1], x[3], y[5], y[4], y[7], y[2], y[1],
           y[3], z[5], z[4], z[7], z[2], z[1], z[3], &dvdx[6], &dvdy[6],
           &dvdz[6]);
  volu_der(x[6], x[5], x[4], x[3], x[2], x[0], y[6], y[5], y[4], y[3], y[2],
           y[0], z[6], z[5], z[4], z[3], z[2], z[0], &dvdx[7], &dvdy[7],
           &dvdz[7]);
}

static double calc_elem_char_len(const double x[8], const double y[8],
                                 const double z[8], double vol) {
  double cl = 0;
#define AF(a, b, c, d)                                                         \
  do {                                                                         \
    double fx = (x[c] - x[a]) - (x[d] - x[b]),                                 \
           fy = (y[c] - y[a]) - (y[d] - y[b]),                                 \
           fz = (z[c] - z[a]) - (z[d] - z[b]);                                 \
    double gx = (x[c] - x[a]) + (x[d] - x[b]),                                 \
           gy = (y[c] - y[a]) + (y[d] - y[b]),                                 \
           gz = (z[c] - z[a]) + (z[d] - z[b]);                                 \
    double ar =                                                                \
        (fx * fx + fy * fy + fz * fz) * (gx * gx + gy * gy + gz * gz) -        \
        (fx * gx + fy * gy + fz * gz) * (fx * gx + fy * gy + fz * gz);         \
    if (ar > cl)                                                               \
      cl = ar;                                                                 \
  } while (0)
  AF(0, 1, 2, 3);
  AF(4, 5, 6, 7);
  AF(0, 1, 5, 4);
  AF(1, 2, 6, 5);
  AF(2, 3, 7, 6);
  AF(3, 0, 4, 7);
#undef AF
  return 4.0 * vol / sqrt(cl);
}

static const double ghg[4][8] = {{1, 1, -1, -1, -1, -1, 1, 1},
                                 {1, -1, -1, 1, -1, 1, 1, -1},
                                 {1, -1, 1, -1, 1, -1, 1, -1},
                                 {-1, 1, -1, 1, 1, -1, 1, -1}};

static void calc_fb_hg_force(const double xd[8], const double yd[8],
                             const double zd[8], const double hg[8][4],
                             double coef, double hfx[8], double hfy[8],
                             double hfz[8]) {
  for (int m = 0; m < 3; m++) {
    const double *v = (m == 0) ? xd : (m == 1) ? yd : zd;
    double *hf = (m == 0) ? hfx : (m == 1) ? hfy : hfz;
    double h[4];
    for (int i = 0; i < 4; i++) {
      h[i] = 0;
      for (int j = 0; j < 8; j++)
        h[i] += hg[j][i] * v[j];
    }
    for (int i = 0; i < 8; i++)
      hf[i] = coef * (hg[i][0] * h[0] + hg[i][1] * h[1] + hg[i][2] * h[2] +
                      hg[i][3] * h[3]);
  }
}

/* calc_elem_forces_pq: takes pq, ss, vol as direct args instead of estate_db.
 * On iteration 0, p+q=0, ss=0, vol=1.0 (initial values from init).
 * On subsequent iterations, these come from force_db's cache fields
 * (written by previous iteration's visc_eos_time_edt). */
static void calc_elem_forces_pq(const_db_header_t *ch, pos_vel_db_header_t *pvh,
                                int k, double pq_val, double ss_val,
                                double vol_val, double fx[8], double fy[8],
                                double fz[8]) {
  int *en = TILE_PTR(ch, ch->off_elem_node, int);
  double *px = TILE_PTR(pvh, pvh->off_pos_x, double),
         *py = TILE_PTR(pvh, pvh->off_pos_y, double),
         *pz = TILE_PTR(pvh, pvh->off_pos_z, double);
  double *vxp = TILE_PTR(pvh, pvh->off_vel_x, double),
         *vyp = TILE_PTR(pvh, pvh->off_vel_y, double),
         *vzp = TILE_PTR(pvh, pvh->off_vel_z, double);
  double xl[8], yl[8], zl[8], xd[8], yd[8], zd[8];
  for (int c = 0; c < 8; c++) {
    int nd = en[k * 8 + c];
    xl[c] = px[nd];
    yl[c] = py[nd];
    zl[c] = pz[nd];
    xd[c] = vxp[nd];
    yd[c] = vyp[nd];
    zd[c] = vzp[nd];
  }
  double pfx[8], pfy[8], pfz[8];
  calc_node_normals(pfx, pfy, pfz, xl, yl, zl);
  for (int c = 0; c < 8; c++) {
    fx[c] = pq_val * pfx[c];
    fy[c] = pq_val * pfy[c];
    fz[c] = pq_val * pfz[c];
  }
  double hgc = ch->hgcoef;
  if (hgc > 0) {
    double volo = TILE_PTR(ch, ch->off_elem_init_vol, double)[k];
    double em = TILE_PTR(ch, ch->off_elem_mass, double)[k];
    double determ = vol_val * volo, volinv = 1.0 / determ;
    double dvdx[8], dvdy[8], dvdz[8];
    calc_elem_vol_deriv(dvdx, dvdy, dvdz, xl, yl, zl);
    double hmx[4], hmy[4], hmz[4];
    for (int i = 0; i < 4; i++) {
      hmx[i] = hmy[i] = hmz[i] = 0;
      for (int j = 0; j < 8; j++) {
        hmx[i] += xl[j] * ghg[i][j];
        hmy[i] += yl[j] * ghg[i][j];
        hmz[i] += zl[j] * ghg[i][j];
      }
    }
    double hgam[8][4];
    for (int i = 0; i < 8; i++)
      for (int j = 0; j < 4; j++)
        hgam[i][j] = ghg[j][i] - volinv * (dvdx[i] * hmx[j] + dvdy[i] * hmy[j] +
                                           dvdz[i] * hmz[j]);
    double coef = -hgc * 0.01 * ss_val * em / cbrt(determ);
    double hfx[8], hfy[8], hfz[8];
    calc_fb_hg_force(xd, yd, zd, (const double(*)[4])hgam, coef, hfx, hfy, hfz);
    for (int c = 0; c < 8; c++) {
      fx[c] += hfx[c];
      fy[c] += hfy[c];
      fz[c] += hfz[c];
    }
  }
}

static void calc_monoq_gradients(const_db_header_t *ch,
                                 pos_vel_db_header_t *pvh,
                                 grad_db_header_t *gh) {
  int ne = ch->num_elems;
  int *en = TILE_PTR(ch, ch->off_elem_node, int);
  double *px = TILE_PTR(pvh, pvh->off_pos_x, double),
         *py = TILE_PTR(pvh, pvh->off_pos_y, double),
         *pz = TILE_PTR(pvh, pvh->off_pos_z, double);
  double *vxp = TILE_PTR(pvh, pvh->off_vel_x, double),
         *vyp = TILE_PTR(pvh, pvh->off_vel_y, double),
         *vzp = TILE_PTR(pvh, pvh->off_vel_z, double);
  double *dvxi = TILE_PTR(gh, gh->off_delv_xi, double),
         *dveta = TILE_PTR(gh, gh->off_delv_eta, double),
         *dvzeta = TILE_PTR(gh, gh->off_delv_zeta, double);
  double *dxxi = TILE_PTR(gh, gh->off_delx_xi, double),
         *dxeta = TILE_PTR(gh, gh->off_delx_eta, double),
         *dxzeta = TILE_PTR(gh, gh->off_delx_zeta, double);
  double *eivol = TILE_PTR(ch, ch->off_elem_init_vol, double);
  double *vnew = TILE_PTR(gh, gh->off_new_volume, double);
  double ptiny = 1.e-36;
  for (int i = 0; i < ne; i++) {
    int n0 = en[i * 8 + 0], n1 = en[i * 8 + 1], n2 = en[i * 8 + 2],
        n3 = en[i * 8 + 3], n4 = en[i * 8 + 4], n5 = en[i * 8 + 5],
        n6 = en[i * 8 + 6], n7 = en[i * 8 + 7];
    double vol = eivol[i] * vnew[i], norm = 1.0 / (vol + ptiny);
    double dxj = -0.25 * ((px[n0] + px[n1] + px[n5] + px[n4]) -
                          (px[n3] + px[n2] + px[n6] + px[n7]));
    double dyj = -0.25 * ((py[n0] + py[n1] + py[n5] + py[n4]) -
                          (py[n3] + py[n2] + py[n6] + py[n7]));
    double dzj = -0.25 * ((pz[n0] + pz[n1] + pz[n5] + pz[n4]) -
                          (pz[n3] + pz[n2] + pz[n6] + pz[n7]));
    double dxi = 0.25 * ((px[n1] + px[n2] + px[n6] + px[n5]) -
                         (px[n0] + px[n3] + px[n7] + px[n4]));
    double dyi = 0.25 * ((py[n1] + py[n2] + py[n6] + py[n5]) -
                         (py[n0] + py[n3] + py[n7] + py[n4]));
    double dzi = 0.25 * ((pz[n1] + pz[n2] + pz[n6] + pz[n5]) -
                         (pz[n0] + pz[n3] + pz[n7] + pz[n4]));
    double dxk = 0.25 * ((px[n4] + px[n5] + px[n6] + px[n7]) -
                         (px[n0] + px[n1] + px[n2] + px[n3]));
    double dyk = 0.25 * ((py[n4] + py[n5] + py[n6] + py[n7]) -
                         (py[n0] + py[n1] + py[n2] + py[n3]));
    double dzk = 0.25 * ((pz[n4] + pz[n5] + pz[n6] + pz[n7]) -
                         (pz[n0] + pz[n1] + pz[n2] + pz[n3]));
    double ax, ay, az, dxv, dyv, dzv;
    ax = dyi * dzj - dzi * dyj;
    ay = dzi * dxj - dxi * dzj;
    az = dxi * dyj - dyi * dxj;
    dxzeta[i] = vol / sqrt(ax * ax + ay * ay + az * az + ptiny);
    ax *= norm;
    ay *= norm;
    az *= norm;
    dxv = 0.25 * ((vxp[n4] + vxp[n5] + vxp[n6] + vxp[n7]) -
                  (vxp[n0] + vxp[n1] + vxp[n2] + vxp[n3]));
    dyv = 0.25 * ((vyp[n4] + vyp[n5] + vyp[n6] + vyp[n7]) -
                  (vyp[n0] + vyp[n1] + vyp[n2] + vyp[n3]));
    dzv = 0.25 * ((vzp[n4] + vzp[n5] + vzp[n6] + vzp[n7]) -
                  (vzp[n0] + vzp[n1] + vzp[n2] + vzp[n3]));
    dvzeta[i] = ax * dxv + ay * dyv + az * dzv;
    ax = dyj * dzk - dzj * dyk;
    ay = dzj * dxk - dxj * dzk;
    az = dxj * dyk - dyj * dxk;
    dxxi[i] = vol / sqrt(ax * ax + ay * ay + az * az + ptiny);
    ax *= norm;
    ay *= norm;
    az *= norm;
    dxv = 0.25 * ((vxp[n1] + vxp[n2] + vxp[n6] + vxp[n5]) -
                  (vxp[n0] + vxp[n3] + vxp[n7] + vxp[n4]));
    dyv = 0.25 * ((vyp[n1] + vyp[n2] + vyp[n6] + vyp[n5]) -
                  (vyp[n0] + vyp[n3] + vyp[n7] + vyp[n4]));
    dzv = 0.25 * ((vzp[n1] + vzp[n2] + vzp[n6] + vzp[n5]) -
                  (vzp[n0] + vzp[n3] + vzp[n7] + vzp[n4]));
    dvxi[i] = ax * dxv + ay * dyv + az * dzv;
    ax = dyk * dzi - dzk * dyi;
    ay = dzk * dxi - dxk * dzi;
    az = dxk * dyi - dyk * dxi;
    dxeta[i] = vol / sqrt(ax * ax + ay * ay + az * az + ptiny);
    ax *= norm;
    ay *= norm;
    az *= norm;
    dxv = -0.25 * ((vxp[n0] + vxp[n1] + vxp[n5] + vxp[n4]) -
                   (vxp[n3] + vxp[n2] + vxp[n6] + vxp[n7]));
    dyv = -0.25 * ((vyp[n0] + vyp[n1] + vyp[n5] + vyp[n4]) -
                   (vyp[n3] + vyp[n2] + vyp[n6] + vyp[n7]));
    dzv = -0.25 * ((vzp[n0] + vzp[n1] + vzp[n5] + vzp[n4]) -
                   (vzp[n3] + vzp[n2] + vzp[n6] + vzp[n7]));
    dveta[i] = ax * dxv + ay * dyv + az * dzv;
  }
}

static void calc_monoq_region(const_db_header_t *ch, grad_db_header_t *gh,
                              grad_db_header_t *nbr_gh[6], int *nbr_dir,
                              int nfn) {
  int T = ch->tile_elems, ne = ch->num_elems;
  double *dvxi = TILE_PTR(gh, gh->off_delv_xi, double),
         *dveta = TILE_PTR(gh, gh->off_delv_eta, double),
         *dvzeta = TILE_PTR(gh, gh->off_delv_zeta, double);
  double *dxxi = TILE_PTR(gh, gh->off_delx_xi, double),
         *dxeta = TILE_PTR(gh, gh->off_delx_eta, double),
         *dxzeta = TILE_PTR(gh, gh->off_delx_zeta, double);
  double *vdov_a = TILE_PTR(gh, gh->off_vdov, double),
         *ql_a = TILE_PTR(gh, gh->off_ql, double),
         *qq_a = TILE_PTR(gh, gh->off_qq, double);
  double *emass = TILE_PTR(ch, ch->off_elem_mass, double),
         *eivol = TILE_PTR(ch, ch->off_elem_init_vol, double);
  double *vnew = TILE_PTR(gh, gh->off_new_volume, double);
  int *sx = TILE_PTR(ch, ch->off_sym_x, int),
      *sy = TILE_PTR(ch, ch->off_sym_y, int),
      *sz = TILE_PTR(ch, ch->off_sym_z, int);
  int *en = TILE_PTR(ch, ch->off_elem_node, int);
  double lm = ch->monoq_limiter_mult, ms = ch->monoq_max_slope,
         qlc = ch->qlc_monoq, qqc = ch->qqc_monoq, ptiny = 1.e-36;
  double *n_dvxi[6] = {0, 0, 0, 0, 0, 0}, *n_dveta[6] = {0, 0, 0, 0, 0, 0},
         *n_dvzeta[6] = {0, 0, 0, 0, 0, 0};
  for (int fn = 0; fn < nfn; fn++) {
    int d = nbr_dir[fn];
    if (nbr_gh[fn]) {
      n_dvxi[d] = TILE_PTR(nbr_gh[fn], nbr_gh[fn]->off_delv_xi, double);
      n_dveta[d] = TILE_PTR(nbr_gh[fn], nbr_gh[fn]->off_delv_eta, double);
      n_dvzeta[d] = TILE_PTR(nbr_gh[fn], nbr_gh[fn]->off_delv_zeta, double);
    }
  }
  for (int i = 0; i < ne; i++) {
    int ez = i / (T * T), ey = (i / T) % T, ex = i % T;
    double phixi, phieta, phizeta, delvm, delvp;
    double norm_xi = 1.0 / (dvxi[i] + ptiny);
    if (ex > 0)
      delvm = dvxi[eidx(T, ex - 1, ey, ez)];
    else if (n_dvxi[0])
      delvm = n_dvxi[0][eidx(T, T - 1, ey, ez)];
    else if (sx[en[i * 8 + 0]])
      delvm = dvxi[i];
    else
      delvm = 0;
    if (ex < T - 1)
      delvp = dvxi[eidx(T, ex + 1, ey, ez)];
    else if (n_dvxi[1])
      delvp = n_dvxi[1][eidx(T, 0, ey, ez)];
    else if (sx[en[i * 8 + 1]])
      delvp = dvxi[i];
    else
      delvp = 0;
    delvm *= norm_xi;
    delvp *= norm_xi;
    phixi = .5 * (delvm + delvp);
    delvm *= lm;
    delvp *= lm;
    if (delvm < phixi)
      phixi = delvm;
    if (delvp < phixi)
      phixi = delvp;
    if (phixi < 0)
      phixi = 0;
    if (phixi > ms)
      phixi = ms;
    double norm_eta = 1.0 / (dveta[i] + ptiny);
    if (ey > 0)
      delvm = dveta[eidx(T, ex, ey - 1, ez)];
    else if (n_dveta[2])
      delvm = n_dveta[2][eidx(T, ex, T - 1, ez)];
    else if (sy[en[i * 8 + 0]])
      delvm = dveta[i];
    else
      delvm = 0;
    if (ey < T - 1)
      delvp = dveta[eidx(T, ex, ey + 1, ez)];
    else if (n_dveta[3])
      delvp = n_dveta[3][eidx(T, ex, 0, ez)];
    else if (sy[en[i * 8 + 2]])
      delvp = dveta[i];
    else
      delvp = 0;
    delvm *= norm_eta;
    delvp *= norm_eta;
    phieta = .5 * (delvm + delvp);
    delvm *= lm;
    delvp *= lm;
    if (delvm < phieta)
      phieta = delvm;
    if (delvp < phieta)
      phieta = delvp;
    if (phieta < 0)
      phieta = 0;
    if (phieta > ms)
      phieta = ms;
    double norm_zeta = 1.0 / (dvzeta[i] + ptiny);
    if (ez > 0)
      delvm = dvzeta[eidx(T, ex, ey, ez - 1)];
    else if (n_dvzeta[4])
      delvm = n_dvzeta[4][eidx(T, ex, ey, T - 1)];
    else if (sz[en[i * 8 + 0]])
      delvm = dvzeta[i];
    else
      delvm = 0;
    if (ez < T - 1)
      delvp = dvzeta[eidx(T, ex, ey, ez + 1)];
    else if (n_dvzeta[5])
      delvp = n_dvzeta[5][eidx(T, ex, ey, 0)];
    else if (sz[en[i * 8 + 4]])
      delvp = dvzeta[i];
    else
      delvp = 0;
    delvm *= norm_zeta;
    delvp *= norm_zeta;
    phizeta = .5 * (delvm + delvp);
    delvm *= lm;
    delvp *= lm;
    if (delvm < phizeta)
      phizeta = delvm;
    if (delvp < phizeta)
      phizeta = delvp;
    if (phizeta < 0)
      phizeta = 0;
    if (phizeta > ms)
      phizeta = ms;
    if (vdov_a[i] > 0) {
      ql_a[i] = 0;
      qq_a[i] = 0;
    } else {
      double dvxxi = dvxi[i] * dxxi[i], dvxeta = dveta[i] * dxeta[i],
             dvxzeta = dvzeta[i] * dxzeta[i];
      if (dvxxi > 0)
        dvxxi = 0;
      if (dvxeta > 0)
        dvxeta = 0;
      if (dvxzeta > 0)
        dvxzeta = 0;
      double rho = emass[i] / (eivol[i] * vnew[i]);
      ql_a[i] = -qlc * rho *
                (dvxxi * (1 - phixi) + dvxeta * (1 - phieta) +
                 dvxzeta * (1 - phizeta));
      qq_a[i] = qqc * rho *
                (dvxxi * dvxxi * (1 - phixi * phixi) +
                 dvxeta * dvxeta * (1 - phieta * phieta) +
                 dvxzeta * dvxzeta * (1 - phizeta * phizeta));
    }
  }
}

static void eval_eos_tile(const_db_header_t *ch, elem_state_db_header_t *esh,
                          grad_db_header_t *gh) {
  int ne = ch->num_elems;
  double *en_a = TILE_PTR(esh, esh->off_energy, double),
         *pr_a = TILE_PTR(esh, esh->off_pressure, double);
  double *vi_a = TILE_PTR(esh, esh->off_viscosity, double),
         *ss_a = TILE_PTR(esh, esh->off_sound_speed, double);
  double *vol_a = TILE_PTR(esh, esh->off_volume, double);
  double *ql_a = TILE_PTR(gh, gh->off_ql, double),
         *qq_a = TILE_PTR(gh, gh->off_qq, double);
  double *vnew = TILE_PTR(gh, gh->off_new_volume, double);
  double ecut = ch->e_cut, pcut = ch->p_cut, qcut = ch->q_cut, emin = ch->emin,
         pmin = ch->pmin;
  double rho0 = ch->refdens, eosvmax = ch->eosvmax, eosvmin = ch->eosvmin;
  double c1s = 2.0 / 3.0, sixth = 1.0 / 6.0;
  for (int i = 0; i < ne; i++) {
    double vnewc = vnew[i], delvc = vnewc - vol_a[i];
    double e_old = en_a[i], p_old = pr_a[i], q_old = vi_a[i], ql_i = ql_a[i],
           qq_i = qq_a[i];
    double comp = 1.0 / vnewc - 1.0, vchalf = vnewc - delvc * 0.5,
           compHS = 1.0 / vchalf - 1.0;
    if (eosvmin != 0 && vnewc <= eosvmin)
      compHS = comp;
    if (eosvmax != 0 && vnewc >= eosvmax) {
      p_old = 0;
      comp = 0;
      compHS = 0;
    }
    double bvc = c1s * (comp + 1.0), pbvc = c1s, bvcHS = c1s * (compHS + 1.0);
    double e_new = e_old - 0.5 * delvc * (p_old + q_old);
    if (e_new < emin)
      e_new = emin;
    double pHS = bvcHS * e_new;
    if (fabs(pHS) < pcut)
      pHS = 0;
    if (vnewc >= eosvmax)
      pHS = 0;
    if (pHS < pmin)
      pHS = pmin;
    double q_new, vhalf = 1.0 / (1.0 + compHS);
    if (delvc > 0)
      q_new = 0;
    else {
      double ssc = (pbvc * e_new + vhalf * vhalf * bvcHS * pHS) / rho0;
      if (ssc <= .1111111e-36)
        ssc = .3333333e-18;
      else
        ssc = sqrt(ssc);
      q_new = ssc * ql_i + qq_i;
    }
    e_new += 0.5 * delvc * (3.0 * (p_old + q_old) - 4.0 * (pHS + q_new));
    if (fabs(e_new) < ecut)
      e_new = 0;
    if (e_new < emin)
      e_new = emin;
    double p_new = bvc * e_new;
    if (fabs(p_new) < pcut)
      p_new = 0;
    if (vnewc >= eosvmax)
      p_new = 0;
    if (p_new < pmin)
      p_new = pmin;
    double q_tilde;
    if (delvc > 0)
      q_tilde = 0;
    else {
      double ssc = (pbvc * e_new + vnewc * vnewc * bvc * p_new) / rho0;
      if (ssc <= .1111111e-36)
        ssc = .3333333e-18;
      else
        ssc = sqrt(ssc);
      q_tilde = ssc * ql_i + qq_i;
    }
    e_new -= (7.0 * (p_old + q_old) - 8.0 * (pHS + q_new) + (p_new + q_tilde)) *
             delvc * sixth;
    if (fabs(e_new) < ecut)
      e_new = 0;
    if (e_new < emin)
      e_new = emin;
    p_new = bvc * e_new;
    if (fabs(p_new) < pcut)
      p_new = 0;
    if (vnewc >= eosvmax)
      p_new = 0;
    if (p_new < pmin)
      p_new = pmin;
    if (delvc <= 0) {
      double ssc = (pbvc * e_new + vnewc * vnewc * bvc * p_new) / rho0;
      if (ssc <= .1111111e-36)
        ssc = .3333333e-18;
      else
        ssc = sqrt(ssc);
      q_new = ssc * ql_i + qq_i;
      if (fabs(q_new) < qcut)
        q_new = 0;
    } else
      q_new = 0;
    en_a[i] = e_new;
    pr_a[i] = p_new;
    vi_a[i] = q_new;
    double sst = (pbvc * e_new + vnewc * vnewc * bvc * p_new) / rho0;
    if (sst <= .1111111e-36)
      sst = .3333333e-18;
    else
      sst = sqrt(sst);
    ss_a[i] = sst;
  }
}

/* ========================================================================= */
/*  Topology                                                                 */
/* ========================================================================= */

static void build_topo(int N, int T, int nn) {
  int tpd = N / T, nt = tpd * tpd * tpd;
  g_init.num_tiles = nt;
  g_init.tiles_per_dim = tpd;
  g_init.tile_elems = T;
  g_init.global_edge_elems = N;
  for (int tid = 0; tid < nt; tid++) {
    int tz = tid / (tpd * tpd), ty = (tid / tpd) % tpd, tx = tid % tpd;
    g_init.tile_owner[tid] = (unsigned)(tid % nn);
    int nf = 0;
    if (tx > 0) {
      g_init.face_nbrs[tid][nf] = (tz * tpd + ty) * tpd + (tx - 1);
      g_init.face_nbr_dir[tid][nf] = 0;
      nf++;
    }
    if (tx < tpd - 1) {
      g_init.face_nbrs[tid][nf] = (tz * tpd + ty) * tpd + (tx + 1);
      g_init.face_nbr_dir[tid][nf] = 1;
      nf++;
    }
    if (ty > 0) {
      g_init.face_nbrs[tid][nf] = (tz * tpd + (ty - 1)) * tpd + tx;
      g_init.face_nbr_dir[tid][nf] = 2;
      nf++;
    }
    if (ty < tpd - 1) {
      g_init.face_nbrs[tid][nf] = (tz * tpd + (ty + 1)) * tpd + tx;
      g_init.face_nbr_dir[tid][nf] = 3;
      nf++;
    }
    if (tz > 0) {
      g_init.face_nbrs[tid][nf] = ((tz - 1) * tpd + ty) * tpd + tx;
      g_init.face_nbr_dir[tid][nf] = 4;
      nf++;
    }
    if (tz < tpd - 1) {
      g_init.face_nbrs[tid][nf] = ((tz + 1) * tpd + ty) * tpd + tx;
      g_init.face_nbr_dir[tid][nf] = 5;
      nf++;
    }
    g_init.num_face_nbrs[tid] = nf;
    for (int i = nf; i < 6; i++) {
      g_init.face_nbrs[tid][i] = -1;
      g_init.face_nbr_dir[tid][i] = -1;
    }
    int na = 0;
    for (int dz = -1; dz <= 1; dz++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
          if (!dx && !dy && !dz)
            continue;
          int nx_ = tx + dx, ny_ = ty + dy, nz_ = tz + dz;
          if (nx_ >= 0 && nx_ < tpd && ny_ >= 0 && ny_ < tpd && nz_ >= 0 &&
              nz_ < tpd)
            g_init.all_nbrs[tid][na++] = (nz_ * tpd + ny_) * tpd + nx_;
        }
    g_init.num_all_nbrs[tid] = na;
  }
}

/* ========================================================================= */
/*  EDT declarations                                                         */
/* ========================================================================= */

void init_tile_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]);
void post_init_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]);
void forces_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]);
void reduce_kin_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]);
void elem_props_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]);
void visc_eos_time_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]);
void reduce_dt_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]);
void next_iter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]);
static void launch_iteration(int iter, double dt, double elapsed);

/* ========================================================================= */
/*  init_tile_edt                                                            */
/*  paramv: [tid, N, T, tpd, max_iter,                                      */
/*           const_guid, posvel_guid, estate_guid, grad_guid,                */
/*           force_guid, dt_result_guid, init_done_guid]                     */
/*  depv: none (depc=0)                                                      */
/* ========================================================================= */

void init_tile_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int tid = (int)paramv[0], N = (int)paramv[1], T = (int)paramv[2],
      tpd = (int)paramv[3], max_iter = (int)paramv[4];
  arts_guid_t cg = (arts_guid_t)paramv[5];
  arts_guid_t pg = (arts_guid_t)paramv[6];
  arts_guid_t es_guid = (arts_guid_t)paramv[7];
  arts_guid_t gg = (arts_guid_t)paramv[8];
  arts_guid_t fg = (arts_guid_t)paramv[9];
  arts_guid_t dt_guid = (arts_guid_t)paramv[10];
  arts_guid_t init_done = (arts_guid_t)paramv[11];

  /* Create 6 DBs with pre-reserved GUIDs on this node */
  size_t csz = compute_const_db_size(T), pvsz = compute_pos_vel_db_size(T);
  size_t essz = compute_elem_state_db_size(T), gsz = compute_grad_db_size(T),
         fsz = compute_force_db_size(T);

  const_db_header_t *ch =
      arts_db_create_with_guid(cg, (uint64_t)csz, ARTS_DB_DEFAULT, NULL, NULL);
  memset(ch, 0, csz);
  pos_vel_db_header_t *pvh =
      arts_db_create_with_guid(pg, (uint64_t)pvsz, ARTS_DB_DEFAULT, NULL, NULL);
  memset(pvh, 0, pvsz);
  elem_state_db_header_t *esh = arts_db_create_with_guid(
      es_guid, (uint64_t)essz, ARTS_DB_DEFAULT, NULL, NULL);
  memset(esh, 0, essz);
  grad_db_header_t *gh =
      arts_db_create_with_guid(gg, (uint64_t)gsz, ARTS_DB_DEFAULT, NULL, NULL);
  memset(gh, 0, gsz);
  force_db_header_t *fh =
      arts_db_create_with_guid(fg, (uint64_t)fsz, ARTS_DB_DEFAULT, NULL, NULL);
  memset(fh, 0, fsz);
  dt_result_db_t *dr = arts_db_create_with_guid(
      dt_guid, (uint64_t)sizeof(dt_result_db_t), ARTS_DB_DEFAULT, NULL, NULL);
  dr->dt_courant = 1.e20;
  dr->dt_hydro = 1.e20;

  set_const_offsets(ch, T);
  set_pos_vel_offsets(pvh, T);
  set_elem_state_offsets(esh, T);
  set_grad_offsets(gh, T);
  set_force_offsets(fh, T);
  int tz = tid / (tpd * tpd), ty = (tid / tpd) % tpd, tx = tid % tpd,
      ne = T * T * T, nn = (T + 1) * (T + 1) * (T + 1);
  ch->tile_id = tid;
  ch->tile_ix = tx;
  ch->tile_iy = ty;
  ch->tile_iz = tz;
  ch->tile_elems = T;
  ch->num_elems = ne;
  ch->num_nodes = nn;
  ch->global_edge_elems = N;
  ch->tiles_per_dim = tpd;
  ch->max_iterations = max_iter;
  ch->num_face_nbrs = g_init.num_face_nbrs[tid];
  ch->num_all_nbrs = g_init.num_all_nbrs[tid];
  for (int i = 0; i < 6; i++)
    ch->face_nbr_dir[i] = g_init.face_nbr_dir[tid][i];
  ch->hgcoef = 3.0;
  ch->ss4o3 = 4.0 / 3.0;
  ch->qstop = 1.e12;
  ch->monoq_max_slope = 1.0;
  ch->monoq_limiter_mult = 2.0;
  ch->qlc_monoq = 0.5;
  ch->qqc_monoq = 2.0 / 3.0;
  ch->qqc = 2.0;
  ch->eosvmax = 1.e9;
  ch->eosvmin = 1.e-9;
  ch->pmin = 0;
  ch->emin = -1.e15;
  ch->dvovmax = 0.1;
  ch->refdens = 1.0;
  ch->e_cut = 1.e-7;
  ch->p_cut = 1.e-7;
  ch->q_cut = 1.e-7;
  ch->v_cut = 1.e-10;
  ch->u_cut = 1.e-7;
  ch->stop_time = 1.e-2;
  ch->max_delta_time = 1.e-2;
  ch->deltatimemultlb = 1.1;
  ch->deltatimemultub = 1.2;
  double delta = 1.125 / (double)N;
  double *px = TILE_PTR(pvh, pvh->off_pos_x, double),
         *py = TILE_PTR(pvh, pvh->off_pos_y, double),
         *pz = TILE_PTR(pvh, pvh->off_pos_z, double);
  double *vxp = TILE_PTR(pvh, pvh->off_vel_x, double),
         *vyp = TILE_PTR(pvh, pvh->off_vel_y, double),
         *vzp = TILE_PTR(pvh, pvh->off_vel_z, double);
  double *nm = TILE_PTR(ch, ch->off_node_mass, double);
  for (int lz = 0; lz <= T; lz++)
    for (int ly = 0; ly <= T; ly++)
      for (int lx = 0; lx <= T; lx++) {
        int ni = nidx(T, lx, ly, lz);
        px[ni] = delta * (tx * T + lx);
        py[ni] = delta * (ty * T + ly);
        pz[ni] = delta * (tz * T + lz);
        vxp[ni] = vyp[ni] = vzp[ni] = nm[ni] = 0;
      }
  int *en = TILE_PTR(ch, ch->off_elem_node, int),
      *ee = TILE_PTR(ch, ch->off_elem_elem, int);
  double *emass = TILE_PTR(ch, ch->off_elem_mass, double),
         *eivol = TILE_PTR(ch, ch->off_elem_init_vol, double);
  double *pr = TILE_PTR(esh, esh->off_pressure, double),
         *eg = TILE_PTR(esh, esh->off_energy, double);
  double *vi = TILE_PTR(esh, esh->off_viscosity, double),
         *ss = TILE_PTR(esh, esh->off_sound_speed, double);
  double *vol = TILE_PTR(esh, esh->off_volume, double),
         *pvol = TILE_PTR(esh, esh->off_prev_volume, double);
  for (int lz = 0; lz < T; lz++)
    for (int ly = 0; ly < T; ly++)
      for (int lx = 0; lx < T; lx++) {
        int ei = eidx(T, lx, ly, lz);
        en[ei * 8 + 0] = nidx(T, lx, ly, lz);
        en[ei * 8 + 1] = nidx(T, lx + 1, ly, lz);
        en[ei * 8 + 2] = nidx(T, lx + 1, ly + 1, lz);
        en[ei * 8 + 3] = nidx(T, lx, ly + 1, lz);
        en[ei * 8 + 4] = nidx(T, lx, ly, lz + 1);
        en[ei * 8 + 5] = nidx(T, lx + 1, ly, lz + 1);
        en[ei * 8 + 6] = nidx(T, lx + 1, ly + 1, lz + 1);
        en[ei * 8 + 7] = nidx(T, lx, ly + 1, lz + 1);
        ee[ei * 6 + 0] = (lx > 0) ? eidx(T, lx - 1, ly, lz) : -1;
        ee[ei * 6 + 1] = (lx < T - 1) ? eidx(T, lx + 1, ly, lz) : -1;
        ee[ei * 6 + 2] = (ly > 0) ? eidx(T, lx, ly - 1, lz) : -1;
        ee[ei * 6 + 3] = (ly < T - 1) ? eidx(T, lx, ly + 1, lz) : -1;
        ee[ei * 6 + 4] = (lz > 0) ? eidx(T, lx, ly, lz - 1) : -1;
        ee[ei * 6 + 5] = (lz < T - 1) ? eidx(T, lx, ly, lz + 1) : -1;
        double xc[8], yc[8], zc[8];
        for (int c = 0; c < 8; c++) {
          xc[c] = px[en[ei * 8 + c]];
          yc[c] = py[en[ei * 8 + c]];
          zc[c] = pz[en[ei * 8 + c]];
        }
        double v = calc_elem_volume(xc, yc, zc);
        vol[ei] = 1.0;
        pvol[ei] = 1.0;
        eivol[ei] = v;
        emass[ei] = v;
        pr[ei] = eg[ei] = vi[ei] = ss[ei] = 0;
        for (int c = 0; c < 8; c++)
          nm[en[ei * 8 + c]] += v / 8.0;
      }
  if (tx == 0 && ty == 0 && tz == 0) {
    double einit = 3.948746e+7 * pow((double)N / 45.0, 3.0);
    eg[0] = einit;
  }
  /* Initialize force_db arrays and cache fields to zero */
  memset(TILE_PTR(fh, fh->off_elem_fx, double), 0,
         (size_t)ne * 8 * sizeof(double));
  memset(TILE_PTR(fh, fh->off_elem_fy, double), 0,
         (size_t)ne * 8 * sizeof(double));
  memset(TILE_PTR(fh, fh->off_elem_fz, double), 0,
         (size_t)ne * 8 * sizeof(double));
  /* Cache fields: pq=0 (pressure+viscosity), ss=0, vol=1.0 at init */
  double *pq_cache = TILE_PTR(fh, fh->off_pq_cache, double);
  double *ss_cache = TILE_PTR(fh, fh->off_ss_cache, double);
  double *vol_cache = TILE_PTR(fh, fh->off_vol_cache, double);
  for (int k = 0; k < ne; k++) {
    pq_cache[k] = 0;
    ss_cache[k] = 0;
    vol_cache[k] = 1.0;
  }
  /* Initialize grad_db arrays to zero */
  memset(TILE_PTR(gh, gh->off_vdov, double), 0, (size_t)ne * sizeof(double));
  memset(TILE_PTR(gh, gh->off_arealg, double), 0, (size_t)ne * sizeof(double));
  memset(TILE_PTR(gh, gh->off_delv_xi, double), 0, (size_t)ne * sizeof(double));
  memset(TILE_PTR(gh, gh->off_delv_eta, double), 0,
         (size_t)ne * sizeof(double));
  memset(TILE_PTR(gh, gh->off_delv_zeta, double), 0,
         (size_t)ne * sizeof(double));
  memset(TILE_PTR(gh, gh->off_delx_xi, double), 0, (size_t)ne * sizeof(double));
  memset(TILE_PTR(gh, gh->off_delx_eta, double), 0,
         (size_t)ne * sizeof(double));
  memset(TILE_PTR(gh, gh->off_delx_zeta, double), 0,
         (size_t)ne * sizeof(double));
  memset(TILE_PTR(gh, gh->off_ql, double), 0, (size_t)ne * sizeof(double));
  memset(TILE_PTR(gh, gh->off_qq, double), 0, (size_t)ne * sizeof(double));
  memset(TILE_PTR(gh, gh->off_new_volume, double), 0,
         (size_t)ne * sizeof(double));
  int *nem = TILE_PTR(ch, ch->off_node_elem, int);
  memset(nem, -1, (size_t)nn * 8 * sizeof(int));
  int *cnt = (int *)calloc((size_t)nn, sizeof(int));
  for (int ei = 0; ei < ne; ei++)
    for (int c = 0; c < 8; c++) {
      int nd = en[ei * 8 + c];
      if (cnt[nd] < 8)
        nem[nd * 8 + cnt[nd]++] = ei;
    }
  free(cnt);
  int *sxf = TILE_PTR(ch, ch->off_sym_x, int),
      *syf = TILE_PTR(ch, ch->off_sym_y, int),
      *szf = TILE_PTR(ch, ch->off_sym_z, int);
  memset(sxf, 0, (size_t)nn * sizeof(int));
  memset(syf, 0, (size_t)nn * sizeof(int));
  memset(szf, 0, (size_t)nn * sizeof(int));
  for (int lz = 0; lz <= T; lz++)
    for (int ly = 0; ly <= T; ly++)
      for (int lx = 0; lx <= T; lx++) {
        int ni = nidx(T, lx, ly, lz);
        if (tx * T + lx == 0)
          sxf[ni] = 1;
        if (ty * T + ly == 0)
          syf[ni] = 1;
        if (tz * T + lz == 0)
          szf[ni] = 1;
      }
  /* Signal init_done latch */
  arts_event_satisfy_slot(init_done, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* ========================================================================= */
/*  post_init_edt                                                            */
/*  depv: [0]=init_done(NULL), [1]=const[0](RO), [2]=elem_state[0](RO)      */
/*  Computes initial delta_time from tile 0 and launches first iteration.    */
/* ========================================================================= */

void post_init_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const_db_header_t *ch = depv[1].ptr;
  elem_state_db_header_t *esh = depv[2].ptr;
  double *eivol = TILE_PTR(ch, ch->off_elem_init_vol, double);
  double *eg = TILE_PTR(esh, esh->off_energy, double);
  double dt = 0.5 * cbrt(eivol[0]) / sqrt(2.0 * eg[0]);
  fprintf(stderr, "  Initial delta_time: %12.6e\n", dt);
  /* Spawn next_iter_edt trampoline instead of calling launch_iteration
   * directly. This ensures post_init_edt's DB deps are released before
   * launch_iteration creates new EDTs that need the same DBs (avoiding EW
   * ordering deadlock). */
  uint64_t p[3] = {0, 0, 0};
  memcpy(&p[1], &dt, sizeof(double));
  double zero = 0.0;
  memcpy(&p[2], &zero, sizeof(double));
  arts_edt_create(next_iter_edt, 3, p, 0, &(arts_hint_t){.route = 0});
}

/* ========================================================================= */
/*  launch_iteration                                                         */
/*                                                                           */
/*  Called on rank 0 only. Uses g_init for topology + DB GUIDs.              */
/*                                                                           */
/*  Event structure (7 per tile per iteration):                              */
/*    fd[t] (count=1):    forces done                                        */
/*    rd[t] (count=1):    reduce done                                        */
/*    ed[t] (count=1):    elem_props done                                    */
/*    vd[t] (count=1):    visc_eos done                                      */
/*    rr[t] (count=1+na): ready for reduce (own fd + all nbr fds)           */
/*    er[t] (count=1+nf): ready for elem_props (own rd + face nbr rds)      */
/*    vr[t] (count=1+nf): ready for visc_eos (own ed + face nbr eds)        */
/*                                                                           */
/*  Plus: setup_done (count=1) gates forces_edts                             */
/*        dr (count=nt) aggregates all vd                                    */
/*                                                                           */
/*  Phase 1 forces_edt(t):                                                   */
/*    depv: [0]=setup_done(NULL), [1]=const(RO), [2]=posvel(RO),             */
/*          [3]=force(EW)                                                    */
/*                                                                           */
/*  Phase 2 reduce_kin_edt(t):                                               */
/*    depv: [0]=rr(NULL), [1]=force[t](RO), [2..2+na-1]=force[nbrs](RO),    */
/*          [2+na]=const(RO), [3+na]=posvel(EW)                              */
/*                                                                           */
/*  Phase 3 elem_props_edt(t):                                               */
/*    depv: [0]=er(NULL), [1]=const(RO), [2]=posvel(RO), [3]=grad(EW)       */
/*                                                                           */
/*  Phase 4 visc_eos_time_edt(t):                                            */
/*    depv: [0]=vr(NULL), [1]=grad[t](RO), [2..2+nf-1]=grad[face_nbrs](RO), */
/*          [2+nf]=const(RO), [3+nf]=estate(EW), [4+nf]=dt_result(EW),      */
/*          [5+nf]=force(EW)                                                 */
/*                                                                           */
/*  Phase 5 reduce_dt_edt:                                                   */
/*    depv: [0]=dr(NULL), [1..nt]=dt_result[0..nt-1](RO),                    */
/*          [nt+1]=const[0](RO)                                              */
/* ========================================================================= */

static void launch_iteration(int iter, double dt, double elapsed) {
#ifdef LULESH_PROFILE
  uint64_t _prof_t0 = prof_now_ns();
  if (g_prof_first_ns == 0)
    g_prof_first_ns = _prof_t0;
#endif
  int nt = g_init.num_tiles;
  uint64_t dtb, elb;
  memcpy(&dtb, &dt, sizeof(double));
  memcpy(&elb, &elapsed, sizeof(double));

  /* Step 1: Create 7*nt events on rank 0.
   * ARTS events are location-independent. */
  arts_guid_t fd_guids[MAX_TILES], rd_guids[MAX_TILES], ed_guids[MAX_TILES],
      vd_guids[MAX_TILES];
  arts_guid_t rr_guids[MAX_TILES], er_guids[MAX_TILES], vr_guids[MAX_TILES];
  for (int t = 0; t < nt; t++) {
    int na = g_init.num_all_nbrs[t], nf = g_init.num_face_nbrs[t];
    fd_guids[t] = arts_event_create(0, ARTS_EVENT_LATCH, 1, NULL_GUID);
    rd_guids[t] = arts_event_create(0, ARTS_EVENT_LATCH, 1, NULL_GUID);
    ed_guids[t] = arts_event_create(0, ARTS_EVENT_LATCH, 1, NULL_GUID);
    vd_guids[t] = arts_event_create(0, ARTS_EVENT_LATCH, 1, NULL_GUID);
    rr_guids[t] =
        arts_event_create(0, ARTS_EVENT_LATCH, (unsigned)(1 + na), NULL_GUID);
    er_guids[t] =
        arts_event_create(0, ARTS_EVENT_LATCH, (unsigned)(1 + nf), NULL_GUID);
    vr_guids[t] =
        arts_event_create(0, ARTS_EVENT_LATCH, (unsigned)(1 + nf), NULL_GUID);
  }

  /* Step 2: Wire event-to-event deps (fd->rr, rd->er, ed->vr) */
  for (int t = 0; t < nt; t++) {
    int na = g_init.num_all_nbrs[t], nf = g_init.num_face_nbrs[t];
    /* fd[t] -> rr[t] (own forces done) */
    arts_add_dependence(fd_guids[t], rr_guids[t], 0, DB_MODE_NULL);
    /* fd[nbr] -> rr[t] (neighbor forces done) */
    for (int n = 0; n < na; n++)
      arts_add_dependence(fd_guids[g_init.all_nbrs[t][n]], rr_guids[t], 0,
                          DB_MODE_NULL);
    /* rd[t] -> er[t] (own reduce done) */
    arts_add_dependence(rd_guids[t], er_guids[t], 0, DB_MODE_NULL);
    /* rd[face_nbr] -> er[t] */
    for (int n = 0; n < nf; n++)
      arts_add_dependence(rd_guids[g_init.face_nbrs[t][n]], er_guids[t], 0,
                          DB_MODE_NULL);
    /* ed[t] -> vr[t] (own elem done) */
    arts_add_dependence(ed_guids[t], vr_guids[t], 0, DB_MODE_NULL);
    /* ed[face_nbr] -> vr[t] */
    for (int n = 0; n < nf; n++)
      arts_add_dependence(ed_guids[g_init.face_nbrs[t][n]], vr_guids[t], 0,
                          DB_MODE_NULL);
  }

  /* Step 3: Create setup_done + dr latch */
  arts_guid_t setup_done = arts_event_create(0, ARTS_EVENT_LATCH, 1, NULL_GUID);
  arts_guid_t dr =
      arts_event_create(0, ARTS_EVENT_LATCH, (unsigned)nt, NULL_GUID);

  /* Step 4: Wire vd -> dr deps */
  for (int t = 0; t < nt; t++)
    arts_add_dependence(vd_guids[t], dr, 0, DB_MODE_NULL);

  /* Step 5: reduce_dt_edt will be created AFTER phase EDTs in Step 7.
   * This ensures phase EDTs register their DB EW entries in the frontier
   * before reduce_dt's EW entries, which preserves the EW→EW→EW chain
   * order and lets phase EDTs run before the iteration barrier. */

  /* Step 6: Create 4*nt phase EDTs with DB + event deps */
  /* Wire EDT DB deps in PHASE ORDER to ensure CDAG frontier registration
   * matches event execution order.  With tile-order wiring, a cross-tile
   * neighbor dep (reduce_kin_edt[A] RO on force_db[B]) could be registered
   * AFTER visc_eos_time_edt[B] EW on force_db[B], inverting the frontier:
   *   EW(forces) → RO(reduce_own) → EW(visc_eos) → RO(reduce_nbr)
   * Event ordering requires reduce_nbr BEFORE visc_eos → circular wait.
   * Phase-order ensures: EW(forces) → RO(reduce_all) → EW(visc_eos). */

  /* Phase EDTs use RO for reads and EW for writes. The runtime's RO→EW
   * frontier progress (ARTS_DB_DEFAULT frontier) guarantees ordering:
   * when the last reader releases and a writer is waiting in frontier->next,
   * the frontier is progressed so the writer can proceed. */

  /* Phase 1: forces_edt */
  for (int t = 0; t < nt; t++) {
    unsigned o = g_init.tile_owner[t];
    uint64_t p[2] = {(uint64_t)t, (uint64_t)fd_guids[t]};
    arts_guid_t e =
        arts_edt_create(forces_edt, 2, p, 4, &(arts_hint_t){.route = o});
    arts_add_dependence(setup_done, e, 0, DB_MODE_NULL);
    arts_add_dependence(g_init.const_guids[t], e, 1, DB_MODE_RO);
    arts_add_dependence(g_init.pos_vel_guids[t], e, 2, DB_MODE_RO);
    arts_add_dependence(g_init.force_guids[t], e, 3, DB_MODE_EW);
  }

  /* Phase 2: reduce_kin_edt */
  for (int t = 0; t < nt; t++) {
    unsigned o = g_init.tile_owner[t];
    int na = g_init.num_all_nbrs[t];
    uint64_t p[3] = {(uint64_t)t, dtb, (uint64_t)rd_guids[t]};
    uint32_t depc2 = (uint32_t)(1 + 1 + na + 1 + 1);
    arts_guid_t e = arts_edt_create(reduce_kin_edt, 3, p, depc2,
                                    &(arts_hint_t){.route = o});
    arts_add_dependence(rr_guids[t], e, 0, DB_MODE_NULL);
    arts_add_dependence(g_init.force_guids[t], e, 1, DB_MODE_RO);
    for (int n = 0; n < na; n++)
      arts_add_dependence(g_init.force_guids[g_init.all_nbrs[t][n]], e,
                          (uint32_t)(2 + n), DB_MODE_RO);
    arts_add_dependence(g_init.const_guids[t], e, (uint32_t)(2 + na),
                        DB_MODE_RO);
    arts_add_dependence(g_init.pos_vel_guids[t], e, (uint32_t)(3 + na),
                        DB_MODE_EW);
  }

  /* Phase 3: elem_props_edt */
  for (int t = 0; t < nt; t++) {
    unsigned o = g_init.tile_owner[t];
    uint64_t p[3] = {(uint64_t)t, dtb, (uint64_t)ed_guids[t]};
    arts_guid_t e =
        arts_edt_create(elem_props_edt, 3, p, 4, &(arts_hint_t){.route = o});
    arts_add_dependence(er_guids[t], e, 0, DB_MODE_NULL);
    arts_add_dependence(g_init.const_guids[t], e, 1, DB_MODE_RO);
    arts_add_dependence(g_init.pos_vel_guids[t], e, 2, DB_MODE_RO);
    arts_add_dependence(g_init.grad_guids[t], e, 3, DB_MODE_EW);
  }

  /* Phase 4: visc_eos_time_edt */
  for (int t = 0; t < nt; t++) {
    unsigned o = g_init.tile_owner[t];
    int nf = g_init.num_face_nbrs[t];
    uint64_t p[2] = {(uint64_t)t, (uint64_t)vd_guids[t]};
    uint32_t depc2 = (uint32_t)(1 + 1 + nf + 1 + 1 + 1 + 1);
    arts_guid_t e = arts_edt_create(visc_eos_time_edt, 2, p, depc2,
                                    &(arts_hint_t){.route = o});
    arts_add_dependence(vr_guids[t], e, 0, DB_MODE_NULL);
    arts_add_dependence(g_init.grad_guids[t], e, 1, DB_MODE_RO);
    for (int n = 0; n < nf; n++)
      arts_add_dependence(g_init.grad_guids[g_init.face_nbrs[t][n]], e,
                          (uint32_t)(2 + n), DB_MODE_RO);
    arts_add_dependence(g_init.const_guids[t], e, (uint32_t)(2 + nf),
                        DB_MODE_RO);
    arts_add_dependence(g_init.elem_state_guids[t], e, (uint32_t)(3 + nf),
                        DB_MODE_EW);
    arts_add_dependence(g_init.dt_result_guids[t], e, (uint32_t)(4 + nf),
                        DB_MODE_EW);
    arts_add_dependence(g_init.force_guids[t], e, (uint32_t)(5 + nf),
                        DB_MODE_EW);
  }

  /* Reduction: reduce_dt_edt — reads all dt_result DBs (RO) + const[0] (RO).
   * The dr latch ensures all visc_eos_time_edts have fired rd before this
   * EDT starts.  Additional per-tile DB deps are not needed because the
   * event chain (vd→dr) already guarantees ordering. */
  {
    uint64_t p[3] = {(uint64_t)iter, dtb, elb};
    uint32_t depc = (uint32_t)(1 + nt + 1);
    arts_guid_t e =
        arts_edt_create(reduce_dt_edt, 3, p, depc, &(arts_hint_t){.route = 0});
    arts_add_dependence(dr, e, 0, DB_MODE_NULL);
    for (int t = 0; t < nt; t++)
      arts_add_dependence(g_init.dt_result_guids[t], e, (uint32_t)(1 + t),
                          DB_MODE_RO);
    arts_add_dependence(g_init.const_guids[0], e, (uint32_t)(1 + nt),
                        DB_MODE_RO);
  }

  /* Step 7: Satisfy setup_done to release all forces_edts */
  arts_event_satisfy_slot(setup_done, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
#ifdef LULESH_PROFILE
  g_prof_launch_ns += prof_now_ns() - _prof_t0;
  g_prof_launch_count++;
#endif
}

/* ========================================================================= */
/*  forces_edt                                                               */
/*  paramv: [tid, done_guid]                                                 */
/*  depv: [0]=setup_done(NULL), [1]=const(RO), [2]=posvel(RO),              */
/*        [3]=force(EW)                                                      */
/* ========================================================================= */

void forces_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t done = (arts_guid_t)paramv[1];
  const_db_header_t *ch = depv[1].ptr;
  pos_vel_db_header_t *pvh = depv[2].ptr;
  force_db_header_t *fh = depv[3].ptr;
  int ne = ch->num_elems;
  /* Read p+q, ss, vol from force_db's cache fields (written by prev iter's
   * visc_eos_time) */
  double *pq_cache = TILE_PTR(fh, fh->off_pq_cache, double);
  double *ss_cache = TILE_PTR(fh, fh->off_ss_cache, double);
  double *vol_cache = TILE_PTR(fh, fh->off_vol_cache, double);
  double *efx = TILE_PTR(fh, fh->off_elem_fx, double),
         *efy = TILE_PTR(fh, fh->off_elem_fy, double),
         *efz = TILE_PTR(fh, fh->off_elem_fz, double);
  memset(efx, 0, (size_t)ne * 8 * sizeof(double));
  memset(efy, 0, (size_t)ne * 8 * sizeof(double));
  memset(efz, 0, (size_t)ne * 8 * sizeof(double));
  for (int k = 0; k < ne; k++) {
    double fx[8], fy[8], fz[8];
    calc_elem_forces_pq(ch, pvh, k, pq_cache[k], ss_cache[k], vol_cache[k], fx,
                        fy, fz);
    for (int c = 0; c < 8; c++) {
      efx[k * 8 + c] = fx[c];
      efy[k * 8 + c] = fy[c];
      efz[k * 8 + c] = fz[c];
    }
  }
  arts_event_satisfy_slot(done, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* ========================================================================= */
/*  reduce_kin_edt                                                           */
/*  paramv: [tid, dt_bits, done_guid]                                        */
/*  depv: [0]=rr(NULL), [1]=force[t](RO), [2..2+na-1]=force[nbrs](RO),     */
/*        [2+na]=const(RO), [3+na]=posvel(EW)                               */
/*  na = depc - 4 (inferred from depv layout)                                */
/* ========================================================================= */

void reduce_kin_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  double dt;
  memcpy(&dt, &paramv[1], sizeof(double));
  arts_guid_t done = (arts_guid_t)paramv[2];
  int na = (int)depc - 4;
  force_db_header_t *fh = depv[1].ptr;
  const_db_header_t *ch = depv[2 + na].ptr;
  pos_vel_db_header_t *pvh = depv[3 + na].ptr;
  int T = ch->tile_elems, nn = ch->num_nodes, tpd = ch->tiles_per_dim;
  int tx_t = ch->tile_ix, ty_t = ch->tile_iy, tz_t = ch->tile_iz;
  double *px = TILE_PTR(pvh, pvh->off_pos_x, double),
         *py = TILE_PTR(pvh, pvh->off_pos_y, double),
         *pz = TILE_PTR(pvh, pvh->off_pos_z, double);
  double *vxp = TILE_PTR(pvh, pvh->off_vel_x, double),
         *vyp = TILE_PTR(pvh, pvh->off_vel_y, double),
         *vzp = TILE_PTR(pvh, pvh->off_vel_z, double);
  double *nm = TILE_PTR(ch, ch->off_node_mass, double);
  int *en = TILE_PTR(ch, ch->off_elem_node, int),
      *nem = TILE_PTR(ch, ch->off_node_elem, int);
  int *sxf = TILE_PTR(ch, ch->off_sym_x, int),
      *syf = TILE_PTR(ch, ch->off_sym_y, int),
      *szf = TILE_PTR(ch, ch->off_sym_z, int);
  double *efx = TILE_PTR(fh, fh->off_elem_fx, double),
         *efy = TILE_PTR(fh, fh->off_elem_fy, double),
         *efz = TILE_PTR(fh, fh->off_elem_fz, double);
  static const int corner_lut[2][2][2] = {{{0, 4}, {3, 7}}, {{1, 5}, {2, 6}}};

  /* Build a mapping from neighbor tile_id to depv index.
   * We infer neighbor tile IDs from the const_db's topology fields. */
  for (int ni = 0; ni < nn; ni++) {
    double fx = 0, fy = 0, fz = 0;
    for (int lei = 0; lei < 8; lei++) {
      int ei = nem[ni * 8 + lei];
      if (ei < 0)
        continue;
      int corner = -1;
      for (int c = 0; c < 8; c++)
        if (en[ei * 8 + c] == ni) {
          corner = c;
          break;
        }
      if (corner >= 0) {
        fx += efx[ei * 8 + corner];
        fy += efy[ei * 8 + corner];
        fz += efz[ei * 8 + corner];
      }
    }
    int lz = ni / ((T + 1) * (T + 1)), ly = (ni / (T + 1)) % (T + 1),
        lx = ni % (T + 1);
    for (int dz = -1; dz <= 1; dz++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
          if (!dx && !dy && !dz)
            continue;
          if (dx == -1 && lx != 0)
            continue;
          if (dx == 1 && lx != T)
            continue;
          if (dy == -1 && ly != 0)
            continue;
          if (dy == 1 && ly != T)
            continue;
          if (dz == -1 && lz != 0)
            continue;
          if (dz == 1 && lz != T)
            continue;
          int nx_ = tx_t + dx, ny_ = ty_t + dy, nz_ = tz_t + dz;
          if (nx_ < 0 || nx_ >= tpd || ny_ < 0 || ny_ >= tpd || nz_ < 0 ||
              nz_ >= tpd)
            continue;
          int nbr_t = (nz_ * tpd + ny_) * tpd + nx_;
          /* Find this neighbor in depv[2..2+na-1] by comparing GUIDs.
           * The depv ordering matches g_init.all_nbrs[tid], but we don't
           * have g_init access. Instead, search depv by tile_id in force_db
           * headers -- but force_db has no tile_id. Instead we use a simpler
           * approach: iterate depv[2..2+na-1] and find the one whose
           * force_db pointer matches the neighbor tile. Since all neighbors
           * are ordered by g_init.all_nbrs which uses the same 3D iteration
           * order as here, we can compute the depv index directly. */
          int di = -1;
          /* Compute neighbor index in sorted all_nbrs order.
           * all_nbrs is built by iterating dz,dy,dx in [-1,1]^3 and
           * checking bounds. We need to find the rank of
           * (nz_-tz_t,ny_-ty_t,nx_-tx_t) among all valid neighbors. */
          int rank = 0;
          for (int dz2 = -1; dz2 <= 1; dz2++)
            for (int dy2 = -1; dy2 <= 1; dy2++)
              for (int dx2 = -1; dx2 <= 1; dx2++) {
                if (!dx2 && !dy2 && !dz2)
                  continue;
                int cx = tx_t + dx2, cy = ty_t + dy2, cz = tz_t + dz2;
                if (cx < 0 || cx >= tpd || cy < 0 || cy >= tpd || cz < 0 ||
                    cz >= tpd)
                  continue;
                if (dx2 == dx && dy2 == dy && dz2 == dz) {
                  di = 2 + rank;
                  break;
                }
                rank++;
              }
          if (di < 0 || di >= 2 + na)
            continue;
          force_db_header_t *nfh = depv[di].ptr;
          if (!nfh)
            continue;
          int nlx = (dx == -1)  ? T
                    : (dx == 1) ? 0
                                : lx,
              nly = (dy == -1)  ? T
                    : (dy == 1) ? 0
                                : ly,
              nlz = (dz == -1)  ? T
                    : (dz == 1) ? 0
                                : lz;
          double *nfx = TILE_PTR(nfh, nfh->off_elem_fx, double),
                 *nfy = TILE_PTR(nfh, nfh->off_elem_fy, double),
                 *nfz = TILE_PTR(nfh, nfh->off_elem_fz, double);
          for (int dz2 = -1; dz2 <= 0; dz2++)
            for (int dy2 = -1; dy2 <= 0; dy2++)
              for (int dx2 = -1; dx2 <= 0; dx2++) {
                int ex_ = nlx + dx2, ey_ = nly + dy2, ez_ = nlz + dz2;
                if (ex_ < 0 || ex_ >= T || ey_ < 0 || ey_ >= T || ez_ < 0 ||
                    ez_ >= T)
                  continue;
                int nei = eidx(T, ex_, ey_, ez_);
                int corner = corner_lut[nlx - ex_][nly - ey_][nlz - ez_];
                fx += nfx[nei * 8 + corner];
                fy += nfy[nei * 8 + corner];
                fz += nfz[nei * 8 + corner];
              }
        }
    if (sxf[ni])
      fx = 0;
    if (syf[ni])
      fy = 0;
    if (szf[ni])
      fz = 0;
    double mi = 1.0 / nm[ni];
    vxp[ni] += fx * mi * dt;
    vyp[ni] += fy * mi * dt;
    vzp[ni] += fz * mi * dt;
    if (fabs(vxp[ni]) < ch->u_cut)
      vxp[ni] = 0;
    if (fabs(vyp[ni]) < ch->u_cut)
      vyp[ni] = 0;
    if (fabs(vzp[ni]) < ch->u_cut)
      vzp[ni] = 0;
    px[ni] += vxp[ni] * dt;
    py[ni] += vyp[ni] * dt;
    pz[ni] += vzp[ni] * dt;
  }
  arts_event_satisfy_slot(done, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* ========================================================================= */
/*  elem_props_edt                                                           */
/*  paramv: [tid, dt_bits, done_guid]                                        */
/*  depv: [0]=er(NULL), [1]=const(RO), [2]=posvel(RO), [3]=grad(EW)         */
/* ========================================================================= */

void elem_props_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  double dt;
  memcpy(&dt, &paramv[1], sizeof(double));
  arts_guid_t done = (arts_guid_t)paramv[2];
  const_db_header_t *ch = depv[1].ptr;
  pos_vel_db_header_t *pvh = depv[2].ptr;
  grad_db_header_t *gh = depv[3].ptr;
  int ne = ch->num_elems;
  int *en = TILE_PTR(ch, ch->off_elem_node, int);
  double *eivol = TILE_PTR(ch, ch->off_elem_init_vol, double);
  double *px = TILE_PTR(pvh, pvh->off_pos_x, double),
         *py = TILE_PTR(pvh, pvh->off_pos_y, double),
         *pz = TILE_PTR(pvh, pvh->off_pos_z, double);
  double *vxp = TILE_PTR(pvh, pvh->off_vel_x, double),
         *vyp = TILE_PTR(pvh, pvh->off_vel_y, double),
         *vzp = TILE_PTR(pvh, pvh->off_vel_z, double);
  double *vdov = TILE_PTR(gh, gh->off_vdov, double),
         *arealg_a = TILE_PTR(gh, gh->off_arealg, double),
         *vnew = TILE_PTR(gh, gh->off_new_volume, double);
  for (int k = 0; k < ne; k++) {
    double xl[8], yl[8], zl[8], xd[8], yd[8], zd[8];
    for (int c = 0; c < 8; c++) {
      int nd = en[k * 8 + c];
      xl[c] = px[nd];
      yl[c] = py[nd];
      zl[c] = pz[nd];
      xd[c] = vxp[nd];
      yd[c] = vyp[nd];
      zd[c] = vzp[nd];
    }
    double v = calc_elem_volume(xl, yl, zl);
    vnew[k] = v / eivol[k];
    arealg_a[k] = calc_elem_char_len(xl, yl, zl, v);
    double dt2 = 0.5 * dt, xh[8], yh[8], zh[8];
    for (int c = 0; c < 8; c++) {
      xh[c] = xl[c] - dt2 * xd[c];
      yh[c] = yl[c] - dt2 * yd[c];
      zh[c] = zl[c] - dt2 * zd[c];
    }
    double B[3][8], detJ;
    calc_elem_sfd(xh, yh, zh, B, &detJ);
    double D[6];
    calc_elem_vel_grad(xd, yd, zd, B, detJ, D);
    vdov[k] = D[0] + D[1] + D[2];
    if (vnew[k] <= 0) {
      fprintf(stderr, "LULESH ERROR: negative volume elem %d tile %d\n", k,
              ch->tile_id);
      arts_abort(1);
    }
  }
  calc_monoq_gradients(ch, pvh, gh);
  arts_event_satisfy_slot(done, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* ========================================================================= */
/*  visc_eos_time_edt                                                        */
/*  paramv: [tid, done_guid]                                                 */
/*  depv: [0]=vr(NULL), [1]=grad[t](RO), [2..2+nf-1]=grad[face_nbrs](RO),  */
/*        [2+nf]=const(RO), [3+nf]=estate(EW), [4+nf]=dt_result(EW),        */
/*        [5+nf]=force(EW)                                                   */
/*  nf = depc - 6 (inferred)                                                 */
/* ========================================================================= */

void visc_eos_time_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  arts_guid_t done = (arts_guid_t)paramv[1];
  int nf = (int)depc - 6;
  grad_db_header_t *gh = depv[1].ptr;
  const_db_header_t *ch = depv[2 + nf].ptr;
  elem_state_db_header_t *esh = depv[3 + nf].ptr;
  dt_result_db_t *dr = depv[4 + nf].ptr;
  force_db_header_t *fh = depv[5 + nf].ptr;

  /* Build neighbor grad_db pointers and directions from const_db */
  grad_db_header_t *nbr_gh[6] = {0, 0, 0, 0, 0, 0};
  int nbr_dir[6];
  for (int n = 0; n < nf; n++) {
    nbr_gh[n] = depv[2 + n].ptr;
    nbr_dir[n] = ch->face_nbr_dir[n];
  }
  calc_monoq_region(ch, gh, nbr_gh, nbr_dir, nf);
  eval_eos_tile(ch, esh, gh);

  int ne = ch->num_elems;
  double *vnew = TILE_PTR(gh, gh->off_new_volume, double);
  double *vol = TILE_PTR(esh, esh->off_volume, double),
         *pvol = TILE_PTR(esh, esh->off_prev_volume, double);
  for (int k = 0; k < ne; k++) {
    pvol[k] = vol[k];
    vol[k] = vnew[k];
  }

  /* Compute per-tile dt constraints and store in dt_result_db */
  double *ss_a = TILE_PTR(esh, esh->off_sound_speed, double);
  double *al = TILE_PTR(gh, gh->off_arealg, double);
  double *vd = TILE_PTR(gh, gh->off_vdov, double);
  double qqc2 = 64.0 * ch->qqc * ch->qqc;
  double dtcourant = 1.e20, dthydro = 1.e20;
  for (int i = 0; i < ne; i++) {
    double dtf = ss_a[i] * ss_a[i];
    if (vd[i] < 0)
      dtf += qqc2 * al[i] * al[i] * vd[i] * vd[i];
    dtf = sqrt(dtf);
    dtf = al[i] / dtf;
    if (vd[i] != 0 && dtf < dtcourant)
      dtcourant = dtf;
    if (vd[i] != 0) {
      double dtdvov = ch->dvovmax / (fabs(vd[i]) + 1.e-20);
      if (dtdvov < dthydro)
        dthydro = dtdvov;
    }
  }
  dr->dt_courant = dtcourant;
  dr->dt_hydro = dthydro;
  /* Tile 0 stores origin energy for final output */
  if (ch->tile_id == 0)
    dr->origin_energy = TILE_PTR(esh, esh->off_energy, double)[0];

  /* Write p+q, ss, vol cache into force_db for next iteration's forces_edt */
  double *pr = TILE_PTR(esh, esh->off_pressure, double);
  double *vi = TILE_PTR(esh, esh->off_viscosity, double);
  double *pq_cache = TILE_PTR(fh, fh->off_pq_cache, double);
  double *ss_cache = TILE_PTR(fh, fh->off_ss_cache, double);
  double *vol_cache = TILE_PTR(fh, fh->off_vol_cache, double);
  for (int k = 0; k < ne; k++) {
    pq_cache[k] = pr[k] + vi[k];
    ss_cache[k] = ss_a[k];
    vol_cache[k] = vol[k];
  }

  arts_event_satisfy_slot(done, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* ========================================================================= */
/*  reduce_dt_edt                                                            */
/*  paramv: [iter, dt_bits, elapsed_bits]                                    */
/*  depv: [0]=dr(NULL), [1..nt]=dt_result[0..nt-1](RO),                     */
/*        [nt+1]=const[0](RO)                                               */
/*  nt = depc - 2                                                            */
/* ========================================================================= */

void reduce_dt_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  int iter = (int)paramv[0];
  double dt_old;
  memcpy(&dt_old, &paramv[1], sizeof(double));
  double elapsed;
  memcpy(&elapsed, &paramv[2], sizeof(double));
  int nt = (int)depc - 2;
  const_db_header_t *ch0 = depv[nt + 1].ptr;

  /* Print iteration info */
  /* Read origin energy from tile 0's dt_result */
  double origin_energy = ((dt_result_db_t *)depv[1].ptr)->origin_energy;
  fprintf(stderr, "iteration %d, delta time %f, energy %f\n", iter + 1, dt_old,
          origin_energy);

  double stop = ch0->stop_time;
  int max_iter = ch0->max_iterations;
  double dtcourant = 1.e20, dthydro = 1.e20;
  for (int t = 0; t < nt; t++) {
    dt_result_db_t *dr_t = depv[1 + t].ptr;
    if (dr_t->dt_courant < dtcourant)
      dtcourant = dr_t->dt_courant;
    if (dr_t->dt_hydro < dthydro)
      dthydro = dr_t->dt_hydro;
  }
  double dt_new = dt_old;
  {
    double gnewdt = 1.e20;
    if (dtcourant < gnewdt)
      gnewdt = dtcourant / 2.0;
    if (dthydro < gnewdt)
      gnewdt = dthydro * 2.0 / 3.0;
    double ratio = gnewdt / dt_old;
    if (ratio >= 1.0) {
      if (ratio < ch0->deltatimemultlb)
        gnewdt = dt_old;
      else if (ratio > ch0->deltatimemultub)
        gnewdt = dt_old * ch0->deltatimemultub;
    }
    if (gnewdt > ch0->max_delta_time)
      gnewdt = ch0->max_delta_time;
    dt_new = gnewdt;
  }
  double new_elapsed = elapsed + dt_old;
  double targetdt = stop - new_elapsed;
  if (targetdt > dt_new && targetdt < 4.0 * dt_new / 3.0)
    targetdt = 2.0 * dt_new / 3.0;
  if (targetdt < dt_new)
    dt_new = targetdt;
  int finished = (new_elapsed >= stop) || (iter + 1 >= max_iter);
  if (finished) {
    fprintf(stderr, "Run completed:\n");
    fprintf(stderr, "   Problem size        = %d\n", ch0->global_edge_elems);
    fprintf(stderr, "   Iteration count     = %d\n", iter + 1);
    fprintf(stderr, "   Final Origin Energy = %12.6e\n", origin_energy);
    fprintf(stderr, "   Elapsed time        = %12.6e\n", new_elapsed);
#ifdef LULESH_PROFILE
    uint64_t total_ns = prof_now_ns() - g_prof_first_ns;
    fprintf(stderr,
            "PROFILE: launch_iteration total=%.3f ms (%lu calls, avg=%.3f ms); "
            "wall since first launch=%.3f ms; serial_fraction=%.2f%%\n",
            g_prof_launch_ns / 1e6, g_prof_launch_count,
            (g_prof_launch_ns / 1e6) / (double)g_prof_launch_count,
            total_ns / 1e6, 100.0 * g_prof_launch_ns / (double)total_ns);
#endif
    fflush(stdout);
    arts_shutdown();
  } else {
    /* Create next_iter_edt(depc=0) as trampoline.
     * This ensures reduce_dt_edt's DB deps are fully released
     * before the next iteration starts. */
    uint64_t p[4] = {(uint64_t)(iter + 1), 0, 0, 0};
    memcpy(&p[1], &dt_new, sizeof(double));
    memcpy(&p[2], &new_elapsed, sizeof(double));
    arts_edt_create(next_iter_edt, 3, p, 0, &(arts_hint_t){.route = 0});
  }
}

/* ========================================================================= */
/*  next_iter_edt -- trampoline to launch next iteration                     */
/*  paramv: [iter, dt_bits, elapsed_bits]                                    */
/*  depv: none (depc=0)                                                      */
/* ========================================================================= */

void next_iter_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int iter = (int)paramv[0];
  double dt;
  memcpy(&dt, &paramv[1], sizeof(double));
  double elapsed;
  memcpy(&elapsed, &paramv[2], sizeof(double));
  launch_iteration(iter, dt, elapsed);
}

/* ========================================================================= */
/*  main_edt                                                                 */
/* ========================================================================= */

static int compute_tile_elems(int N, int num_nodes) {
  int target_tpd = 5;
  if (num_nodes > 1) {
    int min_tpd = (int)ceil(cbrt((double)num_nodes));
    if (target_tpd < min_tpd)
      target_tpd = min_tpd;
  }
  int best = N;
  for (int tpd = target_tpd; tpd <= N; tpd++) {
    if (N % tpd == 0) {
      best = N / tpd;
      break;
    }
  }
  if (best <= 0)
    best = N;
  return best;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  int N = 30, max_iter = 9999999, T = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-s") && i + 1 < argc)
      N = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-i") && i + 1 < argc)
      max_iter = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-t") && i + 1 < argc)
      T = atoi(argv[++i]);
  }
  unsigned nn = arts_get_total_nodes();
  if (T <= 0)
    T = compute_tile_elems(N, (int)nn);
  if (N % T) {
    printf("Error: %d not divisible by %d\n", N, T);
    arts_shutdown();
    return;
  }
  int tpd = N / T, nt = tpd * tpd * tpd;
  if (nt > MAX_TILES) {
    printf("Error: too many tiles\n");
    arts_shutdown();
    return;
  }
  fprintf(stderr, "LULESH ARTS 6-DB-partition implementation\n");
  fprintf(stderr,
          "  Edge elements:  %d\n  Tile elements:  %d\n  Tiles:          %d\n  "
          "ARTS nodes:     %u\n",
          N, T, nt, nn);
  build_topo(N, T, (int)nn);

  /* Pre-reserve 6*nt DB GUIDs on tile owner nodes */
  for (int t = 0; t < nt; t++) {
    unsigned o = g_init.tile_owner[t];
    g_init.const_guids[t] = arts_guid_reserve(ARTS_DB, o);
    g_init.pos_vel_guids[t] = arts_guid_reserve(ARTS_DB, o);
    g_init.elem_state_guids[t] = arts_guid_reserve(ARTS_DB, o);
    g_init.grad_guids[t] = arts_guid_reserve(ARTS_DB, o);
    g_init.force_guids[t] = arts_guid_reserve(ARTS_DB, o);
    g_init.dt_result_guids[t] = arts_guid_reserve(ARTS_DB, o);
  }

  /* Create init_done latch (count=nt) and post_init_edt */
  arts_guid_t init_done =
      arts_event_create(0, ARTS_EVENT_LATCH, (unsigned)nt, NULL_GUID);
  arts_guid_t post_init =
      arts_edt_create(post_init_edt, 0, NULL, 3, &(arts_hint_t){.route = 0});
  arts_add_dependence(init_done, post_init, 0, DB_MODE_NULL);
  arts_add_dependence(g_init.const_guids[0], post_init, 1, DB_MODE_RO);
  arts_add_dependence(g_init.elem_state_guids[0], post_init, 2, DB_MODE_RO);

  /* Create init_tile_edts with GUIDs in paramv */
  for (int t = 0; t < nt; t++) {
    unsigned o = g_init.tile_owner[t];
    uint64_t p[12] = {(uint64_t)t,
                      (uint64_t)N,
                      (uint64_t)T,
                      (uint64_t)tpd,
                      (uint64_t)max_iter,
                      (uint64_t)g_init.const_guids[t],
                      (uint64_t)g_init.pos_vel_guids[t],
                      (uint64_t)g_init.elem_state_guids[t],
                      (uint64_t)g_init.grad_guids[t],
                      (uint64_t)g_init.force_guids[t],
                      (uint64_t)g_init.dt_result_guids[t],
                      (uint64_t)init_done};
    arts_edt_create(init_tile_edt, 12, p, 0, &(arts_hint_t){.route = o});
  }
}

/* Called on EVERY node before main_edt -- ensures g_init is available
 * everywhere */
void init_per_node(unsigned int node_id, int argc, char **argv) {
  (void)node_id;
  int N = 30, T = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-s") && i + 1 < argc)
      N = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-t") && i + 1 < argc)
      T = atoi(argv[++i]);
  }
  unsigned nn = arts_get_total_nodes();
  if (T <= 0)
    T = compute_tile_elems(N, (int)nn);
  if (N % T)
    return;
  build_topo(N, T, (int)nn);
}

int main(int argc, char *argv[]) {
  arts_rt(argc, argv);
  return 0;
}
