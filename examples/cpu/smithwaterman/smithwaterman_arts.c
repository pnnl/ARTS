/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

/*
 * Smith-Waterman (local alignment) — native ARTS port.
 *
 * Ports third_party/ocr-apps/apps/smithwaterman/ocr/smithwaterman.c to native
 * ARTS in LULESH style, with ocrPNNL's distributed tile-owner strategy
 * (round-robin chunk=8) reimplemented via arts_hint_t.route.
 *
 * Algorithm:
 *   Standard wavefront DP.  Strings are laid out as a 2-D tile grid of
 *   (n_tiles_h x n_tiles_w) tiles of size (tile_h x tile_w) cells.  Tile
 *   (i,j) needs 3 halos from neighbors: right-column of (i,j-1), bottom-
 *   row of (i-1,j), and bottom-right of (i-1,j-1).  Each tile emits its
 *   own 3 halos for downstream tiles.  Boundary tiles receive GAP-penalty
 *   halos from main_edt.
 *
 * DB granularity (per tile):
 *   3 output halo DBs — right_col (tile_h * int32), bottom_row (tile_w *
 *   int32), bottom_right (1 * int32).  Each halo DB is created once,
 *   consumed once, destroyed.  No concurrent access, no CDAG frontier
 *   pressure.
 *
 * Synchronization:
 *   Per tile, 3 ARTS_EVENT_ONCE events carry halo DBs.  Tile (i,j)'s
 *   consumer EDTs (tile (i+1,j), (i,j+1), (i+1,j+1)) wire in advance via
 *   arts_add_dependence on those events.  Producer satisfies events via
 *   arts_event_satisfy_slot with DB payload.  Pattern A release-before-
 *   satisfy is used on every signaled halo for CXL retrofit.
 *
 * Distribution:
 *   tile_owner(i,j) = (tile_linear_index(i,j) / CHUNK_SIZE) % num_nodes,
 *   CHUNK_SIZE = 8 (ocrPNNL default).
 *
 * Shared data:
 *   One params DB holds tile/grid dims and both strings.  Every tile EDT
 *   has an RO dep on it at slot 3.  Read-only, never modified.
 */

#include "arts.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GAP_PENALTY (-1)
#define TRANSITION_PENALTY (-2)
#define TRANSVERSION_PENALTY (-4)
#define MATCH (2)
#define CHUNK_SIZE (8)

enum Nucleotide { GAP = 0, ADENINE, CYTOSINE, GUANINE, THYMINE };

static const int alignment_score_matrix[5][5] = {
    {GAP_PENALTY, GAP_PENALTY, GAP_PENALTY, GAP_PENALTY, GAP_PENALTY},
    {GAP_PENALTY, MATCH, TRANSVERSION_PENALTY, TRANSITION_PENALTY,
     TRANSVERSION_PENALTY},
    {GAP_PENALTY, TRANSVERSION_PENALTY, MATCH, TRANSVERSION_PENALTY,
     TRANSITION_PENALTY},
    {GAP_PENALTY, TRANSITION_PENALTY, TRANSVERSION_PENALTY, MATCH,
     TRANSVERSION_PENALTY},
    {GAP_PENALTY, TRANSVERSION_PENALTY, TRANSITION_PENALTY,
     TRANSVERSION_PENALTY, MATCH}};

static inline int char_map(char c) {
  switch (c) {
  case 'A':
    return ADENINE;
  case 'C':
    return CYTOSINE;
  case 'G':
    return GUANINE;
  case 'T':
    return THYMINE;
  default:
    return -1;
  }
}

/* ========================================================================= */
/*  Global state (built on every node by init_per_node)                      */
/* ========================================================================= */

typedef struct {
  int n_tiles_h;    /* number of row tiles */
  int n_tiles_w;    /* number of col tiles */
  int tile_h;       /* cells per tile (row direction) */
  int tile_w;       /* cells per tile (col direction) */
  int string1_len;  /* length of string 1 (laid along width) */
  int string2_len;  /* length of string 2 (laid along height) */
  int verify_score; /* expected final DP score (from file) */
} sw_topo_t;

static sw_topo_t g_topo;

static inline int n_events_per_type(void) {
  return (g_topo.n_tiles_h + 1) * (g_topo.n_tiles_w + 1);
}
static inline int idx_ev(int i, int j) {
  return i * (g_topo.n_tiles_w + 1) + j;
}

static inline unsigned tile_owner(int i, int j) {
  int linear = (i - 1) * g_topo.n_tiles_w + (j - 1);
  return (unsigned)((linear / CHUNK_SIZE) % arts_get_total_nodes());
}

/* ========================================================================= */
/*  Shared params DB layout                                                   */
/*                                                                           */
/*  [int tile_w, tile_h, n_tiles_h, n_tiles_w, string1_len, string2_len,    */
/*   string1_off, string2_off, verify_score, reserved[7],                   */
/*   string1[string1_len], string2[string2_len]]                             */
/* ========================================================================= */

#define PARAMS_HDR_WORDS 16

static inline int params_tile_w(const int64_t *p) { return (int)p[0]; }
static inline int params_tile_h(const int64_t *p) { return (int)p[1]; }
static inline int params_n_h(const int64_t *p) { return (int)p[2]; }
static inline int params_n_w(const int64_t *p) { return (int)p[3]; }
static inline int params_s1_len(const int64_t *p) { return (int)p[4]; }
static inline int params_s2_len(const int64_t *p) { return (int)p[5]; }
static inline const int8_t *params_s1(const int64_t *p) {
  return (const int8_t *)&p[PARAMS_HDR_WORDS];
}
static inline const int8_t *params_s2(const int64_t *p) {
  int s1_bytes = params_s1_len(p);
  int s1_words = (s1_bytes + 7) / 8;
  return (const int8_t *)&p[PARAMS_HDR_WORDS + s1_words];
}
static inline int params_score(const int64_t *p) { return (int)p[8]; }

/* ========================================================================= */
/*  Forward decls                                                            */
/* ========================================================================= */

void sw_tile_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]);

void done_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]);

/* ========================================================================= */
/*  done_edt  (always runs on node 0)                                        */
/*                                                                           */
/*  paramv[0] = expected score (verify_score from params)                    */
/*  depv[0]   = score DB (int32_t[1]) RO — final DP score from last tile     */
/* ========================================================================= */

void done_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int expected = (int)paramv[0];
  const int32_t *score_db = (const int32_t *)depv[0].ptr;
  int final_score = score_db[0];
  arts_printf("SW-ARTS final score = %d (expected %d) %s\n", final_score,
              expected, (final_score == expected) ? "PASS" : "FAIL");
  arts_shutdown();
}

/* ========================================================================= */
/*  sw_tile_edt                                                              */
/*                                                                           */
/*  paramv[0] = i, paramv[1] = j                                             */
/*  paramv[2..4] = halo event GUIDs this EDT will satisfy when done:         */
/*     [2] right_col event for (i,j+1)                                       */
/*     [3] bottom_row event for (i+1,j)                                      */
/*     [4] bottom_right event for (i+1,j+1)                                  */
/*  paramv[5] = done_guid event (NULL_GUID for non-last tiles)               */
/*  depv[0] = left halo (right_col of (i, j-1)) RO                           */
/*  depv[1] = top  halo (bottom_row of (i-1, j)) RO                          */
/*  depv[2] = NW   halo (bottom_right of (i-1, j-1)) RO                      */
/*  depv[3] = params DB RO                                                   */
/* ========================================================================= */

void sw_tile_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int i = (int)paramv[0];
  int j = (int)paramv[1];
  arts_guid_t rc_event = (arts_guid_t)paramv[2];
  arts_guid_t br_row_event = (arts_guid_t)paramv[3];
  arts_guid_t br_corner_event = (arts_guid_t)paramv[4];
  arts_guid_t done_guid = (arts_guid_t)paramv[5];

  const int32_t *left_col = (const int32_t *)depv[0].ptr;
  const int32_t *top_row = (const int32_t *)depv[1].ptr;
  const int32_t *nw_br = (const int32_t *)depv[2].ptr;
  const int64_t *params = (const int64_t *)depv[3].ptr;

  int tile_w = params_tile_w(params);
  int tile_h = params_tile_h(params);
  int n_tiles_w = params_n_w(params);
  int n_tiles_h = params_n_h(params);
  int string1_len = params_s1_len(params);
  int string2_len = params_s2_len(params);
  const int8_t *string_1 = params_s1(params);
  const int8_t *string_2 = params_s2(params);

  int eff_w = tile_w;
  int eff_h = tile_h;
  if (j == n_tiles_w) {
    int rem = string1_len - (j - 1) * tile_w;
    if (rem < tile_w && rem > 0)
      eff_w = rem;
  }
  if (i == n_tiles_h) {
    int rem = string2_len - (i - 1) * tile_h;
    if (rem < tile_h && rem > 0)
      eff_h = rem;
  }

  /* Local DP scratchpad: (tile_h+1) x (tile_w+1) ints */
  size_t cells = (size_t)(tile_h + 1) * (size_t)(tile_w + 1);
  int32_t *M = (int32_t *)malloc(cells * sizeof(int32_t));
  int H = tile_h + 1;
  int W = tile_w + 1;
  (void)H;
#define Mref(r, c) M[(size_t)(r) * (size_t)W + (size_t)(c)]

  /* Seed halo */
  Mref(0, 0) = nw_br[0];
  for (int r = 1; r <= eff_h; ++r)
    Mref(r, 0) = left_col[r - 1];
  for (int c = 1; c <= eff_w; ++c)
    Mref(0, c) = top_row[c - 1];

  /* DP */
  for (int ii = 1; ii <= eff_h; ++ii) {
    for (int jj = 1; jj <= eff_w; ++jj) {
      int c1 = string_1[(j - 1) * tile_w + (jj - 1)];
      int c2 = string_2[(i - 1) * tile_h + (ii - 1)];
      int diag = Mref(ii - 1, jj - 1) + alignment_score_matrix[c2][c1];
      int left = Mref(ii, jj - 1) + alignment_score_matrix[c1][GAP];
      int top = Mref(ii - 1, jj) + alignment_score_matrix[GAP][c2];
      int bigger = (left > top) ? left : top;
      Mref(ii, jj) = (bigger > diag) ? bigger : diag;
    }
  }

  /* Emit 3 halo DBs and satisfy events.
   * Pattern A: arts_db_release before each satisfy. */
  arts_guid_t rc_guid = NULL_GUID;
  arts_guid_t br_row_guid = NULL_GUID;
  arts_guid_t br_corner_guid = NULL_GUID;
  unsigned cur = arts_get_current_node();

  /* Bottom-right corner */
  if (br_corner_event != NULL_GUID) {
    int32_t *br_corner_data;
#if ARTS_USE_CXL
    br_corner_guid = arts_db_create((void **)&br_corner_data, sizeof(int32_t),
                                    ARTS_DB_CXL, 0, NULL);
#else
    br_corner_guid =
        arts_db_create((void **)&br_corner_data, sizeof(int32_t),
                       ARTS_DB_DEFAULT, 0, &(arts_db_hint_t){.rank = cur});
#endif
    br_corner_data[0] = Mref(eff_h, eff_w);
#if ARTS_USE_CXL
    arts_cxl_producer_flush(br_corner_guid); // flush CXL FAM writes before release
#endif
    arts_db_release(br_corner_guid); // Pattern A
    arts_event_satisfy_slot(br_corner_event, br_corner_guid,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }

  /* Right column */
  if (rc_event != NULL_GUID) {
    int32_t *rc_data;
#if ARTS_USE_CXL
    rc_guid = arts_db_create((void **)&rc_data, sizeof(int32_t) * tile_h,
                             ARTS_DB_CXL, 0, NULL);
#else
    rc_guid = arts_db_create((void **)&rc_data, sizeof(int32_t) * tile_h,
                             ARTS_DB_DEFAULT, 0, &(arts_db_hint_t){.rank = cur});
#endif
    for (int r = 0; r < eff_h; ++r)
      rc_data[r] = Mref(r + 1, eff_w);
    for (int r = eff_h; r < tile_h; ++r)
      rc_data[r] = 0;         /* pad */
#if ARTS_USE_CXL
    arts_cxl_producer_flush(rc_guid); // flush CXL FAM writes before release
#endif
    arts_db_release(rc_guid); // Pattern A
    arts_event_satisfy_slot(rc_event, rc_guid, ARTS_EVENT_LATCH_DECR_SLOT);
  }

  /* Bottom row */
  if (br_row_event != NULL_GUID) {
    int32_t *br_row_data;
#if ARTS_USE_CXL
    br_row_guid = arts_db_create((void **)&br_row_data,
                                 sizeof(int32_t) * tile_w, ARTS_DB_CXL, 0, NULL);
#else
    br_row_guid =
        arts_db_create((void **)&br_row_data, sizeof(int32_t) * tile_w,
                       ARTS_DB_DEFAULT, 0, &(arts_db_hint_t){.rank = cur});
#endif
    for (int c = 0; c < eff_w; ++c)
      br_row_data[c] = Mref(eff_h, c + 1);
    for (int c = eff_w; c < tile_w; ++c)
      br_row_data[c] = 0;         /* pad */
#if ARTS_USE_CXL
    arts_cxl_producer_flush(br_row_guid); // flush CXL FAM writes before release
#endif
    arts_db_release(br_row_guid); // Pattern A
    arts_event_satisfy_slot(br_row_event, br_row_guid,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }

  int final_score = Mref(eff_h, eff_w);
  free(M);

  /* If this is the last tile (bottom-right most), signal done_guid on node 0
   * which will print the result and shut down. */
  if (i == n_tiles_h && j == n_tiles_w && done_guid != NULL_GUID) {
    int32_t *score_data;
#if ARTS_USE_CXL
    arts_guid_t score_db_guid = arts_db_create(
        (void **)&score_data, sizeof(int32_t), ARTS_DB_CXL, 0, NULL);
#else
    arts_guid_t score_db_guid =
        arts_db_create((void **)&score_data, sizeof(int32_t), ARTS_DB_DEFAULT,
                       0, &(arts_db_hint_t){.rank = cur});
#endif
    score_data[0] = final_score;
#if ARTS_USE_CXL
    arts_cxl_producer_flush(score_db_guid); // flush CXL FAM writes before release
#endif
    arts_db_release(score_db_guid); // Pattern A
    arts_event_satisfy_slot(done_guid, score_db_guid,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }
#undef Mref
}

/* ========================================================================= */
/*  File I/O (reads on rank 0 only, from main_edt)                           */
/* ========================================================================= */

static char *read_file_whitespace_mapped(const char *path, int *out_len) {
  FILE *f = fopen(path, "r");
  if (!f) {
    arts_printf("SW: cannot open %s\n", path);
    return NULL;
  }
  fseek(f, 0L, SEEK_END);
  long fsz = ftell(f);
  fseek(f, 0L, SEEK_SET);
  char *buf = (char *)malloc((size_t)fsz + 1);
  size_t got = fread(buf, 1, (size_t)fsz, f);
  (void)got;
  buf[fsz] = '\0';
  fclose(f);

  /* Strip whitespace, map to enum */
  int n = 0;
  for (long k = 0; k < fsz; ++k) {
    char c = buf[k];
    if (c == 'A' || c == 'C' || c == 'G' || c == 'T') {
      buf[n++] = (char)char_map(c);
    }
  }
  *out_len = n;
  return buf;
}

/* ========================================================================= */
/*  main_edt                                                                 */
/* ========================================================================= */

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  if (argc < 6) {
    arts_printf("Usage: %s <tile_w> <tile_h> <string1_file> <string2_file> "
                "<score_file>\n",
                argv[0]);
    arts_shutdown();
    return;
  }
  int tile_w = atoi(argv[1]);
  int tile_h = atoi(argv[2]);
  int s1_len = 0;
  int s2_len = 0;
  char *s1 = read_file_whitespace_mapped(argv[3], &s1_len);
  char *s2 = read_file_whitespace_mapped(argv[4], &s2_len);
  if (!s1 || !s2) {
    arts_shutdown();
    return;
  }
  int score_len = 0;
  char *score_buf = read_file_whitespace_mapped(argv[5], &score_len);
  (void)score_buf;
  /* read_file_whitespace_mapped strips chars — for score file read raw */
  int verify_score = 0;
  {
    FILE *f = fopen(argv[5], "r");
    if (f) {
      if (fscanf(f, "%d", &verify_score) != 1)
        verify_score = 0;
      fclose(f);
    }
  }
  int n_tiles_w = (s1_len + tile_w - 1) / tile_w;
  int n_tiles_h = (s2_len + tile_h - 1) / tile_h;

  g_topo.tile_w = tile_w;
  g_topo.tile_h = tile_h;
  g_topo.n_tiles_w = n_tiles_w;
  g_topo.n_tiles_h = n_tiles_h;
  g_topo.string1_len = s1_len;
  g_topo.string2_len = s2_len;
  g_topo.verify_score = verify_score;

  unsigned nn = arts_get_total_nodes();
  arts_printf("SW-ARTS: tile_w=%d tile_h=%d string1=%d string2=%d "
              "n_tiles=%dx%d nodes=%u score_expect=%d\n",
              tile_w, tile_h, s1_len, s2_len, n_tiles_h, n_tiles_w, nn,
              verify_score);

  /* --- Build params DB (shared RO across all tile EDTs) --- */
  int s1_words = (s1_len + 7) / 8;
  int s2_words = (s2_len + 7) / 8;
  size_t params_bytes = ((size_t)PARAMS_HDR_WORDS + s1_words + s2_words) * 8;
  int64_t *params;
#if ARTS_USE_CXL
  arts_guid_t params_guid =
      arts_db_create((void **)&params, params_bytes, ARTS_DB_CXL, 0, NULL);
#else
  arts_guid_t params_guid =
      arts_db_create((void **)&params, params_bytes, ARTS_DB_DEFAULT,
                     0, &(arts_db_hint_t){.rank = 0});
#endif
  memset(params, 0, params_bytes);
  params[0] = tile_w;
  params[1] = tile_h;
  params[2] = n_tiles_h;
  params[3] = n_tiles_w;
  params[4] = s1_len;
  params[5] = s2_len;
  params[8] = verify_score;
  memcpy((char *)&params[PARAMS_HDR_WORDS], s1, (size_t)s1_len);
  memcpy((char *)&params[PARAMS_HDR_WORDS + s1_words], s2, (size_t)s2_len);
  free(s1);
  free(s2);
  if (score_buf)
    free(score_buf);
#if ARTS_USE_CXL
  arts_cxl_producer_flush(params_guid); // flush CXL FAM writes before release
#endif
  arts_db_release(params_guid); // WRITE — shared params ready

  /* --- Create halo events for every (i,j) including border row/col --- */
  int nev = (n_tiles_h + 1) * (n_tiles_w + 1);
  arts_guid_t *ev_rc = (arts_guid_t *)malloc(sizeof(arts_guid_t) * nev);
  arts_guid_t *ev_brow = (arts_guid_t *)malloc(sizeof(arts_guid_t) * nev);
  arts_guid_t *ev_bcorner = (arts_guid_t *)malloc(sizeof(arts_guid_t) * nev);
  for (int i = 0; i <= n_tiles_h; ++i) {
    for (int j = 0; j <= n_tiles_w; ++j) {
      int idx = i * (n_tiles_w + 1) + j;
      /* All events home-rooted at rank 0 for now. Cross-node consumer
       * EDTs depend on them; ARTS handles the data forwarding. */
      ev_rc[idx] = arts_event_create(0, ARTS_EVENT_ONCE, 1, NULL_GUID);
      ev_brow[idx] = arts_event_create(0, ARTS_EVENT_ONCE, 1, NULL_GUID);
      ev_bcorner[idx] = arts_event_create(0, ARTS_EVENT_ONCE, 1, NULL_GUID);
    }
  }

  /* --- Create done_edt on node 0 — triggered by the last tile --- */
  arts_guid_t done_guid = arts_event_create(0, ARTS_EVENT_ONCE, 1, NULL_GUID);
  {
    uint64_t dp[1] = {(uint64_t)verify_score};
    arts_guid_t done_e =
        arts_edt_create(done_edt, 1, dp, 1, &(arts_edt_hint_t){.rank = 0});
    arts_add_dependence(done_guid, done_e, 0, DB_MODE_RO);
  }

  /* --- Create tile EDTs for (i in 1..n_tiles_h, j in 1..n_tiles_w) --- */
  for (int i = 1; i <= n_tiles_h; ++i) {
    for (int j = 1; j <= n_tiles_w; ++j) {
      unsigned o = tile_owner(i, j);
      int idx_self = i * (n_tiles_w + 1) + j;
      /* Pass the downstream event guids that this EDT will satisfy. */
      arts_guid_t my_rc_ev = ev_rc[idx_self];
      arts_guid_t my_brow_ev = ev_brow[idx_self];
      arts_guid_t my_bcorner_ev = ev_bcorner[idx_self];
      int is_last = (i == n_tiles_h && j == n_tiles_w);
      uint64_t p[6] = {(uint64_t)i, (uint64_t)j, (uint64_t)my_rc_ev,
                       (uint64_t)my_brow_ev, (uint64_t)my_bcorner_ev,
                       (uint64_t)(is_last ? done_guid : NULL_GUID)};
      arts_guid_t e =
          arts_edt_create(sw_tile_edt, 6, p, 4, &(arts_edt_hint_t){.rank = o});
      /* left halo from (i, j-1) */
      arts_add_dependence(ev_rc[i * (n_tiles_w + 1) + (j - 1)], e, 0,
                          DB_MODE_RO);
      /* top halo from (i-1, j) */
      arts_add_dependence(ev_brow[(i - 1) * (n_tiles_w + 1) + j], e, 1,
                          DB_MODE_RO);
      /* NW halo from (i-1, j-1) */
      arts_add_dependence(ev_bcorner[(i - 1) * (n_tiles_w + 1) + (j - 1)], e, 2,
                          DB_MODE_RO);
      /* params DB */
      arts_add_dependence(params_guid, e, 3, DB_MODE_RO);
    }
  }

  /* --- Initialize boundary halos --- */
  /* (0,0) bottom-right corner = 0 */
  {
    int32_t *p0;
#if ARTS_USE_CXL
    arts_guid_t g0 =
        arts_db_create((void **)&p0, sizeof(int32_t), ARTS_DB_CXL, 0, NULL);
#else
    arts_guid_t g0 =
        arts_db_create((void **)&p0, sizeof(int32_t), ARTS_DB_DEFAULT,
                       0, &(arts_db_hint_t){.rank = 0});
#endif
    p0[0] = 0;
#if ARTS_USE_CXL
    arts_cxl_producer_flush(g0); // flush CXL FAM writes before release
#endif
    arts_db_release(g0); // Pattern A
    arts_event_satisfy_slot(ev_bcorner[0], g0, ARTS_EVENT_LATCH_DECR_SLOT);
  }
  /* Top row of halos: for (0, j) the bottom_row and bottom_right values.
   * A real tile (1, j) reads (0, j) bottom_row as its top halo. */
  for (int j = 1; j <= n_tiles_w; ++j) {
    int eff_w = tile_w;
    if (j == n_tiles_w) {
      int rem = s1_len - (j - 1) * tile_w;
      if (rem < tile_w && rem > 0)
        eff_w = rem;
    }
    /* bottom_row for (0,j): values GAP*(n) along string 1 */
    int32_t *brow;
#if ARTS_USE_CXL
    arts_guid_t g_brow = arts_db_create(
        (void **)&brow, sizeof(int32_t) * tile_w, ARTS_DB_CXL, 0, NULL);
#else
    arts_guid_t g_brow =
        arts_db_create((void **)&brow, sizeof(int32_t) * tile_w,
                       ARTS_DB_DEFAULT, 0, &(arts_db_hint_t){.rank = 0});
#endif
    for (int c = 0; c < eff_w; ++c)
      brow[c] = GAP_PENALTY * ((j - 1) * tile_w + c + 1);
    for (int c = eff_w; c < tile_w; ++c)
      brow[c] = 0;
#if ARTS_USE_CXL
    arts_cxl_producer_flush(g_brow); // flush CXL FAM writes before release
#endif
    arts_db_release(g_brow); // Pattern A
    arts_event_satisfy_slot(ev_brow[0 * (n_tiles_w + 1) + j], g_brow,
                            ARTS_EVENT_LATCH_DECR_SLOT);

    /* bottom_right for (0, j): scalar GAP*((j-1)*tile_w + eff_w) */
    int32_t *bcor;
#if ARTS_USE_CXL
    arts_guid_t g_bcor =
        arts_db_create((void **)&bcor, sizeof(int32_t), ARTS_DB_CXL, 0, NULL);
#else
    arts_guid_t g_bcor =
        arts_db_create((void **)&bcor, sizeof(int32_t), ARTS_DB_DEFAULT,
                       0, &(arts_db_hint_t){.rank = 0});
#endif
    bcor[0] = GAP_PENALTY * ((j - 1) * tile_w + eff_w);
#if ARTS_USE_CXL
    arts_cxl_producer_flush(g_bcor); // flush CXL FAM writes before release
#endif
    arts_db_release(g_bcor); // Pattern A
    arts_event_satisfy_slot(ev_bcorner[0 * (n_tiles_w + 1) + j], g_bcor,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }
  /* Left column of halos */
  for (int i = 1; i <= n_tiles_h; ++i) {
    int eff_h = tile_h;
    if (i == n_tiles_h) {
      int rem = s2_len - (i - 1) * tile_h;
      if (rem < tile_h && rem > 0)
        eff_h = rem;
    }
    int32_t *rc;
#if ARTS_USE_CXL
    arts_guid_t g_rc = arts_db_create((void **)&rc, sizeof(int32_t) * tile_h,
                                      ARTS_DB_CXL, 0, NULL);
#else
    arts_guid_t g_rc =
        arts_db_create((void **)&rc, sizeof(int32_t) * tile_h, ARTS_DB_DEFAULT,
                       0, &(arts_db_hint_t){.rank = 0});
#endif
    for (int r = 0; r < eff_h; ++r)
      rc[r] = GAP_PENALTY * ((i - 1) * tile_h + r + 1);
    for (int r = eff_h; r < tile_h; ++r)
      rc[r] = 0;
#if ARTS_USE_CXL
    arts_cxl_producer_flush(g_rc); // flush CXL FAM writes before release
#endif
    arts_db_release(g_rc); // Pattern A
    arts_event_satisfy_slot(ev_rc[i * (n_tiles_w + 1) + 0], g_rc,
                            ARTS_EVENT_LATCH_DECR_SLOT);

    int32_t *bcor;
#if ARTS_USE_CXL
    arts_guid_t g_bcor =
        arts_db_create((void **)&bcor, sizeof(int32_t), ARTS_DB_CXL, 0, NULL);
#else
    arts_guid_t g_bcor =
        arts_db_create((void **)&bcor, sizeof(int32_t), ARTS_DB_DEFAULT,
                       0, &(arts_db_hint_t){.rank = 0});
#endif
    bcor[0] = GAP_PENALTY * ((i - 1) * tile_h + eff_h);
#if ARTS_USE_CXL
    arts_cxl_producer_flush(g_bcor); // flush CXL FAM writes before release
#endif
    arts_db_release(g_bcor); // Pattern A
    arts_event_satisfy_slot(ev_bcorner[i * (n_tiles_w + 1) + 0], g_bcor,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }

  free(ev_rc);
  free(ev_brow);
  free(ev_bcorner);
}

void init_per_node(unsigned int node_id, int argc, char **argv) {
  (void)node_id;
  /* Parse same args so g_topo is populated on each node for sw_tile_edt. */
  if (argc < 3)
    return;
  g_topo.tile_w = atoi(argv[1]);
  g_topo.tile_h = atoi(argv[2]);
  /* Other fields are carried via paramv/depv deps (params DB); g_topo
   * isn't strictly needed on non-rank-0 nodes for sw_tile_edt. */
}

int main(int argc, char *argv[]) {
  arts_rt(argc, argv);
  return 0;
}
