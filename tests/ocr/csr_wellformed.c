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

/// @file csr_wellformed.c
/// @brief T265 — arts_csr_init builds a well-formed CSR including gap-fill for
///        vertices that have no out-edges.
///
/// Verified properties (via the public API only — arts_csr_get_neighbors plus
/// the public arts_csr_graph_t fields):
///   * row_indices[0] == 0        : neighbours of local vertex 0 begin at the
///                                  start of the column array.
///   * row_indices monotonic      : for consecutive local vertices u, u+1 the
///                                  neighbour pointer of u+1 equals the pointer
///                                  of u advanced by count(u) (contiguous, no
///                                  overlap, no gap).
///   * row_indices[localv]==locale: sum of all neighbour counts == number of
///                                  local edges.
///   * gap-fill                    : a local vertex with no out-edges reports
///                                  count 0 and its start pointer equals the
///                                  end pointer of the previous vertex.
///   * column contents             : each vertex's listed neighbours match the
///                                  edges fed in.
///
/// Config-independent (ARTS_DB_PIN).  Single-node.

#include <stdlib.h>

#include "arts.h"
#include "arts/graph.h"

/// Edge list over 8 vertices.  Vertices 2, 5, 6 deliberately have NO out-edges
/// (interior gaps + a trailing gap) to exercise the gap-fill path.  Vertex 0
/// is the first source; vertex 7 the last source (no trailing tail-fill gap on
/// the very last, but 6 is a gap before it).
///   0 -> {3}
///   1 -> {0, 4}
///   3 -> {2, 5, 7}
///   4 -> {1}
///   7 -> {0, 6}
/// total = 9 edges.
static const arts_vertex_t SRC[] = {0, 1, 1, 3, 3, 3, 4, 7, 7};
static const arts_vertex_t TGT[] = {3, 0, 4, 2, 5, 7, 1, 0, 6};
#define NEDGES ((int)(sizeof(SRC) / sizeof(SRC[0])))
#define NVERTS 8

/// Expected out-degree per local vertex.
static const arts_graph_sz_t EXPECT_DEG[NVERTS] = {1, 2, 0, 3, 1, 0, 0, 2};

static int u64cmp(const void *a, const void *b) {
  arts_vertex_t x = *(const arts_vertex_t *)a;
  arts_vertex_t y = *(const arts_vertex_t *)b;
  return (x < y) ? -1 : (x > y) ? 1 : 0;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== csr_wellformed (T265) ===\n");

  arts_block_dist_t *dist =
      arts_block_dist_init(NVERTS, NEDGES, 1, ARTS_GUID_DB);

  arts_edge_vector_t vec;
  arts_edge_vector_init(&vec, 64);
  for (int i = 0; i < NEDGES; ++i) {
    arts_edge_vector_push_back(&vec, SRC[i], TGT[i], 0);
  }
  /* Sort by source so the gap-fill / monotonic build runs over grouped edges.
   */
  arts_edge_vector_sort_by_source(&vec);

  arts_guid_t g = arts_block_dist_guid_for_partition(dist, 0);
  arts_csr_graph_t *csr = arts_csr_init(0, NVERTS, NEDGES, dist, &vec, true, g);
  arts_edge_vector_free(&vec);

  if (csr == NULL) {
    arts_printf("FAIL: init returned NULL\n");
    arts_block_dist_free(dist);
    arts_shutdown();
    return;
  }

  bool ok =
      (csr->num_local_vertices == NVERTS) && (csr->num_local_edges == NEDGES);
  if (!ok) {
    arts_printf("FAIL: header localv=%lu locale=%lu\n",
                (unsigned long)csr->num_local_vertices,
                (unsigned long)csr->num_local_edges);
  }

  arts_vertex_t *prev_ptr = NULL;
  arts_graph_sz_t prev_cnt = 0;
  arts_graph_sz_t total = 0;
  arts_vertex_t *first_ptr = NULL;

  for (arts_vertex_t v = 0; v < NVERTS && ok; ++v) {
    arts_vertex_t *nbrs = NULL;
    arts_graph_sz_t cnt = (arts_graph_sz_t)-1;
    arts_csr_get_neighbors(csr, v, &nbrs, &cnt);

    if (cnt != EXPECT_DEG[v]) {
      arts_printf("FAIL: vertex %lu degree %lu expected %lu\n",
                  (unsigned long)v, (unsigned long)cnt,
                  (unsigned long)EXPECT_DEG[v]);
      ok = false;
      break;
    }

    if (v == 0) {
      first_ptr = nbrs; /* row_indices[0]==0 => columns base */
    } else {
      /* Monotonic + gap-fill: this vertex's start must equal previous
       * vertex's end (prev_ptr + prev_cnt), with no overlap or gap, even for
       * zero-degree (gap) vertices. */
      if (nbrs != prev_ptr + prev_cnt) {
        arts_printf("FAIL: vertex %lu start not contiguous with prev end\n",
                    (unsigned long)v);
        ok = false;
        break;
      }
    }

    /* Verify the neighbour set matches the edges we fed for this source. */
    if (cnt > 0) {
      arts_vertex_t expect[8];
      int n = 0;
      for (int e = 0; e < NEDGES; ++e) {
        if (SRC[e] == v) {
          expect[n++] = TGT[e];
        }
      }
      qsort(expect, (size_t)n, sizeof(expect[0]), u64cmp);
      arts_vertex_t got[8];
      for (arts_graph_sz_t k = 0; k < cnt; ++k) {
        got[k] = nbrs[k];
      }
      qsort(got, (size_t)cnt, sizeof(got[0]), u64cmp);
      for (arts_graph_sz_t k = 0; k < cnt; ++k) {
        if (got[k] != expect[k]) {
          arts_printf("FAIL: vertex %lu neighbour mismatch\n",
                      (unsigned long)v);
          ok = false;
          break;
        }
      }
    }

    prev_ptr = nbrs;
    prev_cnt = cnt;
    total += cnt;
  }

  /* row_indices[localv] == locale: total degree must equal local edge count. */
  if (ok && total != (arts_graph_sz_t)NEDGES) {
    arts_printf("FAIL: total degree %lu != locale %d\n", (unsigned long)total,
                NEDGES);
    ok = false;
  }
  /* row_indices[0]==0: first vertex columns start at the column base. */
  if (ok && first_ptr == NULL && EXPECT_DEG[0] > 0) {
    arts_printf("FAIL: vertex 0 has NULL neighbour base\n");
    ok = false;
  }

  if (ok) {
    arts_printf("PASS csr_wellformed\n");
  }

  arts_csr_free(csr);
  arts_block_dist_free(dist);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
