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

/// @file csr_get_neighbors.c
/// @brief T267 — arts_csr_get_neighbors plus the local-index <-> vertex
///        round-trip.
///
/// Covers:
///   * Zero-neighbour vertex: count == 0, and the returned base pointer is NOT
///     dereferenced by the caller (we must not read past it).
///   * First local vertex and last local vertex neighbour retrieval.
///   * Round-trip: for every local vertex v, arts_csr_get_local_index(v) maps
///     back to v via the partition start (v == partition_start + local_index).
///     With a single partition starting at 0 this means local_index == v.
///
/// Config-independent (ARTS_DB_PIN).  Single-node.

#include <stdlib.h>

#include "arts.h"
#include "arts/graph.h"

/// 6 vertices, single partition.
///   0 -> {1, 2}
///   1 -> {}            (zero-neighbour, interior)
///   2 -> {0}
///   3 -> {}            (zero-neighbour, interior)
///   4 -> {5}
///   5 -> {0, 1, 4}     (last vertex, multiple)
static const arts_vertex_t SRC[] = {0, 0, 2, 4, 5, 5, 5};
static const arts_vertex_t TGT[] = {1, 2, 0, 5, 0, 1, 4};
#define NEDGES ((int)(sizeof(SRC) / sizeof(SRC[0])))
#define NVERTS 6
static const arts_graph_sz_t EXPECT_DEG[NVERTS] = {2, 0, 1, 0, 1, 3};

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

  arts_printf("=== csr_get_neighbors (T267) ===\n");

  arts_block_dist_t *dist =
      arts_block_dist_init(NVERTS, NEDGES, 1, ARTS_GUID_DB);

  arts_edge_vector_t vec;
  arts_edge_vector_init(&vec, 32);
  for (int i = 0; i < NEDGES; ++i) {
    arts_edge_vector_push_back(&vec, SRC[i], TGT[i], 0);
  }
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

  bool ok = true;
  arts_vertex_t pstart = arts_block_dist_partition_start(0, dist);

  for (arts_vertex_t v = 0; v < NVERTS && ok; ++v) {
    arts_vertex_t *nbrs = NULL;
    arts_graph_sz_t cnt = (arts_graph_sz_t)-1;
    arts_csr_get_neighbors(csr, v, &nbrs, &cnt);

    if (cnt != EXPECT_DEG[v]) {
      arts_printf("FAIL: vertex %lu count %lu expected %lu\n", (unsigned long)v,
                  (unsigned long)cnt, (unsigned long)EXPECT_DEG[v]);
      ok = false;
      break;
    }

    /* Zero-neighbour vertex: do NOT dereference nbrs. */
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
          arts_printf("FAIL: vertex %lu neighbour mismatch at %lu\n",
                      (unsigned long)v, (unsigned long)k);
          ok = false;
          break;
        }
      }
    }

    /* Round-trip: local index recovers v via partition start. */
    arts_local_index_t li = arts_csr_get_local_index(v, csr);
    if (pstart + (arts_vertex_t)li != v) {
      arts_printf("FAIL: round-trip vertex %lu local %lu start %lu\n",
                  (unsigned long)v, (unsigned long)li, (unsigned long)pstart);
      ok = false;
    }
  }

  if (ok) {
    arts_printf("PASS csr_get_neighbors\n");
  }

  arts_csr_free(csr);
  arts_block_dist_free(dist);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
