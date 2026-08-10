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

/// @file csr_empty_partition.c
/// @brief T264 — arts_csr_init with a local partition that has ZERO edges.
///
/// B-csr-empty-ub: arts_csr_init reads edges->edge_array[0].source/.target
/// UNCONDITIONALLY (before the (edges->used)?1:0 guard).  For a local
/// partition with no local edges (edges->used == 0) this is an out-of-bounds /
/// uninitialised read of element 0, and columns[0]=t writes garbage.
///
/// The CORRECT behaviour for an empty partition is: build a valid empty CSR
/// (row_indices all zero, num_local_edges == 0, get_neighbors of any local
/// vertex returns count 0) with no OOB access.  This test asserts that correct
/// behaviour, so under ASan it is EXPECTED TO FAIL until the bug is fixed.
/// exposes_runtime_bug = true (B-csr-empty-ub).
///
/// We make the empty edge vector zero-capacity (arts_malloc(0) -> NULL) so the
/// unconditional edge_array[0] read is a hard NULL deref, giving a
/// deterministic sanitizer/SEGV exposure rather than a silent garbage read.
///
/// Config-independent (CSR DBs are ARTS_DB_PIN).  Single-node.

#include "arts.h"
#include "../test_failure_status.h"
#include "arts/graph.h"

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== csr_empty_partition (T264) ===\n");

  /* Single local partition covering 8 vertices, but with NO edges. */
  arts_block_dist_t *dist = arts_block_dist_init(8, 0, 1, ARTS_GUID_DB);

  arts_edge_vector_t vec;
  /* Zero capacity: edge_array == NULL, used == 0.  A correct empty-partition
   * build must NOT touch edge_array at all. */
  arts_edge_vector_init(&vec, 0);

  arts_guid_t g = arts_block_dist_guid_for_partition(dist, 0);
  /* localv = 8 (all local vertices), locale = 0 (no local edges). */
  arts_csr_graph_t *csr = arts_csr_init(0, 8, 0, dist, &vec, true, g);

  arts_edge_vector_free(&vec);

  if (csr == NULL) {
    arts_test_fail();
    arts_printf("FAIL: empty local partition init returned NULL\n");
    arts_block_dist_free(dist);
    arts_shutdown();
    return;
  }

  bool ok = (csr->num_local_edges == 0) && (csr->num_local_vertices == 8);

  /* Every local vertex must report zero neighbours and never deref. */
  for (arts_vertex_t v = 0; v < 8 && ok; ++v) {
    arts_vertex_t *nbrs = NULL;
    arts_graph_sz_t cnt = (arts_graph_sz_t)-1;
    arts_csr_get_neighbors(csr, v, &nbrs, &cnt);
    if (cnt != 0) {
      arts_test_fail();
      arts_printf("FAIL: vertex %lu of empty partition has %lu neighbours\n",
                  (unsigned long)v, (unsigned long)cnt);
      ok = false;
    }
  }

  if (ok) {
    arts_printf("PASS csr_empty_partition\n");
  } else {
    arts_test_fail();
    arts_printf("FAIL csr_empty_partition\n");
  }

  arts_csr_free(csr);
  arts_block_dist_free(dist);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Two verdicts to merge: what arts_rt saw of the ranks it spawned (their exit
     status reaches nobody else) and what this rank's own checks found. */
  int rc = arts_rt(argc, argv);
  return rc != 0 ? 1 : arts_test_status();
}
