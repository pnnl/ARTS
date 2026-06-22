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

/// @file csr_remote_null.c
/// @brief T271 — multinode: arts_csr_init returns NULL for a remote partition
///        and per-rank local-partition selection is exclusive.
///
/// With one block per rank, exactly one partition is local on each rank.
/// On every rank we verify:
///   * arts_csr_init for a NON-local (remote) block GUID returns NULL and
///     materialises nothing (arts_csr_from_partition of that index is NULL).
///   * arts_csr_init for THIS rank's own (local) block GUID succeeds and
///     arts_csr_from_partition of that index is non-NULL.
///   * arts_csr_from_partition for a remote partition is NULL.
///   * The set of local partitions is disjoint across ranks (each rank owns
///     exactly the partition whose round-robin assignment matches its rank).
///
/// Config-independent (ARTS_DB_PIN).  Requires >= 2 ranks; SKIPs otherwise.

#include "arts.h"
#include "arts/gas/guid.h" /* ARTS_GUID_GET_RANK */
#include "arts/graph.h"

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int ranks = arts_get_total_ranks();
  unsigned int me = arts_get_current_rank();
  arts_printf("=== csr_remote_null (T271) rank %u/%u ===\n", me, ranks);
  if (ranks < 2) {
    arts_printf("SKIP csr_remote_null: needs >= 2 ranks\n");
    arts_shutdown();
    return;
  }

  /* One block per rank: partition index i is owned by rank i. */
  unsigned int num_blocks = ranks;
  arts_block_dist_t *dist =
      arts_block_dist_init(num_blocks * 8, 0, num_blocks, ARTS_GUID_DB);

  bool ok = true;
  unsigned int local_count = 0;
  unsigned int my_partition = (unsigned int)-1;

  for (unsigned int i = 0; i < num_blocks && ok; ++i) {
    arts_guid_t g = arts_block_dist_guid_for_partition(dist, i);
    bool is_local = arts_guid_is_local(g);
    if (is_local) {
      local_count++;
      my_partition = i;
    }

    /* Build an empty-ish edge vector for an arts_csr_init attempt on this
     * block.  For a remote block, arts_csr_init must short-circuit to NULL
     * before touching the edge vector. */
    arts_edge_vector_t vec;
    arts_edge_vector_init(&vec, 8);
    if (is_local) {
      /* Local vertices for partition i start at i*block_sz. */
      arts_vertex_t base = arts_block_dist_partition_start(i, dist);
      arts_edge_vector_push_back(&vec, base, base + 1, 0);
      arts_edge_vector_sort_by_source(&vec);
    }

    arts_graph_sz_t localv = arts_block_dist_block_size(i, dist);
    arts_csr_graph_t *csr =
        arts_csr_init(i, localv, vec.used, dist, &vec, true, g);
    arts_edge_vector_free(&vec);

    if (is_local) {
      if (csr == NULL) {
        arts_printf("FAIL: rank %u local partition %u init returned NULL\n", me,
                    i);
        ok = false;
      } else if (arts_csr_from_partition(i, dist) == NULL) {
        arts_printf("FAIL: rank %u local partition %u not in route table\n", me,
                    i);
        ok = false;
      }
      if (csr) {
        arts_csr_free(csr);
      }
    } else {
      if (csr != NULL) {
        arts_printf("FAIL: rank %u remote partition %u init returned %p\n", me,
                    i, (void *)csr);
        ok = false;
      }
      if (arts_csr_from_partition(i, dist) != NULL) {
        arts_printf("FAIL: rank %u remote partition %u lookup not NULL\n", me,
                    i);
        ok = false;
      }
    }
  }

  /* Per-rank exclusivity: exactly one local partition, and it is the one whose
   * round-robin rank equals this rank. */
  if (ok && local_count != 1) {
    arts_printf("FAIL: rank %u owns %u local partitions (expected 1)\n", me,
                local_count);
    ok = false;
  }
  if (ok && my_partition != me) {
    arts_printf("FAIL: rank %u local partition is %u (expected %u)\n", me,
                my_partition, me);
    ok = false;
  }

  arts_block_dist_free(dist);

  if (ok) {
    arts_printf("PASS csr_remote_null rank %u\n", me);
  }
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
