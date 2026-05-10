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
#include "arts/graph.h"

#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"
#include "arts/system/print.h"
#include "arts/utils/malloc.h"

void internal_init_block_distribution(arts_block_dist_t *dist, graph_sz_t n,
                                      graph_sz_t m, unsigned int num_blocks) {
  dist->num_vertices = n;
  dist->num_edges = m;
  dist->num_blocks = num_blocks;
  dist->block_sz = (num_blocks > 0) ? ((n + num_blocks - 1) / num_blocks) : 0;
}

arts_block_dist_t *init_block_distribution_block(graph_sz_t n, graph_sz_t m,
                                                 unsigned int num_blocks,
                                                 arts_guid_kind_t db_type) {
  arts_block_dist_t *dist = (arts_block_dist_t *)arts_malloc(
      sizeof(arts_block_dist_t) + (sizeof(arts_guid_t) * num_blocks));
  unsigned int blocks_per_node = num_blocks / arts_get_total_ranks();
  unsigned int mod = num_blocks % arts_get_total_ranks();
  unsigned int current = 0;
  for (unsigned int i = 0; i < arts_get_total_ranks(); i++) {
    for (unsigned int j = 0; j < blocks_per_node; j++) {
      dist->graphGuid[current++] = arts_guid_reserve(db_type, i);
    }
    if (mod) {
      dist->graphGuid[current++] = arts_guid_reserve(db_type, i);
      mod--;
    }
  }

  internal_init_block_distribution(dist, n, m, num_blocks);
  return dist;
}

arts_block_dist_t *init_block_distribution(graph_sz_t n, graph_sz_t m) {
  unsigned int num_blocks = arts_get_total_ranks();
  arts_block_dist_t *dist = (arts_block_dist_t *)arts_malloc(
      sizeof(arts_block_dist_t) + (sizeof(arts_guid_t) * num_blocks));
  for (unsigned int i = 0; i < num_blocks; i++) {
    dist->graphGuid[i] = arts_guid_reserve(ARTS_GUID_DB, i);
  }
  internal_init_block_distribution(dist, n, m, num_blocks);
  return dist;
}

arts_block_dist_t *init_block_distribution_with_cmd_line_args(int argc,
                                                              char **argv) {
  uint64_t n = 0;
  uint64_t m = 0;
  for (int i = 0; i < argc; ++i) {
    if (strcmp("--num-vertices", argv[i]) == 0) {
      n = (uint64_t)strtoull(argv[i + 1], NULL, 10);
    }
    if (strcmp("--num-edges", argv[i]) == 0) {
      m = (uint64_t)strtoull(argv[i + 1], NULL, 10);
    }
  }

  if (n && m) {
    unsigned int num_blocks = arts_get_total_ranks();
    arts_block_dist_t *dist = (arts_block_dist_t *)arts_malloc(
        sizeof(arts_block_dist_t) + (sizeof(arts_guid_t) * num_blocks));
    for (unsigned int i = 0; i < num_blocks; i++) {
      dist->graphGuid[i] = arts_guid_reserve(ARTS_GUID_DB, i);
    }
    internal_init_block_distribution(dist, n, m, num_blocks);
    return dist;
  }
  ARTS_INFO("Must set --num-vertices and --num-edges");
  return NULL;
}

void free_distribution(arts_block_dist_t *dist) { arts_free(dist); }

unsigned int get_num_local_blocks(arts_block_dist_t *dist) {
  unsigned int num_local_parts = 0;
  for (unsigned int i = 0; i < dist->num_blocks; i++) {
    if (arts_guid_is_local(get_guid_for_vertex_distr(i, dist))) {
      num_local_parts++;
    }
  }
  return num_local_parts;
}

graph_sz_t get_block_size_for_partition(unsigned int index,
                                        const arts_block_dist_t *const dist) {
  // is this the last block/partition
  if (index == (dist->num_blocks - 1)) {
    return (dist->num_vertices - ((dist->num_blocks - 1) * dist->block_sz));
  }
  return dist->block_sz;
}

unsigned int get_owner_distr(vertex_t v, const arts_block_dist_t *const dist) {
  return (unsigned int)(v / dist->block_sz);
}

vertex_t partition_start_distr(partition_t index,
                               const arts_block_dist_t *const dist) {
  return (vertex_t)((dist->block_sz) * index);
}

vertex_t partition_end_distr(partition_t index,
                             const arts_block_dist_t *const dist) {
  // is this the last block/partition?
  if (index == (dist->num_blocks - 1)) {
    return (vertex_t)(dist->num_vertices - 1);
  }
  return (partition_start_distr(index, dist) + (dist->block_sz - 1));
}

vertex_t get_vertex_from_local_distr(unsigned int local, local_index_t u,
                                     const arts_block_dist_t *const dist) {
  vertex_t v = partition_start_distr(local, dist);
  return (v + u);
}

local_index_t get_local_index_distr(vertex_t v,
                                    const arts_block_dist_t *const dist) {
  unsigned int n = get_owner_distr(v, dist);
  vertex_t base = partition_start_distr(n, dist);
  assert(base <= v);
  return (v - base);
}

arts_guid_t get_guid_for_vertex_distr(vertex_t v,
                                      const arts_block_dist_t *const dist) {
  unsigned int owner = get_owner_distr(v, dist);
  assert(owner < dist->num_blocks);
  return dist->graphGuid[owner];
}

arts_guid_t get_guid_for_partition_distr(const arts_block_dist_t *const dist,
                                         partition_t index) {
  return dist->graphGuid[index];
}
