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

#ifndef ARTS_GRAPH_H
#define ARTS_GRAPH_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"
#include <stdbool.h>
#include <stdint.h>

/* === Graph primitive types === */

typedef uint64_t arts_vertex_t;
typedef uint64_t arts_graph_sz_t;
typedef unsigned int arts_partition_t;
typedef uint32_t arts_edge_data_t;
typedef uint64_t arts_local_index_t;

typedef struct {
  arts_vertex_t source;
  arts_vertex_t target;
  arts_edge_data_t data;
} arts_edge_t;

/* === Edge vector === */

#define ARTS_EDGE_VEC_SZ 10000

typedef struct {
  arts_edge_t *edge_array;
  arts_graph_sz_t used;
  arts_graph_sz_t size;
} arts_edge_vector_t;

void arts_edge_vector_init(arts_edge_vector_t *v, arts_graph_sz_t initial_size);
void arts_edge_vector_push_back(arts_edge_vector_t *v, arts_vertex_t s, arts_vertex_t t,
                    arts_edge_data_t d);
void arts_edge_vector_free(arts_edge_vector_t *v);
void arts_edge_vector_sort_by_source(arts_edge_vector_t *v);
void arts_edge_vector_sort_by_source_and_target(arts_edge_vector_t *v);

/* === Block distribution === */

typedef struct {
  arts_graph_sz_t num_vertices;
  arts_graph_sz_t num_edges;
  arts_graph_sz_t block_sz;
  unsigned int num_blocks;
  arts_guid_t graphGuid[];
} arts_block_dist_t;

arts_block_dist_t *arts_block_dist_init(arts_graph_sz_t n, arts_graph_sz_t m,
                                                 unsigned int num_blocks,
                                                 arts_guid_kind_t db_type);
arts_block_dist_t *arts_block_dist_init_from_args(int argc,
                                                              char **argv);
void arts_block_dist_free(arts_block_dist_t *dist);

arts_graph_sz_t arts_block_dist_block_size(arts_partition_t index,
                                        const arts_block_dist_t *dist);

arts_partition_t arts_block_dist_get_owner(arts_vertex_t v, const arts_block_dist_t *dist);
arts_vertex_t arts_block_dist_partition_start(arts_partition_t index,
                               const arts_block_dist_t *dist);
arts_vertex_t arts_block_dist_partition_end(arts_partition_t index, const arts_block_dist_t *dist);
arts_local_index_t arts_block_dist_get_local_index(arts_vertex_t v, const arts_block_dist_t *dist);

arts_guid_t arts_block_dist_guid_for_vertex(arts_vertex_t v,
                                      const arts_block_dist_t *dist);
arts_guid_t arts_block_dist_guid_for_partition(const arts_block_dist_t *dist,
                                         arts_partition_t index);

/* === CSR graph === */

#define ARTS_GRAPH_MAXCHAR (1024 * 1024)

typedef struct {
  arts_guid_t partGuid;
  arts_graph_sz_t num_local_vertices;
  arts_graph_sz_t num_local_edges;
  arts_graph_sz_t block_sz;
  arts_partition_t index;
  unsigned int num_blocks;
} arts_csr_graph_t;

arts_csr_graph_t *arts_csr_init(arts_partition_t part_index, arts_graph_sz_t localv,
                      arts_graph_sz_t locale, arts_block_dist_t *dist,
                      arts_edge_vector_t *edges, bool sorted_by_src,
                      arts_guid_t block_guid);
int arts_csr_load_no_weight(const char *file_path, arts_block_dist_t *dist,
                         bool flip, bool ignore_self_loops);
int arts_csr_load_no_weight_csr(const char *file_path, arts_block_dist_t *dist,
                             bool flip, bool ignore_self_loops);
int arts_csr_load_from_args(arts_block_dist_t *dist, int argc,
                                   char **argv);
void arts_csr_free(arts_csr_graph_t *csr);
void arts_csr_print(arts_csr_graph_t *csr);
void arts_csr_get_neighbors(arts_csr_graph_t *csr, arts_vertex_t v, arts_vertex_t **out,
                   arts_graph_sz_t *neighborcount);
arts_csr_graph_t *arts_csr_from_guid(arts_guid_t guid);
arts_csr_graph_t *arts_csr_from_partition(arts_partition_t part_index,
                                      arts_block_dist_t *dist);
arts_local_index_t arts_csr_get_local_index(arts_vertex_t v, const arts_csr_graph_t *part);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_GRAPH_H */
