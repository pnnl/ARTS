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

typedef uint64_t vertex_t;
typedef uint64_t graph_sz_t;
typedef unsigned int partition_t;
typedef uint32_t edge_data_t;
typedef uint64_t local_index_t;

typedef struct {
  vertex_t source;
  vertex_t target;
  edge_data_t data;
} edge_t;

/* === Edge vector === */

#define EDGE_VEC_SZ 10000

typedef struct {
  edge_t *edge_array;
  graph_sz_t used;
  graph_sz_t size;
} arts_edge_vector_t;

void init_edge_vector(arts_edge_vector_t *v, graph_sz_t initial_size);
void push_back_edge(arts_edge_vector_t *v, vertex_t s, vertex_t t,
                    edge_data_t d);
void free_edge_vector(arts_edge_vector_t *v);
void sort_by_source(arts_edge_vector_t *v);
void sort_by_source_and_target(arts_edge_vector_t *v);
void print_edge_vector(const arts_edge_vector_t *v);

/* === Block distribution === */

typedef struct {
  graph_sz_t num_vertices;
  graph_sz_t num_edges;
  graph_sz_t block_sz;
  unsigned int num_blocks;
  arts_guid_t graphGuid[];
} arts_block_dist_t;

arts_block_dist_t *init_block_distribution_block(graph_sz_t n, graph_sz_t m,
                                                 unsigned int num_blocks,
                                                 arts_guid_kind_t db_type);
arts_block_dist_t *init_block_distribution(graph_sz_t n, graph_sz_t m);
arts_block_dist_t *init_block_distribution_with_cmd_line_args(int argc,
                                                              char **argv);
void free_distribution(arts_block_dist_t *dist);

unsigned int get_num_local_blocks(arts_block_dist_t *dist);

graph_sz_t get_block_size_for_partition(partition_t index,
                                        const arts_block_dist_t *dist);

partition_t get_owner_distr(vertex_t v, const arts_block_dist_t *dist);
vertex_t partition_start_distr(partition_t index,
                               const arts_block_dist_t *dist);
vertex_t partition_end_distr(partition_t index, const arts_block_dist_t *dist);
vertex_t get_vertex_from_local_distr(partition_t local, local_index_t u,
                                     const arts_block_dist_t *dist);
local_index_t get_local_index_distr(vertex_t v, const arts_block_dist_t *dist);

arts_guid_t get_guid_for_vertex_distr(vertex_t v,
                                      const arts_block_dist_t *dist);
arts_guid_t get_guid_for_partition_distr(const arts_block_dist_t *dist,
                                         partition_t index);

/* === CSR graph === */

#define MAXCHAR (1024 * 1024)

typedef struct {
  arts_guid_t partGuid;
  graph_sz_t num_local_vertices;
  graph_sz_t num_local_edges;
  graph_sz_t block_sz;
  partition_t index;
  unsigned int num_blocks;
} csr_graph_t;

csr_graph_t *init_csr(partition_t part_index, graph_sz_t localv,
                      graph_sz_t locale, arts_block_dist_t *dist,
                      arts_edge_vector_t *edges, bool sorted_by_src,
                      arts_guid_t block_guid);
int load_graph_no_weight(const char *file_path, arts_block_dist_t *dist,
                         bool flip, bool ignore_self_loops);
int load_graph_no_weight_csr(const char *file_path, arts_block_dist_t *dist,
                             bool flip, bool ignore_self_loops);
int load_graph_using_cmd_line_args(arts_block_dist_t *dist, int argc,
                                   char **argv);
void free_csr(csr_graph_t *csr);
void print_csr(csr_graph_t *csr);
void get_neighbors(csr_graph_t *csr, vertex_t v, vertex_t **out,
                   graph_sz_t *neighborcount);
csr_graph_t *get_graph_from_guid(arts_guid_t guid);
csr_graph_t *get_graph_from_partition(partition_t part_index,
                                      arts_block_dist_t *dist);
local_index_t get_local_index_csr(vertex_t v, const csr_graph_t *part);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_GRAPH_H */
