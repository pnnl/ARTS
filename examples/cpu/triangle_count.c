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

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>

#include "arts/block_distribution.h"
#include "arts/csr.h"
#include "arts.h"
#include "arts/utils/atomics.h"

arts_block_dist_t *distribution;
csr_graph_t *graph;

arts_guid_t epoch_guid = NULL_GUID;
arts_guid_t start_reduce_guid = NULL_GUID;
arts_guid_t final_reduce_guid = NULL_GUID;

vertex_t dist_start = 0;
vertex_t dist_end = 0;
uint64_t block_size = 0;
uint64_t over_sub = 16;

uint64_t other_count = 0;
uint64_t local_triangle_count = 0;
uint64_t time = 0;

uint64_t local = 0;
uint64_t remote = 0;
uint64_t incoming = 0;

void final_reduce(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]);
void local_reduce(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]);
void start_reduce(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]);
void visit_vertex(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]);

// Only support up to 64 nodes
static inline unsigned int check_and_set(uint64_t *mask, unsigned int index) {
  uint64_t bit = 1 << index;
  if (((*mask) & bit) == 0) {
    (*mask) |= bit;
    return 1;
  }
  return 0;
}

void final_reduce(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  uint64_t count = 0;
  for (unsigned int i = 0; i < depc; i++) {
    count += (uint64_t)depv[i].guid;
  }
  time = arts_get_time_stamp() - time;
  arts_printf("Triangle Count: %lu Time: %lu\n", count, time);
  arts_shutdown();
}

void local_reduce(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_printf("Local: %lu Remote: %lu Incoming: %lu\n", local, remote, incoming);
  arts_signal_edt_value(final_reduce_guid, arts_get_current_node(),
                     local_triangle_count + other_count);
}

void start_reduce(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    arts_edt_create_dep(local_reduce, 0, NULL, 0, false, &(arts_hint_t){.route = i});
  }
}

static inline uint64_t lower_bound(vertex_t value, uint64_t start, uint64_t end,
                                  const vertex_t *edges) {
  while ((start < end) && (edges[start] < value)) {
    start++;
}
  return start;
}

static inline uint64_t upper_bound(vertex_t value, uint64_t start, uint64_t end,
                                  const vertex_t *edges) {
  while ((start < end) && (value < edges[end - 1])) {
    end--;
}
  return end;
}

static inline uint64_t count_triangles(const vertex_t *a, uint64_t a_start,
                                      uint64_t a_end, const vertex_t *b,
                                      uint64_t b_start, uint64_t b_end) {
  uint64_t count = 0;
  while ((a_start < a_end) && (b_start < b_end)) {
    if (a[a_start] < b[b_start]) {
      a_start++;
    } else if (a[a_start] > b[b_start]) {
      b_start++;
    } else {
      count++;
      a_start++;
      b_start++;
    }
  }
  return count;
}

static inline uint64_t process_vertex(vertex_t i, vertex_t *neighbors,
                                     uint64_t neighbor_count,
                                     uint64_t *visit_mask, uint64_t *proc_local,
                                     uint64_t *proc_remote) {
  uint64_t local_count = 0;

  uint64_t first_pred = lower_bound(i, 0, neighbor_count, neighbors);
  uint64_t last_pred = neighbor_count;
  //    arts_printf("%lu = %lu %lu\n", i, first_pred, last_pred);
  for (uint64_t next_pred = first_pred + 1; next_pred < last_pred; next_pred++) {
    vertex_t j = neighbors[next_pred];
    unsigned int owner = get_owner_distr(j, distribution);
    if (get_owner_distr(j, distribution) == arts_get_current_node()) {
      vertex_t *j_neighbors = NULL;
      uint64_t j_neighbor_count = 0;
      get_neighbors(graph, j, &j_neighbors, &j_neighbor_count);
      uint64_t first_succ = lower_bound(i, 0, j_neighbor_count, j_neighbors);
      uint64_t last_succ = upper_bound(j, 0, j_neighbor_count, j_neighbors);
      uint64_t temp = count_triangles(neighbors, first_pred, next_pred, j_neighbors,
                                     first_succ, last_succ);
      local_count += temp;
      //            arts_printf("%lu %lu -- %lu\n", i, j, temp);
      (*proc_local)++;
    } else if (check_and_set(visit_mask, owner)) {
      uint64_t args[3];
      args[0] = i;
      args[1] = i;
      args[2] = neighbor_count;
      arts_guid_t guid = arts_edt_create(visit_vertex, 3, args, 1, &(arts_hint_t){.route = owner});
      arts_signal_edt_ptr(guid, 0, neighbors, sizeof(vertex_t) * neighbor_count);
      (*proc_remote)++;
    }
  }
  return local_count;
}

void visit_vertex(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  uint64_t local_count = 0;

  vertex_t start = paramv[0];
  vertex_t end = paramv[1];

  vertex_t *neighbors = NULL;
  uint64_t neighbor_count = 0;

  uint64_t proc_local = 0;
  uint64_t proc_remote = 0;
  uint64_t proc_incoming = 0;

  if (depc) {
    neighbors = depv[0].ptr;
    neighbor_count = paramv[2];
    uint64_t visit_mask = (uint64_t)-1;
    local_count = process_vertex(start, neighbors, neighbor_count, &visit_mask,
                               &proc_incoming, &proc_remote);
    arts_atomic_add_u64(&other_count, local_count);
    arts_atomic_add_u64(&incoming, proc_incoming);
  } else {
    for (vertex_t i = start; i < end; i++) {
      uint64_t visit_mask = 0;
      get_neighbors(graph, i, &neighbors, &neighbor_count);
      //            arts_printf("Neighbors: %lu %lu\n", i, neighbor_count);
      local_count += process_vertex(i, neighbors, neighbor_count, &visit_mask,
                                  &proc_local, &proc_remote);
    }
    arts_atomic_add_u64(&local_triangle_count, local_count);
    arts_atomic_add_u64(&local, proc_local);
    arts_atomic_add_u64(&remote, proc_remote);
  }
  //    arts_printf("%lu : %lu\n", start, local_count);
}

void init_node(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  unsigned int node_id = arts_get_current_node();

  distribution = init_block_distribution_with_cmd_line_args(argc, argv);
  load_graph_using_cmd_line_args(distribution, argc, argv);
  graph = get_graph_from_partition(node_id, distribution);

  start_reduce_guid = arts_guid_reserve(ARTS_EDT, 0);
  final_reduce_guid = arts_guid_reserve(ARTS_EDT, 0);
  epoch_guid = arts_initialize_epoch(0, start_reduce_guid, 0);

  dist_start = partition_start_distr(node_id, distribution);
  dist_end = partition_end_distr(node_id, distribution);
  block_size = (partition_end_distr(node_id, distribution) -
               partition_start_distr(node_id, distribution)) /
              (arts_get_total_workers() * over_sub);
  if (!block_size) {
    block_size = 1;
  }
}

void start_work(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  unsigned int node_id = arts_get_current_node();

  arts_start_epoch(epoch_guid);

  uint64_t args[2];
  for (vertex_t i = dist_start; i <= dist_end; i += block_size) {
    args[0] = i;
    args[1] = (i + block_size < dist_end) ? i + block_size : dist_end;
    arts_edt_create(visit_vertex, 2, args, 0, &(arts_hint_t){.route = node_id});
  }
}

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;

  // Initialize graph data on every node
  arts_guid_t init_epoch_guid = arts_initialize_and_start_epoch(NULL_GUID, 0);
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    arts_edt_create_with_epoch(init_node, paramc, paramv, 0, init_epoch_guid, &(arts_hint_t){.route = i});
  }
  arts_wait_on_handle(init_epoch_guid);

  // Master creates reduce EDTs
  time = arts_get_time_stamp();
  arts_edt_create_with_guid(start_reduce, start_reduce_guid, 0, NULL, 1);
  arts_edt_create_with_guid(final_reduce, final_reduce_guid, 0, NULL,
                            arts_get_total_nodes());

  // Start work on every node
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    arts_edt_create(start_work, 0, NULL, 0, &(arts_hint_t){.route = i});
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
