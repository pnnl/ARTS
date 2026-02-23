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
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "arts.h"
#include "arts/block_distribution.h"
#include "arts/csr.h"
#include "arts/runtime/compute/shad_adapter.h"

arts_block_dist_t *distribution;
csr_graph_t *graph;
char *file = NULL;
arts_guid_t max_reducer_guid = NULL_GUID;

uint64_t start_time;
uint64_t end_time;

typedef struct {
  arts_guid_t find_intersection_guid;
  vertex_t source;
  unsigned int num_neighbors;
  vertex_t neighbors[];
} source_info_t;

typedef struct {
  vertex_t source;
  uint64_t scan_stat;
} per_vertex_scan_stat_t;

// int compare(const void * a, const void * b)
// {
//   return ( *(uint64_t*)a - *(uint64_t*)b );
// }

// arts_guid_t visit_source(uint32_t paramc, uint64_t * paramv, uint32_t depc,
// arts_edt_dep_t depv[]);

// arts_guid_t exit_program(uint32_t paramc, uint64_t * paramv, uint32_t depc,
// arts_edt_dep_t depv[]) {
//   arts_printf("Called exit\n");
//   arts_shutdown();
// }

void max_reducer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  // std::cout << "In max reducer" << std::endl;
  uint32_t max_scan_stat = 0;
  vertex_t max_vertex = 0;
  for (uint32_t v = 0; v < depc; v++) {
    per_vertex_scan_stat_t *vertex_scan_stat =
        (per_vertex_scan_stat_t *)depv[v].ptr;
    // std::cout << "Vertex: " << vertex_scan_stat->source << " scan_stat: " <<
    // vertex_scan_stat->scan_stat << std::endl;
    if (vertex_scan_stat->scan_stat > max_scan_stat) {
      max_scan_stat = vertex_scan_stat->scan_stat;
      max_vertex = vertex_scan_stat->source;
    }
  }
  std::cout << "Max vertex_t: " << max_vertex << " scan_stat: " << max_scan_stat
            << '\n';
  end_time = arts_get_time_stamp();
  arts_printf("Total execution time: %f s \n",
              (double)(end_time - start_time) / 1000000000.0);
  arts_stop_intro_shad();
  arts_shutdown();
}

void find_intersection(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  uint64_t sum = 0;
  per_vertex_scan_stat_t *local_intersection =
      (per_vertex_scan_stat_t *)depv[0].ptr;
  vertex_t source = local_intersection->source;

  for (uint64_t rank = 0; rank < depc; rank++) {
    per_vertex_scan_stat_t *local_intersection =
        (per_vertex_scan_stat_t *)depv[rank].ptr;
    // std::cout << "Source: " << source << " Rank: " << rank << "Scanstat: " <<
    // local_intersection->scan_stat << std::endl;
    sum += local_intersection->scan_stat;
  }

  vertex_t *neighbors = NULL;
  uint64_t neighbor_cnt = 0;
  get_neighbors(graph, source, &neighbors, &neighbor_cnt);

  sum += neighbor_cnt;

  unsigned int db_size = sizeof(per_vertex_scan_stat_t);
  void *ptr = NULL;
  arts_guid_t db_guid = arts_db_create(&ptr, db_size, ARTS_DB_DEFAULT, NULL);
  per_vertex_scan_stat_t *vertex_scan_stat = (per_vertex_scan_stat_t *)ptr;
  vertex_scan_stat->source = source;
  vertex_scan_stat->scan_stat = sum;
  // std::cout << "Source " << source << " ScanStat: " << sum << std::endl;
  arts_signal_edt(max_reducer_guid, source, db_guid, DB_MODE_EW);
}

void visit_one_hop_neighbor_on_rank(uint32_t paramc, const uint64_t *paramv,
                                    uint32_t depc, arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  (void)paramv;
  source_info_t *src_info = (source_info_t *)depv[0].ptr;
  vertex_t *one_hop_neighbor;
  vertex_t *immediate_neighbors = src_info->neighbors;
  std::vector<vertex_t> local_intersection;
  for (unsigned int i = 0; i < src_info->num_neighbors; i++) {
    vertex_t current_neighbor = src_info->neighbors[i];
    if (get_owner_distr(current_neighbor, distribution) ==
        arts_get_current_node()) {
      // std::cout << "Source " << src_info->source << " Current_neighbor: " <<
      // current_neighbor << std::endl;
      vertex_t *one_hop_neighbors = NULL;
      uint64_t neighbor_cnt = 0;
      get_neighbors(graph, current_neighbor, &one_hop_neighbors, &neighbor_cnt);
      for (unsigned int j = 0; j < neighbor_cnt; j++) {
        // std::cout << "One-hop neighbor for " <<  src_info->source << " is: "
        // << one_hop_neighbors[j] << std::endl;
      }
      std::set_intersection(immediate_neighbors,
                            immediate_neighbors + src_info->num_neighbors,
                            one_hop_neighbors, one_hop_neighbors + neighbor_cnt,
                            std::back_inserter(local_intersection));
    }
  }

  unsigned int db_size = sizeof(per_vertex_scan_stat_t);
  void *ptr = NULL;
  arts_guid_t db_guid = arts_db_create(&ptr, db_size, ARTS_DB_DEFAULT, NULL);
  per_vertex_scan_stat_t *vertex_scan_stat = (per_vertex_scan_stat_t *)ptr;
  vertex_scan_stat->source = src_info->source;
  vertex_scan_stat->scan_stat = local_intersection.size();
  // std::cout << "Source: " << src_info->source << " rank: "  <<
  // arts_get_current_node() << " set intersection size: " <<
  // local_intersection.size() <<std::endl;
  arts_signal_edt(src_info->find_intersection_guid, arts_get_current_node(),
                  db_guid, DB_MODE_EW);
}

void visit_source(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  vertex_t *neighbors = NULL;
  uint64_t neighbor_cnt = 0;
  vertex_t source = (vertex_t)paramv[0];
  // std::cout << "Visiting source: " << source <<std::endl;
  get_neighbors(graph, source, &neighbors, &neighbor_cnt);
  if (neighbor_cnt) {
    /*Now spawn an edt that will wait to get oneHopneighbors from all the ranks
     * in slots and calculate the grand count */
    arts_hint_t local_hint = {arts_get_current_node(), 0};
    arts_guid_t find_intersection_guid = arts_edt_create(
        find_intersection, 0, NULL, arts_get_total_nodes(), &local_hint);
    /*For each rank, now spawn an edt that will perform an intersection*/
    for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
      unsigned int db_size =
          sizeof(source_info_t) + (sizeof(vertex_t) * neighbor_cnt);
      void *ptr = NULL;
      arts_guid_t db_guid =
          arts_db_create(&ptr, db_size, ARTS_DB_DEFAULT, NULL);
      source_info_t *src_info = (source_info_t *)ptr;
      src_info->find_intersection_guid = find_intersection_guid;
      src_info->source = source;
      src_info->num_neighbors = neighbor_cnt;
      memcpy(&(src_info->neighbors), neighbors,
             sizeof(vertex_t) * neighbor_cnt);
      /*create the edt to find # one-hop neighbors*/
      arts_hint_t hop_hint = {i, 0};
      arts_guid_t visit_one_hop_neighbor_guid = arts_edt_create(
          visit_one_hop_neighbor_on_rank, 0, NULL, 1, &hop_hint);
      arts_signal_edt(visit_one_hop_neighbor_guid, 0, db_guid, DB_MODE_EW);
    }
  } else {
    /*signal maxreducer*/
    unsigned int db_size = sizeof(per_vertex_scan_stat_t);
    void *ptr = NULL;
    arts_guid_t db_guid = arts_db_create(&ptr, db_size, ARTS_DB_DEFAULT, NULL);
    per_vertex_scan_stat_t *vertex_scan_stat = (per_vertex_scan_stat_t *)ptr;
    vertex_scan_stat->source = source;
    vertex_scan_stat->scan_stat = 1;
    // std::cout << "signaling maxruducer for source " << source << std::endl;
    arts_signal_edt(max_reducer_guid, source, db_guid, DB_MODE_EW);
  }
}

extern "C" void init_node(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  unsigned int node_id = arts_get_current_node();

  arts_printf("Node %u argc %u\n", node_id, argc);
  distribution = init_block_distribution_with_cmd_line_args(argc, argv);
  graph = get_graph_from_partition(node_id, distribution);
  load_graph_using_cmd_line_args(distribution, argc, argv);
  max_reducer_guid = arts_guid_reserve(ARTS_EDT, 0);
}

extern "C" void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;

  // Initialize graph data on every node
  arts_guid_t init_epoch_guid = arts_initialize_and_start_epoch(NULL_GUID, 0);
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    arts_hint_t node_hint = {i, 0};
    arts_edt_create_with_epoch(init_node, paramc, paramv, 0, init_epoch_guid,
                               &node_hint);
  }
  arts_wait_on_handle(init_epoch_guid);

  // Create max reducer EDT
  arts_edt_create_with_guid(max_reducer, max_reducer_guid, 0, NULL,
                            distribution->num_vertices);
  arts_start_intro_shad(5);
  start_time = arts_get_time_stamp();
  for (uint64_t i = 0; i < distribution->num_vertices; ++i) {
    uint64_t source = i;
    partition_t rank = get_owner_distr(source, distribution);
    uint64_t packed_values[1] = {source};
    arts_hint_t rank_hint = {(unsigned int)rank, 0};
    arts_guid_t visit_source_guid = arts_edt_create(
        visit_source, 1, (uint64_t *)&packed_values, 1, &rank_hint);
    arts_signal_edt_value(visit_source_guid, -1, 0);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
