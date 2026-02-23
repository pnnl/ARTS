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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"
#include "arts/graph.h"

arts_block_dist_t *distribution;
csr_graph_t *graph;
uint64_t *level;

void bfs_output() {
  arts_printf("Printing vertex levels....\n");
  uint64_t i;
  for (i = 0; i < graph->num_local_vertices; ++i) {
    arts_printf("Local vertex : %" PRIu64 ", Level : %" PRIu64 "\n", i,
                level[i]);
  }
}

void exit_program(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  bfs_output();
  free(level);
  arts_shutdown();
}

void bfs_send(vertex_t u, uint64_t ulevel);

void relax(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_printf("calling relax\n");
  assert(paramc == 2);
  vertex_t v = (vertex_t)paramv[0];
  uint64_t vlevel = paramv[1];

  local_index_t indexv = get_local_index_distr(v, distribution);
  assert(indexv < graph->num_local_vertices);

  uint64_t oldlevel = level[indexv];
  bool success = false;
  while (vlevel < oldlevel) {
    // NOTE : This call depends on GNU (GCC)
    success =
        __atomic_compare_exchange(&level[indexv], &oldlevel, &vlevel, false,
                                  __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    oldlevel = level[indexv];
  }

  if (success) {
    // notify neighbors
    // get neighbors
    vertex_t *neighbors = NULL;
    uint64_t neighbor_cnt = 0;
    get_neighbors(graph, v, &neighbors, &neighbor_cnt);

    // iterate over neighbors
    uint64_t neigbrlevel = level[indexv] + 1;
    for (uint64_t i = 0; i < neighbor_cnt; ++i) {
      vertex_t u = neighbors[i];

      // route message
      arts_printf("sending u=%" PRIu64 ", level= %" PRIu64 "\n", u,
                  neigbrlevel);
      bfs_send(u, neigbrlevel);
    }
  }
}

void bfs_send(vertex_t u, uint64_t ulevel) {
  arts_guid_t neighb_dbguid = get_guid_for_vertex_distr(u, distribution);
  uint64_t send[2];
  send[0] = u;
  send[1] = ulevel;
  arts_guid_t relax_guid = arts_edt_create(
      relax, 2, send, 1,
      &(arts_hint_t){.route = arts_guid_get_rank(neighb_dbguid)});
  arts_signal_edt(relax_guid, 0, neighb_dbguid, DB_MODE_EW);
}

void kickoff_termination(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  arts_printf("Kick off\n");
  vertex_t source = (vertex_t)paramv[0];
  bfs_send(source, 0);
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

  level = (uint64_t *)malloc(graph->num_local_vertices * sizeof(uint64_t));
  for (uint64_t i = 0; i < graph->num_local_vertices; ++i) {
    level[i] = UINT64_MAX;
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];

  // Initialize graph data on every node
  arts_guid_t init_epoch_guid = arts_initialize_and_start_epoch(NULL_GUID, 0);
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    arts_edt_create_with_epoch(init_node, paramc, paramv, 0, init_epoch_guid,
                               &(arts_hint_t){.route = i});
  }
  arts_wait_on_handle(init_epoch_guid);

  // Find the source vertex
  vertex_t source = 0;
  for (int i = 0; i < argc; ++i) {
    if (strcmp("--source", argv[i]) == 0) {
      source = (uint64_t)strtoull(argv[i + 1], NULL, 10);
    }
  }

  assert(source < distribution->num_vertices);

  arts_guid_t exit_guid =
      arts_edt_create(exit_program, 0, NULL, 1, &(arts_hint_t){.route = 0});
  arts_initialize_and_start_epoch(exit_guid, 0);
  arts_edt_create(kickoff_termination, 1, (uint64_t *)&source, 0,
                  &(arts_hint_t){.route = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
