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
#include <stdio.h>

#include "arts.h"
#include "arts/block_distribution.h"
#include "arts/csr.h"

void arts_main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  // Simple Graph, vertices = 8, edges = 11
  /**
   0 6
  1 2
  2 5
  2 3
  2 4
  1 6
  1 3
  1 7
  1 4
  3 5
  1 5
  **/

  int edge_arr[] = {5, 6, 1, 2, 2, 5, 2, 3, 2, 4, 1,
                    6, 1, 3, 1, 7, 1, 4, 3, 5, 1, 5};

  // Create a block distribution
  arts_block_dist_t *dist = init_block_distribution_block(8,  // global vertices
                                                          11, // global edges
                                                          1,  // partitions
                                                          ARTS_DB_PIN);

  // Create a list of edges, use arts_edge_vector_t
  arts_edge_vector_t vec;
  init_edge_vector(&vec, 100);
  for (int i = 0; i < 11; ++i) {
    push_back_edge(&vec, edge_arr[(ptrdiff_t)i * 2],
                   edge_arr[((ptrdiff_t)i * 2) + 1], 0);
  }
  sort_by_source_and_target(&vec);

  // Create the CSR graph, graphGuid is used to allocate
  // row indices and column array
  csr_graph_t *graph = init_csr(0,
                                8,    // number of "local" vertices
                                11,   // number of "local" edges
                                dist, // distribution
                                &vec, // edges
                                true, /*are edges sorted ?*/
                                get_guid_for_partition_distr(dist, 0));

  // Edge list not needed after creating the CSR
  free_edge_vector(&vec);

  print_csr(graph);

  vertex_t *neighbors = NULL;
  graph_sz_t nbrcnt = 0;
  get_neighbors(graph, (vertex_t)1, &neighbors, &nbrcnt);
  assert(nbrcnt == 6);

  arts_printf("Neighbors of 1 : {");
  for (graph_sz_t i = 0; i < nbrcnt; ++i) {
    printf("%" PRIu64 ", ", neighbors[i]);
  }
  printf("}\n");
  free_csr(graph);

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
