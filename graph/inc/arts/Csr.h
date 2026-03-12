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

#ifndef HAGGLE_CSR_H
#define HAGGLE_CSR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/BlockDistribution.h"
#include "arts/EdgeVector.h"
#include "arts/GraphDefs.h"
#include "arts/runtime/RT.h"

#define MAXCHAR 1024 * 1024

typedef struct {
  artsGuid_t partGuid;
  graph_sz_t num_local_vertices;
  graph_sz_t num_local_edges;
  graph_sz_t block_sz;
  partition_t index;
  unsigned int num_blocks;
} csr_graph_t;

csr_graph_t *initCSR(partition_t partIndex, graph_sz_t _localv,
                     graph_sz_t _locale, arts_block_dist_t *_dist,
                     artsEdgeVector *_edges, bool _sorted_by_src,
                     artsGuid_t blockGuid);
int loadGraphNoWeight(const char *_file, arts_block_dist_t *_dist, bool _flip,
                      bool _ignore_self_loops);
int loadGraphNoWeightCsr(const char *_file, arts_block_dist_t *_dist,
                         bool _flip, bool _ignore_self_loops);
int loadGraphUsingCmdLineArgs(arts_block_dist_t *_dist, int argc, char **argv);
void freeCSR(csr_graph_t *_csr);
void printCSR(csr_graph_t *_csr);
void getNeighbors(csr_graph_t *_csr, vertex_t v, vertex_t **_out,
                  graph_sz_t *_neighborcount);
csr_graph_t *getGraphFromGuid(artsGuid_t guid);
csr_graph_t *getGraphFromPartition(partition_t partIndex,
                                   arts_block_dist_t *dist);
local_index_t getLocalIndexCSR(vertex_t v, const csr_graph_t *const part);
#ifdef __cplusplus
}
#endif

#endif
