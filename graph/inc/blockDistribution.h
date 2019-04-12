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

#ifndef ARTS_BLOCK_DISTRIBUTION_H
#define ARTS_BLOCK_DISTRIBUTION_H
#ifdef __cplusplus
extern "C" {
#endif
#include "graphDefs.h"

struct artsBlockDistribution {
  graph_sz_t num_vertices; // the complete number of vertices
  graph_sz_t num_edges; // the complete number of edges
  graph_sz_t block_sz;
  artsGuid_t * graphGuid;
};

typedef struct artsBlockDistribution arts_block_dist_t;

graph_sz_t getBlockSize(const arts_block_dist_t* const _dist);
graph_sz_t getNodeBlockSize(node_t _node, const arts_block_dist_t* const _dist);
void initBlockDistribution(arts_block_dist_t* _dist,
                           graph_sz_t _n,
                           graph_sz_t _m);
node_t getOwner(vertex v, const arts_block_dist_t* const _dist);
vertex nodeStart(node_t n, const arts_block_dist_t* const _dist);
vertex nodeEnd(node_t n, const arts_block_dist_t* const _dist);

local_index_t getLocalIndex(vertex v, const arts_block_dist_t* const _dist);

// Note : This is always for current rank
// TODO should remove local arg. Added this because linker was complaining
// about artsGlobalRankId. Too tired to debug the problem ....
vertex getVertexId(node_t local, local_index_t, const arts_block_dist_t* const _dist);

void initBlockDistributionWithCmdLineArgs(arts_block_dist_t* _dist,
                                          int argc, 
                                          char** argv);

void freeDistribution(arts_block_dist_t* _dist);

artsGuid_t* getGuidForVertex(vertex v,
                             const arts_block_dist_t* const _dist);

artsGuid_t* getGuidForCurrentNode(const 
                                  arts_block_dist_t* const _dist);
#ifdef __cplusplus
}
#endif

#endif


