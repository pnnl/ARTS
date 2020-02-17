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
#include "arts.h"

typedef struct {
  graph_sz_t num_vertices; //Complete number of vertices
  graph_sz_t num_edges; //Complete number of edges
  graph_sz_t block_sz; //Standard block size
  unsigned int num_blocks; //Total number of blocks
  artsGuid_t graphGuid[]; //Guids for all the partitions
} arts_block_dist_t;

arts_block_dist_t * initBlockDistributionBlock(graph_sz_t n, graph_sz_t m, unsigned int numBlocks, artsType_t dbType);
arts_block_dist_t * initBlockDistribution(graph_sz_t n, graph_sz_t m);
arts_block_dist_t * initBlockDistributionWithCmdLineArgs(int argc, char** argv);
void freeDistribution(arts_block_dist_t* _dist);

unsigned int getNumLocalBlocks(arts_block_dist_t* _dist);

graph_sz_t getBlockSizeForPartition(partition_t index, const arts_block_dist_t* const _dist);

partition_t getOwnerDistr(vertex_t v, const arts_block_dist_t* const _dist);
vertex_t partitionStartDistr(partition_t index, const arts_block_dist_t* const _dist);
vertex_t partitionEndDistr(partition_t index, const arts_block_dist_t* const _dist);
vertex_t getVertexFromLocalDistr(partition_t local, local_index_t u, const arts_block_dist_t* const _dist);
local_index_t getLocalIndexDistr(vertex_t v, const arts_block_dist_t* const _dist);

artsGuid_t getGuidForVertexDistr(vertex_t v, const arts_block_dist_t* const _dist);
artsGuid_t getGuidForPartitionDistr(const arts_block_dist_t* const _dist, partition_t index);

#ifdef __cplusplus
}
#endif

#endif
