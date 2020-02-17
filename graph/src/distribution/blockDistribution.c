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
#include "blockDistribution.h"
#include <string.h>
#include <assert.h>
#include <inttypes.h>

void internalInitBlockDistribution(arts_block_dist_t* _dist, graph_sz_t _n, graph_sz_t _m, unsigned int numBlocks)
{
    _dist->num_vertices = _n;
    _dist->num_edges = _m;
    _dist->num_blocks = numBlocks;
    _dist->block_sz = _n / numBlocks;
}

arts_block_dist_t * initBlockDistributionBlock(graph_sz_t n, graph_sz_t m, unsigned int numBlocks, artsType_t dbType)
{
    arts_block_dist_t * dist = artsMalloc(sizeof(arts_block_dist_t) + sizeof(artsGuid_t)*numBlocks);
    unsigned int blocksPerNode = numBlocks / artsGetTotalNodes();
    unsigned int mod = numBlocks % artsGetTotalNodes();
    unsigned int current = 0;
    for(unsigned int i=0; i<artsGetTotalNodes(); i++) 
    {
        for(unsigned int j=0; j<blocksPerNode; j++) 
        {
            dist->graphGuid[current++] = artsReserveGuidRoute(dbType, i);
        }
        if(mod) 
        {
            dist->graphGuid[current++] = artsReserveGuidRoute(dbType, i);
            mod--;
        }
    }

    internalInitBlockDistribution(dist, n, m, numBlocks);
    return dist;
}

arts_block_dist_t * initBlockDistribution(graph_sz_t n, graph_sz_t m)
{
    unsigned int numBlocks = artsGetTotalNodes();
    arts_block_dist_t * dist = artsMalloc(sizeof(arts_block_dist_t) + sizeof(artsGuid_t) * numBlocks);
    for(unsigned int i=0; i<numBlocks; i++) 
        dist->graphGuid[i] = artsReserveGuidRoute(ARTS_DB_PIN, i);
    internalInitBlockDistribution(dist, n, m, numBlocks);
    return dist;
}

arts_block_dist_t * initBlockDistributionWithCmdLineArgs(int argc, char** argv)
{
    uint64_t n = 0;
    uint64_t m = 0;
    for (int i=0; i < argc; ++i) 
    {
        if (strcmp("--num-vertices", argv[i]) == 0) 
            sscanf(argv[i+1], "%" SCNu64, &n);
        if (strcmp("--num-edges", argv[i]) == 0)
            sscanf(argv[i+1], "%" SCNu64, &m);
    }

    if(n && m) 
    {
        unsigned int numBlocks = artsGetTotalNodes();
        arts_block_dist_t * dist = artsMalloc(sizeof(arts_block_dist_t) + sizeof(artsGuid_t) * numBlocks);
        for(unsigned int i=0; i<numBlocks; i++) 
            dist->graphGuid[i] = artsReserveGuidRoute(ARTS_DB_PIN, i);
        internalInitBlockDistribution(dist, n, m, numBlocks);
        return dist;
    }
    else
        PRINTF("Must set --num-vertices and --num-edges\n");
    return NULL;
}

void freeDistribution(arts_block_dist_t * dist)
{
    artsFree(dist);
}

unsigned int getNumLocalBlocks(arts_block_dist_t* _dist)
{
    unsigned int numLocalParts = 0;
    for(unsigned int i=0; i<_dist->num_blocks; i++) 
    {
        if(artsIsGuidLocal(getGuidForVertexDistr(i, _dist))) 
            numLocalParts++;
    }
    return numLocalParts;
}

graph_sz_t getBlockSizeForPartition(unsigned int index, const arts_block_dist_t* const _dist)
{
    // is this the last node
    if (index == (_dist->num_blocks-1))
        return (_dist->num_vertices - ((_dist->num_blocks-1)*_dist->block_sz));
    else
        return _dist->block_sz;
}

unsigned int getOwnerDistr(vertex_t v, const arts_block_dist_t* const _dist)
{
    return (unsigned int)(v / _dist->block_sz);
}

vertex_t partitionStartDistr(partition_t index, const arts_block_dist_t* const _dist)
{
    return (vertex_t)((_dist->block_sz) * index);
}

vertex_t partitionEndDistr(partition_t index, const arts_block_dist_t* const _dist)
{
    // is this the last node ?
    if (index == (_dist->num_blocks-1))
        return (vertex_t)(_dist->num_vertices - 1);
    else
        return (partitionStartDistr(index, _dist) + (_dist->block_sz-1));
}

vertex_t getVertexFromLocalDistr(unsigned int local, local_index_t u, const arts_block_dist_t* const _dist)
{
    vertex_t v = partitionStartDistr(local, _dist);
    return (v+u);
}

local_index_t getLocalIndexDistr(vertex_t v, const arts_block_dist_t* const _dist)
{
    unsigned int n = getOwnerDistr(v, _dist);
    vertex_t base = partitionStartDistr(n, _dist);
    assert(base <= v);
    return (v - base);
}

artsGuid_t getGuidForVertexDistr(vertex_t v, const arts_block_dist_t* const _dist)
{
    unsigned int owner = getOwnerDistr(v, _dist);
    assert(owner < _dist->num_blocks);
    return _dist->graphGuid[owner];
}

artsGuid_t getGuidForPartitionDistr(const arts_block_dist_t* const _dist, partition_t index)
{
    return _dist->graphGuid[index];
}
