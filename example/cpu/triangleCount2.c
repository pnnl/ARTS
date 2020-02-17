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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "arts.h"
#include "artsGraph.h"
#include "artsTerminationDetection.h"
#include "artsAtomics.h"

arts_block_dist_t distribution;
csr_graph_t graph;

artsGuid_t epochGuid    = NULL_GUID;
artsGuid_t startReduceGuid = NULL_GUID;
artsGuid_t finalReduceGuid = NULL_GUID;

uint64_t localTriangleCount = 0;
uint64_t time = 0;

uint64_t blockSize = 0;
uint64_t numBlocks = 0;

//Only support up to 64 nodes
unsigned int checkAndSet(uint64_t * mask, unsigned int index) {
    uint64_t bit = 1 << index;
    if(((*mask) & bit) == 0) {
        (*mask)|=bit;
        return 1;
    }
    return 0;
}

void finalReduce(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    uint64_t count = 0;
    for (unsigned int i = 0; i < depc; i++) {
        count += (uint64_t)depv[i].guid;
    }
    time = artsGetTimeStamp() - time;
    PRINTF("Triangle Count: %lu Time: %lu\n", count, time);
    artsShutdown();
}

void localReduce(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
//    PRINTF("Local Count: %lu Signal: %lu\n", localTriangleCount, finalEdtGuid);
    artsSignalEdtValue(finalReduceGuid, artsGetCurrentNode(), localTriangleCount);
}

void startReduce(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
//    PRINTF("Local Count: %lu Signal: %lu\n", localTriangleCount, finalEdtGuid);
    for(unsigned int i=0; i<artsGetTotalNodes(); i++) {
        artsEdtCreateDep(localReduce, i, 0, NULL, 0, false);
    }
}

uint64_t lowerBound(vertex_t value, uint64_t start, uint64_t end, vertex_t * edges) {
    while ((start < end) && (edges[start] < value))
        start++;
    return start;
}

uint64_t upperBound(vertex_t value, uint64_t start, uint64_t end, vertex_t * edges) {
    while ((start < end) && (value < edges[end - 1]))
        end--;
    return end;
}

uint64_t count_triangles(vertex_t * a, uint64_t a_start, uint64_t a_end, vertex_t * b, uint64_t b_start, uint64_t b_end) {
    uint64_t count = 0;
    while ((a_start < a_end) && (b_start < b_end)) {
        if (a[a_start] < b[b_start])
            a_start++;
        else if (a[a_start] > b[b_start])
            b_start++;
        else {
            count++;
            a_start++;
            b_start++;
        }
    }
    return count;
}

uint64_t processBlock(uint64_t index) {
    vertex_t * neighbors = NULL;
    uint64_t neighborCount = 0;
    uint64_t localCount = 0;
    
    uint64_t iStart = index*blockSize;
    uint64_t iEnd   = (index+1 == numBlocks) ? nodeEnd(artsGetCurrentNode(), &distribution) : iStart + blockSize;
    
    for (vertex_t i=iStart; i<iEnd; i++) {
        
        getNeighbors(&graph, i, &neighbors, &neighborCount);
        
        uint64_t firstPred = lowerBound(i, 0, neighborCount, neighbors);
        uint64_t lastPred = neighborCount;
        
        for (uint64_t nextPred = firstPred + 1; nextPred < lastPred; nextPred++) {
            vertex_t j = neighbors[nextPred];
            unsigned int owner = getOwner(j, &distribution);
            if (getOwner(j, &distribution) == artsGetCurrentNode()) {
                vertex_t * jNeighbors = NULL;
                uint64_t jNeighborCount = 0;
                getNeighbors(&graph, j, &jNeighbors, &jNeighborCount);
                uint64_t firstSucc = lowerBound(i, 0, jNeighborCount, jNeighbors);
                uint64_t lastSucc = upperBound(j, 0, jNeighborCount, jNeighbors);
                localCount += count_triangles(neighbors, firstPred, nextPred, jNeighbors, firstSucc, lastSucc);
            }
        }
    }
    return localCount;
}

void visitNode(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    uint64_t localCount = 0;
    uint64_t index = paramv[0];
    
    localCount += processBlock(index);
    uint64_t nextIndex = (numBlocks - 1) - index;
    if(nextIndex != index) {
        localCount += processBlock(nextIndex);
    }
    artsAtomicAddU64(&localTriangleCount, localCount);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {
    initBlockDistributionWithCmdLineArgs(&distribution, argc, argv);
    loadGraphUsingCmdLineArgs(&graph, &distribution, argc, argv);

    startReduceGuid = artsReserveGuidRoute(ARTS_EDT,   0);
    finalReduceGuid = artsReserveGuidRoute(ARTS_EDT,   0);
    epochGuid       = artsInitializeEpoch(0, startReduceGuid, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) {
    if(!nodeId && !workerId) {
        time = artsGetTimeStamp();
        artsEdtCreateWithGuid(startReduce, startReduceGuid, 0, NULL, 1);
        artsEdtCreateWithGuid(finalReduce, finalReduceGuid, 0, NULL, artsGetTotalNodes());
    }
    
    artsStartEpoch(epochGuid);
    vertex_t start = nodeStart(nodeId, &distribution);
    vertex_t end   = nodeEnd(nodeId, &distribution);
    
    uint64_t size = end - start;
    blockSize = size / (artsGetTotalWorkers() * 32 * 2);
    numBlocks = size / blockSize;
    if(size % blockSize)
        numBlocks++;
    
    uint64_t half = numBlocks / 2;
    if(numBlocks % 2)
        half++;
    
    for (uint64_t index = 0; index < half; index++) {
        if(index % artsGetTotalWorkers() == workerId) {
            artsEdtCreate(visitNode, nodeId, 1, &index, 0);
        }
    }
}

int main(int argc, char** argv) {
    artsRT(argc, argv);
    return 0;
}
