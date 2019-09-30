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

#define DPRINTF(...)
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

arts_block_dist_t distribution;
csr_graph graph;
uint64_t* level;

void bfs_output() {
    DPRINTF("Printing vertex levels....\n");
    uint64_t i;
    for (i = 0; i < graph.num_local_vertices; ++i) {
        DPRINTF("Local vertex : %" PRIu64 ", Level : %" PRIu64 "\n", i, level[i]);
    }
}

void exitProgram(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    bfs_output();
    artsShutdown();
}

void bfs_send(vertex u, uint64_t ulevel);

void relax(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    DPRINTF("calling relax\n");
    assert(paramc == 2);
    vertex v = (vertex) paramv[0];
    uint64_t vlevel = paramv[1];

    local_index_t indexv = getLocalIndex(v, &distribution);
    assert(indexv < graph.num_local_vertices);

    uint64_t oldlevel = level[indexv];
    bool success = false;
    while (vlevel < oldlevel) {
        // NOTE : This call depends on GNU (GCC)
        success = __atomic_compare_exchange(&level[indexv],
                &oldlevel,
                &vlevel,
                false,
                __ATOMIC_RELAXED,
                __ATOMIC_RELAXED);
        oldlevel = level[indexv];
    }

    if (success) {
        // notify neighbors
        // get neighbors
        vertex* neighbors = NULL;
        uint64_t neighbor_cnt = 0;
        getNeighbors(&graph, v,
                &neighbors,
                &neighbor_cnt);

        // iterate over neighbors
        uint64_t neigbrlevel = level[indexv] + 1;
        for (uint64_t i = 0; i < neighbor_cnt; ++i) {
            vertex u = neighbors[i];

            // route message
            DPRINTF("sending u=%" PRIu64 ", level= %" PRIu64 "\n", u, neigbrlevel);
            bfs_send(u, neigbrlevel);
        }
    }
}

void bfs_send(vertex u, uint64_t ulevel) {
    artsGuid_t* neighbDbguid = getGuidForVertex(u, &distribution);
    uint64_t send[2];
    send[0] = u;
    send[1] = ulevel;
    artsGuid_t relaxGuid = artsActiveMessageWithDb(relax, // function 
            2, // number of parameters
            send, // parameters
            0, // additional deps
            (*neighbDbguid)); // this is the guid to co-locate task with
}

void kickoffTermination(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    PRINTF("Kick off\n");
    vertex source = (vertex) paramv[0];
    bfs_send(source, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {

//    int edge_arr[] = {
//    5,  6,
//    1,  2,
//    2,  5,
//    2,  3,
//    2,  4,
//    1,  6,
//    1,  3,
//    1,  7,
//    1,  4,
//    3,  5,
//    1, 5
//  };
//
//  initBlockDistribution(&distribution, 
//                        8, /*global vertices*/ 
//                        11); /*global edges*/
//    
//  // Create a list of edges, use artsEdgeVector
//  artsEdgeVector vec;
//  initEdgeVector(&vec, 100);
//  for(int i=0; i < 11; ++i) {
//    pushBackEdge(&vec, edge_arr[i*2], edge_arr[(i*2)+1], 0);
//  }
//  
//  initCSR(&graph, // graph structure
//          8, // number of "local" vertices
//          11, // number of "local" edges
//          &distribution, // distribution
//          &vec, // edges
//          false /*are edges sorted ?*/);
//
//  // Edge list not needed after creating the CSR
//  freeEdgeVector(&vec);
    
    
    
  // distribution must be initialized in initPerNode
  initBlockDistributionWithCmdLineArgs(&distribution, 
                                       argc, argv);
  // set-up the graph
  loadGraphUsingCmdLineArgs(&graph,
			    &distribution,
			    argc,
			    argv);

    // should probably encapsulate into something
    level = (uint64_t *) artsMalloc(graph.num_local_vertices * sizeof (uint64_t));
    // initialize the level array
    for (uint64_t i = 0; i < graph.num_local_vertices; ++i) {
        level[i] = UINT64_MAX;
    }
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) {

    if (!workerId) {
        // find the source vertex
        vertex source;
        for (int i = 0; i < argc; ++i) {
            if (strcmp("--source", argv[i]) == 0) {
                sscanf(argv[i + 1], "%" SCNu64, &source);
            }
        }

        assert(source < distribution.num_vertices);

        if (!nodeId) {
            artsGuid_t exitGuid = artsEdtCreate(exitProgram, 0, 0, NULL, 1);
            artsInitializeAndStartEpoch(exitGuid, 0);
            artsGuid_t startGuid = artsEdtCreate(kickoffTermination, 0, 1, (uint64_t*) &source, 0);
        }
    }
}

int main(int argc, char** argv) {
    artsRT(argc, argv);
    return 0;
}
