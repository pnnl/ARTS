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

graph_sz_t getBlockSize(const arts_block_dist_t* _dist) {
  graph_sz_t rem = _dist->num_vertices % artsGetTotalNodes();
  if (!rem)
    return (_dist->num_vertices /  artsGetTotalNodes());
  else
    return ((graph_sz_t)(_dist->num_vertices /  artsGetTotalNodes()) + 1);
}


void initBlockDistribution(arts_block_dist_t* _dist,
                           graph_sz_t _n,
                           graph_sz_t _m) {
  _dist->num_vertices = _n;
  _dist->num_edges = _m;
  _dist->block_sz = getBlockSize(_dist);

  // copied from ssspStart.c
  _dist->graphGuid = artsMalloc(sizeof(artsGuid_t)*artsGetTotalNodes());
  for(unsigned int i=0; i<artsGetTotalNodes(); i++) {
    _dist->graphGuid[i] = artsReserveGuidRoute(ARTS_DB_PIN, i % artsGetTotalNodes());
  }
}

void initBlockDistributionWithCmdLineArgs(arts_block_dist_t* _dist,
                                          int argc, 
                                          char** argv) {
  uint64_t n = 0;
  uint64_t m = 0;

  for (int i=0; i < argc; ++i) {
    if (strcmp("--num-vertices", argv[i]) == 0) {
      sscanf(argv[i+1], "%" SCNu64, &n);
    }

    if (strcmp("--num-edges", argv[i]) == 0) {
      sscanf(argv[i+1], "%" SCNu64, &m);
    }
  }

  if(n && m)
  {
    PRINTF("[INFO] Initializing Block Distribution with following parameters ...\n");
    PRINTF("[INFO] Vertices : %" PRIu64 "\n", n);
    PRINTF("[INFO] Edges : %" PRIu64 "\n", m);

    _dist->num_vertices = n;
    _dist->num_edges = m;
    _dist->block_sz = getBlockSize(_dist);

    // copied from ssspStart.c
    _dist->graphGuid = artsMalloc(sizeof(artsGuid_t)*artsGetTotalNodes());
    for(unsigned int i=0; i<artsGetTotalNodes(); i++) {
      _dist->graphGuid[i] = artsReserveGuidRoute(ARTS_DB_PIN, i % artsGetTotalNodes());
    }
  }
  else
      PRINTF("Must set --num-vertices and --num-edges\n");
}

void freeDistribution(arts_block_dist_t* _dist) {
  artsFree(_dist->graphGuid);
  _dist->graphGuid = NULL;

  _dist->num_vertices = 0;
  _dist->num_edges = 0;
  _dist->block_sz = 0;
}

artsGuid_t* getGuidForVertex(vertex v,
                             const arts_block_dist_t* const _dist) {
  node_t owner = getOwner(v, _dist);
  assert(owner < artsGetTotalNodes());
  return &(_dist->graphGuid[owner]);
}

artsGuid_t* getGuidForCurrentNode(const arts_block_dist_t* const _dist) {
  return &(_dist->graphGuid[artsGetCurrentNode()]);
}

node_t getOwner(vertex v, const arts_block_dist_t* const _dist) {
  return (node_t)(v / _dist->block_sz);
}

vertex nodeStart(node_t n, const arts_block_dist_t* const _dist) {
  return (vertex)((_dist->block_sz) * n);
}

vertex nodeEnd(node_t n, const arts_block_dist_t* const _dist) {
  // is this the last node ?
  if (n == (artsGetTotalNodes()-1)) {
    return (vertex)(_dist->num_vertices - 1);
  } else {
    return (nodeStart(n, _dist) + (_dist->block_sz-1));
  }
}

graph_sz_t getNodeBlockSize(node_t n, const arts_block_dist_t* const _dist) {
  // is this the last node
  if (n == (artsGetTotalNodes()-1)) {
    return (_dist->num_vertices - ((artsGetTotalNodes()-1)*_dist->block_sz));
  } else
    return _dist->block_sz;
}

local_index_t getLocalIndex(vertex v, 
                            const arts_block_dist_t* const _dist) {
  node_t n = getOwner(v, _dist);
  vertex base = nodeStart(n, _dist);
  assert(base <= v);
  return (v - base);
}

vertex getVertexId(node_t local_rank,
                   local_index_t u, const arts_block_dist_t* const _dist) {
  vertex v = nodeStart(local_rank, _dist);
  return (v+u);
}
