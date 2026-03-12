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
#include <stdlib.h>
#include <string.h>

#include "arts/Csr.h"
#include "arts/EdgeVector.h"
#include "arts/arts.h"
#include "arts/gas/RouteTable.h"
#include "arts/system/ArtsPrint.h"

vertex_t *getRowPtr(csr_graph_t *_csr) { return (vertex_t *)(_csr + 1); }

vertex_t *getColPtr(csr_graph_t *_csr) {
  return getRowPtr(_csr) + _csr->num_local_vertices + 1;
}

csr_graph_t *initCSR(partition_t partIndex, graph_sz_t _localv,
                     graph_sz_t _locale, arts_block_dist_t *_dist,
                     artsEdgeVector *_edges, bool _sorted_by_src,
                     artsGuid_t blockGuid) {
  // TODO: what will happen if partition does not have any vertex??
  csr_graph_t *_csr = NULL;
  if (artsIsGuidLocal(blockGuid)) {
    // data is a single array that merges row_indices and columns
    graph_sz_t totsz = (_localv + 1) + _locale;
    unsigned int dbSize = sizeof(csr_graph_t) + totsz * sizeof(vertex_t);

    _csr = (csr_graph_t *)artsDbCreateWithGuid(blockGuid, dbSize);
    _csr->partGuid = blockGuid;
    _csr->num_local_vertices = _localv;
    _csr->num_local_edges = _locale;
    _csr->block_sz = getBlockSizeForPartition(0, _dist);
    _csr->index = partIndex;
    _csr->num_blocks = _dist->num_blocks;

    vertex_t *row_indices = getRowPtr(_csr);
    vertex_t *columns = getColPtr(_csr);

    for (uint64_t i = 0; i <= _localv; ++i)
      row_indices[i] = 0;

    for (uint64_t i = 0; i < _locale; ++i)
      columns[i] = 0;

    if (!_sorted_by_src)
      sortBySource(_edges);

    vertex_t last_src = _edges->edge_array[0].source;
    vertex_t t = _edges->edge_array[0].target;
    vertex_t src_ind = getLocalIndexDistr(last_src, _dist);

    columns[0] = t;
    row_indices[src_ind] = 0;
    row_indices[src_ind + 1] = (_edges->used) ? 1 : 0;

    // populate edges
    for (uint64_t i = 1; i < _edges->used; ++i) {
      vertex_t s = _edges->edge_array[i].source;
      vertex_t t = _edges->edge_array[i].target;
      src_ind = getLocalIndexDistr(s, _dist);

      if (s == last_src) {
        // refers to previous source
        columns[i] = t;
        ++(row_indices[src_ind + 1]);
      } else {
        // if there are vertices without edges, those indexes need to be
        // set
        vertex_t last_src_ind = getLocalIndexDistr(last_src, _dist);
        ++last_src_ind;
        vertex_t val = row_indices[last_src_ind];
        assert(last_src_ind <= src_ind);
        while (last_src_ind != src_ind)
          row_indices[++last_src_ind] = val;

        // new source
        // assert(_csr->row_indices[src_ind] == i);
        // ARTS_INFO("src_ind = %" PRIu64 ", ",  src_ind);
        row_indices[src_ind + 1] = i + 1;
        columns[i] = t;
        last_src = s;
      }
    }

    // initialize until the end of the vertex_t array
    vertex_t last_src_ind = getLocalIndexDistr(last_src, _dist);
    ++last_src_ind;
    vertex_t val = row_indices[last_src_ind];
    if (last_src_ind > _localv) {
      ARTS_INFO("lasr_src index: %lu _localv: %lu", last_src_ind, _localv);
    }
    assert(last_src_ind <= _localv);
    while (last_src_ind < _localv)
      row_indices[++last_src_ind] = val;
  }
  return _csr;
}

void freeCSR(csr_graph_t *_csr) { artsDbDestroy(_csr->partGuid); }

unsigned int getOwnerCSR(vertex_t v, const csr_graph_t *const part) {
  return (unsigned int)(v / part->block_sz);
}

vertex_t indexStartCSR(unsigned int index, const csr_graph_t *const part) {
  return (vertex_t)((part->block_sz) * index);
}

vertex_t indexEndCSR(unsigned int index, const csr_graph_t *const part) {
  // is this the last node ?
  if (index == (part->num_blocks - 1)) {
    return (vertex_t)(part->num_local_vertices - 1);
  }
  return (indexStartCSR(index, part) + (part->block_sz - 1));
}

vertex_t partitionStartCSR(const csr_graph_t *const part) {
  return indexStartCSR(part->index, part);
}

vertex_t partitionEndCSR(const csr_graph_t *const part) {
  return indexEndCSR(part->index, part);
}

vertex_t getVertexFromLocalCSR(local_index_t u, const csr_graph_t *const part) {
  vertex_t v = partitionStartCSR(part);
  return (v + u);
}

local_index_t getLocalIndexCSR(vertex_t v, const csr_graph_t *const part) {
  vertex_t base = indexStartCSR(part->index, part);
  assert(base <= v);
  return (v - base);
}

void printCSR(csr_graph_t *_csr) {
  ARTS_INFO("=============================================");
  ARTS_INFO("[INFO] Number of local vertices : %" PRIu64 "",
            _csr->num_local_vertices);
  ARTS_INFO("[INFO] Number of local edges : %" PRIu64 "",
            _csr->num_local_edges);

  vertex_t *row_indices = getRowPtr(_csr);
  vertex_t *columns = getColPtr(_csr);

  uint64_t i, j;
  uint64_t nedges = 0;
  for (i = 0; i < _csr->num_local_vertices; ++i) {
    if (nedges == _csr->num_local_edges)
      break;

    vertex_t v = getVertexFromLocalCSR(i, _csr);

    for (j = row_indices[i]; j < row_indices[i + 1]; ++j) {
      vertex_t u = columns[j];
      ARTS_INFO("(%" PRIu64 ", %" PRIu64 ")", v, u);
      ++nedges;
    }
  }
  ARTS_INFO("=============================================");
}

void getNeighbors(csr_graph_t *_csr, vertex_t v, vertex_t **_out,
                  graph_sz_t *_neighborcount) {
  vertex_t *row_indices = getRowPtr(_csr);
  vertex_t *columns = getColPtr(_csr);
  // get the local index for the vertex
  local_index_t i = getLocalIndexCSR(v, _csr);
  // get the column start position
  graph_sz_t start = row_indices[i];
  graph_sz_t end = row_indices[i + 1];

  (*_out) = &(columns[start]);
  (*_neighborcount) = (end - start);
}

int loadGraphUsingCmdLineArgs(arts_block_dist_t *_dist, int argc, char **argv) {
  bool flip = false;
  bool keep_self_loops = false;
  bool csr_format = false;
  char *file = NULL;

  for (int i = 0; i < argc; ++i) {
    if (strcmp("--file", argv[i]) == 0)
      file = argv[i + 1];
    if (strcmp("--flip", argv[i]) == 0)
      flip = true;
    if (strcmp("--keep-self-loops", argv[i]) == 0)
      keep_self_loops = true;
    if (strcmp("--csr-format", argv[i]) == 0)
      csr_format = true;
  }

  ARTS_INFO("[INFO] Initializing GraphDB with following parameters ...");
  ARTS_INFO("[INFO] Graph file : %s", file);
  ARTS_INFO("[INFO] Flip ? : %d", flip);
  ARTS_INFO("[INFO] Keep Self-loops ? : %d", keep_self_loops);
  ARTS_INFO("[INFO] Csr-format : %d", csr_format);
  if (csr_format) {
    return loadGraphNoWeightCsr(file, _dist, flip, !keep_self_loops);
  }
  return loadGraphNoWeight(file, _dist, flip, !keep_self_loops);
}

// If we want to read the graph as an undirected graph set _flip = True
int loadGraphNoWeight(const char *_file, arts_block_dist_t *_dist, bool _flip,
                      bool _ignore_self_loops) {
  FILE *file = fopen(_file, "r");
  if (file == NULL) {
    ARTS_INFO("[ERROR] File cannot be opened -- %s", _file);
    return -1;
  }

  unsigned int numLocalParts = 0;
  unsigned int *partIndex;
  artsEdgeVector *vedges;

  for (unsigned int i = 0; i < _dist->num_blocks; i++) {
    if (artsIsGuidLocal(getGuidForPartitionDistr(_dist, i)))
      numLocalParts++;
  }

  partIndex = (unsigned int *)artsCalloc(numLocalParts, sizeof(unsigned int));
  vedges = (artsEdgeVector *)artsCalloc(numLocalParts, sizeof(artsEdgeVector));

  unsigned int j = 0;
  for (unsigned int i = 0; i < _dist->num_blocks; i++) {
    if (artsIsGuidLocal(getGuidForPartitionDistr(_dist, i))) {
      partIndex[j] = i;
      initEdgeVector(&vedges[j], EDGE_VEC_SZ);
      j++;
    }
  }

  char str[MAXCHAR];
  bool ignoreFirst = false;
  while (fgets(str, MAXCHAR, file) != NULL) {
    if (str[0] == '%') {
      ignoreFirst = true; // for mmio
      continue;
    }
    if (str[0] == '#')
      continue;
    if (ignoreFirst) {
      ignoreFirst = false;
      continue;
    }
    // We do not know how many edges we are going to load
    graph_sz_t src, target;
    edge_data_t weight;

    char *token = strtok(str, " \t");
    int i = 0;
    while (token != NULL) {
      if (i == 0) // Source
      {
        src = atoll(token);
        ++i;
      } else if (i == 1) // Target
      {
        target = atoll(token);
        i = 0;
      }

      // printf("src=%lu, target=%lu", src, target);
      token = strtok(NULL, " ");
    }

    if (_ignore_self_loops && (src == target))
      continue;

    // TODO weights
    // source belongs to current node
    unsigned int owner = getOwnerDistr(src, _dist);
    for (unsigned int k = 0; k < numLocalParts; k++) {
      if (owner == partIndex[k]) {
        // ARTS_INFO("src = %lu owner = %u start = %lu end = %lu", src, owner,
        // partitionStartDistr(owner, _dist), partitionEndDistr(owner, _dist));
        pushBackEdge(&vedges[k], src, target, 0 /*weight zeor for the moment*/);
      }
      /*else {
          printf("src = %" PRIu64 ", owner = %d, global rank : %d", src,
          getOwner(src, _dist),
          artsGetCurrentNode());
          assert(false); //TODO remove
      }*/
    }

    if (_flip) {
      owner = getOwnerDistr(target, _dist);
      for (unsigned int k = 0; k < numLocalParts; k++) {
        if (owner == partIndex[k])
          pushBackEdge(&vedges[k], target, src,
                       0 /*weight zeor for the moment*/);
      }
    }
  }

  fclose(file);

  for (unsigned int k = 0; k < numLocalParts; k++) {
    // done loading edge -- sort them by source
    sortBySource(&vedges[k]);
    // ARTS_INFO("getBlockSizeForPartition(partIndex[k], _dist): %lu,
    // vedges[k].used: %lu ", getBlockSizeForPartition(partIndex[k], _dist),
    // vedges[k].used);
    initCSR(partIndex[k], getBlockSizeForPartition(partIndex[k], _dist),
            vedges[k].used, _dist, &vedges[k], true,
            _dist->graphGuid[partIndex[k]]);

    freeEdgeVector(&vedges[k]);
  }
  return 0;
}

int loadGraphNoWeightCsr(const char *_file, arts_block_dist_t *_dist,
                         bool _flip, bool _ignore_self_loops) {
  FILE *file = fopen(_file, "r");
  if (file == NULL) {
    ARTS_INFO("[ERROR] File cannot be opened -- %s", _file);
    return -1;
  }

  uint64_t numVerts = 0;
  uint64_t numEdges = 0;

  unsigned int numLocalParts = 0;
  unsigned int *partIndex;
  artsEdgeVector *vedges;

  for (unsigned int i = 0; i < _dist->num_blocks; i++) {
    if (artsIsGuidLocal(getGuidForPartitionDistr(_dist, i)))
      numLocalParts++;
  }

  partIndex = (unsigned int *)artsCalloc(numLocalParts, sizeof(unsigned int));
  vedges = (artsEdgeVector *)artsCalloc(numLocalParts, sizeof(artsEdgeVector));

  unsigned int j = 0;
  for (unsigned int i = 0; i < _dist->num_blocks; i++) {
    if (artsIsGuidLocal(getGuidForPartitionDistr(_dist, i))) {
      partIndex[j] = i;
      initEdgeVector(&vedges[j], EDGE_VEC_SZ);
      j++;
    }
  }

  char str[MAXCHAR];
  if (fgets(str, MAXCHAR, file) != NULL) {
    char *token = strtok(str, " ");
    numVerts = atoll(token);
    token = strtok(NULL, " ");
    numEdges = atoll(token);
  }

  uint64_t localEdges = 0;
  uint64_t edgeCount = 0;
  graph_sz_t src = 0;
  while (fgets(str, MAXCHAR, file) != NULL) {
    if (str[0] == '%') {
      // ARTS_INFO("%%%%%%");
      continue;
    }

    if (str[0] == '#') {
      // ARTS_INFO("#######");
      continue;
    }

    char *token = strtok(str, " \t\\v\f\r");
    while (token != NULL) {
      graph_sz_t target = atoll(token) - 1;
      token = strtok(NULL, " \t\\v\f\r");

      if (_ignore_self_loops && (src == target)) {
        // ARTS_INFO("SELF LOOP");
        continue;
      }

      unsigned int owner = getOwnerDistr(src, _dist);
      for (unsigned int k = 0; k < numLocalParts; k++) {
        if (owner == partIndex[k]) {
          pushBackEdge(&vedges[k], src, target,
                       0 /*weight zeor for the moment*/);
          localEdges++;
        }
      }

      // if (_flip) {
      //     owner = getOwnerDistr(target, _dist);
      //     for(unsigned int k=0; k<numLocalParts; k++) {
      //         if (owner == partIndex[k]) {
      //             pushBackEdge(&vedges[k], target, src, 0/*weight zeor for
      //             the moment*/); localEdges++;
      //         }
      //     }
      // }

      edgeCount++;
    }
    src++;
  }
  fclose(file);

  if (src == numVerts) {
    for (unsigned int k = 0; k < numLocalParts; k++) {
      // ARTS_INFO("Sorting edges %lu local %lu vert %lu", edgeCount,
      // localEdges, src); done loading edge -- sort them by source
      sortBySource(&vedges[k]);

      initCSR(partIndex[k], getBlockSizeForPartition(partIndex[k], _dist),
              vedges[k].used, _dist, &vedges[k], true,
              _dist->graphGuid[partIndex[k]]);
      freeEdgeVector(&vedges[k]);
    }
  } else {
    ARTS_INFO("SRC: %lu != numVerts %lu.  Check the line length", src,
              numVerts);
  }

  return 0;
}

csr_graph_t *getGraphFromGuid(artsGuid_t guid) {
  struct artsDb *dbRes = (struct artsDb *)artsRouteTableLookupItem(guid);
  if (artsIsGuidLocal(guid) && dbRes)
    return (csr_graph_t *)(dbRes + 1);
  return NULL;
}

csr_graph_t *getGraphFromPartition(partition_t partIndex,
                                   arts_block_dist_t *dist) {
  return getGraphFromGuid(dist->graphGuid[partIndex]);
}
