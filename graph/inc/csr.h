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

#include "graphDefs.h"
#include "blockDistribution.h"
#include "artsEdgeVector.h"

#define MAXCHAR 4096

typedef struct {
  graph_sz_t num_local_vertices;
  graph_sz_t num_local_edges;
  vertex* row_indices;
  vertex* columns;
  arts_block_dist_t* distribution;
  void* data;
} csr_graph;


void initCSR(csr_graph* _csr, 
             graph_sz_t _localv,
             graph_sz_t _locale,
             arts_block_dist_t* _dist,
             artsEdgeVector* _edges,
             bool _sorted_by_src);

int loadGraphNoWeight(const char* _file,
                      csr_graph* _graph,
                      arts_block_dist_t* _dist,
                      bool _flip,
                      bool _ignore_self_loops);

int loadGraphNoWeightCsr(const char* _file,
                        csr_graph* _graph,
                        arts_block_dist_t* _dist,
                        bool _flip,
                        bool _ignore_self_loops);

void printLocalCSR(const csr_graph* _csr);

int loadGraphUsingCmdLineArgs(csr_graph* _graph,
                              arts_block_dist_t* _dist,
                              int argc, char** argv);
void freeCSR(csr_graph* _csr);

void getNeighbors(csr_graph* _csr,
                  vertex v,
                  vertex** _out,
                  graph_sz_t* _neighborcount);
#ifdef __cplusplus
}
#endif

#endif
