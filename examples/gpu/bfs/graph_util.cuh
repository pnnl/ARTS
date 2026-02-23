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
#include <cuda_runtime_api.h>
#include <inttypes.h>

#include "arts/graph.h"

__device__ vertex_t *get_row_ptr_gpu(csr_graph_t *csr);
__device__ vertex_t *get_col_ptr_gpu(csr_graph_t *csr);
__device__ unsigned int get_owner_gpu(vertex_t v, const csr_graph_t *part);
__device__ vertex_t index_start_gpu(unsigned int index,
                                    const csr_graph_t *part);
__device__ vertex_t index_end_gpu(unsigned int index, const csr_graph_t *part);
__device__ vertex_t partition_start_gpu(const csr_graph_t *part);
__device__ vertex_t partition_end_gpu(const csr_graph_t *part);
__device__ vertex_t get_vertex_from_local_gpu(local_index_t u,
                                              const csr_graph_t *part);
__device__ local_index_t get_local_index_gpu(vertex_t v,
                                             const csr_graph_t *part);
__device__ void get_neighbors_gpu(csr_graph_t *csr, vertex_t v, vertex_t **out,
                                  graph_sz_t *neighborcount);

void get_properties(char *filename, unsigned int *num_verts,
                    unsigned int *num_edges);
