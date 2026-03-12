#include <assert.h>
#include <cuda_runtime_api.h>
#include <inttypes.h>

#include "arts/Csr.h"

__device__ vertex_t *getRowPtrGpu(csr_graph_t *_csr);
__device__ vertex_t *getColPtrGpu(csr_graph_t *_csr);
__device__ unsigned int getOwnerGpu(vertex_t v, const csr_graph_t *const part);
__device__ vertex_t indexStartGpu(unsigned int index,
                                  const csr_graph_t *const part);
__device__ vertex_t indexEndGpu(unsigned int index,
                                const csr_graph_t *const part);
__device__ vertex_t partitionStartGpu(const csr_graph_t *const part);
__device__ vertex_t partitionEndGpu(const csr_graph_t *const part);
__device__ vertex_t getVertexFromLocalGpu(local_index_t u,
                                          const csr_graph_t *const part);
__device__ local_index_t getLocalIndexGpu(vertex_t v,
                                          const csr_graph_t *const part);
__device__ void getNeighborsGpu(csr_graph_t *_csr, vertex_t v, vertex_t **_out,
                                graph_sz_t *_neighborcount);

void getProperties(char *filename, unsigned int *numVerts,
                   unsigned int *numEdges);