#include <assert.h>
#include <cuda_runtime_api.h>
#include <inttypes.h>

#include "arts/csr.h"

__device__ vertex_t *get_row_ptr_gpu(csr_graph_t *csr);
__device__ vertex_t *get_col_ptr_gpu(csr_graph_t *csr);
__device__ unsigned int get_owner_gpu(vertex_t v, const csr_graph_t *part);
__device__ vertex_t index_start_gpu(unsigned int index,
                                    const csr_graph_t *part);
__device__ vertex_t index_end_gpu(unsigned int index,
                                  const csr_graph_t *part);
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
