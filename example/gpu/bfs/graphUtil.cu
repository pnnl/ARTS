#include "graphUtil.cuh"

#include <stdio.h>

__device__ vertex_t *getRowPtrGpu(csr_graph_t *_csr) {
  return (vertex_t *)(_csr + 1);
}

__device__ vertex_t *getColPtrGpu(csr_graph_t *_csr) {
  return getRowPtrGpu(_csr) + _csr->num_local_vertices + 1;
}

__device__ unsigned int getOwnerGpu(vertex_t v, const csr_graph_t *const part) {
  return (unsigned int)(v / part->block_sz);
}

__device__ vertex_t indexStartGpu(unsigned int index,
                                  const csr_graph_t *const part) {
  return (vertex_t)((part->block_sz) * index);
}

__device__ vertex_t indexEndGpu(unsigned int index,
                                const csr_graph_t *const part) {
  // is this the last node ?
  if (index == (part->num_blocks - 1)) {
    return (vertex_t)(part->num_local_vertices - 1);
  }
  return (indexStartGpu(index, part) + (part->block_sz - 1));
}

__device__ vertex_t partitionStartGpu(const csr_graph_t *const part) {
  return indexStartGpu(part->index, part);
}

__device__ vertex_t partitionEndGpu(const csr_graph_t *const part) {
  return indexEndGpu(part->index, part);
}

__device__ vertex_t getVertexFromLocalGpu(local_index_t u,
                                          const csr_graph_t *const part) {
  vertex_t v = partitionStartGpu(part);
  return (v + u);
}
__device__ local_index_t getLocalIndexGpu(vertex_t v,
                                          const csr_graph_t *const part) {
  vertex_t base = indexStartGpu(part->index, part);
  assert(base <= v);
  return (v - base);
}

__device__ void getNeighborsGpu(csr_graph_t *_csr, vertex_t v, vertex_t **_out,
                                graph_sz_t *_neighborcount) {
  vertex_t *row_indices = getRowPtrGpu(_csr);
  vertex_t *columns = getColPtrGpu(_csr);
  // get the local index for the vertex
  local_index_t i = getLocalIndexGpu(v, _csr);
  // get the column start position
  graph_sz_t start = row_indices[i];
  graph_sz_t end = row_indices[i + 1];

  (*_out) = &(columns[start]);
  (*_neighborcount) = (end - start);
}

void getProperties(char *filename, unsigned int *numVerts,
                   unsigned int *numEdges) {
  FILE *fp = fopen(filename, "r");
  if (fp) {
    unsigned int nonZero = 0;
    fscanf(fp, "%u %u %u", numVerts, &nonZero, numEdges);
    PRINTF("Verts: %u Edges: %u NonZero: %u\n", *numVerts, *numEdges, nonZero);
    fclose(fp);
  } else
    PRINTF("FAILED TO OPEN %s\n", filename);
}