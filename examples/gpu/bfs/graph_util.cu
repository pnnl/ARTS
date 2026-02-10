#include "graph_util.cuh"

#include <stdio.h>
#include <stdlib.h>

__device__ vertex_t *get_row_ptr_gpu(csr_graph_t *csr) {
  return (vertex_t *)(csr + 1);
}

__device__ vertex_t *get_col_ptr_gpu(csr_graph_t *csr) {
  return get_row_ptr_gpu(csr) + csr->num_local_vertices + 1;
}

__device__ unsigned int get_owner_gpu(vertex_t v, const csr_graph_t *const part) {
  return (unsigned int)(v / part->block_sz);
}

__device__ vertex_t index_start_gpu(unsigned int index,
                                    const csr_graph_t *const part) {
  return (vertex_t)((part->block_sz) * index);
}

__device__ vertex_t index_end_gpu(unsigned int index,
                                  const csr_graph_t *const part) {
  // is this the last node ?
  if (index == (part->num_blocks - 1)) {
    return (vertex_t)(part->num_local_vertices - 1);
  }
  return (index_start_gpu(index, part) + (part->block_sz - 1));
}

__device__ vertex_t partition_start_gpu(const csr_graph_t *const part) {
  return index_start_gpu(part->index, part);
}

__device__ vertex_t partition_end_gpu(const csr_graph_t *const part) {
  return index_end_gpu(part->index, part);
}

__device__ vertex_t get_vertex_from_local_gpu(local_index_t u,
                                              const csr_graph_t *const part) {
  vertex_t v = partition_start_gpu(part);
  return (v + u);
}
__device__ local_index_t get_local_index_gpu(vertex_t v,
                                             const csr_graph_t *const part) {
  vertex_t base = index_start_gpu(part->index, part);
  assert(base <= v);
  return (v - base);
}

__device__ void get_neighbors_gpu(csr_graph_t *csr, vertex_t v, vertex_t **out,
                                  graph_sz_t *neighborcount) {
  vertex_t *row_indices = get_row_ptr_gpu(csr);
  vertex_t *columns = get_col_ptr_gpu(csr);
  // get the local index for the vertex
  local_index_t i = get_local_index_gpu(v, csr);
  // get the column start position
  graph_sz_t start = row_indices[i];
  graph_sz_t end = row_indices[i + 1];

  (*out) = &(columns[start]);
  (*neighborcount) = (end - start);
}

void get_properties(char *filename, unsigned int *num_verts,
                    unsigned int *num_edges) {
  FILE *fp = fopen(filename, "r");
  if (fp) {
    char line[256];
    if (fgets(line, sizeof(line), fp) != NULL) {
      char *end = NULL;
      *num_verts = (unsigned int)strtoul(line, &end, 10);
      unsigned int non_zero = (unsigned int)strtoul(end, &end, 10);
      *num_edges = (unsigned int)strtoul(end, &end, 10);
      arts_printf("Verts: %u Edges: %u NonZero: %u\n", *num_verts, *num_edges,
                  non_zero);
    } else {
      arts_printf("FAILED TO PARSE %s\n", filename);
    }
    (void)fclose(fp);
  } else {
    arts_printf("FAILED TO OPEN %s\n", filename);
  }
}
