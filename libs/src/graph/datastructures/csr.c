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
#include "arts/graph.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts/gas/route_table.h"
#include "arts/system/print.h"
#include "arts/utils/malloc.h"

vertex_t *get_row_ptr(csr_graph_t *csr) { return (vertex_t *)(csr + 1); }

vertex_t *get_col_ptr(csr_graph_t *csr) {
  return get_row_ptr(csr) + csr->num_local_vertices + 1;
}

csr_graph_t *init_csr(partition_t part_index, graph_sz_t localv,
                      graph_sz_t locale, arts_block_dist_t *dist,
                      arts_edge_vector_t *edges, bool sorted_by_src,
                      arts_guid_t block_guid) {
  // TODO: what will happen if partition does not have any vertex??
  csr_graph_t *csr = NULL;
  if (arts_guid_is_local(block_guid)) {
    // data is a single array that merges row_indices and columns
    graph_sz_t totsz = (localv + 1) + locale;
    unsigned int db_size = sizeof(csr_graph_t) + (totsz * sizeof(vertex_t));

    csr = (csr_graph_t *)arts_db_create_with_guid(
        block_guid, db_size, ARTS_DB_PIN, ARTS_DB_PROP_NONE, NULL);
    csr->partGuid = block_guid;
    csr->num_local_vertices = localv;
    csr->num_local_edges = locale;
    csr->block_sz = get_block_size_for_partition(0, dist);
    csr->index = part_index;
    csr->num_blocks = dist->num_blocks;

    vertex_t *row_indices = get_row_ptr(csr);
    vertex_t *columns = get_col_ptr(csr);

    for (uint64_t i = 0; i <= localv; ++i) {
      row_indices[i] = 0;
    }

    for (uint64_t i = 0; i < locale; ++i) {
      columns[i] = 0;
    }

    if (!sorted_by_src) {
      sort_by_source(edges);
    }

    vertex_t last_src = edges->edge_array[0].source;
    vertex_t t = edges->edge_array[0].target;
    vertex_t src_ind = get_local_index_distr(last_src, dist);

    columns[0] = t;
    row_indices[src_ind] = 0;
    row_indices[src_ind + 1] = (edges->used) ? 1 : 0;

    // populate edges
    for (uint64_t i = 1; i < edges->used; ++i) {
      vertex_t s = edges->edge_array[i].source;
      vertex_t t = edges->edge_array[i].target;
      src_ind = get_local_index_distr(s, dist);

      if (s == last_src) {
        // refers to previous source
        columns[i] = t;
        ++(row_indices[src_ind + 1]);
      } else {
        // if there are vertices without edges, those indexes need to be
        // set
        vertex_t last_src_ind = get_local_index_distr(last_src, dist);
        ++last_src_ind;
        vertex_t val = row_indices[last_src_ind];
        assert(last_src_ind <= src_ind);
        while (last_src_ind != src_ind) {
          row_indices[++last_src_ind] = val;
        }

        // new source
        // assert(csr->row_indices[src_ind] == i);
        // ARTS_INFO("src_ind = %" PRIu64 ", ",  src_ind);
        row_indices[src_ind + 1] = i + 1;
        columns[i] = t;
        last_src = s;
      }
    }

    // initialize until the end of the vertex_t array
    vertex_t last_src_ind = get_local_index_distr(last_src, dist);
    ++last_src_ind;
    vertex_t val = row_indices[last_src_ind];
    if (last_src_ind > localv) {
      ARTS_INFO("lasr_src index: %lu localv: %lu", last_src_ind, localv);
    }
    assert(last_src_ind <= localv);
    while (last_src_ind < localv) {
      row_indices[++last_src_ind] = val;
    }
  }
  return csr;
}

void free_csr(csr_graph_t *csr) { arts_db_destroy(csr->partGuid); }

unsigned int get_owner_csr(vertex_t v, const csr_graph_t *const part) {
  return (unsigned int)(v / part->block_sz);
}

vertex_t index_start_csr(unsigned int index, const csr_graph_t *const part) {
  return (vertex_t)((part->block_sz) * index);
}

vertex_t index_end_csr(unsigned int index, const csr_graph_t *const part) {
  // is this the last block/partition?
  if (index == (part->num_blocks - 1)) {
    return (vertex_t)(part->num_local_vertices - 1);
  }
  return (index_start_csr(index, part) + (part->block_sz - 1));
}

vertex_t partition_start_csr(const csr_graph_t *const part) {
  return index_start_csr(part->index, part);
}

vertex_t partition_end_csr(const csr_graph_t *const part) {
  return index_end_csr(part->index, part);
}

vertex_t get_vertex_from_local_csr(local_index_t u,
                                   const csr_graph_t *const part) {
  vertex_t v = partition_start_csr(part);
  return (v + u);
}

local_index_t get_local_index_csr(vertex_t v, const csr_graph_t *const part) {
  vertex_t base = index_start_csr(part->index, part);
  assert(base <= v);
  return (v - base);
}

void print_csr(csr_graph_t *csr) {
  ARTS_INFO("=============================================");
  ARTS_INFO("[INFO] Number of local vertices : %" PRIu64 "",
            csr->num_local_vertices);
  ARTS_INFO("[INFO] Number of local edges : %" PRIu64 "", csr->num_local_edges);

  vertex_t *row_indices = get_row_ptr(csr);
  vertex_t *columns = get_col_ptr(csr);

  uint64_t i;
  uint64_t j;
  uint64_t nedges = 0;
  for (i = 0; i < csr->num_local_vertices; ++i) {
    if (nedges == csr->num_local_edges) {
      break;
    }

    vertex_t v = get_vertex_from_local_csr(i, csr);

    for (j = row_indices[i]; j < row_indices[i + 1]; ++j) {
      vertex_t u = columns[j];
      ARTS_INFO("(%" PRIu64 ", %" PRIu64 ")", v, u);
      ++nedges;
    }
  }
  ARTS_INFO("=============================================");
}

void get_neighbors(csr_graph_t *csr, vertex_t v, vertex_t **out,
                   graph_sz_t *neighborcount) {
  vertex_t *row_indices = get_row_ptr(csr);
  vertex_t *columns = get_col_ptr(csr);
  // get the local index for the vertex
  local_index_t i = get_local_index_csr(v, csr);
  // get the column start position
  graph_sz_t start = row_indices[i];
  graph_sz_t end = row_indices[i + 1];

  (*out) = &(columns[start]);
  (*neighborcount) = (end - start);
}

int load_graph_using_cmd_line_args(arts_block_dist_t *dist, int argc,
                                   char **argv) {
  bool flip = false;
  bool keep_self_loops = false;
  bool csr_format = false;
  char *file = NULL;

  for (int i = 0; i < argc; ++i) {
    if (strcmp("--file", argv[i]) == 0) {
      file = argv[i + 1];
    }
    if (strcmp("--flip", argv[i]) == 0) {
      flip = true;
    }
    if (strcmp("--keep-self-loops", argv[i]) == 0) {
      keep_self_loops = true;
    }
    if (strcmp("--csr-format", argv[i]) == 0) {
      csr_format = true;
    }
  }

  ARTS_INFO("[INFO] Initializing GraphDB with following parameters ...");
  ARTS_INFO("[INFO] Graph file : %s", file);
  ARTS_INFO("[INFO] Flip ? : %d", flip);
  ARTS_INFO("[INFO] Keep Self-loops ? : %d", keep_self_loops);
  ARTS_INFO("[INFO] Csr-format : %d", csr_format);
  if (csr_format) {
    return load_graph_no_weight_csr(file, dist, flip, !keep_self_loops);
  }
  return load_graph_no_weight(file, dist, flip, !keep_self_loops);
}

// If we want to read the graph as an undirected graph set flip = True
int load_graph_no_weight(const char *file_path, arts_block_dist_t *dist,
                         bool flip, bool ignore_self_loops) {
  if (file_path == NULL) {
    ARTS_INFO("[ERROR] File path is NULL");
    return -1;
  }
  FILE *file = fopen(file_path, "r");
  if (file == NULL) {
    ARTS_INFO("[ERROR] File cannot be opened -- %s", file_path);
    return -1;
  }

  unsigned int num_local_parts = 0;
  unsigned int *part_index;
  arts_edge_vector_t *vedges;

  for (unsigned int i = 0; i < dist->num_blocks; i++) {
    if (arts_guid_is_local(get_guid_for_partition_distr(dist, i))) {
      num_local_parts++;
    }
  }

  part_index =
      (unsigned int *)arts_calloc(num_local_parts, sizeof(unsigned int));
  vedges = (arts_edge_vector_t *)arts_calloc(num_local_parts,
                                             sizeof(arts_edge_vector_t));

  unsigned int j = 0;
  for (unsigned int i = 0; i < dist->num_blocks; i++) {
    if (arts_guid_is_local(get_guid_for_partition_distr(dist, i))) {
      part_index[j] = i;
      init_edge_vector(&vedges[j], EDGE_VEC_SZ);
      j++;
    }
  }

  char str[MAXCHAR];
  bool ignore_first = false;
  while (fgets(str, MAXCHAR, file) != NULL) {
    if (str[0] == '%') {
      ignore_first = true; // for mmio
      continue;
    }
    if (str[0] == '#') {
      continue;
    }
    if (ignore_first) {
      ignore_first = false;
      continue;
    }
    // We do not know how many edges we are going to load
    graph_sz_t src = 0;
    graph_sz_t target = 0;
    edge_data_t weight;

    char *token = strtok(str, " \t");
    int i = 0;
    while (token != NULL) {
      if (i == 0) // Source
      {
        src = strtoll(token, NULL, 10);
        ++i;
      } else if (i == 1) // Target
      {
        target = strtoll(token, NULL, 10);
        i = 0;
      }

      // printf("src=%lu, target=%lu", src, target);
      token = strtok(NULL, " ");
    }

    if (ignore_self_loops && (src == target)) {
      continue;
    }

    // TODO weights
    // source belongs to current node
    unsigned int owner = get_owner_distr(src, dist);
    for (unsigned int k = 0; k < num_local_parts; k++) {
      if (owner == part_index[k]) {
        // ARTS_INFO("src = %lu owner = %u start = %lu end = %lu", src, owner,
        // partition_start_distr(owner, dist), partition_end_distr(owner,
        // dist));
        push_back_edge(&vedges[k], src, target,
                       0 /*weight zeor for the moment*/);
      }
      /*else {
          printf("src = %" PRIu64 ", owner = %d, global rank : %d", src,
          getOwner(src, dist),
          arts_get_current_rank());
          assert(false); //TODO remove
      }*/
    }

    if (flip) {
      owner = get_owner_distr(target, dist);
      for (unsigned int k = 0; k < num_local_parts; k++) {
        if (owner == part_index[k]) {
          push_back_edge(&vedges[k], target, src,
                         0 /*weight zeor for the moment*/);
        }
      }
    }
  }

  (void)fclose(file);

  for (unsigned int k = 0; k < num_local_parts; k++) {
    // done loading edge -- sort them by source
    sort_by_source(&vedges[k]);
    // ARTS_INFO("get_block_size_for_partition(part_index[k], dist): %lu,
    // vedges[k].used: %lu ", get_block_size_for_partition(part_index[k], dist),
    // vedges[k].used);
    init_csr(part_index[k], get_block_size_for_partition(part_index[k], dist),
             vedges[k].used, dist, &vedges[k], true,
             dist->graphGuid[part_index[k]]);

    free_edge_vector(&vedges[k]);
  }
  return 0;
}

int load_graph_no_weight_csr(const char *file_path, arts_block_dist_t *dist,
                             bool flip, bool ignore_self_loops) {
  (void)flip;
  if (file_path == NULL) {
    ARTS_INFO("[ERROR] File path is NULL");
    return -1;
  }
  FILE *file = fopen(file_path, "r");
  if (file == NULL) {
    ARTS_INFO("[ERROR] File cannot be opened -- %s", file_path);
    return -1;
  }

  uint64_t num_verts = 0;
  uint64_t num_edges = 0;

  unsigned int num_local_parts = 0;
  unsigned int *part_index;
  arts_edge_vector_t *vedges;

  for (unsigned int i = 0; i < dist->num_blocks; i++) {
    if (arts_guid_is_local(get_guid_for_partition_distr(dist, i))) {
      num_local_parts++;
    }
  }

  part_index =
      (unsigned int *)arts_calloc(num_local_parts, sizeof(unsigned int));
  vedges = (arts_edge_vector_t *)arts_calloc(num_local_parts,
                                             sizeof(arts_edge_vector_t));

  unsigned int j = 0;
  for (unsigned int i = 0; i < dist->num_blocks; i++) {
    if (arts_guid_is_local(get_guid_for_partition_distr(dist, i))) {
      part_index[j] = i;
      init_edge_vector(&vedges[j], EDGE_VEC_SZ);
      j++;
    }
  }

  char str[MAXCHAR];
  if (fgets(str, MAXCHAR, file) != NULL) {
    char *token = strtok(str, " ");
    num_verts = strtoll(token, NULL, 10);
    token = strtok(NULL, " ");
    num_edges = strtoll(token, NULL, 10);
  } else {
    (void)fclose(file);
    arts_free(part_index);
    arts_free(vedges);
    return -1;
  }

  uint64_t local_edges = 0;
  uint64_t edge_count = 0;
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
      graph_sz_t target = strtoll(token, NULL, 10) - 1;
      token = strtok(NULL, " \t\\v\f\r");

      if (ignore_self_loops && (src == target)) {
        // ARTS_INFO("SELF LOOP");
        continue;
      }

      unsigned int owner = get_owner_distr(src, dist);
      for (unsigned int k = 0; k < num_local_parts; k++) {
        if (owner == part_index[k]) {
          push_back_edge(&vedges[k], src, target,
                         0 /*weight zeor for the moment*/);
          local_edges++;
        }
      }

      // if (flip) {
      //     owner = get_owner_distr(target, dist);
      //     for(unsigned int k=0; k<num_local_parts; k++) {
      //         if (owner == part_index[k]) {
      //             push_back_edge(&vedges[k], target, src, 0/*weight zeor for
      //             the moment*/); local_edges++;
      //         }
      //     }
      // }

      edge_count++;
    }
    src++;
  }
  (void)fclose(file);

  if (src == num_verts) {
    for (unsigned int k = 0; k < num_local_parts; k++) {
      // ARTS_INFO("Sorting edges %lu local %lu vert %lu", edge_count,
      // local_edges, src); done loading edge -- sort them by source
      sort_by_source(&vedges[k]);

      init_csr(part_index[k], get_block_size_for_partition(part_index[k], dist),
               vedges[k].used, dist, &vedges[k], true,
               dist->graphGuid[part_index[k]]);
      free_edge_vector(&vedges[k]);
    }
  } else {
    ARTS_INFO("SRC: %lu != num_verts %lu.  Check the line length", src,
              num_verts);
  }

  return 0;
}

csr_graph_t *get_graph_from_guid(arts_guid_t guid) {
  arts_shared_ptr_t h = arts_route_table_lookup_db(guid);
  struct arts_db_s *db_res = (struct arts_db_s *)arts_shared_get(h);
  if (arts_guid_is_local(guid) && db_res) {
    /* Note: caller uses the returned pointer without holding the cb ref.
     * Safe because graph DBs are never destroyed during computation and
     * callers always access data within an EDT lifetime — the descriptor
     * outlives this call, so we drop the ref immediately. */
    csr_graph_t *out = (csr_graph_t *)(db_res + 1);
    arts_shared_release(&h);
    return out;
  }
  if (db_res != NULL) {
    arts_shared_release(&h);
  }
  return NULL;
}

csr_graph_t *get_graph_from_partition(partition_t part_index,
                                      arts_block_dist_t *dist) {
  return get_graph_from_guid(dist->graphGuid[part_index]);
}
