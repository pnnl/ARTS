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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"
#include "arts/array_db.h"
#include "arts/compute/shad.h"
#include "arts/graph.h"

unsigned int intro_start = 5;

arts_block_dist_t *distribution;
csr_graph_t *graph;
char *graph_file = NULL;
char *id_file = NULL;
arts_guid_t vertex_property_map_guid = NULL_GUID;
arts_guid_t vertex_id_map_guid = NULL_GUID;

uint64_t start_time;
uint64_t end_time;

/*Default values as in python code*/
int num_seeds = 25;
int num_steps = 1500;

int fixed_seed = -1;

typedef struct {
  vertex_t v;
  double propertyVal;
} vertex_property_t;

typedef struct {
  vertex_t v;
  vertex_t id;
} vertex_id_t;

typedef struct {
  vertex_t source;
  unsigned int step;
  unsigned int num_neighbors;
  vertex_t seed;
  // vertex_t neighbors[];
} source_info_t;

void visit_source(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]);

void exit_program(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  end_time = arts_get_time_stamp();
  arts_printf("Total execution time: %f s \n",
              (double)(end_time - start_time) / 1000000000.0);
  arts_stop_intro_shad();
  arts_shutdown();
}

void gather_neighbor_property_val(uint32_t paramc, const uint64_t *paramv,
                                  uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  source_info_t *src_info = (source_info_t *)depv[depc - 1].ptr;
  vertex_property_t *max_weighted_neighbor = (vertex_property_t *)depv[0].ptr;
  for (unsigned int i = 0; i < src_info->num_neighbors; i++) {
    vertex_property_t *data = (vertex_property_t *)depv[i].ptr;
    // TODO: For now, its inefficiently getting both v and id, could have
    // discarded v.
    vertex_id_t *v_id = (vertex_id_t *)depv[i + src_info->num_neighbors].ptr;
    /*For now, just printing in-place*/
    //    arts_printf("Seed: %u, Step: %u, Neighbor: %u, neibID: %llu Weight:
    //    %f, Visited: %d, Indicator computation: \n", src_info->seed, num_steps
    //    - src_info->step + 1, data->v,v_id->id, data->propertyVal,
    //    src_info->source
    //    == data->v ? 1 : 0);
    /*For now we are doing in-place max-weighted sampling for next source*/
    if (data->propertyVal > max_weighted_neighbor->propertyVal) {
      max_weighted_neighbor->v = data->v;
      max_weighted_neighbor->propertyVal = data->propertyVal;
    }
  }

  /*spawn next step*/
  if (src_info->step > 0) {
    vertex_t source = max_weighted_neighbor->v;
    partition_t rank = get_owner_distr(source, distribution);
    /*Spawn an edt at rank that is the owner of current seed vertex*/
    uint64_t packed_values[3] = {source, src_info->step - 1, src_info->seed};
    arts_guid_t visit_source_guid =
        arts_edt_create(visit_source, 3, (uint64_t *)&packed_values, 2,
                        &(arts_hint_t){.route = rank});
    //        arts_printf("New Edt: %lu Source is located on rank %d
    //        Guid:%lu\n", visit_source_guid, rank, vertex_property_map_guid);
    arts_signal_edt(visit_source_guid, 0, vertex_property_map_guid, DB_MODE_EW);
    arts_signal_edt(visit_source_guid, 1, vertex_id_map_guid, DB_MODE_EW);
  }
}

void visit_source(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)depc;
  (void)paramc;
  //  arts_start_intro_shad(intro_start);
  vertex_t *neighbors = NULL;
  uint64_t neighbor_cnt = 0;
  vertex_t source = (vertex_t)paramv[0];
  int n_steps = (int)paramv[1];
  vertex_t seed = (vertex_t)paramv[2];
  //    arts_printf("Current Source  %" PRIu64 "\n", source);

  get_neighbors(graph, source, &neighbors, &neighbor_cnt);
  if (neighbor_cnt) {
    unsigned int db_size =
        sizeof(source_info_t); // + neighbor_cnt * sizeof(vertex_t);
    void *ptr = NULL;
    arts_guid_t db_guid = arts_guid_reserve(ARTS_DB, 0);
    ptr = arts_db_create_with_guid(db_guid, db_size, ARTS_DB_LOCAL, NULL, NULL);
    source_info_t *src_info = (source_info_t *)ptr;
    src_info->source = source;
    src_info->step = n_steps;
    src_info->seed = seed;
    src_info->num_neighbors = neighbor_cnt;
    // arts_printf("Exploring from Source  %" PRIu64 " steps: %d with neighbors
    // %d\n", source, num_steps + 1 - n_steps, neighbor_cnt);
    // memcpy(&(src_info->neighbors), &neighbors, neighbor_cnt *
    // sizeof(vertex_t));
    /* //... keep filling in */
    arts_guid_t gather_neighbor_property_val_guid = arts_edt_create(
        gather_neighbor_property_val, 0, NULL, (2 * neighbor_cnt) + 1,
        &(arts_hint_t){.route = arts_get_current_node()});

    arts_signal_edt(gather_neighbor_property_val_guid, 2 * neighbor_cnt,
                    db_guid, DB_MODE_EW);

    arts_array_db_t *vertex_property_map = (arts_array_db_t *)depv[0].ptr;
    for (unsigned int i = 0; i < neighbor_cnt; i++) {
      vertex_t neib = neighbors[i];
      arts_get_from_array_db(gather_neighbor_property_val_guid, i,
                             vertex_property_map, neib);
    }

    arts_array_db_t *vertex_id_map = (arts_array_db_t *)depv[1].ptr;
    for (unsigned int i = 0; i < neighbor_cnt; i++) {
      vertex_t neib = neighbors[i];
      // arts_printf("Vertex=%llu indexing at %u \n", neib, neighbor_cnt + i);
      arts_get_from_array_db(gather_neighbor_property_val_guid,
                             neighbor_cnt + i, vertex_id_map, neib);
    }
  }
}

void check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  for (unsigned int i = 0; i < depc; i++) {
    vertex_property_t *data = (vertex_property_t *)depv[i].ptr;
    //        arts_printf("%d %f: %u\n", i, data->v, data->propertyVal);
  }

  arts_shutdown();
}

void end_vertex_id_map_read(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  (void)paramc;
  (void)paramv;
  arts_guid_t exit_guid =
      arts_edt_create(exit_program, 0, NULL, 1, &(arts_hint_t){.route = 0});
  arts_initialize_and_start_epoch(exit_guid, 0);

  uint64_t *seeds = (uint64_t *)calloc((size_t)num_seeds, sizeof(uint64_t));
  if (!seeds) {
    arts_shutdown();
    return;
  }

  /*A sanity check that the data is put in properly*/
  /* arts_guid_t edt_guid = arts_edt_create(check, 0, NULL,
   * distribution.num_vertices, &(arts_hint_t){.route = 0}); */
  /* for(unsigned int i = 0; i < distribution.num_vertices; i++) */
  /*   arts_get_from_array_db(edt_guid, i, vertexPropertymap, i); */

  /*Sample seeds*/
  if (fixed_seed > -1) {
    for (int i = 0; i < num_seeds; i++) {
      seeds[i] = (uint64_t)fixed_seed;
    }
  } else {
    for (int i = 0; i < num_seeds; i++) {
      seeds[i] = arc4random_uniform(distribution->num_vertices);
      //	arts_printf("Seed chosen %d,\n", seeds[i]);
    }
  }
  arts_start_intro_shad(intro_start);
  start_time = arts_get_time_stamp();
  /*Start walk from each seed in parallel*/
  for (int i = 0; i < num_seeds; i++) {
    vertex_t source = (vertex_t)seeds[i];
    partition_t rank = get_owner_distr(source, distribution);
    // arts_printf("Source is located on rank %d\n", rank);
    /*Spawn an edt at rank that is the owner of current seed vertex*/
    uint64_t packed_values[3] = {source, (uint64_t)num_steps, source};
    arts_guid_t visit_source_guid =
        arts_edt_create(visit_source, 3, (uint64_t *)&packed_values, 2,
                        &(arts_hint_t){.route = rank});
    // TODO: why pass vertexpropertguid as an argument?
    arts_signal_edt(visit_source_guid, 0, vertex_property_map_guid, DB_MODE_EW);

    arts_signal_edt(visit_source_guid, 1, vertex_id_map_guid, DB_MODE_EW);
  }
  free(seeds);
}

void end_vertex_property_read(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {

  (void)depc;

  (void)depv;

  (void)paramc;

  (void)paramv;

  /*Now read in the vertex ID map*/

  // Start an epoch to read in the ID value
  arts_guid_t end_vertex_id_map_read_epoch_guid = arts_edt_create(
      end_vertex_id_map_read, 0, NULL, 2, &(arts_hint_t){.route = 0});

  // TODO: Is the following line necessary ?
  // Signal the ID map guid
  arts_signal_edt(end_vertex_id_map_read_epoch_guid, 1, vertex_id_map_guid,
                  DB_MODE_EW);

  // Start the epoch
  arts_initialize_and_start_epoch(end_vertex_id_map_read_epoch_guid, 0);

  // Allocate vertex ID map and populate it from node 0
  arts_array_db_t *vertex_id_map = arts_new_array_db_with_guid(
      vertex_id_map_guid, sizeof(vertex_id_t), distribution->num_vertices);

  // Read in property file
  arts_printf("[INFO] Reading in and constructing the vertex id map ...\n");
  FILE *file = fopen(id_file, "r");
  arts_printf("File to be opened %s\n", id_file);
  if (file == NULL) {
    arts_printf("[ERROR] File containing vertex ids  can't be open -- %s",
                id_file);
    arts_shutdown();
    return;
  }

  arts_printf("Started reading the vertex ids file..\n");

  char str[MAXCHAR];
  uint64_t index = 0;
  while (fgets(str, MAXCHAR, file) != NULL) {
    graph_sz_t vertex = 0;
    graph_sz_t id = 0;
    char *token = strtok(str, "\t");
    int i = 0;
    while (token != NULL) {
      if (i == 0) { // vertex
        vertex = strtoll(token, NULL, 10);
        // arts_printf("Vertex=%llu ", vertex);
        ++i;
      } else if (i == 1) { // id
        id = strtoll(token, NULL, 10);
        // arts_printf("id=%llu\n", id);
        i = 0;
      }
      token = strtok(NULL, " ");
    }
    vertex_id_t v_id_info = {.v = vertex, .id = id};

    arts_put_in_array_db(&v_id_info, NULL_GUID, 0, vertex_id_map, index);
    index++;
  }
  (void)fclose(file);
}

void init_node(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  unsigned int node_id = arts_get_current_node();

  vertex_property_map_guid = arts_guid_reserve(ARTS_DB, 0);
  vertex_id_map_guid = arts_guid_reserve(ARTS_DB, 0);

  distribution = init_block_distribution_with_cmd_line_args(argc, argv);
  load_graph_using_cmd_line_args(distribution, argc, argv);
  graph = get_graph_from_partition(node_id, distribution);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];

  // Initialize graph data on every node
  arts_guid_t init_epoch_guid = arts_initialize_and_start_epoch(NULL_GUID, 0);
  for (unsigned int i = 0; i < arts_get_total_nodes(); i++) {
    arts_edt_create_with_epoch(init_node, paramc, paramv, 0, init_epoch_guid,
                               &(arts_hint_t){.route = i});
  }
  arts_wait_on_handle(init_epoch_guid);

  // Parse command line arguments
  for (int i = 0; i < argc; ++i) {
    if (strcmp("--propertyfile", argv[i]) == 0) {
      graph_file = argv[i + 1];
    }
  }

  for (int i = 0; i < argc; ++i) {
    if (strcmp("--num-seeds", argv[i]) == 0) {
      num_seeds = (int)strtol(argv[i + 1], NULL, 10);
    }
  }

  for (int i = 0; i < argc; ++i) {
    if (strcmp("--num-steps", argv[i]) == 0) {
      num_steps = (int)strtol(argv[i + 1], NULL, 10);
    }
  }

  for (int i = 0; i < argc; ++i) {
    if (strcmp("--idfile", argv[i]) == 0) {
      id_file = argv[i + 1];
    }
  }

  // Start an epoch to read in the property value
  arts_guid_t end_vertex_property_read_epoch_guid = arts_edt_create(
      end_vertex_property_read, 0, NULL, 2, &(arts_hint_t){.route = 0});

  // Signal the property map guid
  arts_signal_edt(end_vertex_property_read_epoch_guid, 1,
                  vertex_property_map_guid, DB_MODE_EW);

  // Start the epoch
  arts_initialize_and_start_epoch(end_vertex_property_read_epoch_guid, 0);

  // Allocate vertex property map and populate it from node 0
  arts_array_db_t *vertex_property_map = arts_new_array_db_with_guid(
      vertex_property_map_guid, sizeof(vertex_property_t),
      distribution->num_vertices);

  // Read in property file
  arts_printf(
      "[INFO] Reading in and constructing the vertex property map ...\n");
  FILE *file = fopen(graph_file, "r");
  arts_printf("File to be opened %s\n", graph_file);
  if (file == NULL) {
    arts_printf("[ERROR] File containing property value can't be open -- %s",
                graph_file);
    arts_shutdown();
    return;
  }

  arts_printf("Started reading the vertex property file..\n");
  char str[MAXCHAR];
  uint64_t index = 0;
  while (fgets(str, MAXCHAR, file) != NULL) {
    graph_sz_t vertex = 0;
    double v_property_val = 0.0;
    char *token = strtok(str, "\t");
    int i = 0;
    while (token != NULL) {
      if (i == 0) { // vertex
        vertex = strtoll(token, NULL, 10);
        ++i;
      } else if (i == 1) { // property
        v_property_val = strtod(token, NULL);
        i = 0;
      }
      token = strtok(NULL, " ");
    }
    vertex_property_t v_prop_val = {.v = vertex, .propertyVal = v_property_val};

    arts_put_in_array_db(&v_prop_val, NULL_GUID, 0, vertex_property_map, index);
    index++;
  }
  (void)fclose(file);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
