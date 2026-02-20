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
#include "arts/edge_vector.h"

#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>

#include "arts/system/arts_print.h"
#include "arts/utils/malloc.h"

#define INCREASE_SZ_BY 2

// comparators
int compare_by_source(const void *e1, const void *e2) {
  edge_t *pe1 = (edge_t *)e1;
  edge_t *pe2 = (edge_t *)e2;

  if (pe1->source < pe2->source) {
    return -1;
  }
  if (pe1->source == pe2->source) {
    return 0;
  }
  return 1;
}

int compare_by_source_and_target(const void *e1, const void *e2) {
  edge_t *pe1 = (edge_t *)e1;
  edge_t *pe2 = (edge_t *)e2;

  if (pe1->source < pe2->source) {
    return -1;
  }
  if (pe1->source == pe2->source) {
    if (pe1->target < pe2->target) {
      return -1;
    }
    if (pe1->target == pe2->target) {
      return 0;
    }
    return 1;
  }
  return 1;
}

// end comparators

void init_edge_vector(arts_edge_vector_t *v, graph_sz_t initial_size) {
  v->edge_array = (edge_t *)arts_malloc(initial_size * sizeof(edge_t));
  v->used = 0;
  v->size = initial_size;
}

void push_back_edge(arts_edge_vector_t *v, vertex_t s, vertex_t t,
                    edge_data_t d) {
  if (v->used == v->size) {
    v->size *= INCREASE_SZ_BY;
    edge_t *new_edge_array =
        (edge_t *)arts_realloc(v->edge_array, v->size * sizeof(edge_t));
    if (!new_edge_array) {
      ARTS_INFO("[ERROR] Unable to reallocate memory. Cannot continue.");
      assert(false);
      return;
    }
    v->edge_array = new_edge_array;
  }

  v->edge_array[v->used].source = s;
  v->edge_array[v->used].target = t;
  v->edge_array[v->used++].data = d;
}

void print_edge_vector(const arts_edge_vector_t *v) {
  for (uint64_t i = 0; i < v->used; ++i) {
    ARTS_INFO("(%" PRIu64 ", %" PRIu64 ")", v->edge_array[i].source,
              v->edge_array[i].target);
  }
}

void free_edge_vector(arts_edge_vector_t *v) {
  arts_free(v->edge_array);
  v->edge_array = NULL;
  v->used = 0;
  v->size = 0;
}

void sort_by_source(arts_edge_vector_t *v) {
  qsort((void *)v->edge_array, v->used, sizeof(edge_t), compare_by_source);
}

void sort_by_source_and_target(arts_edge_vector_t *v) {
  qsort((void *)v->edge_array, v->used, sizeof(edge_t),
        compare_by_source_and_target);
}
