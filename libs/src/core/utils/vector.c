/* SPDX-License-Identifier: Apache-2.0 */
#include "arts/utils/vector.h"

#include "arts/utils/malloc.h"

#include <string.h>

#define ARTS_VECTOR_MIN_CAPACITY 8u

void arts_vector_init(arts_vector_t *v, size_t element_size, uint64_t initial) {
  v->data = NULL;
  v->element_size = element_size;
  v->count = 0;
  v->capacity = (initial != 0) ? initial : ARTS_VECTOR_MIN_CAPACITY;
}

void arts_vector_free(arts_vector_t *v) {
  if (v->data != NULL) {
    arts_free(v->data);
    v->data = NULL;
  }
  v->count = 0;
}

void arts_vector_push(arts_vector_t *v, const void *element) {
  if (v->data == NULL) {
    if (v->capacity == 0) {
      v->capacity = ARTS_VECTOR_MIN_CAPACITY;
    }
    v->data = arts_malloc(v->element_size * v->capacity);
  } else if (v->count == v->capacity) {
    /* Double rather than grow by a constant: a run of appends then costs
     * amortised O(1) each, and the copies total one pass over the data. */
    uint64_t next = v->capacity * 2u;
    void *bigger = arts_malloc(v->element_size * next);
    memcpy(bigger, v->data, (size_t)(v->element_size * v->count));
    arts_free(v->data);
    v->data = bigger;
    v->capacity = next;
  }
  memcpy((char *)v->data + (size_t)(v->element_size * v->count), element,
         v->element_size);
  v->count++;
}

void *arts_vector_at(const arts_vector_t *v, uint64_t index) {
  if (index >= v->count) {
    return NULL;
  }
  return (char *)v->data + (size_t)(v->element_size * index);
}

void arts_vector_swap_remove(arts_vector_t *v, uint64_t index) {
  if (index >= v->count) {
    return;
  }
  v->count--;
  if (index != v->count) {
    memcpy((char *)v->data + (size_t)(v->element_size * index),
           (char *)v->data + (size_t)(v->element_size * v->count),
           v->element_size);
  }
}
