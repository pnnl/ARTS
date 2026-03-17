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
#include "arts/utils/malloc.h"

#include <stdlib.h>
#include <string.h>

#include "arts/defs.h"
#include "arts/system/print.h"
#ifdef ARTS_USE_CXL
#include "arts/cxl/wrapper.h"
#endif

#define ALIGNMENT 16
#define IS_POWER_OF_TWO(x) (!((x) & ((x) - 1)))

typedef struct ARTS_ALIGNED(16) header_s {
  size_t size;
  size_t align; // 0 if not aligned
  void *base;
} header_t;

static inline void *align_pointer(void *ptr, size_t align) {
  return (void *)(((uintptr_t)ptr + align - 1) & ~(align - 1));
}

void *arts_malloc(size_t size) {
  if (!size) {
    return NULL;
  }

  header_t *base = (header_t *)malloc(size + sizeof(header_t));
  if (!base) {
    ARTS_ERROR("arts_malloc: system malloc failed (size=%zu)", size);
  }
  INCREMENT_BYTES_MEMORY_FOOTPRINT_BY(size);

  base->size = size;
  base->align = 0;
  base->base = base;

  return base + 1;
}

void *arts_malloc_align(size_t size, size_t align) {
  if (!size || align < ALIGNMENT || !IS_POWER_OF_TWO(align)) {
    ARTS_ERROR("arts_malloc_align: invalid params (size=%zu, align=%zu)", size,
               align);
  }

  void *base = malloc(size + align - 1 + sizeof(header_t));
  if (!base) {
    ARTS_ERROR("arts_malloc_align: system malloc failed (size=%zu, align=%zu)",
               size, align);
  }
  INCREMENT_BYTES_MEMORY_FOOTPRINT_BY(size);

  void *aligned = align_pointer((char *)base + sizeof(header_t), align);
  header_t *hdr = (header_t *)aligned - 1;

  hdr->size = size;
  hdr->align = align;
  hdr->base = base;

  return aligned;
}

void *arts_calloc(size_t nmemb, size_t size) {
  if (!nmemb || !size) {
    return NULL;
  }
  if (size > SIZE_MAX / nmemb) {
    ARTS_ERROR("arts_calloc: overflow (nmemb=%zu, size=%zu)", nmemb, size);
    return NULL;
  }

  size_t total_size = nmemb * size;
  void *ptr = arts_malloc(total_size);
  memset(ptr, 0, total_size);

  return ptr;
}

void *arts_calloc_align(size_t nmemb, size_t size, size_t align) {
  if (!nmemb || !size) {
    return NULL;
  }
  if (size > SIZE_MAX / nmemb || align < ALIGNMENT || !IS_POWER_OF_TWO(align)) {
    ARTS_ERROR(
        "arts_calloc_align: invalid params (nmemb=%zu, size=%zu, align=%zu)",
        nmemb, size, align);
    return NULL;
  }

  size_t total_size = nmemb * size;
  void *ptr = arts_malloc_align(total_size, align);
  memset(ptr, 0, total_size);

  return ptr;
}

void *arts_realloc(void *ptr, size_t size) {
  if (!ptr) {
    return arts_malloc(size);
  }
  if (!size) {
    arts_free(ptr);
    return NULL;
  }

  header_t *old_hdr = (header_t *)ptr - 1;
  size_t old_size = old_hdr->size;
  if (size <= old_size) {
    old_hdr->size = size;
    return ptr;
  }

  size_t align = old_hdr->align;

  void *new_ptr = align ? arts_malloc_align(size, align) : arts_malloc(size);
  memcpy(new_ptr, ptr, old_size);
  arts_free(ptr);
  return new_ptr;
}

void arts_free(void *ptr) {
  if (!ptr) {
    return;
  }
#ifdef ARTS_USE_CXL
  if (IS_CXL_PTR(ptr)) {
    return; /* CXL arena-managed memory, not individually freeable */
  }
#endif
  header_t *hdr = (header_t *)ptr - 1;
  size_t size = hdr->size;
  free(hdr->base);
  DECREMENT_BYTES_MEMORY_FOOTPRINT_BY(size);
}
