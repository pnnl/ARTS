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
#include "arts/arts.h"

#include <stdlib.h>
#include <string.h>

#include "arts/introspection/Metrics.h"
#include "arts/runtime/Globals.h"
#include "arts/system/Debug.h"
#include "arts/utils/Queue.h"

#define IS_POWER_OF_TWO(x) (!((x) & ((x) - 1)))

typedef struct header {
  size_t size;
  size_t align; // 0 if not aligned
  void *base;
} header_t;

static inline void *alignPointer(void *ptr, size_t align) {
  return (void *)(((uintptr_t)ptr + align - 1) & ~(align - 1));
}

void *artsMalloc(size_t size) {
  MALLOC_MEMORY_START();

  if (!size) {
    MALLOC_MEMORY_STOP();
    artsDebugGenerateSegFault();
  }

  header_t *base = (header_t *)malloc(size + sizeof(header_t));
  if (!base) {
    MALLOC_MEMORY_STOP();
    artsDebugGenerateSegFault();
  }
  INCREMENT_MEMORY_FOOTPRINT_BY(size);

  base->size = size;
  base->align = 0;
  base->base = base;

  if (artsThreadInfo.mallocTrace) {
    artsMetricsTriggerEvent(artsMallocBW, artsThread, size);
  }
  MALLOC_MEMORY_STOP();

  return base + 1;
}

void *artsMallocAlign(size_t size, size_t align) {
  MALLOC_MEMORY_START();

  if (!size || align < ALIGNMENT || !IS_POWER_OF_TWO(align)) {
    MALLOC_MEMORY_STOP();
    artsDebugGenerateSegFault();
  }

  void *base = malloc(size + align - 1 + sizeof(header_t));
  if (!base) {
    MALLOC_MEMORY_STOP();
    artsDebugGenerateSegFault();
  }
  INCREMENT_MEMORY_FOOTPRINT_BY(size);

  void *aligned = alignPointer((char *)base + sizeof(header_t), align);
  header_t *hdr = (header_t *)aligned - 1;

  hdr->size = size;
  hdr->align = align;
  hdr->base = base;

  if (artsThreadInfo.mallocTrace) {
    artsMetricsTriggerEvent(artsMallocBW, artsThread, size);
  }
  MALLOC_MEMORY_STOP();

  return aligned;
}

void *artsCalloc(size_t nmemb, size_t size) {
  CALLOC_MEMORY_START();

  if (!nmemb || !size || size > SIZE_MAX / nmemb) {
    CALLOC_MEMORY_STOP();
    artsDebugGenerateSegFault();
  }

  size_t totalSize = nmemb * size;
  void *ptr = artsMalloc(totalSize);
  memset(ptr, 0, totalSize);

  if (artsThreadInfo.mallocTrace) {
    artsMetricsTriggerEvent(artsMallocBW, artsThread, size);
  }
  CALLOC_MEMORY_STOP();

  return ptr;
}

void *artsCallocAlign(size_t nmemb, size_t size, size_t align) {
  CALLOC_MEMORY_START();

  if (!nmemb || !size || size > SIZE_MAX / nmemb || align < ALIGNMENT ||
      !IS_POWER_OF_TWO(align)) {
    CALLOC_MEMORY_STOP();
    artsDebugGenerateSegFault();
  }

  size_t totalSize = nmemb * size;
  void *ptr = artsMallocAlign(totalSize, align);
  memset(ptr, 0, totalSize);

  if (artsThreadInfo.mallocTrace) {
    artsMetricsTriggerEvent(artsMallocBW, artsThread, size);
  }
  CALLOC_MEMORY_STOP();

  return ptr;
}

void *artsRealloc(void *ptr, size_t size) {
  if (!ptr) {
    return artsMalloc(size);
  }
  if (!size) {
    artsFree(ptr);
    return NULL;
  }

  header_t *old_hdr = (header_t *)ptr - 1;
  size_t old_size = old_hdr->size;
  if (size <= old_size) {
    old_hdr->size = size;
    return ptr;
  }

  size_t align = old_hdr->align;

  void *new_ptr = align ? artsMallocAlign(size, align) : artsMalloc(size);
  memcpy(new_ptr, ptr, old_size);
  artsFree(ptr);
  return new_ptr;
}

void artsFree(void *ptr) {
  FREE_MEMORY_START();

  if (!ptr) {
    FREE_MEMORY_STOP();
    return;
  }
  header_t *hdr = (header_t *)ptr - 1;
  size_t size = hdr->size;
  free(hdr->base);
  DECREMENT_MEMORY_FOOTPRINT_BY(size);

  if (artsThreadInfo.mallocTrace) {
    artsMetricsTriggerEvent(artsFreeBW, artsThread, size);
  }
  FREE_MEMORY_STOP();
}
