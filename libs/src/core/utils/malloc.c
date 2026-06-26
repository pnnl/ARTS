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
/* Thin wrappers over the active allocator (mimalloc when ARTS_MALLOC_MIMALLOC,
 * else the C library).  Validation is delegated to the allocator wherever it
 * already does it: overflow (nmemb*size) and bad alignment (zero / not a power
 * of two) both surface as a NULL return, which the wrappers turn into a fatal
 * error.  The only ARTS-level guards are the size==0 -> NULL contract and a
 * uniform minimum alignment, so over-alignment behaves identically regardless
 * of which allocator the build selected.
 *
 * No per-allocation bookkeeping header is stored: the allocator already tracks
 * every block's size internally (it must, for free to work), so the
 * memory-footprint counter reads that size back with the allocator's own
 * usable-size query rather than duplicating it.  Aligned allocations use the
 * allocator's native aligned entry points, which return a directly-freeable
 * pointer — no manual over-allocate-and-offset, no stored base.  Consequences:
 * the footprint counts the allocator's usable size (>= the requested size — the
 * true resident cost, rounding included), and arts_realloc yields only the base
 * allocator alignment (an over-aligned block must not be grown through realloc;
 * nothing in the tree does). */
#include "arts/utils/malloc.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "arts/defs.h"
#include "arts/system/print.h"
#ifdef ARTS_USE_CXL
#include "arts/cxl/wrapper.h"
#endif

#ifdef ARTS_MALLOC_MIMALLOC
#include <mimalloc.h>
#define ARTS_SYS_MALLOC(sz) mi_malloc(sz)
#define ARTS_SYS_CALLOC(n, sz) mi_calloc((n), (sz))
#define ARTS_SYS_REALLOC(p, sz) mi_realloc((p), (sz))
#define ARTS_SYS_FREE(p) mi_free(p)
#define ARTS_SYS_USABLE(p) mi_usable_size(p)
#else
#include <malloc.h> /* malloc_usable_size */
#define ARTS_SYS_MALLOC(sz) malloc(sz)
#define ARTS_SYS_CALLOC(n, sz) calloc((n), (sz))
#define ARTS_SYS_REALLOC(p, sz) realloc((p), (sz))
#define ARTS_SYS_FREE(p) free(p)
#define ARTS_SYS_USABLE(p) malloc_usable_size(p)
#endif

/* Uniform minimum (and base) alignment.  Every power-of-two >= ALIGNMENT is
 * also a multiple of sizeof(void*), which is exactly posix_memalign's extra
 * requirement, so the two allocator backends accept the same alignment set. */
#define ALIGNMENT 16

/* Native aligned allocation returning a directly-freeable pointer.  Returns
 * NULL for an invalid (zero / non-power-of-two) alignment — the allocator
 * validates it. */
static inline void *sys_malloc_aligned(size_t size, size_t align) {
#ifdef ARTS_MALLOC_MIMALLOC
  return mi_malloc_aligned(size, align);
#else
  void *p = NULL;
  return posix_memalign(&p, align, size) == 0 ? p : NULL;
#endif
}

/* Native aligned + zeroed allocation.  mimalloc has a native entry that zeroes,
 * overflow-checks, and validates the alignment; the C library has none, so fall
 * back to aligned + memset with an explicit product-overflow guard. */
static inline void *sys_calloc_aligned(size_t nmemb, size_t size,
                                       size_t align) {
#ifdef ARTS_MALLOC_MIMALLOC
  return mi_calloc_aligned(nmemb, size, align);
#else
  if (size > SIZE_MAX / nmemb) {
    return NULL;
  }
  size_t total = nmemb * size;
  void *p = NULL;
  if (posix_memalign(&p, align, total) != 0) {
    return NULL;
  }
  memset(p, 0, total);
  return p;
#endif
}

void *arts_malloc(size_t size) {
  if (!size) {
    return NULL;
  }
  void *p = ARTS_SYS_MALLOC(size);
  if (!p) {
    ARTS_ERROR("arts_malloc: out of memory (size=%zu)", size);
  }
  INCREMENT_BYTES_MEMORY_FOOTPRINT_BY(ARTS_SYS_USABLE(p));
  return p;
}

void *arts_malloc_aligned(size_t size, size_t align) {
  if (!size) {
    return NULL;
  }
  if (align < ALIGNMENT) {
    ARTS_ERROR("arts_malloc_aligned: align %zu below minimum %d", align,
               ALIGNMENT);
  }
  void *p = sys_malloc_aligned(size, align);
  if (!p) {
    ARTS_ERROR("arts_malloc_aligned: bad alignment or out of memory "
               "(size=%zu, align=%zu)",
               size, align);
  }
  INCREMENT_BYTES_MEMORY_FOOTPRINT_BY(ARTS_SYS_USABLE(p));
  return p;
}

void *arts_calloc(size_t nmemb, size_t size) {
  if (!nmemb || !size) {
    return NULL;
  }
  void *p =
      ARTS_SYS_CALLOC(nmemb, size); /* zeroes + overflow-checks natively */
  if (!p) {
    ARTS_ERROR("arts_calloc: out of memory or overflow (nmemb=%zu, size=%zu)",
               nmemb, size);
  }
  INCREMENT_BYTES_MEMORY_FOOTPRINT_BY(ARTS_SYS_USABLE(p));
  return p;
}

void *arts_calloc_aligned(size_t nmemb, size_t size, size_t align) {
  if (!nmemb || !size) {
    return NULL;
  }
  if (align < ALIGNMENT) {
    ARTS_ERROR("arts_calloc_aligned: align %zu below minimum %d", align,
               ALIGNMENT);
  }
  void *p = sys_calloc_aligned(nmemb, size, align);
  if (!p) {
    ARTS_ERROR("arts_calloc_aligned: bad alignment, overflow, or out of memory "
               "(nmemb=%zu, size=%zu, align=%zu)",
               nmemb, size, align);
  }
  INCREMENT_BYTES_MEMORY_FOOTPRINT_BY(ARTS_SYS_USABLE(p));
  return p;
}

void *arts_realloc(void *ptr, size_t size) {
  if (!ptr) {
    return arts_malloc(size);
  }
  if (!size) {
    arts_free(ptr);
    return NULL;
  }
  size_t old_usable = ARTS_SYS_USABLE(ptr); /* query before realloc frees it */
  void *p = ARTS_SYS_REALLOC(ptr, size);
  if (!p) {
    ARTS_ERROR("arts_realloc: out of memory (size=%zu)", size);
  }
  DECREMENT_BYTES_MEMORY_FOOTPRINT_BY(old_usable);
  INCREMENT_BYTES_MEMORY_FOOTPRINT_BY(ARTS_SYS_USABLE(p));
  return p;
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
  DECREMENT_BYTES_MEMORY_FOOTPRINT_BY(ARTS_SYS_USABLE(ptr));
  ARTS_SYS_FREE(ptr);
}
