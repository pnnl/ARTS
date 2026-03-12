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
#ifndef ARTS_CXL_DEQUE_H
#define ARTS_CXL_DEQUE_H
#ifdef __cplusplus
extern "C" {
#endif

// #include "sharedAlloc.h"
#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>

#include "arts/cxl/Lock.h"
#include "arts/cxl/Wrapper.h"

extern unsigned int artsGlobalRankCount;
extern unsigned int artsGlobalRankId;

// #define DPRINTF( ... ) printf( __VA_ARGS__ )
#define DPRINTF(...)
// #define DPRINTF(...) \
    do { \
        printf(__VA_ARGS__); \
        fflush(stdout); \
    } while (0)
// #define DPRINTF(...)                                                       \
    do {                                                                   \
        FILE *fp__ = get_dprintf_file();                                   \
        if (fp__) {                                                        \
            fprintf(fp__, "[TID %lu] ", (unsigned long)pthread_self());    \
            fprintf(fp__, __VA_ARGS__);                                    \
            fflush(fp__);                                                  \
        }                                                                  \
    } while (0)

#define MEM_DEQ_LENGTH 1000000
// #define CHUNK_SIZE 80 // sizeof(struct artsEdt)
// #define CHUNK_SIZE 104
#define CHUNK_SIZE 1024
// #define NUM_NODES 2
#define NUM_NODES artsGlobalRankCount
#define PROFILE 0

typedef union CXLArena {
  struct {
    char *base;
    char *head;
    char *max_size;
    bool initialized;
  };
  uint8_t pad[(sizeof(char *) * 3 + sizeof(bool) + CACHELINE_SIZE - 1) /
              CACHELINE_SIZE * (CACHELINE_SIZE / sizeof(uint8_t))];
} CXLArena;

// Memory metadata
typedef struct memchunk {
  uint64_t size;
  uint64_t node;
  char mem[CHUNK_SIZE];
} memchunk;

typedef union CacheLine_memchunk {
  memchunk value;
  uint64_t pad[((sizeof(memchunk) + CACHELINE_SIZE - 1) / CACHELINE_SIZE) *
               (CACHELINE_SIZE / sizeof(uint64_t))];
} CacheLine_memchunk;

typedef union CacheLine_ptr {
  struct base {
    char *ptr;
    size_t size;
  } base;
  uint8_t pad[((sizeof(struct base) + CACHELINE_SIZE - 1) / CACHELINE_SIZE) *
              (CACHELINE_SIZE / sizeof(uint8_t))];
} CacheLine_ptr;

typedef union dequeConsts {
  struct {
    int maxSize;
    CXLArena *memArena;
    CXLArena *dbArena;
    TournamentLock_t *lock;
  };
  uint8_t pad[(sizeof(int) + sizeof(CXLArena *) * 2 +
               sizeof(TournamentLock_t *) + CACHELINE_SIZE - 1) /
              CACHELINE_SIZE * (CACHELINE_SIZE / sizeof(uint8_t))];
} dequeConsts;

typedef union dequeIndices {
  struct {
    int frontIdx;
    int backIdx;
  };
  uint8_t pad[(sizeof(int) * 2 + CACHELINE_SIZE - 1) / CACHELINE_SIZE *
              (CACHELINE_SIZE / sizeof(uint8_t))];
} dequeIndices;

typedef union {
  struct {
    dequeConsts consts;
    dequeIndices indices;
    CacheLine_ptr data[MEM_DEQ_LENGTH];
  };
  uint8_t pad[(sizeof(dequeConsts) + sizeof(dequeIndices) +
               sizeof(CacheLine_ptr) + CACHELINE_SIZE - 1) /
              CACHELINE_SIZE * (CACHELINE_SIZE / sizeof(uint8_t))];
} artsCXLDeque;

static inline void initCXLArena(CXLArena **arena, size_t bytes) {
  // Allocate the arena structure
  *arena = GLOBAL_MALLOC(sizeof(CXLArena));
  // Allocate the memory the arena will manage
  char *memory = GLOBAL_MALLOC(bytes);
  // Initialize the arena
  (*arena)->base = memory;
  (*arena)->head = memory;
  (*arena)->initialized = 1;
  (*arena)->max_size = memory + bytes;
}

static inline void *mallocCXLArena(CXLArena *arena, size_t bytes) {
  FLUSH_FENCE_CONSUMER(arena, sizeof(CXLArena));

  // Align allocation up to a multiple of CACHELINE_SIZE
  size_t aligned_bytes = ALIGN_UP(bytes, CACHELINE_SIZE);

  // Align the current head pointer to CACHELINE_SIZE
  uintptr_t head_addr = (uintptr_t)arena->head;
  uintptr_t aligned_head = ALIGN_UP(head_addr, CACHELINE_SIZE);

  // Compute new head after allocating
  uintptr_t new_head = aligned_head + aligned_bytes;

  if (new_head <= (uintptr_t)arena->max_size) {
    arena->head = (char *)new_head;
    FLUSH_FENCE_PRODUCER(arena, sizeof(CXLArena));
    return (void *)aligned_head;
  }
  return NULL; // out of memory
}

// Create deque on shared FAM
static inline artsCXLDeque *artsCXLDequeCreate() {
  artsCXLDeque *memDeque = (artsCXLDeque *)SHARED_MALLOC(sizeof(artsCXLDeque));

  memDeque->indices.frontIdx = -1;
  memDeque->indices.backIdx = 0;
  memDeque->consts.maxSize = MEM_DEQ_LENGTH;

  for (unsigned int i = 0; i < MEM_DEQ_LENGTH; i++) {
    memDeque->data[i].base.ptr = NULL;
    memDeque->data[i].base.size = 0;
  }
  initCXLArena(&(memDeque->consts.memArena), 1000000000); // ~1GB
  initCXLArena(&(memDeque->consts.dbArena), 1000000000);  // ~1GB
  memDeque->consts.lock = newTournamentLock(NUM_NODES);
  assert((sizeof(artsCXLDeque) % CACHELINE_SIZE) == 0 &&
         "artsCXLDeque SharedFam is cache aligned");
  FLUSH_FENCE_PRODUCER(memDeque, sizeof(artsCXLDeque));
  SHARED_MALLOC_INITIALIZED(memDeque);
  return memDeque;
}

// Get created shared FAM deque
static inline artsCXLDeque *artsCXLDequeGet() {
  artsCXLDeque *memDeque =
      (artsCXLDeque *)LAST_SHARED_MALLOC(sizeof(artsCXLDeque));
  FLUSH_FENCE_CONSUMER(memDeque, sizeof(artsCXLDeque));
  return memDeque;
}

static inline bool dequePtrCheck(void *ptr) {
  static void *check = NULL;
  if (check == NULL) {
    check = ptr;
    return 1;
  }
  return (check == ptr);
}

// Simplified Wrapper around create and get
static inline artsCXLDeque *artsCXLDequeInit() {
  if (!artsGlobalRankId) {
    artsCXLDeque *ret = artsCXLDequeCreate();
    assert(dequePtrCheck(ret) && "Rank 0 ptr check init");
    assert(ret != NULL && "Created CXL Deque not NULL");
    return ret;
  }
  artsCXLDeque *ret = artsCXLDequeGet();
  assert(dequePtrCheck(ret) && "Rank 1 ptr check init");
  assert(ret != NULL && "Gotten CXL Deque not NULL");
  return ret;
}

// Free deque from shared FAM
static inline void artsCXLDequeFree(artsCXLDeque *memDeque) {
  if (!artsGlobalRankId) {
    GLOBAL_FREE(memDeque->consts.memArena);
    GLOBAL_FREE(memDeque->consts.dbArena);
    SHARED_FREE(memDeque);
  }
}

static inline int artsCXLDequeFull(artsCXLDeque *deque) {
  FLUSH_FENCE_CONSUMER(&(deque->indices), sizeof(dequeIndices));
  FLUSH_FENCE_CONSUMER(&(deque->consts), sizeof(dequeConsts));
  return ((deque->indices.frontIdx == 0 &&
           deque->indices.backIdx == deque->consts.maxSize - 1) ||
          (deque->indices.frontIdx == deque->indices.backIdx + 1));
}

static inline void *artsCXLDequeBack(artsCXLDeque *deque, size_t *size) {
  assert(dequePtrCheck(deque) && "ptr check cxlDequeBack");
  FLUSH_FENCE_CONSUMER(&(deque->indices), sizeof(dequeIndices));
  if (deque->indices.frontIdx != -1) { // not empty
    FLUSH_FENCE_CONSUMER(&(deque->data[deque->indices.backIdx]),
                         sizeof(CacheLine_ptr));
    *size = deque->data[deque->indices.backIdx].base.size;
    return deque->data[deque->indices.backIdx].base.ptr;
  }
  return NULL;
}

static inline int artsCXLDequeEmpty(artsCXLDeque *deque) {
  FLUSH_FENCE_CONSUMER(&(deque->indices), sizeof(dequeIndices));
  return (deque->indices.frontIdx == -1);
}

static inline int artsCXLDequePopBack(artsCXLDeque *deque) {
  if (!artsCXLDequeEmpty(deque)) {
    deque->data[deque->indices.backIdx].base.ptr = NULL;
    deque->data[deque->indices.backIdx].base.size = 0;

    FLUSH_FENCE_PRODUCER(&(deque->data[deque->indices.backIdx]),
                         sizeof(CacheLine_ptr));

    if (deque->indices.frontIdx == deque->indices.backIdx) { // one element only
      deque->indices.frontIdx = -1;
      deque->indices.backIdx = 0;
    } else {
      if (deque->indices.backIdx == 0)
        deque->indices.backIdx = deque->consts.maxSize - 1;
      else
        deque->indices.backIdx -= 1;
    }
    FLUSH_FENCE_PRODUCER(&(deque->indices), sizeof(dequeIndices));
    return 1;
  }
  return 0;
}

// Pop memory from deque
static inline int artsCXLDequePop(artsCXLDeque *memDeque,
                                  pthread_mutex_t *localLock, void **dest_buf) {
  *dest_buf = NULL;
#if PROFILE
  struct timespec start, end;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &start);
#endif
  lockTournament(memDeque->consts.lock, localLock, artsGlobalRankId);
#if PROFILE
  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  PRINTF("Lock time: %.9f seconds\n", elapsed);
#endif
  size_t size;
  char *ptr = artsCXLDequeBack(memDeque, &size);
  if (ptr) {
    size_t aligned_size = ALIGN_UP(size, CACHELINE_SIZE);
    FLUSH_FENCE_CONSUMER(ptr, aligned_size);
    assert(((uintptr_t)ptr % CACHELINE_SIZE) == 0 &&
           "Ptr is aligned (artsCXLDequePop)");
    *dest_buf = (void *)ptr;
    artsCXLDequePopBack(memDeque);
    unlockTournament(memDeque->consts.lock, localLock, artsGlobalRankId);
    return 1;
  }
  // debug print
  // printf("No header\n");
  unlockTournament(memDeque->consts.lock, localLock, artsGlobalRankId);
  return 0;
}

static inline int artsCXLDequePushFront(artsCXLDeque *deque, void *newThing,
                                        size_t size) {
  if (!artsCXLDequeFull(deque)) {
    if (deque->indices.frontIdx == -1) { // uninitialized
      deque->indices.frontIdx = 0;
      deque->indices.backIdx = 0;
    } else if (deque->indices.frontIdx == 0)
      deque->indices.frontIdx = deque->consts.maxSize - 1;
    else
      deque->indices.frontIdx -= 1;

    char *ptr = (char *)mallocCXLArena(deque->consts.memArena, size);
    assert(ptr != NULL && "Arena allocated ptr not NULL");
    deque->data[deque->indices.frontIdx].base.ptr = ptr;
    deque->data[deque->indices.frontIdx].base.size = size;
    memcpy(ptr, newThing, size);

    // AA: Get the aligned size for flushes
    size_t aligned_size = ALIGN_UP(size, CACHELINE_SIZE);
    FLUSH_FENCE_PRODUCER(&(deque->data[deque->indices.frontIdx]),
                         sizeof(CacheLine_ptr));
    FLUSH_FENCE_PRODUCER(deque->data[deque->indices.frontIdx].base.ptr,
                         aligned_size);
    FLUSH_FENCE_PRODUCER(&(deque->indices), sizeof(dequeIndices));
    return 1;
  }
  return 0;
}

// Push memory to deque
static inline int artsCXLDequePush(artsCXLDeque *memDeque,
                                   pthread_mutex_t *localLock, size_t size,
                                   void *src_buf) {
#if PROFILE
  struct timespec start, end;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &start);
#endif
  lockTournament(memDeque->consts.lock, localLock, artsGlobalRankId);
#if PROFILE
  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  PRINTF("Lock time: %.9f seconds\n", elapsed);
#endif
  if (!artsCXLDequeFull(memDeque)) {
    artsCXLDequePushFront(memDeque, src_buf, size);
    unlockTournament(memDeque->consts.lock, localLock, artsGlobalRankId);
    return 1;
  }
  unlockTournament(memDeque->consts.lock, localLock, artsGlobalRankId);
  return 0;
}

static inline void *artsCXLDequeDbMalloc(artsCXLDeque *memDeque,
                                         pthread_mutex_t *localLock,
                                         size_t size) {
  lockTournament(memDeque->consts.lock, localLock, artsGlobalRankId);
  FLUSH_FENCE_CONSUMER(memDeque->consts.dbArena, sizeof(CXLArena));
  void *ptr = mallocCXLArena(memDeque->consts.dbArena, size);
  FLUSH_FENCE_PRODUCER(memDeque->consts.dbArena, sizeof(CXLArena));
  unlockTournament(memDeque->consts.lock, localLock, artsGlobalRankId);
  return ptr;
}

static inline int artsCXLDequeGetDbArenaRange(artsCXLDeque *memDeque,
                                              void **start, void **end) {
  FLUSH_FENCE_CONSUMER(&(memDeque->consts), sizeof(dequeConsts));
  FLUSH_FENCE_CONSUMER(memDeque->consts.dbArena, sizeof(CXLArena));
  if (!memDeque->consts.dbArena->initialized) {
    return 0;
  }
  *start = memDeque->consts.dbArena->base;
  *end = memDeque->consts.dbArena->max_size;
  return 1;
}

#ifdef __cplusplus
}
#endif
#endif /* ARTS_CXL_DEQUE_H */