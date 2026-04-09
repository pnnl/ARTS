/******************************************************************************
 * Copyright 2019 Battelle Memorial Institute
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#ifndef ARTS_CXL_DEQUE_H
#define ARTS_CXL_DEQUE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "arts/cxl/lock.h"
#include "arts/cxl/wrapper.h"

extern unsigned int arts_global_rank_count;
extern unsigned int arts_global_rank_id;

#define ARTS_CXL_DEQUE_LENGTH 1000000
#define ARTS_CXL_CHUNK_SIZE 1024
#define ARTS_CXL_NUM_NODES arts_global_rank_count

/* ── Arena allocator (bump pointer on CXL global memory) ────────────────────
 */

typedef union {
  struct {
    char *base;
    char *head;
    char *max_size;
    bool initialized;
  };
  uint8_t pad[((sizeof(char *) * 3 + sizeof(bool) + CACHELINE_SIZE - 1) /
               CACHELINE_SIZE) *
              (CACHELINE_SIZE / sizeof(uint8_t))];
} arts_cxl_arena_t;

/* ── Deque element (cache-line padded pointer + size) ───────────────────────
 */

typedef union {
  struct {
    char *ptr;
    size_t size;
  } base;
  uint8_t pad[((sizeof(char *) + sizeof(size_t) + CACHELINE_SIZE - 1) /
               CACHELINE_SIZE) *
              (CACHELINE_SIZE / sizeof(uint8_t))];
} arts_cxl_deque_elem_t;

/* Maximum number of CXL devices supported for DB arena allocation. */
#define ARTS_CXL_MAX_DEVICES 16

/* ── Deque constants (cache-line padded) ────────────────────────────────────
 */

typedef union {
  struct {
    int max_size;
    arts_cxl_arena_t *mem_arena;
    arts_cxl_arena_t *db_arenas[ARTS_CXL_MAX_DEVICES]; /**< One arena per CXL device. */
    unsigned int db_arena_count; /**< Number of active db_arenas entries. */
    arts_cxl_tournament_lock_t *lock;
  };
  uint8_t pad[((sizeof(int) + sizeof(arts_cxl_arena_t *) +
                sizeof(arts_cxl_arena_t *) * ARTS_CXL_MAX_DEVICES +
                sizeof(unsigned int) +
                sizeof(arts_cxl_tournament_lock_t *) + CACHELINE_SIZE - 1) /
               CACHELINE_SIZE) *
              (CACHELINE_SIZE / sizeof(uint8_t))];
} arts_cxl_deque_consts_t;

/* ── Deque indices (cache-line padded) ──────────────────────────────────────
 */

typedef union {
  struct {
    int front_idx;
    int back_idx;
  };
  uint8_t pad[((sizeof(int) * 2 + CACHELINE_SIZE - 1) / CACHELINE_SIZE) *
              (CACHELINE_SIZE / sizeof(uint8_t))];
} arts_cxl_deque_indices_t;

/* ── CXL shared-memory deque ────────────────────────────────────────────────
 */

typedef union {
  struct {
    arts_cxl_deque_consts_t consts;
    arts_cxl_deque_indices_t indices;
    arts_cxl_deque_elem_t data[ARTS_CXL_DEQUE_LENGTH];
  };
  uint8_t
      pad[((sizeof(arts_cxl_deque_consts_t) + sizeof(arts_cxl_deque_indices_t) +
            sizeof(arts_cxl_deque_elem_t) + CACHELINE_SIZE - 1) /
           CACHELINE_SIZE) *
          (CACHELINE_SIZE / sizeof(uint8_t))];
} arts_cxl_deque_t;

/* ── Arena operations ───────────────────────────────────────────────────────
 */

static inline void arts_cxl_arena_init(arts_cxl_arena_t **arena, size_t bytes) {
  *arena = (arts_cxl_arena_t *)GLOBAL_MALLOC(sizeof(arts_cxl_arena_t));
  char *memory = (char *)GLOBAL_MALLOC(bytes);
  (*arena)->base = memory;
  (*arena)->head = memory;
  (*arena)->initialized = true;
  (*arena)->max_size = memory + bytes;
}

/**
 * arts_cxl_arena_init_dev — Allocate a DB arena on a specific CXL device.
 *
 * Uses GLOBAL_MALLOC_DEV to place the backing memory on @p dev_id.
 * The arena metadata struct itself is allocated with GLOBAL_MALLOC (any device).
 */
static inline void arts_cxl_arena_init_dev(arts_cxl_arena_t **arena,
                                            size_t bytes, uint64_t dev_id) {
  *arena = (arts_cxl_arena_t *)GLOBAL_MALLOC(sizeof(arts_cxl_arena_t));
  char *memory = (char *)GLOBAL_MALLOC_DEV(bytes, dev_id);
  (*arena)->base = memory;
  (*arena)->head = memory;
  (*arena)->initialized = true;
  (*arena)->max_size = memory + bytes;
}

static inline void arts_cxl_arena_free(arts_cxl_arena_t *arena) {
  if (arena) {
    GLOBAL_FREE(arena->base);
    GLOBAL_FREE(arena);
  }
}

static inline void *arts_cxl_arena_malloc(arts_cxl_arena_t *arena,
                                          size_t bytes) {
  FLUSH_FENCE_CONSUMER(arena, sizeof(arts_cxl_arena_t));

  size_t aligned_bytes = ALIGN_UP(bytes, CACHELINE_SIZE);
  uintptr_t head_addr = (uintptr_t)arena->head;
  uintptr_t aligned_head = ALIGN_UP(head_addr, CACHELINE_SIZE);
  uintptr_t new_head = aligned_head + aligned_bytes;

  if (new_head <= (uintptr_t)arena->max_size) {
    arena->head = (char *)new_head;
    FLUSH_FENCE_PRODUCER(arena, sizeof(arts_cxl_arena_t));
    return (void *)aligned_head;
  }
  else {
    printf("Ran out of space in arena!\n");
    fflush(stdout);
  }
  return NULL;
}

/* ── Deque lifecycle ────────────────────────────────────────────────────────
 */

/**
 * arts_cxl_deque_create — Create a CXL deque with a single DB arena on
 * device 0 (legacy / static-device-0 path).
 */
static inline arts_cxl_deque_t *arts_cxl_deque_create(void) {
  arts_cxl_deque_t *dq =
      (arts_cxl_deque_t *)SHARED_MALLOC(sizeof(arts_cxl_deque_t));

  dq->indices.front_idx = -1;
  dq->indices.back_idx = 0;
  dq->consts.max_size = ARTS_CXL_DEQUE_LENGTH;

  for (unsigned int i = 0; i < ARTS_CXL_DEQUE_LENGTH; i++) {
    dq->data[i].base.ptr = NULL;
    dq->data[i].base.size = 0;
  }
  arts_cxl_arena_init(&dq->consts.mem_arena, 5000000000); /* ~5 GB */

  /* Single DB arena on device 0 (default / static strategy). */
  arts_cxl_arena_init_dev(&dq->consts.db_arenas[0], 5000000000, 0);
  for (unsigned int i = 1; i < ARTS_CXL_MAX_DEVICES; i++) {
    dq->consts.db_arenas[i] = NULL;
  }
  dq->consts.db_arena_count = 1;

  dq->consts.lock = arts_cxl_tournament_lock_new(ARTS_CXL_NUM_NODES);

  assert((sizeof(arts_cxl_deque_t) % CACHELINE_SIZE) == 0 &&
         "arts_cxl_deque_t must be cache-line aligned");
  FLUSH_FENCE_PRODUCER(dq, sizeof(arts_cxl_deque_t));
  SHARED_MALLOC_INITIALIZED(dq);
  return dq;
}

/**
 * arts_cxl_deque_create_with_arenas — Create a CXL deque with DB arenas
 * allocated on specific devices.
 *
 * @param dev_ids   Array of device IDs to allocate arenas on.
 * @param dev_count Number of devices (length of dev_ids).
 *
 * For the static strategy, pass a single-element array with the chosen device.
 * For round-robin, pass all device IDs.
 */
static inline arts_cxl_deque_t *
arts_cxl_deque_create_with_arenas(const uint64_t *dev_ids,
                                   unsigned int dev_count) {
  assert(dev_count > 0 && dev_count <= ARTS_CXL_MAX_DEVICES &&
         "dev_count must be in [1, ARTS_CXL_MAX_DEVICES]");

  arts_cxl_deque_t *dq =
      (arts_cxl_deque_t *)SHARED_MALLOC(sizeof(arts_cxl_deque_t));

  assert(dq && "Allocated CXL deque pointer is valid");

  dq->indices.front_idx = -1;
  dq->indices.back_idx = 0;
  dq->consts.max_size = ARTS_CXL_DEQUE_LENGTH;

  for (unsigned int i = 0; i < ARTS_CXL_DEQUE_LENGTH; i++) {
    dq->data[i].base.ptr = NULL;
    dq->data[i].base.size = 0;
  }
  arts_cxl_arena_init(&dq->consts.mem_arena, 5000000000); /* ~5 GB */

  for (unsigned int i = 0; i < dev_count; i++) {
    arts_cxl_arena_init_dev(&dq->consts.db_arenas[i], 5000000000, dev_ids[i]);
  }
  for (unsigned int i = dev_count; i < ARTS_CXL_MAX_DEVICES; i++) {
    dq->consts.db_arenas[i] = NULL;
  }
  dq->consts.db_arena_count = dev_count;

  dq->consts.lock = arts_cxl_tournament_lock_new(ARTS_CXL_NUM_NODES);

  assert((sizeof(arts_cxl_deque_t) % CACHELINE_SIZE) == 0 &&
         "arts_cxl_deque_t must be cache-line aligned");
  FLUSH_FENCE_PRODUCER(dq, sizeof(arts_cxl_deque_t));
  SHARED_MALLOC_INITIALIZED(dq);
  return dq;
}

static inline arts_cxl_deque_t *arts_cxl_deque_get(void) {
  arts_cxl_deque_t *dq =
      (arts_cxl_deque_t *)LAST_SHARED_MALLOC(sizeof(arts_cxl_deque_t));
  FLUSH_FENCE_CONSUMER(dq, sizeof(arts_cxl_deque_t));
  return dq;
}

static inline arts_cxl_deque_t *arts_cxl_deque_init(void) {
#ifdef ARTS_CXL_NATIVE
  /* Real CXL: rank 0 creates in shared memory, others retrieve it */
  if (!arts_global_rank_id) {
    arts_cxl_deque_t *dq = arts_cxl_deque_create();
    assert(dq != NULL && "CXL deque creation failed");
    return dq;
  }
  arts_cxl_deque_t *dq = arts_cxl_deque_get();
  assert(dq != NULL && "CXL deque get failed");
  return dq;
#else
  /* Stub mode: each rank creates its own independent deque */
  arts_cxl_deque_t *dq = arts_cxl_deque_create();
  assert(dq != NULL && "CXL deque creation failed");
  return dq;
#endif
}

static inline void arts_cxl_deque_free(arts_cxl_deque_t *dq) {
#ifdef ARTS_CXL_NATIVE
  /* Real CXL: only rank 0 frees shared resources */
  if (!arts_global_rank_id) {
    arts_cxl_arena_free(dq->consts.mem_arena);
    for (unsigned int i = 0; i < dq->consts.db_arena_count; i++) {
      arts_cxl_arena_free(dq->consts.db_arenas[i]);
    }
    arts_cxl_tournament_lock_delete(dq->consts.lock);
    SHARED_FREE(dq);
  }
#else
  /* Stub mode: each rank frees its own */
  arts_cxl_arena_free(dq->consts.mem_arena);
  for (unsigned int i = 0; i < dq->consts.db_arena_count; i++) {
    arts_cxl_arena_free(dq->consts.db_arenas[i]);
  }
  arts_cxl_tournament_lock_delete(dq->consts.lock);
  SHARED_FREE(dq);
#endif
}

/* ── Deque queries ──────────────────────────────────────────────────────────
 */

static inline bool arts_cxl_deque_full(arts_cxl_deque_t *dq) {
  FLUSH_FENCE_CONSUMER(&dq->indices, sizeof(arts_cxl_deque_indices_t));
  FLUSH_FENCE_CONSUMER(&dq->consts, sizeof(arts_cxl_deque_consts_t));
  return (dq->indices.front_idx == 0 &&
          dq->indices.back_idx == dq->consts.max_size - 1) ||
         (dq->indices.front_idx == dq->indices.back_idx + 1);
}

static inline bool arts_cxl_deque_empty(arts_cxl_deque_t *dq) {
  FLUSH_FENCE_CONSUMER(&dq->indices, sizeof(arts_cxl_deque_indices_t));
  return dq->indices.front_idx == -1;
}

/* ── Internal: back peek / pop ──────────────────────────────────────────────
 */

static inline void *arts_cxl_deque_back(arts_cxl_deque_t *dq, size_t *size) {
  FLUSH_FENCE_CONSUMER(&dq->indices, sizeof(arts_cxl_deque_indices_t));
  if (dq->indices.front_idx != -1) {
    FLUSH_FENCE_CONSUMER(&dq->data[dq->indices.back_idx],
                         sizeof(arts_cxl_deque_elem_t));
    *size = dq->data[dq->indices.back_idx].base.size;
    return dq->data[dq->indices.back_idx].base.ptr;
  }
  return NULL;
}

static inline int arts_cxl_deque_pop_back(arts_cxl_deque_t *dq) {
  if (!arts_cxl_deque_empty(dq)) {
    dq->data[dq->indices.back_idx].base.ptr = NULL;
    dq->data[dq->indices.back_idx].base.size = 0;
    FLUSH_FENCE_PRODUCER(&dq->data[dq->indices.back_idx],
                         sizeof(arts_cxl_deque_elem_t));

    if (dq->indices.front_idx == dq->indices.back_idx) {
      dq->indices.front_idx = -1;
      dq->indices.back_idx = 0;
    } else {
      if (dq->indices.back_idx == 0) {
        dq->indices.back_idx = dq->consts.max_size - 1;
      } else {
        dq->indices.back_idx -= 1;
      }
    }
    FLUSH_FENCE_PRODUCER(&dq->indices, sizeof(arts_cxl_deque_indices_t));
    return 1;
  }
  return 0;
}

/* ── Internal: front push ───────────────────────────────────────────────────
 */

static inline int arts_cxl_deque_push_front_internal(arts_cxl_deque_t *dq,
                                                     void *item, size_t size) {
  if (!arts_cxl_deque_full(dq)) {
    if (dq->indices.front_idx == -1) {
      dq->indices.front_idx = 0;
      dq->indices.back_idx = 0;
    } else if (dq->indices.front_idx == 0) {
      dq->indices.front_idx = dq->consts.max_size - 1;
    } else {
      dq->indices.front_idx -= 1;
    }

    char *ptr = (char *)arts_cxl_arena_malloc(dq->consts.mem_arena, size);
    assert(ptr != NULL && "CXL arena allocation failed");
    dq->data[dq->indices.front_idx].base.ptr = ptr;
    dq->data[dq->indices.front_idx].base.size = size;
    memcpy(ptr, item, size);

    size_t aligned_size = ALIGN_UP(size, CACHELINE_SIZE);
    FLUSH_FENCE_PRODUCER(&dq->data[dq->indices.front_idx],
                         sizeof(arts_cxl_deque_elem_t));
    FLUSH_FENCE_PRODUCER(ptr, aligned_size);
    FLUSH_FENCE_PRODUCER(&dq->indices, sizeof(arts_cxl_deque_indices_t));
    return 1;
  }
  return 0;
}

/* ── Public: locked pop (back) ──────────────────────────────────────────────
 */

static inline int arts_cxl_deque_pop(arts_cxl_deque_t *dq,
                                     pthread_mutex_t *local_lock,
                                     void **dest_buf) {
  *dest_buf = NULL;
  arts_cxl_tournament_lock_acquire(dq->consts.lock, local_lock,
                                   arts_global_rank_id);
  size_t size;
  char *ptr = (char *)arts_cxl_deque_back(dq, &size);
  if (ptr) {
    size_t aligned_size = ALIGN_UP(size, CACHELINE_SIZE);
    FLUSH_FENCE_CONSUMER(ptr, aligned_size);
    *dest_buf = (void *)ptr;
    arts_cxl_deque_pop_back(dq);
    arts_cxl_tournament_lock_release(dq->consts.lock, local_lock,
                                     arts_global_rank_id);
    return 1;
  }
  arts_cxl_tournament_lock_release(dq->consts.lock, local_lock,
                                   arts_global_rank_id);
  return 0;
}

/* ── Public: locked push (front) ────────────────────────────────────────────
 */

static inline int arts_cxl_deque_push(arts_cxl_deque_t *dq,
                                      pthread_mutex_t *local_lock, size_t size,
                                      void *src_buf) {
  arts_cxl_tournament_lock_acquire(dq->consts.lock, local_lock,
                                   arts_global_rank_id);
  if (!arts_cxl_deque_full(dq)) {
    arts_cxl_deque_push_front_internal(dq, src_buf, size);
    arts_cxl_tournament_lock_release(dq->consts.lock, local_lock,
                                     arts_global_rank_id);
    return 1;
  }
  arts_cxl_tournament_lock_release(dq->consts.lock, local_lock,
                                   arts_global_rank_id);
  return 0;
}

/* ── Public: locked DB arena malloc (device-indexed) ────────────────────────
 */

/**
 * arts_cxl_deque_db_malloc_dev — Allocate from a specific device's DB arena.
 *
 * @param dq         The CXL deque.
 * @param local_lock Per-node pthread mutex for local serialization.
 * @param size       Allocation size in bytes.
 * @param dev_idx    Index into dq->consts.db_arenas[] (0-based).
 *                   Must be < dq->consts.db_arena_count.
 */
static inline void *arts_cxl_deque_db_malloc_dev(arts_cxl_deque_t *dq,
                                                  pthread_mutex_t *local_lock,
                                                  size_t size,
                                                  unsigned int dev_idx) {
  assert(dev_idx < dq->consts.db_arena_count && "dev_idx out of range");
  arts_cxl_tournament_lock_acquire(dq->consts.lock, local_lock,
                                   arts_global_rank_id);
  arts_cxl_arena_t *arena = dq->consts.db_arenas[dev_idx];
  FLUSH_FENCE_CONSUMER(arena, sizeof(arts_cxl_arena_t));
  void *ptr = arts_cxl_arena_malloc(arena, size);
  FLUSH_FENCE_PRODUCER(arena, sizeof(arts_cxl_arena_t));
  arts_cxl_tournament_lock_release(dq->consts.lock, local_lock,
                                   arts_global_rank_id);
  return ptr;
}

/**
 * arts_cxl_deque_db_malloc — Allocate from device-0 DB arena (legacy path).
 */
static inline void *arts_cxl_deque_db_malloc(arts_cxl_deque_t *dq,
                                             pthread_mutex_t *local_lock,
                                             size_t size) {
  return arts_cxl_deque_db_malloc_dev(dq, local_lock, size, 0);
}

/* ── Public: get DB arena address range (device 0) ──────────────────────────
 */

static inline int arts_cxl_deque_get_db_arena_range(arts_cxl_deque_t *dq,
                                                    void **start, void **end) {
  FLUSH_FENCE_CONSUMER(&dq->consts, sizeof(arts_cxl_deque_consts_t));
  if (dq->consts.db_arena_count == 0 || !dq->consts.db_arenas[0]) {
    return 0;
  }
  FLUSH_FENCE_CONSUMER(dq->consts.db_arenas[0], sizeof(arts_cxl_arena_t));
  if (!dq->consts.db_arenas[0]->initialized) {
    return 0;
  }
  *start = dq->consts.db_arenas[0]->base;
  *end = dq->consts.db_arenas[0]->max_size;
  return 1;
}

#ifdef __cplusplus
}
#endif
#endif /* ARTS_CXL_DEQUE_H */
