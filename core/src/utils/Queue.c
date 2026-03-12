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

#include "arts/utils/Queue.h"

#include "arts/arts.h"

#define CACHE_ALIGN __attribute__((aligned(64)))

#define FAA64(ptr, inc) __sync_fetch_and_add((ptr), (inc))

#define CAS64(ptr, old, new) __sync_bool_compare_and_swap((ptr), (old), (new))

#define CASPTR CAS64

#define StorePrefetch(val)                                                     \
  do {                                                                         \
  } while (0)

#define likely(x) __builtin_expect(!!(x), 1)

#define unlikely(x) __builtin_expect(!!(x), 0)

#ifdef USE_SIMPLE_ARCH

#include <pthread.h>

pthread_mutex_t amtx = PTHREAD_MUTEX_INITIALIZER;

#define __CAS2(ptr, o1, o2, n1, n2)                                            \
  ({                                                                           \
    char __ret;                                                                \
    uint64_t __junk;                                                           \
    uint64_t __old1 = (o1);                                                    \
    uint64_t __old2 = (o2);                                                    \
    uint64_t __new1 = (n1);                                                    \
    uint64_t __new2 = (n2);                                                    \
    pthread_mutex_lock(&amtx);                                                 \
    uint64_t *tmp = (uint64_t *)ptr;                                           \
    if (tmp[0] == o1 && tmp[1] == o2) {                                        \
      tmp[0] = n1;                                                             \
      tmp[1] = n2;                                                             \
      __ret = 1;                                                               \
    } else {                                                                   \
      __ret = 0;                                                               \
    }                                                                          \
    pthread_mutex_unlock(&amtx);                                               \
    __ret;                                                                     \
  })

#define CAS2(ptr, o1, o2, n1, n2) __CAS2(ptr, o1, o2, n1, n2)

#define BIT_TEST_AND_SET(ptr, b)                                               \
  ({                                                                           \
    char __ret;                                                                \
    pthread_mutex_lock(&amtx);                                                 \
    uint64_t mask = 1ULL << b;                                                 \
    __ret = (*ptr & mask) ? 1 : 0;                                             \
    *ptr = *ptr | mask;                                                        \
    pthread_mutex_unlock(&amtx);                                               \
    __ret;                                                                     \
  })

#else

#ifdef __aarch64__
// ARM64 implementation using simpler atomic operations
#define __CAS2(ptr, o1, o2, n1, n2)                                            \
  ({                                                                           \
    char __ret;                                                                \
    __typeof__(o2) __junk;                                                     \
    __typeof__(*(ptr)) __old1 = (o1);                                          \
    __typeof__(o2) __old2 = (o2);                                              \
    __typeof__(*(ptr)) __new1 = (n1);                                          \
    __typeof__(o2) __new2 = (n2);                                              \
    /* Use __sync_bool_compare_and_swap_16 for ARM64 */                        \
    __ret = __sync_bool_compare_and_swap_16(ptr, __old1, __new1);              \
    __ret;                                                                     \
  })

#define BIT_TEST_AND_SET(ptr, b)                                               \
  ({                                                                           \
    char __ret;                                                                \
    uint64_t old_val = *ptr;                                                   \
    uint64_t new_val = old_val | (1ULL << b);                                  \
    __ret = __sync_bool_compare_and_swap_8(ptr, old_val, new_val);             \
    __ret;                                                                     \
  })
#else
// x86_64 implementation using cmpxchg16b
#define __CAS2(ptr, o1, o2, n1, n2)                                            \
  ({                                                                           \
    char __ret;                                                                \
    __typeof__(o2) __junk;                                                     \
    __typeof__(*(ptr)) __old1 = (o1);                                          \
    __typeof__(o2) __old2 = (o2);                                              \
    __typeof__(*(ptr)) __new1 = (n1);                                          \
    __typeof__(o2) __new2 = (n2);                                              \
    asm volatile("lock cmpxchg16b %2;setz %1"                                  \
                 : "=d"(__junk), "=a"(__ret), "+m"(*ptr)                       \
                 : "b"(__new1), "c"(__new2), "a"(__old1), "d"(__old2));        \
    __ret;                                                                     \
  })

#define BIT_TEST_AND_SET(ptr, b)                                               \
  ({                                                                           \
    char __ret;                                                                \
    asm volatile("lock btsq $63, %0; setnc %1"                                 \
                 : "+m"(*ptr), "=a"(__ret)                                     \
                 :                                                             \
                 : "cc");                                                      \
    __ret;                                                                     \
  })
#endif

#endif

#define CAS2(ptr, o1, o2, n1, n2) __CAS2(ptr, o1, o2, n1, n2)

void initRing(RingQueue *r) {
  for (int i = 0; i < RING_SIZE; i++) {
    r->array[i].val = -1;
    r->array[i].idx = i;
  }
  r->head = r->tail = 0;
  r->next = NULL;
}

int isEmpty(uint64_t v) { return (v == (uint64_t)-1); }

uint64_t nodeIndex(uint64_t i) { return (i & ~(1ull << 63)); }

uint64_t setUnsafe(uint64_t i) { return (i | (1ull << 63)); }

uint64_t nodeUnsafe(uint64_t i) { return (i & (1ull << 63)); }

uint64_t tailIndex(uint64_t t) { return (t & ~(1ull << 63)); }

int crqIsClosed(uint64_t t) { return (t & (1ull << 63)) != 0; }

artsQueue *artsNewQueue() {
  artsQueue *queue = (artsQueue *)artsCallocAlign(1, sizeof(artsQueue), 128);
  RingQueue *rq = (RingQueue *)artsCallocAlign(1, sizeof(RingQueue), 128);
  initRing(rq);
  queue->head = queue->tail = rq;
  return queue;
}

void fixState(RingQueue *rq) {
  uint64_t t, h, n;
  while (1) {
    uint64_t t = FAA64(&rq->tail, 0);
    uint64_t h = FAA64(&rq->head, 0);

    if (unlikely(rq->tail != t))
      continue;

    if (h > t) {
      if (CAS64(&rq->tail, t, h))
        break;
      continue;
    }
    break;
  }
}

int closeCrq(RingQueue *rq, const uint64_t t, const int tries) {
  if (tries < 10) {
    return CAS64(&rq->tail, t + 1, (t + 1) | (1ull << 63));
  }
  return BIT_TEST_AND_SET(&rq->tail, 63);
}

void enqueue(Object arg, artsQueue *queue) {
  RingQueue *nrq;
  int try_close = 0;
  while (1) {
    RingQueue *rq = queue->tail;
    RingQueue *next = rq->next;

    if (unlikely(next != NULL)) {
      CASPTR(&queue->tail, rq, next);
      continue;
    }

    uint64_t t = FAA64(&rq->tail, 1);

    if (crqIsClosed(t)) {
    alloc:
      //            ARTS_INFO("Allocing!");
      nrq = (RingQueue *)artsMallocAlign(sizeof(RingQueue), 128);
      initRing(nrq);

      // Solo enqueue
      nrq->tail = 1;
      nrq->array[0].val = arg;
      nrq->array[0].idx = 0;

      if (CASPTR(&rq->next, NULL, nrq)) {
        CASPTR(&queue->tail, rq, nrq);
        nrq = NULL;
        return;
      }
      artsFree(nrq);
      continue;
    }

    RingNode *cell = &rq->array[t & (RING_SIZE - 1)];
    StorePrefetch(cell);

    uint64_t idx = cell->idx;
    uint64_t val = cell->val;

    if (likely(isEmpty(val))) {
      if (likely(nodeIndex(idx) <= t)) {
        if ((likely(!nodeUnsafe(idx)) || rq->head < t) &&
            CAS2((uint64_t *)cell, -1, idx, arg, t)) {
          return;
        }
      }
    }

    uint64_t h = rq->head;
    if (unlikely(t >= RING_SIZE + h) && closeCrq(rq, t, ++try_close)) {
      goto alloc;
    }
  }
}

Object dequeue(artsQueue *queue) {
  while (1) {
    RingQueue *rq = queue->head;
    RingQueue *next;
    uint64_t h = FAA64(&rq->head, 1);
    RingNode *cell = &rq->array[h & (RING_SIZE - 1)];
    StorePrefetch(cell);

    uint64_t tt;
    int r = 0;

    while (1) {
      uint64_t cell_idx = cell->idx;
      uint64_t unsafe = nodeUnsafe(cell_idx);
      uint64_t idx = nodeIndex(cell_idx);
      uint64_t val = cell->val;

      if (unlikely(idx > h))
        break;

      if (likely(!isEmpty(val))) {
        if (likely(idx == h)) {
          if (CAS2((uint64_t *)cell, val, cell_idx, -1,
                   unsafe | (h + RING_SIZE)))
            return val;
        } else {
          if (CAS2((uint64_t *)cell, val, cell_idx, val, setUnsafe(idx))) {
            break;
          }
        }
      } else {
        if ((r & ((1ull << 10) - 1)) == 0)
          tt = rq->tail;

        // Optimization: try to bail quickly if queue is closed.
        int crq_closed = crqIsClosed(tt);
        uint64_t t = tailIndex(tt);

        if (unlikely(unsafe)) // Nothing to do, move along
        {
          if (CAS2((uint64_t *)cell, val, cell_idx, val,
                   unsafe | (h + RING_SIZE)))
            break;
        } else if (t <= h + 1 || r > 200000 || crq_closed) {
          if (CAS2((uint64_t *)cell, val, idx, val, h + RING_SIZE))
            break;
        } else {
          ++r;
        }
      }
    }

    if (tailIndex(rq->tail) <= h + 1) {
      fixState(rq);
      // try to return empty
      next = rq->next;
      if (next == NULL) {
        return 0; // EMPTY
      }
      CASPTR(&queue->head, rq, next);
    }
  }
}
