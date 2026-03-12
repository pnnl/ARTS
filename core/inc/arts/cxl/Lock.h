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
#ifndef ARTS_CXL_LOCK_H
#define ARTS_CXL_LOCK_H
#ifdef __cplusplus
extern "C" {
#endif

#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "arts/cxl/Wrapper.h"

#define SINGLE_ALLOC 1

static inline FILE *get_dprintf_file(void) {
  static FILE *fp = NULL;

  if (!fp) {
    char hostname[256] = {0};
    if (gethostname(hostname, sizeof(hostname)) != 0) {
      strncpy(hostname, "unknown", sizeof(hostname) - 1);
    }

    char fname[300];
    snprintf(fname, sizeof(fname), "file_%s.txt", hostname);

    fp = fopen(fname, "a");
    if (!fp) {
      // If opening the file fails, fall back to stderr
      fp = stderr;
    }
  }
  return fp;
}

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

typedef struct {
  volatile int64_t value;
  char pad[56];
} CacheLine_t;

typedef struct {
  CacheLine_t *level;
  CacheLine_t *lastToEnter;
  uint64_t numProcs;
} FilterLock_t;

typedef union {
  struct {
    CacheLine_t flag[2];
    CacheLine_t turn;
    CacheLine_t owner;
  } lock;
  uint64_t pad[(sizeof(CacheLine_t) * 4 + CACHELINE_SIZE - 1) / CACHELINE_SIZE *
               (CACHELINE_SIZE / sizeof(uint64_t))];
} PetersonLock_t;

typedef union {
  struct {
    uint64_t numProcs;
    uint64_t k;
    pthread_mutex_t localLock;
    PetersonLock_t *locks;
  } lock;
  uint64_t pad[(sizeof(uint64_t) * 2 + sizeof(pthread_mutex_t) +
                sizeof(PetersonLock_t *) + CACHELINE_SIZE - 1) /
               CACHELINE_SIZE * (CACHELINE_SIZE / sizeof(uint64_t))];
} TournamentLock_t;

static inline void initPetersonLock(PetersonLock_t *newLock) {
  newLock->lock.flag[0].value = 0;
  newLock->lock.flag[1].value = 0;
  newLock->lock.turn.value = 0;
  newLock->lock.owner.value = -1;
  FLUSH_FENCE_PRODUCER(newLock, sizeof(PetersonLock_t));
}

static inline PetersonLock_t *newPetersonLock() {
  PetersonLock_t *newLock =
      (PetersonLock_t *)GLOBAL_MALLOC(sizeof(PetersonLock_t));
  initPetersonLock(newLock);
  return newLock;
}

static inline void deletePetersonLock(PetersonLock_t *lock) {
  GLOBAL_FREE(lock);
}

static inline void lockPeterson(PetersonLock_t *lock, int id) {
  FLUSH_FENCE_CONSUMER(lock, sizeof(PetersonLock_t)); // debug
  unsigned int time = 1;
  int other = 1 - id;

  lock->lock.flag[id].value = 1;
  FLUSH_FENCE_PRODUCER(&(lock->lock.flag[id]), sizeof(CacheLine_t));

  COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();

  lock->lock.turn.value = other;
  FLUSH_FENCE_PRODUCER(&(lock->lock.turn), sizeof(CacheLine_t));

  HW_MEMORY_FENCE();

  FLUSH_FENCE_CONSUMER(lock, sizeof(PetersonLock_t));

  while (lock->lock.flag[other].value && lock->lock.turn.value == other) {
    usleep(time);
    // JBMF: changing the delay to 1 sec
    if (time < 1000000)
      time *= 2;

    FLUSH_FENCE_CONSUMER(lock, sizeof(PetersonLock_t));
  }
  COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
  lock->lock.owner.value = id;
  FLUSH_FENCE_PRODUCER(&(lock->lock.owner), sizeof(CacheLine_t));
}

static inline void unlockPeterson(PetersonLock_t *lock) {
  int64_t id = lock->lock.owner.value;
  lock->lock.owner.value = -1;
  lock->lock.flag[id].value = 0;
  FLUSH_FENCE_PRODUCER(lock, sizeof(PetersonLock_t));
}

/**********************************************************/

static inline TournamentLock_t *newTournamentLock(unsigned int numProcs) {
  TournamentLock_t *newLock =
      (TournamentLock_t *)GLOBAL_MALLOC(sizeof(TournamentLock_t));
  unsigned int x = numProcs;
  uint64_t k = 0;
  while (x >>= 1)
    k++;
  if (1 << k < numProcs)
    k++;
  newLock->lock.numProcs = (1 << k);
  newLock->lock.k = k;
  pthread_mutex_init(&newLock->lock.localLock, NULL);
  unsigned int numLocks = newLock->lock.numProcs - 1;
  newLock->lock.locks =
      (PetersonLock_t *)GLOBAL_MALLOC(sizeof(PetersonLock_t) * numLocks);
  for (unsigned int i = 0; i < numLocks; i++) {
    initPetersonLock(&newLock->lock.locks[i]);
  }

  FLUSH_FENCE_PRODUCER(newLock, sizeof(TournamentLock_t));
  FLUSH_FENCE_PRODUCER(newLock->lock.locks, sizeof(PetersonLock_t) * numLocks);

  return newLock;
}

static inline void deleteTournamentLock(TournamentLock_t *lock) {
  pthread_mutex_destroy(&lock->lock.localLock);
  GLOBAL_FREE(lock->lock.locks);
  GLOBAL_FREE(lock);
}

static inline void lockTournament(TournamentLock_t *lock,
                                  pthread_mutex_t *localLock, int id) {
  pthread_mutex_lock(localLock);
  unsigned int nodeId = id + (lock->lock.numProcs - 1);
  for (unsigned int i = 0; i < lock->lock.k; i++) {
    unsigned int pid = (nodeId + 1) % 2;
    nodeId = (nodeId - 1) / 2;
    lockPeterson(&lock->lock.locks[nodeId], pid);
  }
}

static inline void unlockTournament(TournamentLock_t *lock,
                                    pthread_mutex_t *localLock, int id) {
  unsigned int nodeId = 0;
  for (unsigned int i = 0; i < lock->lock.k; i++) {
    unsigned int pid = lock->lock.locks[nodeId].lock.owner.value;
    unlockPeterson(&lock->lock.locks[nodeId]);
    nodeId = 2 * nodeId + 1 + pid;
  }
  pthread_mutex_unlock(localLock);
}

#ifdef __cplusplus
}
#endif
#endif /* ARTS_CXL_LOCK_H */
