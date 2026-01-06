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
#ifndef ARTS_RUNTIME_SYNC_GLOBALS_H
#define ARTS_RUNTIME_SYNC_GLOBALS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/introspection/ArtsIdCounter.h"
#include "arts/introspection/Counter.h"
#include "arts/runtime/RT.h"

struct atomicCreateBarrierInfo {
  volatile unsigned int wait;
  volatile unsigned int result;
};

struct artsRuntimeShared {
  volatile unsigned int sendLock;
  char pad1[56];
  volatile unsigned int recvLock;
  char pad2[56];
  volatile unsigned int stealRequestLock;
  char pad3[56];
  bool (*scheduler)();
  struct artsDeque **deque;
  struct artsDeque **receiverDeque;
  struct artsDeque **gpuDeque;
  struct artsRouteTable **routeTable;
  struct artsRouteTable **gpuRouteTable;
  struct artsRouteTable *remoteRouteTable;
  volatile bool **localSpin;
  unsigned int **memoryMoves;
  struct atomicCreateBarrierInfo **atomicWaits;
  unsigned int workerThreadCount;
  unsigned int senderThreadCount;
  unsigned int receiverThreadCount;
  unsigned int remoteStealingThreadCount;
  unsigned int totalThreadCount;
  volatile unsigned int readyToPush;
  volatile unsigned int readyToParallelStart;
  volatile unsigned int readyToInspect;
  volatile unsigned int readyToExecute;
  volatile unsigned int readyToClean;
  volatile unsigned int readyToShutdown;
  char *buf;
  int packetSize;
  bool shutdownStarted;
  volatile unsigned int shutdownCount;
  uint64_t shutdownTimeout;
  uint64_t shutdownForceTimeout;
  unsigned int printNodeStats;
  artsGuid_t shutdownEpoch;
  unsigned int shadLoopStride;
  bool tMT;
  unsigned int gpu;
  unsigned int gpuLocality;
  unsigned int gpuFit;
  unsigned int gpuLCSync;
  unsigned int gpuMaxEdts;
  unsigned int gpuP2P;
  unsigned int gpuRouteTableSize;
  unsigned int gpuRouteTableEntries;
  uint64_t gpuMaxMemory;
  bool freeDbAfterGpuRun;
  bool runGpuGcIdle;
  bool runGpuGcPreEdt;
  bool deleteZerosGpuGc;
  bool gpuBuffOn;
  unsigned int pinThreads;
  uint64_t **keys;
  uint64_t *globalGuidThreadId;
  const char *counterFolder;
  artsCounter **liveCounters;           // [threadId] -> pointer to thread's __thread counters
  artsCounter **savedCounters;          // [threadId][counterIndex] - final counter values
  artsArrayList ***captureArrays;       // [threadId][counterIndex] - capture history (PERIODIC)
  uint64_t counterCaptureInterval;
  // arts_id reduced metrics computed at output time (not stored during runtime)
} __attribute__((aligned(64)));

struct artsRuntimePrivate {
  struct artsDeque *myDeque;
  struct artsDeque *myNodeDeque;
  struct artsDeque *myGpuDeque;
  unsigned int coreId;
  unsigned int threadId;
  unsigned int groupId;
  unsigned int clusterId;
  unsigned int backOff;
  volatile unsigned int oustandingMemoryMoves;
  struct atomicCreateBarrierInfo atomicWait;
  volatile bool alive;
  volatile bool worker;
  volatile bool networkSend;
  volatile bool networkReceive;
  volatile bool statusSend;
  artsGuid_t currentEdtGuid;
  int mallocType;
  int mallocTrace;
  int edtFree;
  int localCounting;
  unsigned int shadLock;
  unsigned short drand_buf[3];
  // Thread's counter storage accessed via artsThreadLocalCounterCaptures
};

extern struct artsRuntimeShared artsNodeInfo;
extern __thread struct artsRuntimePrivate artsThreadInfo;

extern unsigned int artsGlobalRankId;
extern unsigned int artsGlobalRankCount;
extern unsigned int artsGlobalMasterRankId;
extern bool artsGlobalIWillPrint;
extern uint64_t artsGuidMin;
extern uint64_t artsGuidMax;

#define MASTER_PRINTF(...)                                                     \
  if (artsGlobalRankId == artsGlobalMasterRankId)                              \
  PRINTF(__VA_ARGS__)
#define ONCE_PRINTF(...)                                                       \
  if (artsGlobalIWillPrint == true)                                            \
  PRINTF(__VA_ARGS__)

#define artsLookUpConfig(name) artsNodeInfo.name

#define artsTypeName                                                           \
  const char *const _artsTypeName[] = {"ARTS_NULL",                            \
                                       "ARTS_EDT",                             \
                                       "ARTS_GPU_EDT",                         \
                                       "ARTS_EVENT",                           \
                                       "ARTS_PERSISTENT_EVENT",                \
                                       "ARTS_EPOCH",                           \
                                       "ARTS_CALLBACK",                        \
                                       "ARTS_BUFFER",                          \
                                       "ARTS_DB_READ",                         \
                                       "ARTS_DB_WRITE",                        \
                                       "ARTS_DB_PIN",                          \
                                       "ARTS_DB_ONCE",                         \
                                       "ARTS_DB_ONCE_LOCAL",                   \
                                       "ARTS_DB_GPU_READ",                     \
                                       "ARTS_DB_GPU_WRITE",                    \
                                       "ARTS_DB_LC",                           \
                                       "ARTS_LAST_TYPE",                       \
                                       "ARTS_SINGLE_VALUE",                    \
                                       "ARTS_PTR",                             \
                                       "ARTS_DB_LC_SYNC",                      \
                                       "ARTS_DB_LC_NO_COPY",                   \
                                       "ARTS_DB_GPU_MEMSET"}

#define getTypeName(x) _artsTypeName[x]

extern const char *const _artsTypeName[];

extern volatile uint64_t outstandingEdts;
void checkOutEdts(uint64_t threashold);

// #ifdef CHECK_NO_EDT
#define incOustandingEdts(numEdts)                                             \
  artsAtomicFetchAddU64(&outstandingEdts, numEdts)
#define decOustandingEdts(numEdts)                                             \
  artsAtomicFetchSubU64(&outstandingEdts, numEdts)
#define checkOutstandingEdts(threashold) checkOutEdts(threashold)
// #else
// #define incOustandingEdts(numEdts)
// #define decOustandingEdts(numEdts)
// #define checkOutstandingEdts(threashold)
// #endif

#ifdef __cplusplus
}
#endif

#endif
