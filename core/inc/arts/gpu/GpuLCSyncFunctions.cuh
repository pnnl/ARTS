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
#ifndef ARTS_GPU_GPULCSYNCFUNCTIONS_H
#define ARTS_GPU_GPULCSYNCFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime/RT.h"

typedef struct {
  uint64_t guid;
  void *data;
  uint64_t dataSize;
  volatile unsigned int *hostVersion;
  unsigned int *hostTimeStamp;
  unsigned int gpuVersion;
  unsigned int gpuTimeStamp;
  int gpu;
  volatile unsigned int *readLock;
  volatile unsigned int *writeLock;
} artsLCMeta_t;

typedef void (*artsLCSyncFunction_t)(artsLCMeta_t *host, artsLCMeta_t *dev);
extern artsLCSyncFunction_t lcSyncFunction[];

typedef void (*artsLCSyncFunctionGpu_t)(struct artsDb *src, struct artsDb *dst);
extern artsLCSyncFunctionGpu_t lcSyncFunctionGpu[];

extern unsigned int lcSyncElementSize[];

void *makeLCShadowCopy(struct artsDb *db);

void artsMemcpyGpuDb(artsLCMeta_t *host, artsLCMeta_t *dev);
void artsGetLatestGpuDb(artsLCMeta_t *host, artsLCMeta_t *dev);
void artsGetRandomGpuDb(artsLCMeta_t *host, artsLCMeta_t *dev);
void artsGetNonZerosUnsignedInt(artsLCMeta_t *host, artsLCMeta_t *dev);
void artsGetMinDbUnsignedInt(artsLCMeta_t *host, artsLCMeta_t *dev);
void artsAddDbUnsignedInt(artsLCMeta_t *host, artsLCMeta_t *dev);
void artsXorDbUint64(artsLCMeta_t *host, artsLCMeta_t *dev);

unsigned int gpuLCReduce(artsGuid_t guid, struct artsDb *db,
                         artsLCSyncFunctionGpu_t dbFn, bool *copyOnly);

__global__ void artsCopyGpuDb(struct artsDb *src, struct artsDb *dst);
__global__ void artsMinGpuDbUnsignedInt(struct artsDb *src, struct artsDb *dst);
__global__ void artsNonZeroGpuDbUnsignedInt(struct artsDb *src,
                                            struct artsDb *dst);
__global__ void artsAddGpuDbUnsignedInt(struct artsDb *src, struct artsDb *dst);
__global__ void artsXorGpuDbUint64(struct artsDb *sink, struct artsDb *src);

#ifdef __cplusplus
}
#endif

#endif
