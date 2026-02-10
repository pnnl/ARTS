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
#include "arts/gpu/gpu_lc_sync_functions.cuh"

#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu/gpu_route_table.h"
#include "arts/gpu/gpu_stream_buffer.h"
#include "arts/runtime/globals.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/utils/atomics.h"

// To use this lock the unlock must be an even number
unsigned int versionLock(artsLCMeta_t *meta) {
  arts_writer_lock(meta->read_lock, meta->write_lock);
  return *meta->hostVersion;
}

bool tryVersionLock(artsLCMeta_t *meta) {
  return arts_writer_try_lock(meta->read_lock, meta->write_lock);
}

void versionUnlock(artsLCMeta_t *meta) {
  arts_atomic_add(meta->hostVersion, 2U);
  arts_writer_unlock(meta->write_lock);
}

void *makeLCShadowCopy(struct arts_db *db) {
  unsigned int size = db->header.size;
  void *dest = (void *)(((char *)db) + size);
  struct arts_db *shadow_copy = (struct arts_db *)dest;

  arts_writer_lock(&db->reader, &db->writer);
  unsigned int hostVersion = db->version;
  if (!shadow_copy->version || hostVersion != shadow_copy->version) {
    memcpy(dest, (void *)db, size);
  }
  arts_writer_unlock(&db->writer);
  return dest;
}

inline void artsPrintDbMetaData(artsLCMeta_t *db) {
  ARTS_DEBUG("guid: %lu ptr: %p dataSize: %lu hostVersion: %u gpuVersion: %u "
             "gpuTimeStamp: %u gpu: %d",
             db->guid, db->data, db->dataSize, *db->hostVersion,
             *db->hostTimeStamp, db->gpuVersion, db->gpuTimeStamp, db->gpu);
}

void artsMemcpyGpuDb(artsLCMeta_t *host, artsLCMeta_t *dev) {
  unsigned int hostVersion = versionLock(host);
  memcpy(host->data, dev->data, host->dataSize);
  *host->hostTimeStamp = dev->gpuTimeStamp;
  versionUnlock(host);
}

void artsGetLatestGpuDb(artsLCMeta_t *host, artsLCMeta_t *dev) {
  unsigned int hostVersion = versionLock(host);
  if (*host->hostTimeStamp < dev->gpuTimeStamp) {
    memcpy(host->data, dev->data, host->dataSize);
    host->gpuVersion = dev->gpuVersion;
    host->gpuTimeStamp = dev->gpuTimeStamp;
    *host->hostTimeStamp = dev->gpuTimeStamp;
    host->gpu = dev->gpu;
  }
  versionUnlock(host);
}

void artsGetRandomGpuDb(artsLCMeta_t *host, artsLCMeta_t *dev) {
  bool first_flag = (host->gpu == -1);
  bool randomFlag = ((arts_thread_safe_random() & 1) == 0);
  if (first_flag || randomFlag) {
    if (tryVersionLock(host)) {
      memcpy(host->data, dev->data, host->dataSize);
      host->gpuVersion = dev->gpuVersion;
      host->gpuTimeStamp = dev->gpuTimeStamp;
      *host->hostTimeStamp = dev->gpuTimeStamp;
      host->gpu = dev->gpu;
      // if(!first_flag && randomFlag)
      // arts_gpu_invalidate_route_tables(host->guid, (unsigned int) -1);
      versionUnlock(host);
    }
  }
}

void artsGetNonZerosUnsignedInt(artsLCMeta_t *host, artsLCMeta_t *dev) {
  unsigned int numElem = host->dataSize / sizeof(unsigned int);
  unsigned int *dst = (unsigned int *)host->data;
  unsigned int *src = (unsigned int *)dev->data;
  unsigned int hostVersion = versionLock(host);
  for (unsigned int i = 0; i < numElem; i++) {
    ARTS_DEBUG("src: %u dest: %u", src[i], dst[i]);
    if (src[i])
      dst[i] = src[i];
  }
  versionUnlock(host);
}

void artsGetMinDbUnsignedInt(artsLCMeta_t *host, artsLCMeta_t *dev) {
  unsigned int count = 0;
  unsigned int count2 = 0;
  unsigned int numElem = host->dataSize / sizeof(unsigned int);
  unsigned int *dst = (unsigned int *)host->data;
  unsigned int *src = (unsigned int *)dev->data;
  unsigned int hostVersion = versionLock(host);
  for (unsigned int i = 0; i < numElem; i++) {
    if (src[i] < dst[i]) {
      ARTS_DEBUG("src: %u dst: %u", src[i], dst[i]);
      dst[i] = src[i];
      count++;
    }
    if (src[i] != (unsigned int)-1)
      count2++;
  }
  ARTS_DEBUG("%lu %u %u", host->guid, count, count2);
  versionUnlock(host);
}

void artsAddDbUnsignedInt(artsLCMeta_t *host, artsLCMeta_t *dev) {
  unsigned int count = 0;
  unsigned int numElem = host->dataSize / sizeof(unsigned int);
  unsigned int *dst = (unsigned int *)host->data;
  unsigned int *src = (unsigned int *)dev->data;
  unsigned int hostVersion = versionLock(host);
  for (unsigned int i = 0; i < numElem; i++) {
    dst[i] += src[i];
  }
  ARTS_DEBUG("%lu %u", host->guid, count);
  versionUnlock(host);
}

void artsXorDbUint64(artsLCMeta_t *host, artsLCMeta_t *dev) {
  unsigned int count = 0;
  unsigned int numElem = host->dataSize / sizeof(uint64_t);
  uint64_t *dst = (uint64_t *)host->data;
  uint64_t *src = (uint64_t *)dev->data;
  uint64_t hostVersion = versionLock(host);
  for (unsigned int i = 0; i < numElem; i++) {
    ARTS_DEBUG("xor[%u]: %lu -- %lu = %lu", i, dst[i], src[i], dst[i] ^ src[i]);
    dst[i] ^= src[i];
    count++;
  }
  ARTS_DEBUG("%lu %u", host->guid, count);
  versionUnlock(host);
}

/***********************************************************************/

__global__ void artsCopyGpuDb(struct arts_db *sink, struct arts_db *src) {
  unsigned int *srcData = (unsigned int *)(src + 1);
  unsigned int *sinkData = (unsigned int *)(sink + 1);

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  sinkData[index] = srcData[index];
}

__global__ void artsMinGpuDbUnsignedInt(struct arts_db *sink,
                                        struct arts_db *src) {
  unsigned int *srcData = (unsigned int *)(src + 1);
  unsigned int *sinkData = (unsigned int *)(sink + 1);

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (srcData[index] < sinkData[index])
    sinkData[index] = srcData[index];
}

__global__ void artsNonZeroGpuDbUnsignedInt(struct arts_db *sink,
                                            struct arts_db *src) {
  unsigned int *srcData = (unsigned int *)(src + 1);
  unsigned int *sinkData = (unsigned int *)(sink + 1);

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (sinkData[index] > 0)
    sinkData[index] = srcData[index];
}

__global__ void artsAddGpuDbUnsignedInt(struct arts_db *sink,
                                        struct arts_db *src) {
  unsigned int *srcData = (unsigned int *)(src + 1);
  unsigned int *sinkData = (unsigned int *)(sink + 1);

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  sinkData[index] += srcData[index];
}

__global__ void artsXorGpuDbUint64(struct arts_db *sink, struct arts_db *src) {
  unsigned long long *srcData = (unsigned long long *)(src + 1);
  unsigned long long *sinkData = (unsigned long long *)(sink + 1);

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  sinkData[index] ^= srcData[index];
}

/***********************************************************************/

#define GPUGROUPSIZE 4
#define GPUNUMGROUP 2

void gpuReductionLaunch(int root, int a, int b, unsigned int *remMask,
                        arts_guid_t guid, unsigned int size,
                        artsLCSyncFunctionGpu_t fn_ptr) {
  if (a < 0 || b < 0)
    return;

  if (root != a && root != b) {
    ARTS_INFO("LC Reduction tree invalid root! %d %d %d", root, a, b);
    arts_debug_generate_seg_fault();
  }

  ARTS_DEBUG("A: %d B: %d -> Root: %d guid: %lu", a, b, root, guid);
  unsigned int toRemove = (root == a) ? b : a;
  *remMask &= ~(1 << toRemove);

  void *db_data = arts_gpu_route_table_lookup_db_res(guid, root, NULL, NULL, false);
  void *dst = (void *)(((char *)db_data) + size);
  ARTS_DEBUG("%d %p %p", root, db_data, dst);

  void *src = arts_gpu_route_table_lookup_db_res(guid, toRemove, NULL, NULL, false);
  ARTS_DEBUG("%d %p", toRemove, src);

  ARTS_DEBUG("src: %p dst: %p size: %u", src, dst, size);
  reduce_datafrom_gpus(dst, root, src, toRemove, size, fn_ptr,
                     lcSyncElementSize[arts_node_info.gpu_lc_sync], db_data);
}

void gpuShadowReductionLaunch(int root, arts_guid_t guid, unsigned int size,
                              artsLCSyncFunctionGpu_t fn_ptr) {
  void *sink = arts_gpu_route_table_lookup_db_res(guid, root, NULL, NULL, false);
  void *src = (void *)(((char *)sink) + size);

  do_reduction_now(root, sink, src, fn_ptr, sizeof(unsigned int), size);
}

void gpuCopyLaunch(int root, int a, int b, bool srcShadow, bool dstShadow,
                   arts_guid_t guid, unsigned int size) {

  if (a < 0 || b < 0)
    return;

  if (root != a && root != b) {
    ARTS_INFO("LC Reduction tree invalid root! %d %d %d", root, a, b);
    arts_debug_generate_seg_fault();
  }

  ARTS_DEBUG("A: %d B: %d -> Root: %d", a, b, root);
  unsigned int toRemove = (root == a) ? b : a;

  void *dst = arts_gpu_route_table_lookup_db_res(guid, root, NULL, NULL, false);
  if (dstShadow)
    dst = (void *)(((char *)dst) + size);
  ARTS_DEBUG("%d %p", root, dst);

  void *src = arts_gpu_route_table_lookup_db_res(guid, toRemove, NULL, NULL, false);
  if (srcShadow)
    src = (void *)(((char *)src) + size);
  ARTS_DEBUG("%d %p", toRemove, src);

  ARTS_DEBUG("src: %p dst: %p size: %u", src, dst, size);
  copy_gputo_gpu(dst, root, src, toRemove, size);
}

void findRoots(unsigned int local, int *roots) {
  for (unsigned int i = 0; i < GPUNUMGROUP; i++)
    roots[i] = -1;

  // Make a mask of 4 bits (GPUGROUPSIZE)
  unsigned int mask = 0;
  for (unsigned int j = 0; j < GPUGROUPSIZE; j++) {
    unsigned int bit = 1 << j;
    mask |= bit;
  }

  // Assumes grid... Add shifted local mask with mask and or results
  unsigned int localRoots = (unsigned int)-1;
  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    unsigned int tempLocal = local >> (i * GPUGROUPSIZE);
    unsigned int temp = mask & tempLocal;
    localRoots &= temp;
  }

  // Recover the roots
  for (int i = 0; i < GPUGROUPSIZE; i++) {
    if (localRoots & (1 << i)) {
      ARTS_DEBUG("FOUND MATCHING ROOTS");
      for (unsigned int j = 0; j < GPUNUMGROUP; j++)
        roots[j] = i + j * GPUGROUPSIZE;
      return;
    }
  }

  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    // ARTS_INFO("i: %u", i);
    for (unsigned int j = 0; j < GPUGROUPSIZE; j++) {
      unsigned int bit = i * GPUGROUPSIZE + j;
      // ARTS_INFO("bit: %u", bit);
      if (local & (1 << bit)) {
        roots[i] = bit;
        break;
      }
    }
  }
}

typedef struct {
  int a;
  int b;
  int root;
  int level;
} trav;

void addToTrav(int root, int a, int b, unsigned int level, unsigned int *size,
               trav *ds, unsigned int *maxLevel) {
  if (a < 0 || b < 0)
    return;

  unsigned int index = (*size);
  *size = *size + 1;
  ds[index].a = a;
  ds[index].b = b;
  ds[index].root = root;
  ds[index].level = level;

  *maxLevel = (*maxLevel < level) ? level : *maxLevel;
}

int gpuTreeReductionRec(int root, unsigned int start, unsigned int stop,
                        unsigned int mask, unsigned int level,
                        unsigned int *list_size, trav *list,
                        unsigned int *maxLevel) {
  int localRoot = -1;
  // ARTS_INFO("root: %u start: %u stop: %u", root, start, stop);
  int gpu_id[2] = {(int)start, (int)stop};

  if (stop - start > 1) // Recursive call
  {
    unsigned int middle = (1 + stop - start) / 2;
    gpu_id[0] = gpuTreeReductionRec(root, start, start + middle - 1, mask,
                                   level + 1, list_size, list, maxLevel);
    gpu_id[1] = gpuTreeReductionRec(root, start + middle, stop, mask, level + 1,
                                   list_size, list, maxLevel);
  }

  bool startFound = (gpu_id[0] < 0) ? false : ((mask & (1 << gpu_id[0])) != 0);
  bool stopFound = (gpu_id[1] < 0) ? false : ((mask & (1 << gpu_id[1])) != 0);

  if (startFound && stopFound) // Both are in the mask
  {
    if (root == gpu_id[0] || root == gpu_id[1])
      localRoot = root;
    else
      localRoot = gpu_id[0];            // This is the min
  } else if (startFound && !stopFound) // Only start is in the mask
  {
    gpu_id[1] = -1;
    localRoot = gpu_id[0];
  } else if (!startFound && stopFound) // Only stop is in the mask
  {
    gpu_id[0] = -1;
    localRoot = gpu_id[1];
  } else // Neither start or stop is in the mask
  {
    gpu_id[1] = -1;
    gpu_id[0] = -1;
    // localRoot = -1;
  }

  addToTrav(localRoot, gpu_id[0], gpu_id[1], level, list_size, list, maxLevel);
  return localRoot;
}

void gpuTreeReductionStart(unsigned int mask, unsigned int *list_size,
                           trav *list, unsigned int *maxLevel) {
  int root[GPUNUMGROUP];
  findRoots(mask, root);
  for (unsigned int i = 0; i < GPUNUMGROUP; i++) {
    ARTS_DEBUG("Root[%d]: %d", i, root[i]);
    gpuTreeReductionRec(root[i], i * GPUGROUPSIZE, ((i + 1) * GPUGROUPSIZE) - 1,
                        mask, 2, list_size, list, maxLevel);
  }
  addToTrav(root[0], root[0], root[1], 1, list_size, list, maxLevel);
}

unsigned int gpuTreeReduction(unsigned int mask, arts_guid_t guid,
                              unsigned int db_size,
                              artsLCSyncFunctionGpu_t dbFn) {
  ARTS_DEBUG("mask: %u", mask);
  unsigned int maxLevel = 0;
  unsigned int list_size = 0;
  trav list[GPUNUMGROUP * GPUGROUPSIZE];

  gpuTreeReductionStart(mask, &list_size, list, &maxLevel);

  unsigned int remMask = mask;

  for (unsigned int i = maxLevel; i > 0; i--) {
    for (unsigned int j = 0; j < list_size; j++) {
      if (list[j].level == i)
        gpuReductionLaunch(list[j].root, list[j].a, list[j].b, &remMask, guid,
                           db_size, dbFn);
    }
  }
  ARTS_DEBUG("remMask: %u", remMask);
  return remMask;
}

/***********************************************************/

bool checkMax(unsigned int currentSize, unsigned int *visited,
              unsigned int *maxSize, unsigned int *maxVisited,
              unsigned int cycleSize) {
  if (*maxSize < currentSize) {
    *maxSize = currentSize;
    memcpy(maxVisited, visited, sizeof(unsigned int) * currentSize);
    return (currentSize == cycleSize) &&
           (maxVisited[0] == maxVisited[cycleSize - 1]);
  }
  return false;
}

extern bool **gpuAdjList;
unsigned int gpuDepthFirstRec(unsigned int vertex, unsigned int cycleSize,
                              unsigned int mask, unsigned int current,
                              unsigned int *visited, unsigned int *maxSize,
                              unsigned int *maxVisited) {
  unsigned int order = arts_get_total_gpus();
  visited[current++] = vertex; // Record order visited

  bool ret = checkMax(current, visited, maxSize, maxVisited, cycleSize);

  unsigned int temp = ~(1 << vertex); // Mark off list
  mask &= temp;

  if (current + 1 == cycleSize) // This means the next iteration is the final
                                // one... Lets look for a cycle to make a ring
    mask |= 1 << visited[0];

  if (current < cycleSize) {
    for (unsigned int i = 0; i < order; i++) {
      if ((mask & (1 << i)) && gpuAdjList[vertex][i]) {
        ARTS_INFO("%u -> %u", vertex, i);
        if (gpuDepthFirstRec(i, cycleSize, mask, current, visited, maxSize,
                             maxVisited))
          return true;
      }
    }
  }
  return ret;
}

unsigned int *gpuDepthFirst(unsigned int mask, unsigned int *maxSize) {
  unsigned int *ret = NULL;
  unsigned int cycleSize = 1; // Add one for the backedge
  for (unsigned int i = 0; i < sizeof(mask) * 8; i++) {
    if (mask & (1 << i))
      cycleSize++;
  }

  unsigned int *visited =
      (unsigned int *)arts_calloc(cycleSize, sizeof(unsigned int));
  unsigned int *maxVisited =
      (unsigned int *)arts_calloc(cycleSize, sizeof(unsigned int));
  for (unsigned int i = 0; i < arts_get_total_gpus(); i++) {
    if (mask & (1 << i)) {
      ARTS_INFO("i: %u", i);
      if (gpuDepthFirstRec(i, cycleSize, mask, 0, visited, maxSize,
                           maxVisited)) {
        ret = maxVisited;
        break;
      }
    }
  }
  arts_free(visited);
  if (!ret)
    arts_free(maxVisited);
  return ret;
}

bool gpuRingReduction(unsigned int mask, unsigned int guid, unsigned int db_size,
                      artsLCSyncFunctionGpu_t fn_ptr) {
  // unsigned int remMask = mask;
  unsigned int cycleSize = 0;
  unsigned int *cycle = gpuDepthFirst(mask, &cycleSize);
  if (cycle && cycleSize > 1) {
    unsigned int num_gpus = cycleSize - 1;
    ARTS_INFO("Cycle Size:%u", cycleSize);
    for (unsigned int i = 0; i < 1; i++) {
      for (unsigned int j = 1; j < cycleSize; j++)
        gpuCopyLaunch(cycle[j], cycle[j - 1], cycle[j], (i == 0) ? false : true,
                      true, guid, db_size);

      for (unsigned int j = 0; j < num_gpus; j++)
        gpuShadowReductionLaunch(cycle[j], guid, db_size, fn_ptr);
    }
    return true;
  }
  return false;
}

void gpuLCInvalidate(unsigned int mask, arts_guid_t guid) {
  for (unsigned int i = 0; i < arts_get_total_gpus(); i++) {
    if (mask & (1 << i)) {
      arts_gpu_invalidate_on_route_table(guid, i);
      arts_gpu_route_table_return_db(guid, true, i);
    }
  }
}

unsigned int gpuLCReturnDb(unsigned int mask, arts_guid_t guid) {
  unsigned int remMask = 0;
  for (unsigned int i = 0; i < arts_get_total_gpus(); i++) {
    if (mask & (1 << i)) {
      if (!remMask && i == 2)
        remMask = 1 << i;
      else
        arts_gpu_route_table_return_db(guid, false, i);
    }
  }
  return remMask;
}

unsigned int gpuLCReduce(arts_guid_t guid, struct arts_db *db,
                         artsLCSyncFunctionGpu_t dbFn, bool *copyOnly) {
  *copyOnly = false;
  unsigned int remMask = 0;
  unsigned int size = db->header.size;
  // struct arts_db *shadow_copy = (struct arts_db *)(((char *)db) + size);

  arts_writer_lock(&db->reader, &db->writer);
  unsigned int mask = arts_gpu_lookup_db_fix(guid);
  if (mask) {
    // RING IS NOT WORKING...
    //  if(db->version == shadow_copy->version && gpuRingReduction(mask, guid,
    //  size, dbFn))
    //  {
    //      remMask = gpuLCReturnDb(mask, guid);
    //      *copyOnly = true;
    //  }
    //  else
    {
      remMask = gpuTreeReduction(mask, guid, size, dbFn);
      gpuLCInvalidate(mask & ~remMask, guid);
    }
  }
  remMask = mask;
  arts_writer_unlock(&db->writer);
  return remMask;
}