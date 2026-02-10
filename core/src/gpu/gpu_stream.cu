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

// Some help https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/
// and
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
// Once this *class* works we will put a stream(s) in create a thread local
// stream.  Then we will push stuff!
#include "arts/gpu/gpu_stream.h"

#include "arts.h"
#include "arts/gas/guid.h"
#include "arts/gpu/gpu_lc_sync_functions.cuh"
#include "arts/gpu/gpu_route_table.h"
#include "arts/gpu/gpu_runtime.cuh"
#include "arts/gpu/gpu_stream_buffer.h"
#include "arts/introspection/metrics.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/compute/edt_functions.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/utils/atomics.h"
#include "arts/utils/deque.h"

int random(void *edt_packet);
int allOrNothing(void *edt_packet);
int atleastOne(void *edt_packet);
int hashOnDBZero(void *edt_packet);
int hashLargest(void *edt_packet);
int firstFit(uint64_t mask, uint64_t size, unsigned int total_threads);
int bestFit(uint64_t mask, uint64_t size, unsigned int total_threads);
int worstFit(uint64_t mask, uint64_t size, unsigned int total_threads);
int roundRobinFit(uint64_t mask, uint64_t size, unsigned int total_threads);
bool tryReserve(int gpu, uint64_t size, unsigned int threads);

volatile unsigned int hits = 0;
volatile unsigned int misses = 0;
volatile uint64_t free_bytes = 0;

arts_gpu_t *arts_gpus;

typedef int (*locality_t)(void *edt);

locality_t localityScheme[] = {random, allOrNothing, atleastOne, hashOnDBZero,
                               hashLargest};

locality_t locality; // Locality function ptr

typedef int (*fit_t)(uint64_t mask, uint64_t size, unsigned int total_threads);

fit_t fitScheme[] = {firstFit, bestFit, worstFit, roundRobinFit};

fit_t fit; // Fit function ptr

__thread volatile unsigned int *newEdtLock = 0;
__thread arts_array_list_t *newEdts = NULL;

// These are for the library version of GPU EDTs
// The user can query to get these values
// We still want to collect them for scheduling purposes
__thread dim3 *arts_local_grid;
__thread dim3 *arts_local_block;
__thread cudaStream_t *arts_local_stream;
__thread int arts_local_gpu_id;

#ifdef __cplusplus
extern "C" {
#endif
extern void initPerGpu(unsigned int node_id, int devId, cudaStream_t *stream,
                       int argc, char **argv) __attribute__((weak));
extern void cleanPerGpu(unsigned int node_id, int devId, cudaStream_t *stream)
    __attribute__((weak));
#ifdef __cplusplus
}
#endif

bool **gpuAdjList = NULL;
void artsFullyConnectGpus(bool p2p, bool disconnectP2P) {
  if (!gpuAdjList) {
    gpuAdjList = (bool **)arts_calloc(arts_node_info.gpu, sizeof(bool *));
    for (unsigned int i = 0; i < arts_node_info.gpu; i++)
      gpuAdjList[i] = (bool *)arts_calloc(arts_node_info.gpu, sizeof(bool));
  }
  if (p2p) {
    for (int src = 0; src < arts_node_info.gpu; src++) {
      arts_cuda_set_device(src, false);
      for (int dst = 0; dst < arts_node_info.gpu; dst++) {
        if (src != dst) {
          int hasAccess = 0;
          CHECKCORRECT(cudaDeviceCanAccessPeer(&hasAccess, src, dst));
          if (hasAccess) {
            if (disconnectP2P) {
              CHECKCORRECT(cudaDeviceDisablePeerAccess(dst));
            } else {
              gpuAdjList[src][dst] = 1;
              CHECKCORRECT(cudaDeviceEnablePeerAccess(dst, 0));
            }
          }
        }
      }
    }
  }
}

void arts_node_init_gpus() {
  int numAvailGpus = 0;
  locality = localityScheme[arts_node_info.gpu_locality];
  fit = fitScheme[arts_node_info.gpu_fit];
  CHECKCORRECT(cudaGetDeviceCount(&numAvailGpus));
  if (numAvailGpus < arts_node_info.gpu) {
    ARTS_INFO("Requested %d gpus but only %d available\n", numAvailGpus,
              arts_node_info.gpu);
    arts_node_info.gpu = numAvailGpus;
  }

  ARTS_DEBUG(
      "gpu_route_table_size: %u gpu_route_table_entries: %u free_db_after_gpu_run: "
      "%u run_gpu_gc_idle: %u run_gpu_gc_pre_edt: %u delete_zeros_gpu_gc: %u\n",
      arts_node_info.gpu_route_table_size, arts_node_info.gpu_route_table_entries,
      arts_node_info.free_db_after_gpu_run, arts_node_info.run_gpu_gc_idle,
      arts_node_info.run_gpu_gc_pre_edt, arts_node_info.delete_zeros_gpu_gc);

  ARTS_DEBUG("NUM DEV: %d\n", arts_node_info.gpu);
  arts_gpus = (arts_gpu_t *)arts_calloc(arts_node_info.gpu, sizeof(arts_gpu_t));

  arts_cuda_set_device(-1, true);

  // Initialize arts_gpu with 1 stream/GPU
  for (int i = 0; i < arts_node_info.gpu; ++i) {
    arts_gpus[i].device = i;
    ARTS_DEBUG("Setting %d\n", i);
    arts_cuda_set_device(i, false);
    CHECKCORRECT(cudaStreamCreate(&arts_gpus[i].stream)); // Make it scalable
    arts_node_info.gpu_route_table[i] = arts_gpu_new_route_table(
        arts_node_info.gpu_route_table_entries, arts_node_info.gpu_route_table_size);
    size_t tempFreeMem = 0;
    size_t tempMaxMem = 0;
    CHECKCORRECT(cudaMemGetInfo((size_t *)&tempFreeMem, (size_t *)&tempMaxMem));
    CHECKCORRECT(
        cudaGetDeviceProperties(&arts_gpus[i].prop, arts_gpus[i].device));
    arts_gpus[i].availGlobalMem = (uint64_t)tempFreeMem;
    arts_gpus[i].totalGlobalMem = (uint64_t)tempMaxMem;
    if (arts_gpus[i].availGlobalMem > arts_node_info.gpu_max_memory)
      arts_gpus[i].availGlobalMem = arts_node_info.gpu_max_memory;
    ARTS_DEBUG("to Start: %lu\n", arts_gpus[i].availGlobalMem);
  }

  artsFullyConnectGpus(arts_node_info.gpu_p2p, false);

  arts_cuda_restore_device();
}

void arts_init_per_gpu_wrapper(int argc, char **argv) {
  if (initPerGpu) {
    arts_cuda_set_device(-1, true);
    for (int i = 0; i < arts_node_info.gpu; ++i) {
      ARTS_DEBUG("Set device: %u\n", i);
      arts_cuda_set_device(i, false);
      initPerGpu(arts_global_rank_id, i, &arts_gpus[i].stream, argc, argv);
    }
    arts_cuda_restore_device();
  }
}

void arts_worker_init_gpus() {
  newEdtLock = (unsigned int *)arts_calloc(1, sizeof(unsigned int));
  newEdts = arts_new_array_list(sizeof(void *), 32);
}

void arts_store_new_edts(void *edt) {
  arts_lock(newEdtLock);
  arts_push_to_array_list(newEdts, &edt);
  arts_unlock(newEdtLock);
}

void arts_handle_new_edts() {
  arts_lock(newEdtLock);
  uint64_t size = arts_length_array_list(newEdts);
  if (size) {
    for (uint64_t i = 0; i < size; i++) {
      struct arts_edt **edt =
          (struct arts_edt **)arts_get_from_array_list(newEdts, i);
      if ((*edt)->header.type == ARTS_EDT)
        arts_deque_push_front(arts_thread_info.my_deque, (*edt), 0);
      if ((*edt)->header.type == ARTS_GPU_EDT)
        arts_deque_push_front(arts_thread_info.my_gpu_deque, (*edt), 0);
    }
    arts_reset_array_list(newEdts);
  }
  arts_unlock(newEdtLock);
}

void arts_cleanup_gpus() {
  uint64_t freedSize = 0;
  arts_cuda_set_device(-1, false);

  artsFullyConnectGpus(arts_node_info.gpu_p2p, true);

  for (int i = 0; i < arts_node_info.gpu; i++) {
    arts_cuda_set_device(arts_gpus[i].device, false);
    if (cleanPerGpu)
      cleanPerGpu(arts_global_rank_id, i, &arts_gpus[i].stream);
    freedSize += arts_gpu_free_all(arts_gpus[i].device);
    CHECKCORRECT(cudaStreamSynchronize(arts_gpus[i].stream));
    CHECKCORRECT(cudaStreamDestroy(arts_gpus[i].stream));
  }
  arts_cuda_restore_device();
  ARTS_INFO("Occupancy :\n");
  for (int i = 0; i < arts_get_num_gpus(); ++i)
    ARTS_INFO("\tGPU[%d] = %f\n", i, arts_gpus[i].occupancy);
  ARTS_INFO("HITS: %u MISSES: %u FREED BYTES: %u BYTES FREED ON EXIT %lu\n",
            hits, misses, free_bytes, freedSize);
  ARTS_INFO("HIT RATIO: %lf\n", (double)hits / (double)(hits + misses));
}

void cudart_cb artsWrapUp(cudaStream_t stream, cudaError_t status, void *data) {
  // artsToggleThreadInspection();

  arts_gpu_clean_up_t *gc = (arts_gpu_clean_up_t *)data;

  arts_gpu_t *arts_gpu = &arts_gpus[gc->gpu_id];
  arts_atomic_sub(&arts_gpu->availableEdtSlots, 1U);
  arts_atomic_sub(&arts_gpu->runningEdts, 1U);

  // Shouldn't have to touch newly ready edts regardless of streams and devices
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)gc->edt;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);

  unsigned int total_threads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;
  arts_atomic_sub(&arts_gpu->availableThreads, total_threads);

  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr) {
      if (arts_guid_get_type(depv[i].guid) == ARTS_DB_GPU_WRITE) {
        arts_gpu_invalidate_route_tables(depv[i].guid, gc->gpu_id);
      }
      // True says to mark it for deletion... Change this to false to further
      // delay delete!
      //  bool mark_delete = (arts_guid_get_type(depv[i].guid) != ARTS_DB_GPU_WRITE)
      //  && arts_node_info.free_db_after_gpu_run;
      bool mark_delete = arts_node_info.free_db_after_gpu_run;
      bool res = arts_gpu_route_table_return_db(depv[i].guid, mark_delete, gc->gpu_id);
      // arts_gpu_route_table_return_db(depv[i].guid, arts_node_info.free_db_after_gpu_run,
      // gc->gpu_id);
      ARTS_DEBUG("Returning Db: %lu id: %d res: %u\n", depv[i].guid, gc->gpu_id,
                 res);
    }
  }

  // Definitely mark the dev closure to be deleted as there is no reuse!
  arts_gpu_route_table_return_db(edt->wrapperEdt.current_edt, true, gc->gpu_id);
  newEdtLock = gc->newEdtLock;
  newEdts = gc->newEdts;
  arts_gpu_host_wrap_up(gc->edt, edt->end_guid, edt->slot, edt->data_guid);
  ARTS_DEBUG("FINISHED GPU CALLS %s\n", cudaGetErrorString(status));
  // artsToggleThreadInspection();
}

void cudart_cb artsWrapUpHostFunc(void *data) {
  artsWrapUp(NULL, cudaSuccess, data);
}

void arts_schedule_to_gpu_internal(arts_edt_t fn_ptr, uint32_t paramc,
                               const uint64_t *paramv, uint32_t depc,
                               arts_edt_dep_t *depv, dim3 grid, dim3 block,
                               void *edt_ptr, arts_gpu_t *arts_gpu) {
  //    For now this should push the following into the stream:
  //    1. Copy data from host to device
  //    2. Push kernel
  //    3. Copy data from device to host
  //    4. Call host callback_t function arts_gpu_host_wrap_up

  static volatile unsigned int Gpulock;

  void *devClosure = NULL;
  void *host_closure = NULL;

  uint64_t *devGpuId = NULL;
  uint64_t *devParamv = NULL;
  arts_edt_dep_t *devDepv = NULL;

  arts_gpu_clean_up_t *hostGCPtr = NULL;
  uint64_t *hostGpuId = NULL;
  uint64_t *hostParamv = NULL;
  arts_edt_dep_t *hostDepv = NULL;

  ARTS_DEBUG("Paramc: %u Depc: %u edt: %p\n", paramc, depc, edt_ptr);

  // Get size of closure
  uint64_t devClosureSize =
      sizeof(uint64_t) * (paramc + 1) + sizeof(arts_edt_dep_t) * depc;
  uint64_t hostClosureSize = devClosureSize + sizeof(arts_gpu_clean_up_t);
  ARTS_DEBUG("devClosureSize: %u hostClosureSize: %u\n", devClosureSize,
             hostClosureSize);

  // Allocate Closure for GPU
  if (devClosureSize) {
    devClosure = arts_cuda_malloc(devClosureSize);
    devGpuId = (uint64_t *)devClosure;
    devParamv = devGpuId + 1;
    devDepv = (arts_edt_dep_t *)(devParamv + paramc);
    ARTS_DEBUG("Allocated dev closure\n");
  }

  if (hostClosureSize) {
    // Allocate closure for host
    host_closure = arts_cuda_malloc_host(hostClosureSize);
    hostGCPtr = (arts_gpu_clean_up_t *)host_closure;
    hostGpuId = (uint64_t *)(hostGCPtr + 1);
    hostParamv = hostGpuId + 1;
    hostDepv = (arts_edt_dep_t *)(hostParamv + paramc);
    ARTS_DEBUG("Allocated host closure\n");

    // Fill Host closure
    hostGCPtr->gpu_id = arts_gpu->device;
    hostGCPtr->newEdtLock = newEdtLock;
    hostGCPtr->newEdts = newEdts;
    hostGCPtr->devClosure = devClosure;
    hostGCPtr->edt = (struct arts_edt *)edt_ptr;
    *hostGpuId = (uint64_t)arts_gpu->device;
    for (unsigned int i = 0; i < paramc; i++)
      hostParamv[i] = paramv[i];
    ARTS_DEBUG("Filled host closure\n");

    arts_guid_t edt_guid = hostGCPtr->edt->current_edt;
    // arts_gpu_route_table_add_item_race(hostGCPtr, hostClosureSize, edt_guid,
    // arts_gpu->device);
    arts_gpu_route_table_add_item_race(hostGCPtr, devClosureSize, edt_guid,
                                 arts_gpu->device);
    ARTS_DEBUG("Added edt_guid: %lu size: %u to gpu: %d routing table\n",
               edt_guid, hostClosureSize, arts_gpu->device);
  }

  arts_gpu_edt_t *gpuEdt = (arts_gpu_edt_t *)hostGCPtr->edt;

  // Allocate space for DB on GPU and Move Data
  for (unsigned int i = 0; i < depc; ++i) {
    if (depv[i].ptr) {
      arts_type_t mode =
          arts_guid_get_type(depv[i].guid); // use this mode since it is the type
                                         // of DB depv[i].mode is access type
      unsigned int gpuVersion, time_stamp;
      void *data_ptr = arts_gpu_route_table_lookup_db(depv[i].guid, arts_gpu->device,
                                                &gpuVersion, &time_stamp);
      struct arts_db *db = (struct arts_db *)depv[i].ptr - 1;
      uint64_t size = db->header.size;
      uint64_t alloc_size = (mode == ARTS_DB_LC) ? size * 2 : size;
      if (!data_ptr) {
        bool successfulAdd = false;
        ARTS_DEBUG("WRAPPER SIZE: %lu\n", alloc_size);
        arts_item_wrapper_t *wrapper = arts_gpu_route_table_reserve_item_race(
            &successfulAdd, alloc_size, depv[i].guid, arts_gpu->device,
            false); //(mode == ARTS_DB_LC));

        if (successfulAdd) // We won, so allocate and move data
        {
          ARTS_DEBUG("Adding %lu %u id: %d mode: %s\n", depv[i].guid, alloc_size,
                     arts_gpu->device, arts_type_name[depv[i].mode]);
          data_ptr = arts_cuda_malloc(alloc_size);
          void *src = (void *)db;
          if (mode == ARTS_DB_LC)
            src = makeLCShadowCopy(db);
          if (depv[i].mode == ARTS_DB_LC_NO_COPY ||
              depv[i].mode == ARTS_DB_GPU_MEMSET)
            src = NULL;
          push_data_to_stream(arts_gpu->device, data_ptr, src, size,
                           arts_node_info.gpu_buff_on && !gpuEdt->lib);
          // Must have already launched the memcpy before setting realData or
          // races will ensue
          wrapper->realData = data_ptr;
          ARTS_DEBUG("Malloc[%d]: %p %p\n", arts_gpu->device, wrapper, data_ptr);
          arts_atomic_add(&misses, 1U);
        } else // Someone beat us to creating the data... So we must free
        {
          while (!arts_atomic_fetch_add_u64((uint64_t *)&wrapper->realData, 0))
            ; // Spin till the data memcpy is launched
          data_ptr = (void *)wrapper->realData;
          if (mode == ARTS_DB_GPU_WRITE && depv[i].mode == ARTS_DB_GPU_MEMSET) {
            push_data_to_stream(arts_gpu->device, data_ptr, NULL, size,
                             arts_node_info.gpu_buff_on && !gpuEdt->lib);
          }
          arts_atomic_add_u64(&arts_gpu->availGlobalMem, alloc_size);
          arts_atomic_add(&hits, 1U);
        }
      } else {
        arts_atomic_add_u64(&arts_gpu->availGlobalMem, alloc_size);
        arts_atomic_add(&hits, 1U);
      }
      struct arts_db *new_db = (struct arts_db *)data_ptr;
      hostDepv[i].ptr = (void *)(new_db + 1);
    } else {
      ARTS_DEBUG("Depv: %u is null edt: %lu\n", i,
                 gpuEdt->wrapperEdt.current_edt);
      hostDepv[i].ptr = NULL;
    }

    hostDepv[i].guid = depv[i].guid;
    hostDepv[i].mode = depv[i].mode;
  }
  ARTS_DEBUG("Allocated, added, and moved dbs\n");

  push_data_to_stream(arts_gpu->device, devClosure, (void *)hostGpuId,
                   devClosureSize, arts_node_info.gpu_buff_on && !gpuEdt->lib);
  ARTS_DEBUG("Filled GPU Closure\n");

  if (gpuEdt->lib) {
    arts_local_grid = &gpuEdt->grid;
    arts_local_block = &gpuEdt->block;
    arts_local_stream = &arts_gpu->stream;
    arts_local_gpu_id = arts_gpu->device;
    arts_set_thread_local_edt_info(hostGCPtr->edt);
    arts_route_table_reset_oo(hostGCPtr->edt->current_edt);

    hostGCPtr->edt->func_ptr(paramc, hostParamv, depc, hostDepv);
    ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_EDT, ARTS_METRIC_THREAD, 1);

    arts_unset_thread_local_edt_info();
  } else {
    push_kernel_to_stream(arts_gpu->device, paramc, devParamv, depc, devDepv, fn_ptr,
                       grid, block, arts_node_info.gpu_buff_on);
  }

  // Move data back
  for (unsigned int i = 0; i < depc; i++) {
    arts_type_t mode = arts_guid_get_type(depv[i].guid);
    if (depv[i].ptr && mode == ARTS_DB_GPU_WRITE) {
      struct arts_db *db = (struct arts_db *)depv[i].ptr - 1;
      size_t size = (size_t)(db->header.size - sizeof(struct arts_db));
      get_data_from_stream(arts_gpu->device, depv[i].ptr, hostDepv[i].ptr, size,
                        arts_node_info.gpu_buff_on && !gpuEdt->lib);
      // CHECKCORRECT(cudaStreamSynchronize(arts_gpu->stream));
    }
  }

  push_wrap_up_to_stream(arts_gpu->device, host_closure,
                     arts_node_info.gpu_buff_on && !gpuEdt->lib);
}

void arts_schedule_to_gpu(arts_edt_t fn_ptr, uint32_t paramc, const uint64_t *paramv,
                       uint32_t depc, arts_edt_dep_t *depv, void *edt_ptr,
                       arts_gpu_t *arts_gpu) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_ptr;
  arts_schedule_to_gpu_internal(fn_ptr, paramc, paramv, depc, depv, edt->grid,
                            edt->block, edt_ptr, arts_gpu);
}

void arts_gpu_synchronize(arts_gpu_t *arts_gpu) {
  CHECKCORRECT(cudaStreamSynchronize(arts_gpu->stream));
}

void arts_gpu_stream_busy(arts_gpu_t *arts_gpu) {
  CHECKCORRECT(cudaStreamQuery(arts_gpu->stream));
}

void free_gpu_item(arts_route_item_t *item) {
  arts_type_t type = arts_guid_get_type(item->key);
  arts_item_wrapper_t *wrapper = (arts_item_wrapper_t *)item->data;
  if (type == ARTS_GPU_EDT) {
    arts_gpu_clean_up_t *hostGCPtr = (arts_gpu_clean_up_t *)wrapper->realData;
    ARTS_DEBUG("FREEING DEV PTR: %p\n", hostGCPtr->devClosure);
    arts_cuda_free(hostGCPtr->devClosure);
    ARTS_DEBUG("FREEING HOST PTR: %p\n", hostGCPtr);
    arts_cuda_free_host(hostGCPtr);
  } else if (type == ARTS_DB_LC) {
    int valid_rank = -1;
    struct arts_db *db =
        (struct arts_db *)arts_route_table_lookup_db(item->key, &valid_rank, false);
    if (db) {
      unsigned int size = db->header.size;
      struct arts_db *tempSpace = (struct arts_db *)arts_malloc_align(size, 16);

      artsLCMeta_t host;
      host.guid = item->key;
      host.data = (void *)(db + 1);
      host.dataSize = db->header.size - sizeof(struct arts_db);
      host.hostVersion = &db->version;
      host.hostTimeStamp = &db->time_stamp;
      host.gpuVersion = 0;
      host.gpuTimeStamp = 0;
      host.gpu = -1;
      host.read_lock = &db->reader;
      host.write_lock = &db->writer;

      // arts_cuda_mem_cpy_from_dev(tempSpace, (void*) wrapper->realData, size);
      get_data_from_stream_now(arts_get_current_gpu(), tempSpace,
                           (void *)wrapper->realData, size, false);

      artsLCMeta_t dev;
      dev.guid = item->key;
      dev.data = (void *)(tempSpace + 1);
      dev.dataSize = tempSpace->header.size - sizeof(struct arts_db);
      dev.hostVersion = &tempSpace->version;
      dev.hostTimeStamp = &tempSpace->time_stamp;
      dev.gpuVersion = item->touched;
      dev.gpuTimeStamp = wrapper->time_stamp;
      dev.gpu = -1;
      dev.read_lock = NULL;
      dev.write_lock = NULL;

      lcSyncFunction[arts_node_info.gpu_lc_sync](&host, &dev);

      arts_route_table_return_db(item->key, false);
      arts_free(tempSpace);
      arts_cuda_free((void *)wrapper->realData);

      ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_SYNC_DELETE, ARTS_METRIC_THREAD, 1);
    } else
      ARTS_INFO("Trying to delete an LC but there is no DB to back up to\n");
  } else if (type > ARTS_BUFFER && type < ARTS_LAST_TYPE) // DBs
    arts_cuda_free((void *)wrapper->realData);

  wrapper->realData = NULL;
  wrapper->time_stamp = 0;
  item->key = 0;
  item->lock = 0;
  item->touched = 0;
}

__thread unsigned int runGCFlag = 0;

bool tryReserve(int gpu, uint64_t size, unsigned int threads) {
  arts_gpu_t *arts_gpu = &arts_gpus[gpu];
  ARTS_DEBUG("Trying to reserve %lu of available %lu on GPU[%d]\n", size,
             arts_gpu->availGlobalMem, arts_gpu->device);
  // if(arts_atomic_fetch_add(&arts_gpu->availableThreads, threads) < 1024)
  {
    if (arts_atomic_fetch_add(&arts_gpu->availableEdtSlots, 1U) <
        arts_node_info.gpu_max_edts) {
      volatile uint64_t availSize = arts_gpu->availGlobalMem;
      while (availSize >= size) {
        if (arts_atomic_cswap_u64(&arts_gpu->availGlobalMem, availSize,
                               availSize - size)) {
          runGCFlag = 0;
          return true;
        }
        availSize = arts_gpu->availGlobalMem;
      }
      runGCFlag = gpu + 1;
    }
    arts_atomic_sub(&arts_gpu->availableEdtSlots, 1U);
  }
  // arts_atomic_sub(&arts_gpu->availableThreads, threads);
  ARTS_DEBUG("Failed Avail threads: %u + %u\n", arts_gpu->availableThreads,
             threads);
  ARTS_DEBUG("Failed to reserve %lu of available %lu on GPU[%d]\n", size,
             arts_gpu->availGlobalMem, arts_gpu->device);
  return false;
}

int firstFit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int random = jrand48(arts_thread_info.drand_buf);
  for (int i = 0; i < arts_node_info.gpu; i++) {
    int index = (i + random) % arts_node_info.gpu;
    uint64_t checkMask = 1 << index;
    if (mask && checkMask)
      if (tryReserve(index, size, total_threads)) {
        ARTS_DEBUG("Reserved Successfully on %u\n", index);
        return index;
      }
  }
  return -1;
}

int roundRobinFit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  static volatile unsigned int next = 0;
  unsigned int start = arts_atomic_fetch_add(&next, 1U);
  for (int i = 0; i < arts_node_info.gpu; i++) {
    int index = (i + start) % arts_node_info.gpu;
    uint64_t checkMask = 1 << index;
    if (mask && checkMask)
      if (tryReserve(index, size, total_threads)) {
        ARTS_DEBUG("Reserved Successfully on %u\n", index);
        return index;
      }
  }
  return -1;
}

int bestFit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int selectedGpu = -1;
  uint64_t selectedGpuAvailSize;
  int random = jrand48(arts_thread_info.drand_buf);
  for (int i = 0; i < arts_node_info.gpu; i++) {
    int index = (i + random) % arts_node_info.gpu;
    uint64_t checkMask = 1 << index;
    if (mask && checkMask) {
      if (selectedGpu != -1) {
        if (arts_gpus[index].availGlobalMem - size > selectedGpuAvailSize)
          continue;
      }
      if (tryReserve(index, size, total_threads)) {
        // If successful relinquish previous allocation.
        arts_atomic_add_u64(&arts_gpus[selectedGpu].availGlobalMem, size);
        selectedGpu = index;
        selectedGpuAvailSize = arts_gpus[index].availGlobalMem;
      }
    }
  }
  return selectedGpu;
}

int worstFit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int selectedGpu = -1;
  uint64_t selectedGpuAvailSize;
  int random = jrand48(arts_thread_info.drand_buf);
  for (int i = 0; i < arts_node_info.gpu; i++) {
    int index = (i + random) % arts_node_info.gpu;
    uint64_t checkMask = 1 << index;
    if (mask && checkMask) {
      if (selectedGpu != -1) {
        if (arts_gpus[index].availGlobalMem - size < selectedGpuAvailSize)
          continue;
      }
      if (tryReserve(index, size, total_threads)) {
        // If successful relinquish previous allocation.
        arts_atomic_add_u64(&arts_gpus[selectedGpu].availGlobalMem, size);
        selectedGpu = index;
        selectedGpuAvailSize = arts_gpus[index].availGlobalMem;
      }
    }
  }
  return selectedGpu;
}

uint64_t getDbSizeNeeded(uint32_t depc, arts_edt_dep_t *depv) {
  uint64_t size = 0;
  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr) {
      struct arts_db *db = (struct arts_db *)depv[i].ptr - 1;
      size += db->header.size;
      if (arts_guid_get_type(depv[i].guid) == ARTS_DB_LC)
        size += db->header.size;
    }
  }
  return size;
}

int random(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  // Size to be allocated on the GPU
  uint64_t size = sizeof(uint64_t) * paramc + sizeof(arts_edt_dep_t) * depc +
                  getDbSizeNeeded(depc, depv);
  uint64_t mask = ~0;
  return fit(mask, size, total_threads);
}

int allOrNothing(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  // Size to be allocated on the GPU
  uint64_t size = sizeof(uint64_t) * paramc + sizeof(arts_edt_dep_t) * depc +
                  getDbSizeNeeded(depc, depv);
  uint64_t mask = 0;
  for (unsigned int i = 0; i < depc; ++i)
    mask &= arts_gpu_lookup_db(depv[i].guid);

  ARTS_DEBUG("Mask: %p\n", mask);

  if (mask) { // All DBs in GPU
    return fit(mask, size,
               total_threads); // No need to fit since all Dbs are in a GPU
  }
  return random(edt_packet);
}

int atleastOne(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  // Size to be allocated on the GPU
  uint64_t size = sizeof(uint64_t) * paramc + sizeof(arts_edt_dep_t) * depc +
                  getDbSizeNeeded(depc, depv);
  uint64_t mask = 0;
  for (unsigned int i = 0; i < depc; ++i)
    mask |= arts_gpu_lookup_db(depv[i].guid);

  ARTS_DEBUG("Mask: %p\n", mask);

  if (mask) { // At least one DB in GPU
    return fit(mask, size, total_threads);
  }
  return random(edt_packet);
}

int hashOnDBZero(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  // Size to be allocated on the GPU
  uint64_t size = sizeof(uint64_t) * paramc + sizeof(arts_edt_dep_t) * depc +
                  getDbSizeNeeded(depc, depv);
  uint64_t key = (depv[0].guid) ? arts_get_guid_key(depv[0].guid) : 0;
  uint64_t index = key % (uint64_t)arts_node_info.gpu;
  if (index > arts_node_info.gpu) {
    ARTS_INFO("WHATS WRONG WITH THE HASH %u\n", index);
    arts_debug_generate_seg_fault();
  }
  ARTS_DEBUG("HASH: %lu %u\n", depv[0].guid, index);
  if (tryReserve(index, size, total_threads))
    return index;
  return -1;
}

int hashLargest(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  // Size to be allocated on the GPU
  uint64_t size = sizeof(uint64_t) * paramc + sizeof(arts_edt_dep_t) * depc +
                  getDbSizeNeeded(depc, depv);
  // uint64_t mask = 0;
  uint64_t largest = 0;
  for (unsigned int i = 0; i < depc; ++i) {
    uint64_t key = (depv[i].guid) ? arts_get_guid_key(depv[i].guid) : 0;
    largest = (key > largest) ? key : largest;
  }

  uint64_t index = largest % (uint64_t)arts_node_info.gpu;
  if (tryReserve(index, size, total_threads)) {
    ARTS_DEBUG("Index: %u\n", index);
    return index;
  }
  return -1;
}

int artsReserveEdtRequiredGpu(int *gpu, void *edt_packet) {
  bool ret = false;
  *gpu = -1;
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = edt->grid.x * edt->block.x +
                              edt->grid.y * edt->block.y +
                              edt->grid.z * edt->block.z;

  if (edt->gpuToRunOn > -1) {
    // Size to be allocated on the GPU
    uint64_t size = sizeof(uint64_t) * paramc + sizeof(arts_edt_dep_t) * depc +
                    getDbSizeNeeded(depc, depv);
    if (tryReserve(edt->gpuToRunOn, size, total_threads)) {
      *gpu = edt->gpuToRunOn;
      ret = true;
    }
  }
  return ret;
}

arts_gpu_t *arts_find_gpu(void *data) {
  arts_gpu_t *ret = NULL;
  int gpu;
  if (!artsReserveEdtRequiredGpu(&gpu, data))
    gpu = locality(data);
  ARTS_DEBUG("Choosing gpu: %d\n", gpu);
  if (gpu > -1 && gpu < arts_node_info.gpu)
    ret = &arts_gpus[gpu];

  return ret;
}