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
#include "arts/gpu/gpu_runtime.cuh"

#include "arts/gas/out_of_order.h"
#include "arts/gpu/gpu_lc_sync_functions.cuh"
#include "arts/gpu/gpu_route_table.h"
#include "arts/gpu/gpu_stream.h"
#include "arts/gpu/gpu_stream_buffer.h"
#include "arts/introspection/metrics.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/runtime/compute/edt_functions.h"
#include "arts/runtime/memory/db_functions.h"
#include "arts/runtime/sync/termination_detection.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/utils/atomics.h"
#include "arts/utils/deque.h"

__thread int artsSavedDeviceId = -1;
__thread int artsCurrentDeviceId = -1;

int arts_get_current_gpu() {
  if (artsCurrentDeviceId == -1)
    CHECKCORRECT(cudaGetDevice(&artsCurrentDeviceId));

  return artsCurrentDeviceId;
}

bool arts_cuda_set_device(int id, bool save) {
  if (artsCurrentDeviceId == -1)
    CHECKCORRECT(cudaGetDevice(&artsCurrentDeviceId));

  if (save)
    artsSavedDeviceId = artsCurrentDeviceId;

  if (id > -1 && id < arts_node_info.gpu && id != artsCurrentDeviceId) {
    CHECKCORRECT(cudaSetDevice(id));
    artsCurrentDeviceId = id;
    return true;
  }

  return false;
}

bool arts_cuda_restore_device() {
  return arts_cuda_set_device(artsSavedDeviceId, false);
}

void *arts_cuda_malloc_host(unsigned int size) {
  void *ptr = NULL;
  CHECKCORRECT(cudaMallocHost(&ptr, size));
  // ptr = arts_calloc(1, size);
  if (!ptr) {
    arts_debug_print_stack();
    exit(1);
  }
  return ptr;
}

void arts_cuda_free_host(void *ptr) {
  if (ptr)
    CHECKCORRECT(cudaFreeHost(ptr));
  // arts_free(ptr);
}

void *arts_cuda_malloc(unsigned int size) {
  void *ptr = NULL;
  CHECKCORRECT(cudaMalloc(&ptr, size));
  if (!ptr) {
    ARTS_INFO("arts_cuda_malloc failed %lu\n",
              arts_gpus[artsCurrentDeviceId].availGlobalMem);
    arts_debug_print_stack();
    exit(1);
  }
  return ptr;
}

void arts_cuda_free(void *ptr) {
  if (ptr)
    CHECKCORRECT(cudaFree(ptr));
}

void arts_cuda_mem_cpy_from_dev(void *dst, void *src, size_t count) {
  CHECKCORRECT(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

void arts_cuda_mem_cpy_to_dev(void *dst, void *src, size_t count) {
  CHECKCORRECT(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

dim3 *arts_get_gpu_grid() { return arts_local_grid; }

dim3 *arts_get_gpu_block() { return arts_local_block; }

cudaStream_t *arts_get_gpu_stream() { return arts_local_stream; }

int arts_get_gpu_id() { return arts_local_gpu_id; }

unsigned int arts_get_num_gpus() { return arts_node_info.gpu; }

arts_guid_t internalEdtCreateGpu(arts_edt_t func_ptr, arts_guid_t *guid,
                                unsigned int route, uint32_t paramc,
                                const uint64_t *paramv, uint32_t depc, dim3 grid,
                                dim3 block, arts_guid_t end_guid, uint32_t slot,
                                arts_guid_t data_guid, bool has_depv,
                                bool passThrough, bool lib, int gpuToRunOn) {
  //    ARTSEDTCOUNTERTIMERSTART(EDT_CREATE_COUNTER);
  unsigned int dep_space = (has_depv) ? depc * sizeof(arts_edt_dep_t) : 0;
  unsigned int edt_space =
      sizeof(arts_gpu_edt_t) + paramc * sizeof(uint64_t) + dep_space;

  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)arts_calloc(1, edt_space);
  edt->wrapperEdt.invalidateCount = 1;
  edt->grid = grid;
  edt->block = block;
  edt->gpuToRunOn = gpuToRunOn;
  edt->end_guid = end_guid;
  edt->slot = slot;
  edt->data_guid = data_guid;
  edt->passthrough = passThrough;
  edt->lib = lib;

  // artsIntrospectionEdtCreateBegin();
  bool created = arts_edt_create_internal(
      (struct arts_edt *)edt, ARTS_GPU_EDT, guid, route,
      arts_thread_info.cluster_id, edt_space, NULL_GUID, func_ptr, paramc, paramv,
      depc, true, NULL_GUID, has_depv, 0);
  // artsIntrospectionEdtCreateFinish(created);
  //    ARTSEDTCOUNTERTIMERENDINCREMENT(EDT_CREATE_COUNTER);
  return *guid;
}

arts_guid_t arts_edt_create_gpu_dep(arts_edt_t func_ptr, unsigned int route,
                               uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                               dim3 grid, dim3 block, arts_guid_t end_guid,
                               uint32_t slot, arts_guid_t data_guid,
                               bool has_depv) {
  arts_guid_t guid = NULL_GUID;
  return internalEdtCreateGpu(func_ptr, &guid, route, paramc, paramv, depc, grid,
                              block, end_guid, slot, data_guid, has_depv, false,
                              false, -1);
}

arts_guid_t arts_edt_create_gpu_pt_dep(arts_edt_t func_ptr, unsigned int route,
                                 uint32_t paramc, const uint64_t *paramv,
                                 uint32_t depc, dim3 grid, dim3 block,
                                 arts_guid_t end_guid, uint32_t slot,
                                 unsigned int pass_slot, bool has_depv) {
  arts_guid_t guid = NULL_GUID;
  return internalEdtCreateGpu(func_ptr, &guid, route, paramc, paramv, depc, grid,
                              block, end_guid, slot, (arts_guid_t)pass_slot,
                              has_depv, true, false, -1);
}

arts_guid_t arts_edt_create_gpu(arts_edt_t func_ptr, unsigned int route,
                            uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                            dim3 grid, dim3 block, arts_guid_t end_guid,
                            uint32_t slot, arts_guid_t data_guid) {
  return arts_edt_create_gpu_dep(func_ptr, route, paramc, paramv, depc, grid, block,
                             end_guid, slot, data_guid, true);
}

arts_guid_t arts_edt_create_gpu_with_guid(arts_edt_t func_ptr, arts_guid_t guid,
                                    uint32_t paramc, const uint64_t *paramv,
                                    uint32_t depc, dim3 grid, dim3 block,
                                    arts_guid_t end_guid, uint32_t slot,
                                    arts_guid_t data_guid) {
  return internalEdtCreateGpu(func_ptr, &guid, arts_guid_get_rank(guid), paramc,
                              paramv, depc, grid, block, end_guid, slot,
                              data_guid, true, false, false, -1);
}

arts_guid_t arts_edt_create_gpu_pt(arts_edt_t func_ptr, unsigned int route,
                              uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                              dim3 grid, dim3 block, arts_guid_t end_guid,
                              uint32_t slot, unsigned int pass_slot) {
  return arts_edt_create_gpu_pt_dep(func_ptr, route, paramc, paramv, depc, grid,
                               block, end_guid, slot, pass_slot, true);
}

arts_guid_t arts_edt_create_gpu_pt_with_guid(arts_edt_t func_ptr, arts_guid_t guid,
                                      uint32_t paramc, const uint64_t *paramv,
                                      uint32_t depc, dim3 grid, dim3 block,
                                      arts_guid_t end_guid, uint32_t slot,
                                      unsigned int pass_slot) {
  return internalEdtCreateGpu(func_ptr, &guid, arts_guid_get_rank(guid), paramc,
                              paramv, depc, grid, block, end_guid, slot,
                              (arts_guid_t)pass_slot, true, true, false, -1);
}

arts_guid_t arts_edt_create_gpu_lib(arts_edt_t func_ptr, unsigned int route,
                               uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                               dim3 grid, dim3 block) {
  arts_guid_t guid = NULL_GUID;
  return internalEdtCreateGpu(func_ptr, &guid, route, paramc, paramv, depc, grid,
                              block, NULL_GUID, 0, NULL_GUID, true, false, true,
                              -1);
}

arts_guid_t arts_edt_create_gpu_lib_with_guid(arts_edt_t func_ptr, arts_guid_t guid,
                                       uint32_t paramc, const uint64_t *paramv,
                                       uint32_t depc, dim3 grid, dim3 block) {
  return internalEdtCreateGpu(func_ptr, &guid, arts_guid_get_rank(guid), paramc,
                              paramv, depc, grid, block, NULL_GUID, 0,
                              NULL_GUID, true, false, true, -1);
}

arts_guid_t arts_edt_create_gpu_direct(arts_edt_t func_ptr, unsigned int route,
                                  unsigned int gpu, uint32_t paramc,
                                  const uint64_t *paramv, uint32_t depc, dim3 grid,
                                  dim3 block, arts_guid_t end_guid, uint32_t slot,
                                  arts_guid_t data_guid, bool has_depv) {
  arts_guid_t guid = NULL_GUID;
  return internalEdtCreateGpu(func_ptr, &guid, route, paramc, paramv, depc, grid,
                              block, end_guid, slot, data_guid, has_depv, false,
                              false, gpu);
}

arts_guid_t arts_edt_create_gpu_lib_direct(arts_edt_t func_ptr, unsigned int route,
                                     unsigned int gpu, uint32_t paramc,
                                     const uint64_t *paramv, uint32_t depc, dim3 grid,
                                     dim3 block) {
  arts_guid_t guid = NULL_GUID;
  return internalEdtCreateGpu(func_ptr, &guid, route, paramc, paramv, depc, grid,
                              block, NULL_GUID, 0, NULL_GUID, true, false, true,
                              gpu);
}

void arts_run_gpu(void *edt_packet, arts_gpu_t *arts_gpu) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  arts_edt_t func = edt->wrapperEdt.func_ptr;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);

  arts_cuda_set_device(arts_gpu->device, true);

  if (arts_node_info.run_gpu_gc_pre_edt) {
    // ARTS_INFO("Running Pre Edt GPU GC: %u\n", arts_gpu->device);
    uint64_t freeMemSize = arts_gpu_clean_up_route_table(
        (unsigned int)-1, arts_node_info.delete_zeros_gpu_gc,
        (unsigned int)arts_gpu->device);
    arts_atomic_add_u64(&arts_gpu->availGlobalMem, freeMemSize);
    arts_atomic_add_u64(&free_bytes, freeMemSize);
  }

  arts_atomic_add(&arts_gpu->runningEdts, 1U);

  prep_dbs(depc, depv, true);
  arts_schedule_to_gpu(func, paramc, paramv, depc, depv, edt_packet, arts_gpu);

  arts_cuda_restore_device();
}

void arts_gpu_host_wrap_up(void *edt_packet, arts_guid_t to_signal, uint32_t slot,
                       arts_guid_t data_guid) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);

  release_dbs(depc, depv, true);

  if (edt->lib) {
    edt->wrapperEdt.invalidateCount = 0;
    arts_route_table_fire_oo(edt->wrapperEdt.current_edt, arts_out_of_order_handler);
  } else if (edt->wrapperEdt.epoch_guid) {
    increment_finished_epoch(edt->wrapperEdt.epoch_guid);
  }

  ARTS_DEBUG("TO SIGNAL: %lu -> %lu slot: %u\n", to_signal, data_guid, slot);
  // Signal next
  if (to_signal) {
    if (edt->passthrough)
      arts_signal_edt(to_signal, slot, depv[data_guid].guid);
    else {
      arts_type_t mode = arts_guid_get_type(to_signal);
      if (mode == ARTS_EDT || mode == ARTS_GPU_EDT)
        arts_signal_edt(to_signal, slot, data_guid);
      if (mode == ARTS_EVENT)
        arts_event_satisfy_slot(to_signal, data_guid, slot);
      if (mode ==
          ARTS_BUFFER) // This is for us to be able to block in a host edt
        arts_set_buffer(to_signal, 0, 0);
      if (mode == ARTS_PERSISTENT_EVENT)
        arts_persistent_event_satisfy(to_signal, slot, true);
    }
  }
  arts_edt_delete((struct arts_edt *)edt_packet);
}

struct arts_edt *artsRuntimeStealGpuTask() {
  struct arts_edt *edt = NULL;
  if (arts_node_info.total_thread_count > 1) {
    long unsigned int steal_loc;
    do {
      steal_loc = jrand48(arts_thread_info.drand_buf);
      steal_loc = steal_loc % arts_node_info.total_thread_count;
    } while (steal_loc == arts_thread_info.thread_id);
    edt = (struct arts_edt *)arts_deque_pop_back(arts_node_info.gpu_deque[steal_loc]);
  }
  return edt;
}

bool arts_gpu_scheduler_loop() {
  arts_gpu_t *arts_gpu = NULL;
  arts_handle_new_edts();

  struct arts_edt *edt_found = (struct arts_edt *)NULL;
  if (!(edt_found =
            (struct arts_edt *)arts_deque_pop_front(arts_thread_info.my_gpu_deque))) {
    if (!edt_found)
      edt_found = artsRuntimeStealGpuTask();
  }

  bool ranGpuEdt = false;
  if (edt_found) {
    arts_gpu = arts_find_gpu(edt_found);
    if (arts_gpu) {
      arts_run_gpu(edt_found, arts_gpu);
      ranGpuEdt = true;
    } else
      arts_deque_push_front(arts_thread_info.my_gpu_deque, edt_found, 0);
  }

  if (!ranGpuEdt)
    check_streams(arts_node_info.gpu_buff_on);

  bool ranCpuEdt = arts_default_scheduler_loop();
  if (arts_node_info.run_gpu_gc_idle && !ranGpuEdt && !ranCpuEdt) {
    long unsigned int gpu_id = jrand48(arts_thread_info.drand_buf);
    gpu_id = gpu_id % arts_node_info.gpu;
    arts_gpu = &arts_gpus[gpu_id];
    ARTS_DEBUG("Running Idle GPU GC: %u\n", gpu_id);
    arts_cuda_set_device(arts_gpu->device, true);

    uint64_t freeMemSize = arts_gpu_clean_up_route_table(
        (unsigned int)-1, arts_node_info.delete_zeros_gpu_gc,
        (unsigned int)arts_gpu->device);
    arts_atomic_add_u64(&arts_gpu->availGlobalMem, freeMemSize);
    arts_atomic_add_u64(&free_bytes, freeMemSize);

    arts_cuda_restore_device();
  }

  return ranCpuEdt;
}

#define GCHARDLIMIT 2000000000000
__thread uint64_t backoff = 1;
__thread uint64_t gcCounter = 0;

bool arts_gpu_scheduler_backoff_loop() {
  arts_gpu_t *arts_gpu = NULL;
  arts_handle_new_edts();

  struct arts_edt *edt_found = (struct arts_edt *)NULL;
  if (!(edt_found =
            (struct arts_edt *)arts_deque_pop_front(arts_thread_info.my_gpu_deque))) {
    if (!edt_found)
      edt_found = artsRuntimeStealGpuTask();
  }

  bool ranGpuEdt = false;
  if (edt_found) {
    arts_gpu = arts_find_gpu(edt_found);
    if (arts_gpu) {
      arts_run_gpu(edt_found, arts_gpu);
      ranGpuEdt = true;
    } else
      arts_deque_push_front(arts_thread_info.my_gpu_deque, edt_found, 0);
  }

  if (!ranGpuEdt)
    check_streams(arts_node_info.gpu_buff_on);

  bool ranCpuEdt = arts_default_scheduler_loop();

  if (ranCpuEdt || ranGpuEdt)
    backoff = 1;

  if (!ranGpuEdt && !ranCpuEdt) {
    if (arts_node_info.run_gpu_gc_idle && gcCounter % backoff == 0) {
      long unsigned int gpu_id = jrand48(arts_thread_info.drand_buf);
      gpu_id = gpu_id % arts_node_info.gpu;
      arts_gpu = &arts_gpus[gpu_id];
      ARTS_DEBUG("Running Idle GPU GC: %u\n", gpu_id);
      arts_cuda_set_device(arts_gpu->device, true);

      uint64_t freeMemSize = arts_gpu_clean_up_route_table(
          (unsigned int)-1, arts_node_info.delete_zeros_gpu_gc,
          (unsigned int)arts_gpu->device);
      arts_atomic_add_u64(&arts_gpu->availGlobalMem, freeMemSize);
      arts_atomic_add_u64(&free_bytes, freeMemSize);

      arts_cuda_restore_device();

      if (backoff < GCHARDLIMIT)
        backoff *= 32;
      if (!backoff)
        backoff = 1;
      ARTS_DEBUG("Backoff: %u\n", backoff);
    }
    gcCounter++;
  }

  return ranCpuEdt;
}

extern __thread unsigned int runGCFlag;

bool arts_gpu_scheduler_demand_loop() {
  arts_gpu_t *arts_gpu = NULL;
  arts_handle_new_edts();

  struct arts_edt *edt_found = (struct arts_edt *)NULL;
  if (!(edt_found =
            (struct arts_edt *)arts_deque_pop_front(arts_thread_info.my_gpu_deque))) {
    if (!edt_found)
      edt_found = artsRuntimeStealGpuTask();
  }

  bool ranGpuEdt = false;
  if (edt_found) {
    arts_gpu = arts_find_gpu(edt_found);
    if (arts_gpu) {
      arts_run_gpu(edt_found, arts_gpu);
      ranGpuEdt = true;
    } else
      arts_deque_push_front(arts_thread_info.my_gpu_deque, edt_found, 0);
  }

  if (!ranGpuEdt)
    check_streams(arts_node_info.gpu_buff_on);

  bool ranCpuEdt = arts_default_scheduler_loop();

  if (!ranGpuEdt && !ranCpuEdt) {
    if (arts_node_info.run_gpu_gc_idle && runGCFlag) {
      long unsigned int gpu_id = runGCFlag - 1;
      runGCFlag = 0;

      arts_gpu = &arts_gpus[gpu_id];
      ARTS_DEBUG("Running Idle GPU GC: %u\n", gpu_id);
      arts_cuda_set_device(arts_gpu->device, true);

      uint64_t freeMemSize = arts_gpu_clean_up_route_table(
          (unsigned int)-1, arts_node_info.delete_zeros_gpu_gc,
          (unsigned int)arts_gpu->device);
      arts_atomic_add_u64(&arts_gpu->availGlobalMem, freeMemSize);
      arts_atomic_add_u64(&free_bytes, freeMemSize);

      arts_cuda_restore_device();
    }
  }

  return ranCpuEdt;
}

void arts_put_in_db_from_gpu(void *ptr, arts_guid_t db_guid, unsigned int offset,
                        unsigned int size, bool free_data) {
  unsigned int rank = arts_guid_get_rank(db_guid);
  if (rank == arts_global_rank_id) {
    struct arts_db *db = (struct arts_db *)arts_route_table_lookup_item(db_guid);
    if (db) {
      void *data = (void *)(((char *)(db + 1)) + offset);
      // memcpy(data, ptr, size);
      CHECKCORRECT(cudaMemcpyAsync(data, ptr, size, cudaMemcpyDeviceToHost,
                                   *arts_local_stream));

    } else {
      void *cpy_ptr = arts_malloc(size);
      // memcpy(cpy_ptr, ptr, size);
      CHECKCORRECT(cudaMemcpyAsync(cpy_ptr, ptr, size, cudaMemcpyDeviceToHost,
                                   *arts_local_stream));
      arts_out_of_order_put_in_db(cpy_ptr, NULL_GUID, db_guid, 0, offset, size,
                            NULL_GUID);
    }
    if (free_data)
      arts_gpu_route_table_add_item_to_delete_race(ptr, 0, db_guid, arts_local_gpu_id);
  }
}

artsLCSyncFunction_t lcSyncFunction[] = {
    artsMemcpyGpuDb,         artsGetLatestGpuDb,
    artsGetRandomGpuDb,      artsGetNonZerosUnsignedInt,
    artsGetMinDbUnsignedInt, artsAddDbUnsignedInt,
    artsXorDbUint64};

artsLCSyncFunctionGpu_t lcSyncFunctionGpu[] = {
    artsCopyGpuDb,           artsCopyGpuDb,
    artsCopyGpuDb,           artsNonZeroGpuDbUnsignedInt,
    artsMinGpuDbUnsignedInt, artsAddGpuDbUnsignedInt,
    artsXorGpuDbUint64};

unsigned int lcSyncElementSize[] = {sizeof(unsigned int), sizeof(unsigned int),
                                    sizeof(unsigned int), sizeof(unsigned int),
                                    sizeof(unsigned int), sizeof(unsigned int),
                                    sizeof(uint64_t)};

void internal_lc_sync_gpu(arts_guid_t acq_guid, struct arts_db *db) {
  if (db) {
    artsLCMeta_t host;
    artsLCMeta_t dev;
    host.guid = acq_guid;
    host.data = (void *)(db + 1);
    host.dataSize = db->header.size - sizeof(struct arts_db);
    host.hostVersion = &db->version;
    host.hostTimeStamp = &db->time_stamp;
    host.gpuVersion = 0;
    host.gpuTimeStamp = 0;
    host.gpu = -1;
    host.read_lock = &db->reader;
    host.write_lock = &db->writer;

    arts_cuda_set_device(-1, true);

    bool copyOnly = false;
    unsigned int size = db->header.size;
    struct arts_db *tempSpace = (struct arts_db *)arts_malloc_align(size, 16);

    gpu_gc_write_lock(); // Don't let the gc take our copies...
    ARTS_DEBUG("FUNCTION: %u\n", arts_node_info.gpu_lc_sync);
    unsigned int remMask = gpuLCReduce(
        acq_guid, db, lcSyncFunctionGpu[arts_node_info.gpu_lc_sync], &copyOnly);
    ARTS_DEBUG("RemMask: %u\n", remMask);
    for (int i = 0; i < arts_node_info.gpu; i++) {
      if (remMask & (1 << i)) {
        ARTS_DEBUG("Merging: %u\n", i);
        unsigned int gpuVersion;
        unsigned int time_stamp;
        void *data_ptr = arts_gpu_route_table_lookup_db_res(acq_guid, i, &gpuVersion,
                                                     &time_stamp, false);
        if (data_ptr) {
          if (!copyOnly)
            arts_gpu_invalidate_on_route_table(acq_guid, i);

          arts_cuda_set_device(i, false);
          get_data_from_stream_now(i, tempSpace, data_ptr, size, false);
          arts_gpu_route_table_return_db(acq_guid, !copyOnly, i);

          dev.guid = acq_guid;
          dev.data = (void *)(tempSpace + 1);
          dev.dataSize = tempSpace->header.size - sizeof(struct arts_db);
          dev.hostVersion = &tempSpace->version;
          dev.hostTimeStamp = &tempSpace->time_stamp;
          dev.gpuVersion = gpuVersion;
          dev.gpuTimeStamp = time_stamp;
          dev.gpu = i;
          dev.read_lock = NULL;
          dev.write_lock = NULL;
          if (copyOnly)
            lcSyncFunction[0](&host, &dev);
          else
            lcSyncFunction[arts_node_info.gpu_lc_sync](&host, &dev);

          ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_GPU_SYNC, ARTS_METRIC_THREAD, 1);
        }
      } else {
        ARTS_DEBUG("NO DB COPY ON GPU %d\n", i);
      }
    }
    gpu_gc_write_unlock();
    arts_free(tempSpace);
    arts_cuda_restore_device();
  }
}

void internal_lc_sync_cpu(arts_guid_t acq_guid, struct arts_db *db) {

  // struct arts_db * db = (struct arts_db*) arts_route_table_lookup_item(acq_guid);
  if (db) {
    artsLCMeta_t host;
    artsLCMeta_t dev;
    host.guid = acq_guid;
    host.data = (void *)(db + 1);
    host.dataSize = db->header.size - sizeof(struct arts_db);
    host.hostVersion = &db->version;
    host.hostTimeStamp = &db->time_stamp;
    host.gpuVersion = 0;
    host.gpuTimeStamp = 0;
    host.gpu = -1;
    host.read_lock = &db->reader;
    host.write_lock = &db->writer;

    // arts_cuda_set_device(-1, true);

    unsigned int size = db->header.size;
    struct arts_db *tempSpace = (struct arts_db *)arts_malloc_align(size, 16);
    gpu_gc_write_lock(); // Don't let the gc take our copies...
    for (int i = 0; i < arts_node_info.gpu; i++) {
      unsigned int gpuVersion;
      unsigned int time_stamp;
      ARTS_DEBUG("acq_guid: %lu type: %u i: %u\n", acq_guid,
                 arts_guid_get_type(acq_guid), i);
      void *data_ptr =
          arts_gpu_route_table_lookup_db(acq_guid, i, &gpuVersion, &time_stamp);
      if (data_ptr) {
        ARTS_DEBUG("i: %u %lu\n", i, acq_guid);
        arts_gpu_invalidate_on_route_table(acq_guid, i);
        // arts_cuda_set_device(i, false);

        // arts_cuda_mem_cpy_from_dev(tempSpace, data_ptr, size);
        get_data_from_stream_now(i, tempSpace, data_ptr, size, false);
        arts_gpu_route_table_return_db(acq_guid, true, i);

        dev.guid = acq_guid;
        dev.data = (void *)(tempSpace + 1);
        dev.dataSize = tempSpace->header.size - sizeof(struct arts_db);
        dev.hostVersion = &tempSpace->version;
        dev.hostTimeStamp = &tempSpace->time_stamp;
        dev.gpuVersion = gpuVersion;
        dev.gpuTimeStamp = time_stamp;
        dev.gpu = i;
        dev.read_lock = NULL;
        dev.write_lock = NULL;
        lcSyncFunction[arts_node_info.gpu_lc_sync](&host, &dev);
      } else {
        ARTS_DEBUG("NO DB COPY ON GPU %d\n", i);
      }
    }
    gpu_gc_write_unlock();
    arts_free(tempSpace);
    // arts_cuda_restore_device();
  }
}
