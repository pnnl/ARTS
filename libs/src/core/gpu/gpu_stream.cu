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
#include "arts/db.h"
#include "arts/defs.h"
#include "arts/edt.h"
#include "arts/edt_context.h" /* arts_set/unset_thread_local_edt_info */
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/gpu.h"
#include "arts/gpu/gpu_internal.h"
#include "arts/gpu/gpu_lc.h"
#include "arts/gpu/gpu_route_table.h"
#include "arts/runtime_state.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/deque.h"
#include "arts/utils/malloc.h"

/* File-internal stream/buffer helpers (no cross-TU caller).  Forward-declared
 * here so the runtime/stream code may call them regardless of definition
 * order below. */
typedef struct {
  void *dst;
  void *src;
  size_t count;
} arts_buffer_mem_move_t;

typedef struct {
  uint32_t paramc;
  const uint64_t *paramv;
  uint32_t depc;
  arts_edt_dep_t *depv;
  arts_edt_t fn_ptr;
  unsigned int grid[3];
  unsigned int block[3];
} arts_buffer_kernel_t;

static void check_occupancy(arts_edt_t fn_ptr, unsigned int gpu_id, dim3 block);
static bool push_data_to_stream(unsigned int gpu_id, void *dst, void *src,
                                size_t count, bool buff);
static bool get_data_from_stream(unsigned int gpu_id, void *dst, void *src,
                                 size_t count, bool buff);
static bool push_kernel_to_stream(unsigned int gpu_id, uint32_t paramc,
                                  const uint64_t *paramv, uint32_t depc,
                                  arts_edt_dep_t *depv, arts_edt_t fn_ptr,
                                  dim3 grid, dim3 block, bool buff);
static bool push_wrap_up_to_stream(unsigned int gpu_id, void *host_closure,
                                   bool buff);
static bool flush_mem_stream(unsigned int gpu_id, unsigned int *count,
                             arts_buffer_mem_move_t *buff,
                             enum cudaMemcpyKind kind);
static bool flush_kernel_stream(unsigned int gpu_id);
static bool flush_wrap_up_stream(unsigned int gpu_id);
static bool flush_stream(unsigned int gpu_id);
static bool check_streams(bool buff_on);
static void copy_gputo_gpu(void *dst, unsigned int dst_gpu_id, void *src,
                           unsigned int src_gpu_id, unsigned int size);
static void get_data_from_stream_now(unsigned int gpu_id, void *dst, void *src,
                                     size_t count, bool buff);

volatile unsigned int hits = 0;
volatile unsigned int misses = 0;
volatile uint64_t free_bytes = 0;

arts_gpu_t *arts_gpus;

ARTS_THREAD_LOCAL volatile unsigned int *new_edt_lock = 0;
ARTS_THREAD_LOCAL arts_array_list_t *new_edts = NULL;

// These are for the library version of GPU EDTs
// The user can query to get these values
// We still want to collect them for scheduling purposes
ARTS_THREAD_LOCAL arts_dim3_t *arts_local_grid;
ARTS_THREAD_LOCAL arts_dim3_t *arts_local_block;
ARTS_THREAD_LOCAL cudaStream_t *arts_local_stream;
ARTS_THREAD_LOCAL int arts_local_gpu_id;

#ifdef __cplusplus
extern "C" {
#endif
extern void arts_init_per_gpu(unsigned int node_id, int dev_id,
                              cudaStream_t *stream, int argc,
                              char **argv) ARTS_WEAK_IMPORT;
extern void arts_fini_per_gpu(unsigned int node_id, int dev_id,
                              cudaStream_t *stream) ARTS_WEAK_IMPORT;
#ifdef __cplusplus
}
#endif

bool **gpu_adj_list = NULL;
void arts_fully_connect_gpus(bool p2p, bool disconnect_p2p) {
  if (!gpu_adj_list) {
    gpu_adj_list = (bool **)arts_calloc(arts_node_info.gpu, sizeof(bool *));
    for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
      gpu_adj_list[i] = (bool *)arts_calloc(arts_node_info.gpu, sizeof(bool));
    }
  }
  if (p2p) {
    for (unsigned int src = 0; src < arts_node_info.gpu; src++) {
      arts_cuda_set_device((int)src, false);
      for (unsigned int dst = 0; dst < arts_node_info.gpu; dst++) {
        if (src != dst) {
          int has_access = 0;
          CHECKCORRECT(
              cudaDeviceCanAccessPeer(&has_access, (int)src, (int)dst));
          if (has_access) {
            if (disconnect_p2p) {
              CHECKCORRECT(cudaDeviceDisablePeerAccess((int)dst));
            } else {
              gpu_adj_list[src][dst] = 1;
              CHECKCORRECT(cudaDeviceEnablePeerAccess((int)dst, 0));
            }
          }
        }
      }
    }
  }
}

void arts_node_init_gpus() {
  int num_avail_gpus = 0;
  locality = locality_scheme[arts_node_info.gpu_locality];
  fit = fit_scheme[arts_node_info.gpu_fit];
  CHECKCORRECT(cudaGetDeviceCount(&num_avail_gpus));
  if (num_avail_gpus < (int)arts_node_info.gpu) {
    ARTS_INFO("Requested %d gpus but only %d available\n", num_avail_gpus,
              arts_node_info.gpu);
    arts_node_info.gpu = num_avail_gpus;
  }

  ARTS_DEBUG(
      "gpu_route_table_size: %u gpu_route_table_entries: %u "
      "free_db_after_gpu_run: "
      "%u run_gpu_gc_idle: %u run_gpu_gc_pre_edt: %u delete_zeros_gpu_gc: %u\n",
      arts_node_info.gpu_route_table_size,
      arts_node_info.gpu_route_table_entries,
      arts_node_info.free_db_after_gpu_run, arts_node_info.run_gpu_gc_idle,
      arts_node_info.run_gpu_gc_pre_edt, arts_node_info.delete_zeros_gpu_gc);

  ARTS_DEBUG("NUM DEV: %d\n", arts_node_info.gpu);
  arts_gpus = (arts_gpu_t *)arts_calloc(arts_node_info.gpu, sizeof(arts_gpu_t));

  arts_cuda_set_device(-1, true);

  // Initialize arts_gpu with 1 stream/GPU
  for (unsigned int i = 0; i < arts_node_info.gpu; ++i) {
    arts_gpus[i].device = (int)i;
    ARTS_DEBUG("Setting %u\n", i);
    arts_cuda_set_device((int)i, false);
    CHECKCORRECT(cudaStreamCreate(&arts_gpus[i].stream)); // Make it scalable
    arts_node_info.gpu_route_table[i] =
        arts_gpu_new_route_table(arts_node_info.gpu_route_table_entries,
                                 arts_node_info.gpu_route_table_size);
    size_t temp_free_mem = 0;
    size_t temp_max_mem = 0;
    CHECKCORRECT(
        cudaMemGetInfo((size_t *)&temp_free_mem, (size_t *)&temp_max_mem));
    CHECKCORRECT(
        cudaGetDeviceProperties(&arts_gpus[i].prop, arts_gpus[i].device));
    arts_gpus[i].avail_global_mem = (uint64_t)temp_free_mem;
    arts_gpus[i].total_global_mem = (uint64_t)temp_max_mem;
    if (arts_gpus[i].avail_global_mem > arts_node_info.gpu_max_memory) {
      arts_gpus[i].avail_global_mem = arts_node_info.gpu_max_memory;
    }
    ARTS_DEBUG("to Start: %lu\n", arts_gpus[i].avail_global_mem);
  }

  arts_fully_connect_gpus(arts_node_info.gpu_p2p, false);

  arts_cuda_restore_device();
}

void arts_init_per_gpu_wrapper(int argc, char **argv) {
  if (arts_init_per_gpu) {
    arts_cuda_set_device(-1, true);
    for (unsigned int i = 0; i < arts_node_info.gpu; ++i) {
      ARTS_DEBUG("Set device: %u\n", i);
      arts_cuda_set_device((int)i, false);
      arts_init_per_gpu(arts_global_rank_id, (int)i, &arts_gpus[i].stream, argc,
                        argv);
    }
    arts_cuda_restore_device();
  }
}

void arts_worker_init_gpus() {
  new_edt_lock = (unsigned int *)arts_calloc(1, sizeof(unsigned int));
  new_edts = arts_new_array_list(sizeof(void *), 32);
}

void arts_store_new_edts(void *edt) {
  arts_lock(new_edt_lock);
  arts_push_to_array_list(new_edts, &edt);
  arts_unlock(new_edt_lock);
}

void arts_handle_new_edts() {
  arts_lock(new_edt_lock);
  uint64_t size = arts_length_array_list(new_edts);
  if (size) {
    for (uint64_t i = 0; i < size; i++) {
      struct arts_edt_s **edt =
          (struct arts_edt_s **)arts_get_from_array_list(new_edts, i);
      if ((*edt)->edt_type == ARTS_EDT_GPU) {
        arts_deque_push_front(arts_thread_info.my_gpu_deque, (*edt), 0);
      } else {
        arts_deque_push_front(arts_thread_info.my_deque, (*edt), 0);
      }
    }
    arts_reset_array_list(new_edts);
  }
  arts_unlock(new_edt_lock);
}

void arts_cleanup_gpus() {
  uint64_t freed_size = 0;
  arts_cuda_set_device(-1, false);

  arts_fully_connect_gpus(arts_node_info.gpu_p2p, true);

  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    arts_cuda_set_device(arts_gpus[i].device, false);
    if (arts_fini_per_gpu) {
      arts_fini_per_gpu(arts_global_rank_id, (int)i, &arts_gpus[i].stream);
    }
    freed_size += arts_gpu_free_all((unsigned int)arts_gpus[i].device);
    CHECKCORRECT(cudaStreamSynchronize(arts_gpus[i].stream));
    CHECKCORRECT(cudaStreamDestroy(arts_gpus[i].stream));
  }
  arts_cuda_restore_device();
  ARTS_INFO("Occupancy :\n");
  for (int i = 0; i < (int)arts_get_num_gpus(); ++i) {
    ARTS_INFO("\tGPU[%d] = %f\n", i, arts_gpus[i].occupancy);
  }
  ARTS_INFO("HITS: %u MISSES: %u FREED BYTES: %u BYTES FREED ON EXIT %lu\n",
            hits, misses, free_bytes, freed_size);
  ARTS_INFO("HIT RATIO: %lf\n", (double)hits / (double)(hits + misses));
}

void arts_wrap_up(cudaStream_t stream, cudaError_t status, void *data) {
  (void)stream;
  (void)status;

  arts_gpu_clean_up_t *gc = (arts_gpu_clean_up_t *)data;

  arts_gpu_t *arts_gpu = &arts_gpus[gc->gpu_id];
  arts_atomic_sub(&arts_gpu->available_edt_slots, 1U);
  arts_atomic_sub(&arts_gpu->running_edts, 1U);

  // Shouldn't have to touch newly ready edts regardless of streams and devices
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)gc->edt;
  uint32_t paramc = edt->wrapper_edt.paramc;
  uint32_t depc = edt->wrapper_edt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);

  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);
  arts_atomic_sub(&arts_gpu->available_threads, total_threads);

  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr) {
      struct arts_db_s *db_hdr = (struct arts_db_s *)depv[i].ptr - 1;
      if (db_hdr->db_type == ARTS_DB_GPU_PIN) {
        arts_gpu_invalidate_route_tables(depv[i].guid, gc->gpu_id);
      }
      // True says to mark it for deletion... Change this to false to further
      // delay delete!
      //  bool mark_delete = (arts_guid_get_kind(depv[i].guid) !=
      //  ARTS_DB_GPU_PIN)
      //  && arts_node_info.free_db_after_gpu_run;
      bool mark_delete = arts_node_info.free_db_after_gpu_run;
      bool res =
          arts_gpu_route_table_return_db(depv[i].guid, mark_delete, gc->gpu_id);
      // arts_gpu_route_table_return_db(depv[i].guid,
      // arts_node_info.free_db_after_gpu_run, gc->gpu_id);
      ARTS_DEBUG("Returning Db: %lu id: %d res: %u\n", depv[i].guid, gc->gpu_id,
                 res);
    }
  }

  // Definitely mark the dev closure to be deleted as there is no reuse!
  arts_gpu_route_table_return_db(edt->wrapper_edt.guid, true, gc->gpu_id);
  new_edt_lock = gc->new_edt_lock;
  new_edts = gc->new_edts;
  arts_gpu_host_wrap_up(gc->edt, edt->end_guid, edt->slot, edt->data_guid);
  ARTS_DEBUG("FINISHED GPU CALLS %s\n", cudaGetErrorString(status));
  // artsToggleThreadInspection();
}

void arts_wrap_up_host_func(void *data) {
  arts_wrap_up(NULL, cudaSuccess, data);
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

  static volatile unsigned int gpu_lock;

  void *dev_closure = NULL;
  void *host_closure = NULL;

  uint64_t *dev_gpu_id = NULL;
  uint64_t *dev_paramv = NULL;
  arts_edt_dep_t *dev_depv = NULL;

  arts_gpu_clean_up_t *host_gc_ptr = NULL;
  uint64_t *host_gpu_id = NULL;
  uint64_t *host_paramv = NULL;
  arts_edt_dep_t *host_depv = NULL;

  ARTS_DEBUG("Paramc: %u Depc: %u edt: %p\n", paramc, depc, edt_ptr);

  // Get size of closure
  uint64_t dev_closure_size =
      (sizeof(uint64_t) * (paramc + 1)) + (sizeof(arts_edt_dep_t) * depc);
  uint64_t host_closure_size = dev_closure_size + sizeof(arts_gpu_clean_up_t);
  ARTS_DEBUG("dev_closure_size: %u host_closure_size: %u\n", dev_closure_size,
             host_closure_size);

  // Allocate Closure for GPU
  if (dev_closure_size) {
    dev_closure = arts_cuda_malloc(dev_closure_size);
    dev_gpu_id = (uint64_t *)dev_closure;
    dev_paramv = dev_gpu_id + 1;
    dev_depv = (arts_edt_dep_t *)(dev_paramv + paramc);
    ARTS_DEBUG("Allocated dev closure\n");
  }

  if (host_closure_size) {
    // Allocate closure for host
    host_closure = arts_cuda_malloc_host(host_closure_size);
    host_gc_ptr = (arts_gpu_clean_up_t *)host_closure;
    host_gpu_id = (uint64_t *)(host_gc_ptr + 1);
    host_paramv = host_gpu_id + 1;
    host_depv = (arts_edt_dep_t *)(host_paramv + paramc);
    ARTS_DEBUG("Allocated host closure\n");

    // Fill Host closure
    host_gc_ptr->gpu_id = arts_gpu->device;
    host_gc_ptr->new_edt_lock = new_edt_lock;
    host_gc_ptr->new_edts = new_edts;
    host_gc_ptr->dev_closure = dev_closure;
    host_gc_ptr->edt = (struct arts_edt_s *)edt_ptr;
    *host_gpu_id = (uint64_t)arts_gpu->device;
    for (unsigned int i = 0; i < paramc; i++) {
      host_paramv[i] = paramv[i];
    }
    ARTS_DEBUG("Filled host closure\n");

    arts_guid_t edt_guid = host_gc_ptr->edt->guid;
    // arts_gpu_route_table_add_item(host_gc_ptr, host_closure_size,
    // edt_guid, arts_gpu->device);
    arts_gpu_route_table_add_item(host_gc_ptr, dev_closure_size, edt_guid,
                                  arts_gpu->device);
    ARTS_DEBUG("Added edt_guid: %lu size: %u to gpu: %d routing table\n",
               edt_guid, host_closure_size, arts_gpu->device);
  }

  arts_gpu_edt_t *gpu_edt = (arts_gpu_edt_t *)host_gc_ptr->edt;

  // Allocate space for DB on GPU and Move Data
  for (unsigned int i = 0; i < depc; ++i) {
    if (depv[i].ptr) {
      struct arts_db_s *db = (struct arts_db_s *)depv[i].ptr - 1;
      arts_db_types_t db_subtype = db->db_type;
      unsigned int gpu_version;
      unsigned int time_stamp;
      void *data_ptr = arts_gpu_route_table_lookup_db(
          depv[i].guid, arts_gpu->device, &gpu_version, &time_stamp);
      uint64_t size = arts_db_total_size(db);
      uint64_t alloc_size = (db_subtype == ARTS_DB_GPU) ? (size * 2) : size;
      if (!data_ptr) {
        bool successful_add = false;
        ARTS_DEBUG("WRAPPER SIZE: %lu\n", alloc_size);
        arts_item_wrapper_t *wrapper = arts_gpu_route_table_reserve_item(
            &successful_add, alloc_size, depv[i].guid, arts_gpu->device, true);

        if (successful_add) // We won, so allocate and move data
        {
          ARTS_DEBUG("Adding %lu %u id: %d mode: %s\n", depv[i].guid,
                     alloc_size, arts_gpu->device, db_mode_name[depv[i].mode]);
          data_ptr = arts_cuda_malloc(alloc_size);
          void *src = (void *)db;
          if (db_subtype == ARTS_DB_GPU) {
            src = make_lc_shadow_copy(db);
          }
          if (depv[i].mode == DB_MODE_LC_NO_COPY ||
              depv[i].mode == DB_MODE_MEMSET) {
            src = NULL;
          }
          push_data_to_stream(arts_gpu->device, data_ptr, src, size,
                              arts_node_info.gpu_buff_on && !gpu_edt->lib);
          // Must have already launched the memcpy before setting real_data or
          // races will ensue
          wrapper->real_data = data_ptr;
          ARTS_DEBUG("Malloc[%d]: %p %p\n", arts_gpu->device, wrapper,
                     data_ptr);
          arts_atomic_add(&misses, 1U);
        } else // Someone beat us to creating the data... So we must free
        {
          while (
              !arts_atomic_fetch_add_u64((uint64_t *)&wrapper->real_data, 0)) {
          } // Spin till the data memcpy is launched
          data_ptr = (void *)wrapper->real_data;
          if (db_subtype == ARTS_DB_GPU_PIN && depv[i].mode == DB_MODE_MEMSET) {
            push_data_to_stream(arts_gpu->device, data_ptr, NULL, size,
                                arts_node_info.gpu_buff_on && !gpu_edt->lib);
          }
          arts_atomic_add_u64(&arts_gpu->avail_global_mem, alloc_size);
          arts_atomic_add(&hits, 1U);
        }
      } else {
        arts_atomic_add_u64(&arts_gpu->avail_global_mem, alloc_size);
        arts_atomic_add(&hits, 1U);
      }
      struct arts_db_s *new_db = (struct arts_db_s *)data_ptr;
      host_depv[i].ptr = (void *)(new_db + 1);
    } else {
      ARTS_DEBUG("Depv: %u is null edt: %lu\n", i, gpu_edt->wrapper_edt.guid);
      host_depv[i].ptr = NULL;
    }

    host_depv[i].guid = depv[i].guid;
  }
  ARTS_DEBUG("Allocated, added, and moved dbs\n");

  push_data_to_stream(arts_gpu->device, dev_closure, (void *)host_gpu_id,
                      dev_closure_size,
                      arts_node_info.gpu_buff_on && !gpu_edt->lib);
  ARTS_DEBUG("Filled GPU Closure\n");

  if (gpu_edt->lib) {
    arts_local_grid = &gpu_edt->grid;
    arts_local_block = &gpu_edt->block;
    arts_local_stream = &arts_gpu->stream;
    arts_local_gpu_id = arts_gpu->device;
    arts_set_thread_local_edt_info(host_gc_ptr->edt);
    /* arts_route_table_reset_oo removed: OO list is now lock-free and
     * drained per-installer; no separate reset step needed. */

    host_gc_ptr->edt->func_ptr(paramc, host_paramv, depc, host_depv);

    arts_unset_thread_local_edt_info();
    /* Release DBs created during the lib function NOW, on the worker thread.
       The wrap-up callback runs on the CUDA callback thread whose TLS
       created_db_list is empty, so arts_release_created_dbs() there would be
       a no-op — leaving dependent acquisitions un-progressed and consumer
       EDTs stuck. */
    arts_release_created_dbs();
  } else {
    push_kernel_to_stream(arts_gpu->device, paramc, dev_paramv, depc, dev_depv,
                          fn_ptr, grid, block, arts_node_info.gpu_buff_on);
  }

  // Move data back
  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr) {
      struct arts_db_s *cb_db = (struct arts_db_s *)depv[i].ptr - 1;
      if (cb_db->db_type == ARTS_DB_GPU_PIN &&
          (depv[i].mode == DB_MODE_RW || depv[i].mode == DB_MODE_MEMSET)) {
        size_t size = (size_t)(cb_db->cache.db_size);
        get_data_from_stream(arts_gpu->device, depv[i].ptr, host_depv[i].ptr,
                             size, arts_node_info.gpu_buff_on && !gpu_edt->lib);
      }
    }
  }

  push_wrap_up_to_stream(arts_gpu->device, host_closure,
                         arts_node_info.gpu_buff_on && !gpu_edt->lib);
}

void arts_schedule_to_gpu(arts_edt_t fn_ptr, uint32_t paramc,
                          const uint64_t *paramv, uint32_t depc,
                          arts_edt_dep_t *depv, void *edt_ptr,
                          arts_gpu_t *arts_gpu) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_ptr;
  dim3 grid(edt->grid.x, edt->grid.y, edt->grid.z);
  dim3 block(edt->block.x, edt->block.y, edt->block.z);
  arts_schedule_to_gpu_internal(fn_ptr, paramc, paramv, depc, depv, grid, block,
                                edt_ptr, arts_gpu);
}

void free_gpu_item(arts_route_item_t *item) {
  arts_guid_kind_t type = arts_guid_get_kind(item->key);
  arts_item_wrapper_t *wrapper =
      (arts_item_wrapper_t *)arts_route_item_peek_data(item);
  if (!wrapper) {
    return;
  }
  if (type == ARTS_GUID_EDT) {
    arts_gpu_clean_up_t *host_gc_ptr =
        (arts_gpu_clean_up_t *)wrapper->real_data;
    ARTS_DEBUG("FREEING DEV PTR: %p\n", host_gc_ptr->dev_closure);
    arts_cuda_free(host_gc_ptr->dev_closure);
    ARTS_DEBUG("FREEING HOST PTR: %p\n", host_gc_ptr);
    arts_cuda_free_host(host_gc_ptr);
  } else if (type == ARTS_GUID_DB) {
    arts_shared_ptr_t db_h = arts_route_table_lookup_db(item->key);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
    if (db && db->db_type == ARTS_DB_GPU) {
      unsigned int size = arts_db_total_size(db);
      struct arts_db_s *temp_space =
          (struct arts_db_s *)arts_malloc_align(size, 16);

      arts_lc_meta_t host;
      host.guid = item->key;
      host.data = (void *)(db + 1);
      host.data_size = db->cache.db_size;
      host.host_version = &db->version;
      host.host_time_stamp = &db->time_stamp;
      host.gpu_version = 0;
      host.gpu_time_stamp = 0;
      host.gpu = -1;
      host.read_lock = &db->reader;
      host.write_lock = &db->writer;

      // arts_cuda_mem_cpy_from_dev(temp_space, (void*) wrapper->real_data,
      // size);
      get_data_from_stream_now(arts_get_current_gpu(), temp_space,
                               (void *)wrapper->real_data, size, false);

#if 0 /* FIXME: GPU LC sync needs new model -- task 1a.4 */
      arts_lc_meta_t dev;
      dev.guid = item->key;
      dev.data = (void *)(temp_space + 1);
      dev.data_size = temp_space->cache.db_size;
      dev.host_version = &temp_space->version;
      dev.host_time_stamp = &temp_space->time_stamp;
      dev.gpu_version = item->touched;
      dev.gpu_time_stamp = wrapper->time_stamp;
      dev.gpu = -1;
      dev.read_lock = NULL;
      dev.write_lock = NULL;

      lc_sync_function[arts_node_info.gpu_lc_sync](&host, &dev);
#endif
      (void)host;

      arts_free(temp_space);
      arts_cuda_free((void *)wrapper->real_data);

    } else {
      // Non-LC DB (DEFAULT/GPU) or LC DB not found — just free GPU memory
      arts_cuda_free((void *)wrapper->real_data);
    }
    if (db) {
      arts_shared_release(&db_h);
    }
  }

  wrapper->real_data = NULL;
  wrapper->time_stamp = 0;
  item->key = 0;
  /* item->lock and item->touched fields removed in new route_item model. */
}

/* ======================================================================== */
/* Device / runtime helpers                                                 */
/* ======================================================================== */

ARTS_THREAD_LOCAL int arts_saved_device_id = -1;
ARTS_THREAD_LOCAL int arts_current_device_id = -1;

int arts_get_current_gpu() {
  if (arts_current_device_id == -1) {
    CHECKCORRECT(cudaGetDevice(&arts_current_device_id));
  }

  return arts_current_device_id;
}

bool arts_cuda_set_device(int id, bool save) {
  if (arts_current_device_id == -1) {
    CHECKCORRECT(cudaGetDevice(&arts_current_device_id));
  }

  if (save) {
    arts_saved_device_id = arts_current_device_id;
  }

  if (id > -1 && id < arts_node_info.gpu && id != arts_current_device_id) {
    CHECKCORRECT(cudaSetDevice(id));
    arts_current_device_id = id;
    return true;
  }

  return false;
}

bool arts_cuda_restore_device() {
  return arts_cuda_set_device(arts_saved_device_id, false);
}

void *arts_cuda_malloc_host(unsigned int size) {
  void *ptr = NULL;
  CHECKCORRECT(cudaMallocHost(&ptr, size));
  // ptr = arts_calloc(1, size);
  if (!ptr) {
    ARTS_ERROR("CUDA host malloc failed (size=%u)", size);
  }
  return ptr;
}

void arts_cuda_free_host(void *ptr) {
  if (ptr) {
    CHECKCORRECT(cudaFreeHost(ptr));
  }
  // arts_free(ptr);
}

void *arts_cuda_malloc(unsigned int size) {
  void *ptr = NULL;
  CHECKCORRECT(cudaMalloc(&ptr, size));
  if (!ptr) {
    ARTS_ERROR("CUDA device malloc failed (%lu avail)",
               arts_gpus[arts_current_device_id].avail_global_mem);
  }
  return ptr;
}

void arts_cuda_free(void *ptr) {
  if (ptr) {
    CHECKCORRECT(cudaFree(ptr));
  }
}

void arts_cuda_mem_cpy_from_dev(void *dst, void *src, size_t count) {
  CHECKCORRECT(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

void arts_cuda_mem_cpy_to_dev(void *dst, void *src, size_t count) {
  CHECKCORRECT(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

arts_dim3_t *arts_get_gpu_grid() { return arts_local_grid; }

arts_dim3_t *arts_get_gpu_block() { return arts_local_block; }

void *arts_get_gpu_stream() { return arts_local_stream; }

int arts_get_gpu_id() { return arts_local_gpu_id; }

unsigned int arts_get_num_gpus() { return arts_node_info.gpu; }

arts_guid_t internal_edt_create_gpu(arts_edt_t func_ptr, arts_guid_t *guid,
                                    unsigned int rank, uint32_t paramc,
                                    const uint64_t *paramv, uint32_t depc,
                                    arts_dim3_t grid, arts_dim3_t block,
                                    arts_guid_t end_guid, uint32_t slot,
                                    arts_guid_t data_guid, bool pass_through,
                                    bool lib, int gpu_to_run_on) {
  //    ARTSEDTCOUNTERTIMERSTART(EDT_CREATE_COUNTER);
  unsigned int edt_space = sizeof(arts_gpu_edt_t) +
                           (paramc * sizeof(uint64_t)) +
                           (depc * sizeof(arts_edt_dep_t));

  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)arts_calloc(1, edt_space);
  edt->wrapper_edt.invalidate_count = 1;
  edt->grid = grid;
  edt->block = block;
  edt->gpu_to_run_on = gpu_to_run_on;
  edt->end_guid = end_guid;
  edt->slot = slot;
  edt->data_guid = data_guid;
  edt->passthrough = pass_through;
  edt->lib = lib;

  edt->wrapper_edt.edt_type = ARTS_EDT_GPU;
  // artsIntrospectionEdtCreateBegin();
  (void)arts_edt_create_core((struct arts_edt_s *)edt, ARTS_GUID_EDT, guid,
                             rank, edt_space, func_ptr, paramc, paramv, depc,
                             NULL_GUID, 0, 0);
  // artsIntrospectionEdtCreateFinish(created);
  //    ARTSEDTCOUNTERTIMERENDINCREMENT(EDT_CREATE_COUNTER);
  return *guid;
}

/* ======================================================================== */
/* Unified GPU EDT creation API                                             */
/* ======================================================================== */

arts_guid_t arts_edt_create_gpu(arts_edt_t func_ptr, uint32_t paramc,
                                const uint64_t *paramv, uint32_t depc,
                                arts_dim3_t grid, arts_dim3_t block,
                                const arts_gpu_hint_t *hint) {
  unsigned int rank =
      (hint && hint->rank != ARTS_HINT_CURRENT_RANK) ? hint->rank : 0;
  if (!hint || hint->rank == ARTS_HINT_CURRENT_RANK) {
    rank = arts_global_rank_id;
  }
  arts_guid_t end_guid = hint ? hint->end_guid : NULL_GUID;
  uint32_t slot = hint ? hint->slot : 0;
  arts_guid_t data_guid = hint ? hint->data_guid : NULL_GUID;
  bool passthrough = hint ? hint->passthrough : false;
  bool lib = hint ? hint->lib : false;
  int gpu = hint ? hint->gpu : -1;

  arts_guid_t guid = NULL_GUID;
  return internal_edt_create_gpu(func_ptr, &guid, rank, paramc, paramv, depc,
                                 grid, block, end_guid, slot, data_guid,
                                 passthrough, lib, gpu);
}

arts_guid_t arts_edt_create_gpu_with_guid(arts_edt_t func_ptr, arts_guid_t guid,
                                          uint32_t paramc,
                                          const uint64_t *paramv, uint32_t depc,
                                          arts_dim3_t grid, arts_dim3_t block,
                                          const arts_gpu_hint_t *hint) {
  arts_guid_t end_guid = hint ? hint->end_guid : NULL_GUID;
  uint32_t slot = hint ? hint->slot : 0;
  arts_guid_t data_guid = hint ? hint->data_guid : NULL_GUID;
  bool passthrough = hint ? hint->passthrough : false;
  bool lib = hint ? hint->lib : false;
  int gpu = hint ? hint->gpu : -1;

  return internal_edt_create_gpu(func_ptr, &guid, arts_guid_get_rank(guid),
                                 paramc, paramv, depc, grid, block, end_guid,
                                 slot, data_guid, passthrough, lib, gpu);
}

void arts_run_gpu(void *edt_packet, arts_gpu_t *arts_gpu) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  arts_edt_t func = edt->wrapper_edt.func_ptr;
  uint32_t paramc = edt->wrapper_edt.paramc;
  uint32_t depc = edt->wrapper_edt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);

  arts_cuda_set_device(arts_gpu->device, true);

  if (arts_node_info.run_gpu_gc_pre_edt) {
    // ARTS_INFO("Running Pre Edt GPU GC: %u\n", arts_gpu->device);
    uint64_t free_mem_size = arts_gpu_clean_up_route_table(
        (unsigned int)-1, arts_node_info.delete_zeros_gpu_gc,
        (unsigned int)arts_gpu->device);
    arts_atomic_add_u64(&arts_gpu->avail_global_mem, free_mem_size);
    arts_atomic_add_u64(&free_bytes, free_mem_size);
  }

  arts_atomic_add(&arts_gpu->running_edts, 1U);

  prep_dbs(depc, depv, true);
  arts_schedule_to_gpu(func, paramc, paramv, depc, depv, edt_packet, arts_gpu);

  arts_cuda_restore_device();
}

void arts_gpu_host_wrap_up(void *edt_packet, arts_guid_t to_signal,
                           uint32_t slot, arts_guid_t data_guid) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapper_edt.paramc;
  uint32_t depc = edt->wrapper_edt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);

  release_dbs(depc, depv, true);
  arts_release_created_dbs();

  if (edt->lib) {
    edt->wrapper_edt.invalidate_count = 0;
    arts_ooo_drain_guid(edt->wrapper_edt.guid);
  }

  // Signal next
  if (to_signal) {
    if (edt->passthrough) {
      arts_edt_satisfy_slot(to_signal, slot, depv[data_guid].guid, DB_MODE_RW,
                            NULL, 0);
    } else {
      arts_guid_kind_t mode = arts_guid_get_kind(to_signal);
      if (mode == ARTS_GUID_EDT) {
        arts_edt_satisfy_slot(to_signal, slot, data_guid, DB_MODE_RW, NULL, 0);
      }
      if (mode == ARTS_GUID_EVENT) {
        arts_event_satisfy_slot(to_signal, data_guid, slot);
      }
    }
  }
  arts_edt_delete((struct arts_edt_s *)edt_packet);
}

struct arts_edt_s *arts_runtime_steal_gpu_task() {
  struct arts_edt_s *edt = NULL;
  if (arts_node_info.total_thread_count > 1) {
    long unsigned int steal_loc;
    do {
      steal_loc = jrand48(arts_thread_info.drand_buf);
      steal_loc = steal_loc % arts_node_info.total_thread_count;
    } while (steal_loc == arts_thread_info.thread_id);
    edt = (struct arts_edt_s *)arts_deque_pop_back(
        arts_node_info.gpu_deque[steal_loc]);
  }
  return edt;
}

bool arts_gpu_scheduler_loop() {
  arts_gpu_t *arts_gpu = NULL;
  arts_handle_new_edts();

  struct arts_edt_s *edt_found = (struct arts_edt_s *)NULL;
  if (!(edt_found = (struct arts_edt_s *)arts_deque_pop_front(
            arts_thread_info.my_gpu_deque))) {
    if (!edt_found) {
      edt_found = arts_runtime_steal_gpu_task();
    }
  }

  bool ran_gpu_edt = false;
  if (edt_found) {
    arts_gpu = arts_find_gpu(edt_found);
    if (arts_gpu) {
      arts_run_gpu(edt_found, arts_gpu);
      ran_gpu_edt = true;
    } else {
      arts_deque_push_front(arts_thread_info.my_gpu_deque, edt_found, 0);
    }
  }

  if (!ran_gpu_edt) {
    check_streams(arts_node_info.gpu_buff_on);
  }

  bool ran_cpu_edt = arts_default_scheduler_loop();
  if (arts_node_info.run_gpu_gc_idle && !ran_gpu_edt && !ran_cpu_edt) {
    long unsigned int gpu_id = jrand48(arts_thread_info.drand_buf);
    gpu_id = gpu_id % arts_node_info.gpu;
    arts_gpu = &arts_gpus[gpu_id];
    ARTS_DEBUG("Running Idle GPU GC: %u\n", gpu_id);
    arts_cuda_set_device(arts_gpu->device, true);

    uint64_t free_mem_size = arts_gpu_clean_up_route_table(
        (unsigned int)-1, arts_node_info.delete_zeros_gpu_gc,
        (unsigned int)arts_gpu->device);
    arts_atomic_add_u64(&arts_gpu->avail_global_mem, free_mem_size);
    arts_atomic_add_u64(&free_bytes, free_mem_size);

    arts_cuda_restore_device();
  }

  return ran_cpu_edt;
}

#define GCHARDLIMIT 2000000000000
ARTS_THREAD_LOCAL uint64_t backoff = 1;
ARTS_THREAD_LOCAL uint64_t gc_counter = 0;

bool arts_gpu_scheduler_backoff_loop() {
  arts_gpu_t *arts_gpu = NULL;
  arts_handle_new_edts();

  struct arts_edt_s *edt_found = (struct arts_edt_s *)NULL;
  if (!(edt_found = (struct arts_edt_s *)arts_deque_pop_front(
            arts_thread_info.my_gpu_deque))) {
    if (!edt_found) {
      edt_found = arts_runtime_steal_gpu_task();
    }
  }

  bool ran_gpu_edt = false;
  if (edt_found) {
    arts_gpu = arts_find_gpu(edt_found);
    if (arts_gpu) {
      arts_run_gpu(edt_found, arts_gpu);
      ran_gpu_edt = true;
    } else {
      arts_deque_push_front(arts_thread_info.my_gpu_deque, edt_found, 0);
    }
  }

  if (!ran_gpu_edt) {
    check_streams(arts_node_info.gpu_buff_on);
  }

  bool ran_cpu_edt = arts_default_scheduler_loop();

  if (ran_cpu_edt || ran_gpu_edt) {
    backoff = 1;
  }

  if (!ran_gpu_edt && !ran_cpu_edt) {
    if (arts_node_info.run_gpu_gc_idle && gc_counter % backoff == 0) {
      long unsigned int gpu_id = jrand48(arts_thread_info.drand_buf);
      gpu_id = gpu_id % arts_node_info.gpu;
      arts_gpu = &arts_gpus[gpu_id];
      ARTS_DEBUG("Running Idle GPU GC: %u\n", gpu_id);
      arts_cuda_set_device(arts_gpu->device, true);

      uint64_t free_mem_size = arts_gpu_clean_up_route_table(
          (unsigned int)-1, arts_node_info.delete_zeros_gpu_gc,
          (unsigned int)arts_gpu->device);
      arts_atomic_add_u64(&arts_gpu->avail_global_mem, free_mem_size);
      arts_atomic_add_u64(&free_bytes, free_mem_size);

      arts_cuda_restore_device();

      if (backoff < GCHARDLIMIT) {
        backoff *= 32;
      }
      if (!backoff) {
        backoff = 1;
      }
      ARTS_DEBUG("Backoff: %u\n", backoff);
    }
    gc_counter++;
  }

  return ran_cpu_edt;
}

extern ARTS_THREAD_LOCAL unsigned int run_gc_flag;

bool arts_gpu_scheduler_demand_loop() {
  arts_gpu_t *arts_gpu = NULL;
  arts_handle_new_edts();

  struct arts_edt_s *edt_found = (struct arts_edt_s *)NULL;
  if (!(edt_found = (struct arts_edt_s *)arts_deque_pop_front(
            arts_thread_info.my_gpu_deque))) {
    if (!edt_found) {
      edt_found = arts_runtime_steal_gpu_task();
    }
  }

  bool ran_gpu_edt = false;
  if (edt_found) {
    arts_gpu = arts_find_gpu(edt_found);
    if (arts_gpu) {
      arts_run_gpu(edt_found, arts_gpu);
      ran_gpu_edt = true;
    } else {
      arts_deque_push_front(arts_thread_info.my_gpu_deque, edt_found, 0);
    }
  }

  if (!ran_gpu_edt) {
    check_streams(arts_node_info.gpu_buff_on);
  }

  bool ran_cpu_edt = arts_default_scheduler_loop();

  if (!ran_gpu_edt && !ran_cpu_edt) {
    if (arts_node_info.run_gpu_gc_idle && run_gc_flag) {
      long unsigned int gpu_id = run_gc_flag - 1;
      run_gc_flag = 0;

      arts_gpu = &arts_gpus[gpu_id];
      ARTS_DEBUG("Running Idle GPU GC: %u\n", gpu_id);
      arts_cuda_set_device(arts_gpu->device, true);

      uint64_t free_mem_size = arts_gpu_clean_up_route_table(
          (unsigned int)-1, arts_node_info.delete_zeros_gpu_gc,
          (unsigned int)arts_gpu->device);
      arts_atomic_add_u64(&arts_gpu->avail_global_mem, free_mem_size);
      arts_atomic_add_u64(&free_bytes, free_mem_size);

      arts_cuda_restore_device();
    }
  }

  return ran_cpu_edt;
}

void arts_put_in_db_from_gpu(void *ptr, arts_guid_t db_guid,
                             unsigned int offset, unsigned int size,
                             bool free_data) {
  unsigned int rank = arts_guid_get_rank(db_guid);
  if (rank == arts_global_rank_id) {
    arts_shared_ptr_t db_h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
    if (db) {
      void *data = (void *)(((char *)(db + 1)) + offset);
      // memcpy(data, ptr, size);
      CHECKCORRECT(cudaMemcpyAsync(data, ptr, size, cudaMemcpyDeviceToHost,
                                   *arts_local_stream));
      arts_shared_release(&db_h);
    } else {
      /* GUID homed on this rank but the DB is not yet installed in the route
       * table.  A GPU stage-out targets a DB that must already exist on its
       * home rank, so surface this rather than deferring. */
      ARTS_ERROR("arts_put_in_db_from_gpu: DB[Guid:%lu] not installed on its "
                 "home rank",
                 db_guid);
    }
    if (free_data) {
      arts_gpu_route_table_add_item_to_delete(ptr, 0, db_guid,
                                              arts_local_gpu_id);
    }
  }
}

arts_lc_sync_function_t lc_sync_function[] = {arts_memcpy_gpu_db,
                                              arts_get_latest_gpu_db,
                                              arts_get_random_gpu_db,
                                              arts_get_non_zeros_unsigned_int,
                                              arts_get_min_db_unsigned_int,
                                              arts_add_db_unsigned_int,
                                              arts_xor_db_uint64};

arts_lc_sync_function_gpu_t lc_sync_function_gpu[] = {
    arts_copy_gpu_db,
    arts_copy_gpu_db,
    arts_copy_gpu_db,
    arts_non_zero_gpu_db_unsigned_int,
    arts_min_gpu_db_unsigned_int,
    arts_add_gpu_db_unsigned_int,
    arts_xor_gpu_db_uint64};

unsigned int lc_sync_element_size[] = {
    sizeof(unsigned int), sizeof(unsigned int), sizeof(unsigned int),
    sizeof(unsigned int), sizeof(unsigned int), sizeof(unsigned int),
    sizeof(uint64_t)};

void internal_lc_sync_gpu(arts_guid_t acq_guid, struct arts_db_s *db) {
  if (db) {
    arts_lc_meta_t host;
    arts_lc_meta_t dev;
    host.guid = acq_guid;
    host.data = (void *)(db + 1);
    host.data_size = db->cache.db_size;
    host.host_version = &db->version;
    host.host_time_stamp = &db->time_stamp;
    host.gpu_version = 0;
    host.gpu_time_stamp = 0;
    host.gpu = -1;
    host.read_lock = &db->reader;
    host.write_lock = &db->writer;

    arts_cuda_set_device(-1, true);

    bool copy_only = false;
    unsigned int size = arts_db_total_size(db);
    struct arts_db_s *temp_space =
        (struct arts_db_s *)arts_malloc_align(size, 16);

    gpu_gc_write_lock(); // Don't let the gc take our copies...
    ARTS_DEBUG("FUNCTION: %u\n", arts_node_info.gpu_lc_sync);
    unsigned int rem_mask = gpu_lc_reduce(
        acq_guid, db, lc_sync_function_gpu[arts_node_info.gpu_lc_sync],
        &copy_only);
    ARTS_DEBUG("RemMask: %u\n", rem_mask);
    for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
      if (rem_mask & (1 << i)) {
        ARTS_DEBUG("Merging: %u\n", i);
        unsigned int gpu_version;
        unsigned int time_stamp;
        void *data_ptr = arts_gpu_route_table_lookup_db_res(
            acq_guid, (int)i, &gpu_version, &time_stamp, false);
        if (data_ptr) {
          if (!copy_only) {
            arts_gpu_invalidate_on_route_table(acq_guid, i);
          }

          arts_cuda_set_device((int)i, false);
          get_data_from_stream_now(i, temp_space, data_ptr, size, false);
          arts_gpu_route_table_return_db(acq_guid, !copy_only, i);

          dev.guid = acq_guid;
          dev.data = (void *)(temp_space + 1);
          dev.data_size = temp_space->cache.db_size;
          dev.host_version = &temp_space->version;
          dev.host_time_stamp = &temp_space->time_stamp;
          dev.gpu_version = gpu_version;
          dev.gpu_time_stamp = time_stamp;
          dev.gpu = (int)i;
          dev.read_lock = NULL;
          dev.write_lock = NULL;
          if (copy_only) {
            lc_sync_function[0](&host, &dev);
          } else {
            lc_sync_function[arts_node_info.gpu_lc_sync](&host, &dev);
          }
        }
      } else {
        ARTS_DEBUG("NO DB COPY ON GPU %d\n", i);
      }
    }
    gpu_gc_write_unlock();
    arts_free(temp_space);
    arts_cuda_restore_device();
  }
}

/* ======================================================================== */
/* Per-GPU stream submission buffering                                      */
/* ======================================================================== */

#define CHECKSTREAM 4096
#define MAXSTREAM 32
#define MAXBUFFER 128

volatile unsigned int stream_check_count[MAXSTREAM] = {0};

volatile unsigned int buff_lock[MAXSTREAM] = {0};
unsigned int host_to_dev_count[MAXSTREAM] = {0};
unsigned int kernel_to_dev_count[MAXSTREAM] = {0};
unsigned int dev_to_host_count[MAXSTREAM] = {0};
unsigned int wrap_up_count[MAXSTREAM] = {0};

arts_buffer_mem_move_t host_to_dev_buff[MAXSTREAM][MAXBUFFER];
arts_buffer_kernel_t kernel_to_dev_buff[MAXSTREAM][MAXBUFFER];
arts_buffer_mem_move_t dev_to_host_buff[MAXSTREAM][MAXBUFFER];
void *wrap_up_buff[MAXSTREAM][MAXBUFFER];

static void check_occupancy(arts_edt_t fn_ptr, unsigned int gpu_id,
                            dim3 block) {
  int max_active_blocks;
  int block_size = (int)(block.x * block.y * block.z);
  struct cudaDeviceProp prop = arts_gpus[gpu_id].prop;

  CHECKCORRECT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, (const void *)fn_ptr, block_size, 0));
  float occupancy =
      ((float)(max_active_blocks * block_size) / (float)prop.warpSize) /
      ((float)prop.maxThreadsPerMultiProcessor / (float)prop.warpSize);

  // Cumulative (running) average of occupancy
  arts_lock(&arts_gpus[gpu_id].device_lock);
  arts_gpus[gpu_id].occupancy =
      (occupancy + ((float)(arts_gpus[gpu_id].total_edts - 1) *
                    arts_gpus[gpu_id].occupancy)) /
      (float)(++arts_gpus[gpu_id].total_edts);
  arts_unlock(&arts_gpus[gpu_id].device_lock);
}

static bool push_data_to_stream(unsigned int gpu_id, void *dst, void *src,
                                size_t count, bool buff) {
  if (buff) {
    arts_lock(&buff_lock[gpu_id]);
    host_to_dev_buff[gpu_id][host_to_dev_count[gpu_id]].dst = dst;
    host_to_dev_buff[gpu_id][host_to_dev_count[gpu_id]].src = src;
    host_to_dev_buff[gpu_id][host_to_dev_count[gpu_id]].count = count;
    host_to_dev_count[gpu_id]++;

    bool ret = false;
    if (host_to_dev_count[gpu_id] == MAXBUFFER) {
      ret = flush_stream(gpu_id);
    }
    arts_unlock(&buff_lock[gpu_id]);
    return ret;
  }

  if (src) {
    CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice,
                                 arts_gpus[gpu_id].stream));
  } else {
    CHECKCORRECT(cudaMemsetAsync(dst, 0, count, arts_gpus[gpu_id].stream));
  }
  return true;
}

static bool get_data_from_stream(unsigned int gpu_id, void *dst, void *src,
                                 size_t count, bool buff) {
  if (buff) {
    arts_lock(&buff_lock[gpu_id]);
    dev_to_host_buff[gpu_id][dev_to_host_count[gpu_id]].dst = dst;
    dev_to_host_buff[gpu_id][dev_to_host_count[gpu_id]].src = src;
    dev_to_host_buff[gpu_id][dev_to_host_count[gpu_id]].count = count;
    dev_to_host_count[gpu_id]++;

    bool ret = false;
    if (dev_to_host_count[gpu_id] == MAXBUFFER) {
      ret = flush_stream(gpu_id);
    }
    arts_unlock(&buff_lock[gpu_id]);
    return ret;
  }
  CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost,
                               arts_gpus[gpu_id].stream));
  return true;
}

static bool push_kernel_to_stream(unsigned int gpu_id, uint32_t paramc,
                                  const uint64_t *paramv, uint32_t depc,
                                  arts_edt_dep_t *depv, arts_edt_t fn_ptr,
                                  dim3 grid, dim3 block, bool buff) {
  if (buff) {
    arts_lock(&buff_lock[gpu_id]);
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].paramc = paramc;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].paramv = paramv;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].depc = depc;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].depv = depv;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].fn_ptr = fn_ptr;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].grid[0] = grid.x;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].grid[1] = grid.y;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].grid[2] = grid.z;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].block[0] = block.x;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].block[1] = block.y;
    kernel_to_dev_buff[gpu_id][kernel_to_dev_count[gpu_id]].block[2] = block.z;
    kernel_to_dev_count[gpu_id]++;

    bool ret = false;
    if (kernel_to_dev_count[gpu_id] == MAXBUFFER) {
      ret = flush_stream(gpu_id);
    }
    arts_unlock(&buff_lock[gpu_id]);
    return ret;
  }

  void *kernel_args[] = {&paramc, &paramv, &depc, &depv};
  CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                (void **)kernel_args, (size_t)0,
                                arts_gpus[gpu_id].stream));
  check_occupancy(fn_ptr, gpu_id, block);
  return true;
}

static bool push_wrap_up_to_stream(unsigned int gpu_id, void *host_closure,
                                   bool buff) {
  if (buff) {
    arts_lock(&buff_lock[gpu_id]);
    wrap_up_buff[gpu_id][wrap_up_count[gpu_id]] = host_closure;
    wrap_up_count[gpu_id]++;

    bool ret = false;
    if (wrap_up_count[gpu_id] == MAXBUFFER) {
      ret = flush_stream(gpu_id);
    }
    arts_unlock(&buff_lock[gpu_id]);
    return ret;
  }

#if CUDART_VERSION >= 10000
  CHECKCORRECT(cudaLaunchHostFunc(arts_gpus[gpu_id].stream,
                                  arts_wrap_up_host_func, host_closure));
#else
  CHECKCORRECT(cudaStreamAddCallback(arts_gpus[gpu_id].stream, arts_wrap_up,
                                     host_closure, 0));
#endif
  return true;
}

static bool flush_mem_stream(unsigned int gpu_id, unsigned int *count,
                             arts_buffer_mem_move_t *buff,
                             enum cudaMemcpyKind kind) {
  unsigned int max = *count;
  if (max > 0) {
    uint64_t data_size = 0;
    for (unsigned int i = 0; i < max; i++) {
      if (buff[i].src) {
        // ARTS_INFO("i: %u %p %p %u %p\n", i, buff[i].dst, buff[i].src,
        // buff[i].count,  &arts_gpus[gpu_id].stream);
        CHECKCORRECT(cudaMemcpyAsync(buff[i].dst, buff[i].src, buff[i].count,
                                     kind, arts_gpus[gpu_id].stream));
        data_size += buff[i].count;
      } else {
        CHECKCORRECT(cudaMemsetAsync(buff[i].dst, 0, buff[i].count,
                                     arts_gpus[gpu_id].stream));
      }
    }
    *count = 0;
    return true;
  }
  return false;
}

static bool flush_kernel_stream(unsigned int gpu_id) {
  bool ret = (kernel_to_dev_count[gpu_id] > 0);
  if (ret) {
    for (unsigned int i = 0; i < kernel_to_dev_count[gpu_id]; i++) {
      void *kernel_args[] = {&kernel_to_dev_buff[gpu_id][i].paramc,
                             &kernel_to_dev_buff[gpu_id][i].paramv,
                             &kernel_to_dev_buff[gpu_id][i].depc,
                             &kernel_to_dev_buff[gpu_id][i].depv};
      dim3 grid(kernel_to_dev_buff[gpu_id][i].grid[0],
                kernel_to_dev_buff[gpu_id][i].grid[1],
                kernel_to_dev_buff[gpu_id][i].grid[2]);
      dim3 block(kernel_to_dev_buff[gpu_id][i].block[0],
                 kernel_to_dev_buff[gpu_id][i].block[1],
                 kernel_to_dev_buff[gpu_id][i].block[2]);
      CHECKCORRECT(cudaLaunchKernel(
          (const void *)kernel_to_dev_buff[gpu_id][i].fn_ptr, grid, block,
          (void **)kernel_args, (size_t)0, arts_gpus[gpu_id].stream));
      check_occupancy(kernel_to_dev_buff[gpu_id][i].fn_ptr, gpu_id, block);
    }
    kernel_to_dev_count[gpu_id] = 0;
  }
  return ret;
}

static bool flush_wrap_up_stream(unsigned int gpu_id) {
  bool ret = (wrap_up_count[gpu_id] > 0);
  for (unsigned int i = 0; i < wrap_up_count[gpu_id]; i++) {
#if CUDART_VERSION >= 10000
    CHECKCORRECT(cudaLaunchHostFunc(arts_gpus[gpu_id].stream,
                                    arts_wrap_up_host_func,
                                    wrap_up_buff[gpu_id][i]));
#else
    CHECKCORRECT(cudaStreamAddCallback(arts_gpus[gpu_id].stream, arts_wrap_up,
                                       wrap_up_buff[gpu_id][i], 0));
#endif
  }
  wrap_up_count[gpu_id] = 0;
  return ret;
}

static bool flush_stream(unsigned int gpu_id) {
  ARTS_DEBUG("%u %u %u %u\n", host_to_dev_count[gpu_id],
             kernel_to_dev_count[gpu_id], dev_to_host_count[gpu_id],
             wrap_up_count[gpu_id]);
  if (host_to_dev_count[gpu_id] || kernel_to_dev_count[gpu_id] ||
      dev_to_host_count[gpu_id] || wrap_up_count[gpu_id]) {
    arts_cuda_set_device((int)gpu_id, true);

    flush_mem_stream(gpu_id, &host_to_dev_count[gpu_id],
                     host_to_dev_buff[gpu_id], cudaMemcpyHostToDevice);
    flush_kernel_stream(gpu_id);
    flush_mem_stream(gpu_id, &dev_to_host_count[gpu_id],
                     dev_to_host_buff[gpu_id], cudaMemcpyDeviceToHost);
    flush_wrap_up_stream(gpu_id);

    arts_cuda_restore_device();
    return true;
  }
  return false;
}

static void copy_gputo_gpu(void *dst, unsigned int dst_gpu_id, void *src,
                           unsigned int src_gpu_id, unsigned int size) {
  // We need to lock in a fixed order, so smallest first
  unsigned int first = (dst_gpu_id < src_gpu_id) ? dst_gpu_id : src_gpu_id;
  unsigned int second = (dst_gpu_id == first) ? src_gpu_id : dst_gpu_id;
  arts_lock(&buff_lock[first]);
  arts_lock(&buff_lock[second]);

  // Flush the streams to make sure everything is done
  flush_stream(dst_gpu_id);
  flush_stream(src_gpu_id);
  CHECKCORRECT(cudaStreamSynchronize(arts_gpus[src_gpu_id].stream));

  // Next lets move the data
  CHECKCORRECT(cudaMemcpyPeerAsync(dst, dst_gpu_id, src, src_gpu_id, size,
                                   arts_gpus[dst_gpu_id].stream));

  arts_cuda_restore_device();

  // Unlock in the correct order
  arts_unlock(&buff_lock[second]);
  arts_unlock(&buff_lock[first]);
}

void reduce_datafrom_gpus(void *dst, unsigned int dst_gpu_id, void *src,
                          unsigned int src_gpu_id, unsigned int size,
                          arts_lc_sync_function_gpu_t fn_ptr,
                          unsigned int element_size, void *db_data) {
  ARTS_DEBUG("ELEMENT SIZE: %lu\n", element_size);
  // We need to lock in a fixed order, so smallest first
  unsigned int first = (dst_gpu_id < src_gpu_id) ? dst_gpu_id : src_gpu_id;
  unsigned int second = (dst_gpu_id == first) ? src_gpu_id : dst_gpu_id;
  arts_lock(&buff_lock[first]);
  arts_lock(&buff_lock[second]);

  // Flush the streams to make sure everything is done
  flush_stream(dst_gpu_id);
  flush_stream(src_gpu_id);
  CHECKCORRECT(cudaStreamSynchronize(arts_gpus[src_gpu_id].stream));
  // I think we don't need to synchronize the destination stream since we are
  // just adding to it...
  //  CHECKCORRECT(cudaStreamSynchronize(arts_gpus[dst_gpu_id].stream));

  // Next lets move the data
  CHECKCORRECT(cudaMemcpyPeerAsync(dst, dst_gpu_id, src, src_gpu_id, size,
                                   arts_gpus[dst_gpu_id].stream));

  arts_cuda_set_device((int)dst_gpu_id, true);

  // Lets remove the db header part
  size -= sizeof(struct arts_db_s);

  // Next lets run the reduce function on the db_data and the shadow copy (dst)
  unsigned int tile_size = size / element_size;
  ARTS_DEBUG("TileSize: %u\n", tile_size);
  if (tile_size < 32) {
    dim3 block(tile_size, 1, 1); // For volta...
    dim3 grid(1, 1, 1);
    void *kernel_args[] = {&db_data, &dst};
    ARTS_DEBUG("SRC: %p DST: %p\n", db_data, dst);
    CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                  (void **)kernel_args, (size_t)0,
                                  arts_gpus[dst_gpu_id].stream));
  } else {
    dim3 block(32, 1, 1); // For volta...
    dim3 grid((tile_size + 32 - 1) / 32, 1, 1);
    void *kernel_args[] = {&db_data, &dst};
    ARTS_DEBUG("SRC: %p DST: %p\n", db_data, dst);
    CHECKCORRECT(cudaLaunchKernel((const void *)fn_ptr, grid, block,
                                  (void **)kernel_args, (size_t)0,
                                  arts_gpus[dst_gpu_id].stream));
  }

  // CHECKCORRECT(cudaStreamSynchronize(arts_gpus[src_gpu_id].stream));
  // CHECKCORRECT(cudaStreamSynchronize(arts_gpus[dst_gpu_id].stream));
  arts_cuda_restore_device();

  // Unlock in the correct order
  arts_unlock(&buff_lock[second]);
  arts_unlock(&buff_lock[first]);
}

static void get_data_from_stream_now(unsigned int gpu_id, void *dst, void *src,
                                     size_t count, bool buff) {
  if (buff) {
    arts_lock(&buff_lock[gpu_id]);
    flush_stream(gpu_id);
    arts_unlock(&buff_lock[gpu_id]);
  }
  ARTS_DEBUG("GETTING[%u]: %p %p size: %u\n", gpu_id, dst, src, count);
  CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost,
                               arts_gpus[gpu_id].stream));
  CHECKCORRECT(cudaStreamSynchronize(arts_gpus[gpu_id].stream));
}

static bool check_streams(bool buff_on) {
  if (buff_on) {
    bool ret = false;
    for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
      if (host_to_dev_count[i] || kernel_to_dev_count[i] ||
          dev_to_host_count[i] || wrap_up_count[i]) {
        arts_atomic_fetch_add(&stream_check_count[i], 1U);
        if (stream_check_count[i] % CHECKSTREAM == 0) {
          arts_lock(&buff_lock[i]);
          ret |= flush_stream(i);
          arts_unlock(&buff_lock[i]);
        }
      }
    }
    return ret;
  }
  return false;
}
