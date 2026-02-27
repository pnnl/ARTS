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
#include "arts/defs.h"
#include "arts/gas/guid.h"
#include "arts/gpu/gpu_internal.h"
#include "arts/gpu/gpu_lc_sync_functions.cuh"
#include "arts/gpu/gpu_route_table.h"
#include "arts/gpu/gpu_stream_buffer.h"
#include "arts/compute/edt.h"
#include "arts/memory/db.h"
#include "arts/runtime_state.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/deque.h"
#include "arts/utils/malloc.h"

int random(void *edt_packet);
int all_or_nothing(void *edt_packet);
int atleast_one(void *edt_packet);
int hash_on_db_zero(void *edt_packet);
int hash_largest(void *edt_packet);
int first_fit(uint64_t mask, uint64_t size, unsigned int total_threads);
int best_fit(uint64_t mask, uint64_t size, unsigned int total_threads);
int worst_fit(uint64_t mask, uint64_t size, unsigned int total_threads);
int round_robin_fit(uint64_t mask, uint64_t size, unsigned int total_threads);
bool try_reserve(int gpu, uint64_t size, unsigned int threads);

volatile unsigned int hits = 0;
volatile unsigned int misses = 0;
volatile uint64_t free_bytes = 0;

arts_gpu_t *arts_gpus;

typedef int (*locality_t)(void *edt);

locality_t locality_scheme[] = {random, all_or_nothing, atleast_one,
                                hash_on_db_zero, hash_largest};

locality_t locality; // Locality function ptr

typedef int (*fit_t)(uint64_t mask, uint64_t size, unsigned int total_threads);

fit_t fit_scheme[] = {first_fit, best_fit, worst_fit, round_robin_fit};

fit_t fit; // Fit function ptr

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
    arts_gpus[i].availGlobalMem = (uint64_t)temp_free_mem;
    arts_gpus[i].totalGlobalMem = (uint64_t)temp_max_mem;
    if (arts_gpus[i].availGlobalMem > arts_node_info.gpu_max_memory) {
      arts_gpus[i].availGlobalMem = arts_node_info.gpu_max_memory;
    }
    ARTS_DEBUG("to Start: %lu\n", arts_gpus[i].availGlobalMem);
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
  arts_atomic_sub(&arts_gpu->availableEdtSlots, 1U);
  arts_atomic_sub(&arts_gpu->runningEdts, 1U);

  // Shouldn't have to touch newly ready edts regardless of streams and devices
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)gc->edt;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);

  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);
  arts_atomic_sub(&arts_gpu->availableThreads, total_threads);

  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr) {
      struct arts_db_s *db_hdr = (struct arts_db_s *)depv[i].ptr - 1;
      if (db_hdr->db_type == ARTS_DB_GPU) {
        arts_gpu_invalidate_route_tables(depv[i].guid, gc->gpu_id);
      }
      // True says to mark it for deletion... Change this to false to further
      // delay delete!
      //  bool mark_delete = (arts_guid_get_type(depv[i].guid) !=
      //  ARTS_DB_GPU)
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
  arts_gpu_route_table_return_db(edt->wrapperEdt.current_edt, true, gc->gpu_id);
  new_edt_lock = gc->newEdtLock;
  new_edts = gc->newEdts;
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
    host_gc_ptr->newEdtLock = new_edt_lock;
    host_gc_ptr->newEdts = new_edts;
    host_gc_ptr->devClosure = dev_closure;
    host_gc_ptr->edt = (struct arts_edt_s *)edt_ptr;
    *host_gpu_id = (uint64_t)arts_gpu->device;
    for (unsigned int i = 0; i < paramc; i++) {
      host_paramv[i] = paramv[i];
    }
    ARTS_DEBUG("Filled host closure\n");

    arts_guid_t edt_guid = host_gc_ptr->edt->current_edt;
    // arts_gpu_route_table_add_item_race(host_gc_ptr, host_closure_size,
    // edt_guid, arts_gpu->device);
    arts_gpu_route_table_add_item_race(host_gc_ptr, dev_closure_size, edt_guid,
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
      uint64_t size = db->header.size;
      uint64_t alloc_size = (db_subtype == ARTS_DB_LC) ? (size * 2) : size;
      if (!data_ptr) {
        bool successful_add = false;
        ARTS_DEBUG("WRAPPER SIZE: %lu\n", alloc_size);
        arts_item_wrapper_t *wrapper = arts_gpu_route_table_reserve_item_race(
            &successful_add, alloc_size, depv[i].guid, arts_gpu->device, true);

        if (successful_add) // We won, so allocate and move data
        {
          ARTS_DEBUG("Adding %lu %u id: %d mode: %s\n", depv[i].guid,
                     alloc_size, arts_gpu->device, db_mode_name[depv[i].mode]);
          data_ptr = arts_cuda_malloc(alloc_size);
          void *src = (void *)db;
          if (db_subtype == ARTS_DB_LC) {
            src = make_lc_shadow_copy(db);
          }
          if (depv[i].mode == DB_MODE_LC_NO_COPY || depv[i].mode == DB_MODE_MEMSET) {
            src = NULL;
          }
          push_data_to_stream(arts_gpu->device, data_ptr, src, size,
                              arts_node_info.gpu_buff_on && !gpu_edt->lib);
          // Must have already launched the memcpy before setting realData or
          // races will ensue
          wrapper->realData = data_ptr;
          ARTS_DEBUG("Malloc[%d]: %p %p\n", arts_gpu->device, wrapper,
                     data_ptr);
          arts_atomic_add(&misses, 1U);
        } else // Someone beat us to creating the data... So we must free
        {
          while (
              !arts_atomic_fetch_add_u64((uint64_t *)&wrapper->realData, 0)) {
          } // Spin till the data memcpy is launched
          data_ptr = (void *)wrapper->realData;
          if (db_subtype == ARTS_DB_GPU && depv[i].mode == DB_MODE_MEMSET) {
            push_data_to_stream(arts_gpu->device, data_ptr, NULL, size,
                                arts_node_info.gpu_buff_on && !gpu_edt->lib);
          }
          arts_atomic_add_u64(&arts_gpu->availGlobalMem, alloc_size);
          arts_atomic_add(&hits, 1U);
        }
      } else {
        arts_atomic_add_u64(&arts_gpu->availGlobalMem, alloc_size);
        arts_atomic_add(&hits, 1U);
      }
      struct arts_db_s *new_db = (struct arts_db_s *)data_ptr;
      host_depv[i].ptr = (void *)(new_db + 1);
    } else {
      ARTS_DEBUG("Depv: %u is null edt: %lu\n", i,
                 gpu_edt->wrapperEdt.current_edt);
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
    arts_route_table_reset_oo(host_gc_ptr->edt->current_edt);

    host_gc_ptr->edt->func_ptr(paramc, host_paramv, depc, host_depv);

    arts_unset_thread_local_edt_info();
    /* Release DBs created during the lib function NOW, on the worker thread.
       The wrap-up callback runs on the CUDA callback thread whose TLS
       created_db_list is empty, so arts_release_created_dbs() there would be
       a no-op — leaving frontiers un-progressed and consumer EDTs stuck. */
    arts_release_created_dbs();
  } else {
    push_kernel_to_stream(arts_gpu->device, paramc, dev_paramv, depc, dev_depv,
                          fn_ptr, grid, block, arts_node_info.gpu_buff_on);
  }

  // Move data back
  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr) {
      struct arts_db_s *cb_db = (struct arts_db_s *)depv[i].ptr - 1;
      if (cb_db->db_type == ARTS_DB_GPU &&
          (depv[i].mode == DB_MODE_EW || depv[i].mode == DB_MODE_MEMSET)) {
        size_t size = (size_t)(cb_db->header.size - sizeof(struct arts_db_s));
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

void arts_gpu_synchronize(arts_gpu_t *arts_gpu) {
  CHECKCORRECT(cudaStreamSynchronize(arts_gpu->stream));
}

void arts_gpu_stream_busy(arts_gpu_t *arts_gpu) {
  CHECKCORRECT(cudaStreamQuery(arts_gpu->stream));
}

void free_gpu_item(arts_route_item_t *item) {
  arts_type_t type = arts_guid_get_type(item->key);
  arts_item_wrapper_t *wrapper = (arts_item_wrapper_t *)item->data;
  if (type == ARTS_EDT) {
    arts_gpu_clean_up_t *host_gc_ptr = (arts_gpu_clean_up_t *)wrapper->realData;
    ARTS_DEBUG("FREEING DEV PTR: %p\n", host_gc_ptr->devClosure);
    arts_cuda_free(host_gc_ptr->devClosure);
    ARTS_DEBUG("FREEING HOST PTR: %p\n", host_gc_ptr);
    arts_cuda_free_host(host_gc_ptr);
  } else if (type == ARTS_DB) {
    int valid_rank = -1;
    struct arts_db_s *db = (struct arts_db_s *)arts_route_table_lookup_db(
        item->key, &valid_rank, false);
    if (db && db->db_type == ARTS_DB_LC) {
      unsigned int size = db->header.size;
      struct arts_db_s *temp_space =
          (struct arts_db_s *)arts_malloc_align(size, 16);

      arts_lc_meta_t host;
      host.guid = item->key;
      host.data = (void *)(db + 1);
      host.data_size = db->header.size - sizeof(struct arts_db_s);
      host.host_version = &db->version;
      host.host_time_stamp = &db->time_stamp;
      host.gpu_version = 0;
      host.gpu_time_stamp = 0;
      host.gpu = -1;
      host.read_lock = &db->reader;
      host.write_lock = &db->writer;

      // arts_cuda_mem_cpy_from_dev(temp_space, (void*) wrapper->realData,
      // size);
      get_data_from_stream_now(arts_get_current_gpu(), temp_space,
                               (void *)wrapper->realData, size, false);

      arts_lc_meta_t dev;
      dev.guid = item->key;
      dev.data = (void *)(temp_space + 1);
      dev.data_size = temp_space->header.size - sizeof(struct arts_db_s);
      dev.host_version = &temp_space->version;
      dev.host_time_stamp = &temp_space->time_stamp;
      dev.gpu_version = item->touched;
      dev.gpu_time_stamp = wrapper->time_stamp;
      dev.gpu = -1;
      dev.read_lock = NULL;
      dev.write_lock = NULL;

      lc_sync_function[arts_node_info.gpu_lc_sync](&host, &dev);

      arts_route_table_return_db(item->key, false);
      arts_free(temp_space);
      arts_cuda_free((void *)wrapper->realData);

    } else {
      // Non-LC DB (DEFAULT/GPU) or LC DB not found — just free GPU memory
      arts_cuda_free((void *)wrapper->realData);
    }
  }

  wrapper->realData = NULL;
  wrapper->time_stamp = 0;
  item->key = 0;
  item->lock = 0;
  item->touched = 0;
}

ARTS_THREAD_LOCAL unsigned int run_gc_flag = 0;

bool try_reserve(int gpu, uint64_t size, unsigned int threads) {
  (void)threads;
  arts_gpu_t *arts_gpu = &arts_gpus[gpu];
  ARTS_DEBUG("Trying to reserve %lu of available %lu on GPU[%d]\n", size,
             arts_gpu->availGlobalMem, arts_gpu->device);
  // if(arts_atomic_fetch_add(&arts_gpu->availableThreads, threads) < 1024)
  {
    if (arts_atomic_fetch_add(&arts_gpu->availableEdtSlots, 1U) <
        arts_node_info.gpu_max_edts) {
      volatile uint64_t avail_size = arts_gpu->availGlobalMem;
      while (avail_size >= size) {
        if (arts_atomic_cswap_u64(&arts_gpu->availGlobalMem, avail_size,
                                  avail_size - size)) {
          run_gc_flag = 0;
          return true;
        }
        avail_size = arts_gpu->availGlobalMem;
      }
      run_gc_flag = gpu + 1;
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

int first_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int random = (int)jrand48(arts_thread_info.drand_buf);
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + (unsigned int)random) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask && check_mask) {
      if (try_reserve(index, size, total_threads)) {
        ARTS_DEBUG("Reserved Successfully on %u\n", index);
        return index;
      }
    }
  }
  return -1;
}

int round_robin_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  static volatile unsigned int next = 0;
  unsigned int start = arts_atomic_fetch_add(&next, 1U);
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + start) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask && check_mask) {
      if (try_reserve(index, size, total_threads)) {
        ARTS_DEBUG("Reserved Successfully on %u\n", index);
        return index;
      }
    }
  }
  return -1;
}

int best_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int selected_gpu = -1;
  uint64_t selected_gpu_avail_size = 0;
  int random = (int)jrand48(arts_thread_info.drand_buf);
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + (unsigned int)random) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask && check_mask) {
      if (selected_gpu != -1) {
        if (arts_gpus[index].availGlobalMem - size > selected_gpu_avail_size) {
          continue;
        }
      }
      if (try_reserve(index, size, total_threads)) {
        // If successful relinquish previous allocation (if any).
        if (selected_gpu != -1) {
          arts_atomic_add_u64(&arts_gpus[selected_gpu].availGlobalMem, size);
        }
        selected_gpu = index;
        selected_gpu_avail_size = arts_gpus[index].availGlobalMem;
      }
    }
  }
  return selected_gpu;
}

int worst_fit(uint64_t mask, uint64_t size, unsigned int total_threads) {
  int selected_gpu = -1;
  uint64_t selected_gpu_avail_size = 0;
  int random = (int)jrand48(arts_thread_info.drand_buf);
  for (unsigned int i = 0; i < arts_node_info.gpu; i++) {
    int index = (int)((i + (unsigned int)random) % arts_node_info.gpu);
    uint64_t check_mask = (uint64_t)1 << index;
    if (mask && check_mask) {
      if (selected_gpu != -1) {
        if (arts_gpus[index].availGlobalMem - size < selected_gpu_avail_size) {
          continue;
        }
      }
      if (try_reserve(index, size, total_threads)) {
        // If successful relinquish previous allocation (if any).
        if (selected_gpu != -1) {
          arts_atomic_add_u64(&arts_gpus[selected_gpu].availGlobalMem, size);
        }
        selected_gpu = index;
        selected_gpu_avail_size = arts_gpus[index].availGlobalMem;
      }
    }
  }
  return selected_gpu;
}

uint64_t get_db_size_needed(uint32_t depc, arts_edt_dep_t *depv) {
  uint64_t size = 0;
  for (unsigned int i = 0; i < depc; i++) {
    if (depv[i].ptr) {
      struct arts_db_s *db = (struct arts_db_s *)depv[i].ptr - 1;
      size += db->header.size;
      if (db->db_type == ARTS_DB_LC) {
        size += db->header.size;
      }
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
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  // Size to be allocated on the GPU
  uint64_t size = (sizeof(uint64_t) * paramc) +
                  (sizeof(arts_edt_dep_t) * depc) +
                  get_db_size_needed(depc, depv);
  uint64_t mask = ~0;
  return fit(mask, size, total_threads);
}

int all_or_nothing(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  // Size to be allocated on the GPU
  uint64_t size = (sizeof(uint64_t) * paramc) +
                  (sizeof(arts_edt_dep_t) * depc) +
                  get_db_size_needed(depc, depv);
  uint64_t mask = 0;
  for (unsigned int i = 0; i < depc; ++i) {
    mask &= arts_gpu_lookup_db(depv[i].guid);
  }

  ARTS_DEBUG("Mask: %p\n", mask);

  if (mask) { // All DBs in GPU
    return fit(mask, size,
               total_threads); // No need to fit since all Dbs are in a GPU
  }
  return random(edt_packet);
}

int atleast_one(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  // Size to be allocated on the GPU
  uint64_t size = (sizeof(uint64_t) * paramc) +
                  (sizeof(arts_edt_dep_t) * depc) +
                  get_db_size_needed(depc, depv);
  uint64_t mask = 0;
  for (unsigned int i = 0; i < depc; ++i) {
    mask |= arts_gpu_lookup_db(depv[i].guid);
  }

  ARTS_DEBUG("Mask: %p\n", mask);

  if (mask) { // At least one DB in GPU
    return fit(mask, size, total_threads);
  }
  return random(edt_packet);
}

int hash_on_db_zero(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  // Size to be allocated on the GPU
  uint64_t size = (sizeof(uint64_t) * paramc) +
                  (sizeof(arts_edt_dep_t) * depc) +
                  get_db_size_needed(depc, depv);
  uint64_t key = (depv[0].guid) ? arts_guid_get_key(depv[0].guid) : 0;
  int index = (int)(key % (uint64_t)arts_node_info.gpu);
  if ((unsigned int)index > arts_node_info.gpu) {
    ARTS_ERROR("GPU stream hash failed: index %d >= gpu count %u", index,
               arts_node_info.gpu);
  }
  ARTS_DEBUG("HASH: %lu %d\n", depv[0].guid, index);
  if (try_reserve(index, size, total_threads)) {
    return index;
  }
  return -1;
}

int hash_largest(void *edt_packet) {
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  // Size to be allocated on the GPU
  uint64_t size = (sizeof(uint64_t) * paramc) +
                  (sizeof(arts_edt_dep_t) * depc) +
                  get_db_size_needed(depc, depv);
  // uint64_t mask = 0;
  uint64_t largest = 0;
  for (unsigned int i = 0; i < depc; ++i) {
    uint64_t key = (depv[i].guid) ? arts_guid_get_key(depv[i].guid) : 0;
    largest = (key > largest) ? key : largest;
  }

  int index = (int)(largest % (uint64_t)arts_node_info.gpu);
  if (try_reserve(index, size, total_threads)) {
    ARTS_DEBUG("Index: %d\n", index);
    return index;
  }
  return -1;
}

int arts_reserve_edt_required_gpu(int *gpu, void *edt_packet) {
  bool ret = false;
  *gpu = -1;
  arts_gpu_edt_t *edt = (arts_gpu_edt_t *)edt_packet;
  uint32_t paramc = edt->wrapperEdt.paramc;
  uint32_t depc = edt->wrapperEdt.depc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);
  arts_edt_dep_t *depv = (arts_edt_dep_t *)(paramv + paramc);
  unsigned int total_threads = (edt->grid.x * edt->block.x) +
                               (edt->grid.y * edt->block.y) +
                               (edt->grid.z * edt->block.z);

  if (edt->gpuToRunOn > -1) {
    // Size to be allocated on the GPU
    uint64_t size = (sizeof(uint64_t) * paramc) +
                    (sizeof(arts_edt_dep_t) * depc) +
                    get_db_size_needed(depc, depv);
    if (try_reserve(edt->gpuToRunOn, size, total_threads)) {
      *gpu = edt->gpuToRunOn;
      ret = true;
    }
  }
  return ret;
}

arts_gpu_t *arts_find_gpu(void *data) {
  arts_gpu_t *ret = NULL;
  int gpu;
  if (!arts_reserve_edt_required_gpu(&gpu, data)) {
    gpu = locality(data);
  }
  ARTS_DEBUG("Choosing gpu: %d\n", gpu);
  if (gpu > -1 && gpu < (int)arts_node_info.gpu) {
    ret = &arts_gpus[gpu];
  }

  return ret;
}