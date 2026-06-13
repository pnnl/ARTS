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

/**
 * @file arts/gpu.h
 * @brief GPU extension API for ARTS (CUDA-free public header).
 *
 * This header provides the GPU EDT creation API, device management, and
 * GPU memory utilities.  It is a pure C header — no CUDA types are exposed,
 * so it can be included from any C/C++ translation unit without NVCC.
 *
 * @code
 * #include "arts/gpu.h"
 * @endcode
 */
#ifndef ARTS_GPU_H
#define ARTS_GPU_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"

/* ========================================================================= */
/** @defgroup gpu_types GPU Types
 *  CUDA-free equivalents of GPU-specific types.
 *  @{ */

/** CUDA-free 3D dimension descriptor (equivalent to @c dim3). */
typedef struct {
  unsigned int x;
  unsigned int y;
  unsigned int z;
} arts_dim3_t;

/**
 * @brief Advisory metadata for GPU EDT creation.
 *
 * All fields have natural zero-defaults.  Pass NULL for:
 * route = current node, gpu = auto-select, no completion signal,
 * passthrough = false, lib = false.
 *
 * Initialize with compound literals:
 * @code
 * arts_edt_create_gpu(kernel, paramc, paramv, depc,
 *                     (arts_dim3_t){256,1,1}, (arts_dim3_t){32,1,1},
 *                     &(arts_gpu_hint_t){.end_guid = done, .slot = 0});
 * @endcode
 */
typedef struct {
  unsigned int rank;     /**< Target node rank. ARTS_HINT_CURRENT_RANK = current
                              node (default when NULL hint). */
  uint64_t id;           /**< Compiler-assigned profiling ID. 0 = disabled. */
  int gpu;               /**< GPU device index. -1 = auto-select (default). */
  arts_guid_t end_guid;  /**< EDT/event to signal on completion (NULL_GUID =
                              none). */
  uint32_t slot;         /**< Dependency slot to fill on completion. */
  arts_guid_t data_guid; /**< Data GUID passed with the completion signal. */
  bool passthrough;      /**< Pass-through mode: forward an input dep as the
                              completion data instead of @c data_guid. */
  bool lib; /**< Library EDT mode: function runs on CPU with GPU stream access,
                 not as a CUDA kernel.  Use for cuBLAS-style host functions. */
} arts_gpu_hint_t;

/** @} */ /* end gpu_types */

/* ========================================================================= */
/** @defgroup gpu_edt GPU EDT Creation
 *  Create GPU Event-Driven Tasks.
 *  @{ */

/**
 * @brief Create a GPU EDT.
 *
 * The EDT executes @p func_ptr as a CUDA kernel (or host function if
 * @c hint->lib is true) once all @p depc dependencies are satisfied.
 *
 * @param func_ptr Function to execute on GPU.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of @p paramc uint64_t values copied into the closure.
 * @param depc     Number of dependency slots.
 * @param grid     CUDA grid dimensions.
 * @param block    CUDA block dimensions.
 * @param hint     GPU advisory metadata. NULL = defaults.
 * @return GUID of the newly created GPU EDT.
 */
arts_guid_t arts_edt_create_gpu(arts_edt_t func_ptr, uint32_t paramc,
                                const uint64_t *paramv, uint32_t depc,
                                arts_dim3_t grid, arts_dim3_t block,
                                const arts_gpu_hint_t *hint);

/**
 * @brief Create a GPU EDT with a pre-reserved @p guid.
 *
 * The EDT runs on the home node of @p guid.
 *
 * @param func_ptr Function to execute on GPU.
 * @param guid     Pre-reserved GUID (determines target node).
 * @param paramc   Number of static parameters.
 * @param paramv   Array of parameters.
 * @param depc     Number of dependency slots.
 * @param grid     CUDA grid dimensions.
 * @param block    CUDA block dimensions.
 * @param hint     GPU advisory metadata. NULL = defaults.
 * @return The same @p guid, now associated with the GPU EDT.
 */
arts_guid_t arts_edt_create_gpu_with_guid(arts_edt_t func_ptr, arts_guid_t guid,
                                          uint32_t paramc,
                                          const uint64_t *paramv, uint32_t depc,
                                          arts_dim3_t grid, arts_dim3_t block,
                                          const arts_gpu_hint_t *hint);

/** @} */ /* end gpu_edt */

/* ========================================================================= */
/** @defgroup gpu_device Device Management
 *  Query and select GPU devices.
 *  @{ */

/** @brief Return the device index of the GPU running the current EDT. */
int arts_get_current_gpu(void);

/**
 * @brief Set the active CUDA device.
 *
 * @param id   Device index.
 * @param save If true, the previous device is saved for
 * arts_cuda_restore_device().
 * @return true on success.
 */
bool arts_cuda_set_device(int id, bool save);

/** @brief Restore the CUDA device saved by arts_cuda_set_device(). */
bool arts_cuda_restore_device(void);

/** @brief Return the number of GPUs available on this node. */
unsigned int arts_get_num_gpus(void);

/** @} */ /* end gpu_device */

/* ========================================================================= */
/** @defgroup gpu_mem GPU Memory
 *  Allocate and transfer GPU device/pinned memory.
 *  @{ */

/** @brief Allocate @p size bytes on the current GPU device. */
void *arts_cuda_malloc(unsigned int size);

/** @brief Free device memory allocated by arts_cuda_malloc(). */
void arts_cuda_free(void *ptr);

/** @brief Allocate @p size bytes of pinned (page-locked) host memory. */
void *arts_cuda_malloc_host(unsigned int size);

/** @brief Free pinned host memory allocated by arts_cuda_malloc_host(). */
void arts_cuda_free_host(void *ptr);

/** @brief Copy @p count bytes from device @p src to host @p dst. */
void arts_cuda_mem_cpy_from_dev(void *dst, void *src, size_t count);

/** @brief Copy @p count bytes from host @p src to device @p dst. */
void arts_cuda_mem_cpy_to_dev(void *dst, void *src, size_t count);

/** @} */ /* end gpu_mem */

/* ========================================================================= */
/** @defgroup gpu_data GPU Data Operations
 *  Move data between GPU and the DataBlock layer.
 *  @{ */

/**
 * @brief Write GPU data into a DataBlock.
 *
 * @param ptr      Source pointer on the GPU device.
 * @param db_guid  Target DataBlock GUID.
 * @param offset   Byte offset within the DataBlock.
 * @param size     Number of bytes to write.
 * @param free_data If true, the source pointer is freed after the copy.
 */
void arts_put_in_db_from_gpu(void *ptr, arts_guid_t db_guid,
                             unsigned int offset, unsigned int size,
                             bool free_data);

/**
 * @brief Synchronize an LC DataBlock from GPU to CPU and signal an EDT.
 *
 * Copies the GPU-resident data of the LC (Location Consistency) DataBlock
 * back to the host and signals @p edt_guid at dependency @p slot.
 *
 * @param edt_guid EDT to signal after the sync completes.
 * @param slot     Dependency slot to fill.
 * @param data_guid LC DataBlock GUID to synchronize.
 */
void arts_lc_sync(arts_guid_t edt_guid, uint32_t slot, arts_guid_t data_guid);

/**
 * @brief Signal an EDT slot with GPU memset (zero-initialize on device).
 *
 * Allocates GPU memory for the DataBlock, zeroes it via @c cudaMemset,
 * and signals @p edt_guid at dependency @p slot.
 *
 * @param edt_guid EDT to signal.
 * @param slot     Dependency slot to fill.
 * @param data_guid DataBlock GUID to memset.
 */
void arts_gpu_signal_edt_memset(arts_guid_t edt_guid, uint32_t slot,
                                arts_guid_t data_guid);

/** @} */ /* end gpu_data */

/* ========================================================================= */
/** @defgroup gpu_context GPU Context (inside library EDTs)
 *  Query GPU state from within a library EDT body.
 *  @{ */

/** @brief Return a pointer to the grid dimensions of the current GPU EDT. */
arts_dim3_t *arts_get_gpu_grid(void);

/** @brief Return a pointer to the block dimensions of the current GPU EDT. */
arts_dim3_t *arts_get_gpu_block(void);

/**
 * @brief Return a pointer to the CUDA stream of the current GPU.
 *
 * The returned pointer is actually a @c cudaStream_t* but is exposed as
 * @c void* to avoid CUDA header dependencies.  Cast to @c cudaStream_t*
 * inside @c .cu files.
 */
void *arts_get_gpu_stream(void);

/** @brief Return the device index of the current GPU. */
int arts_get_gpu_id(void);

/** @} */ /* end gpu_context */

/* ========================================================================= */
/** @defgroup gpu_kernel CUDA Kernel Utilities (NVCC only)
 *  Macros available only inside @c .cu translation units.
 *  @{ */

#ifdef __CUDACC__
/**
 * @brief Retrieve the global thread index inside a GPU kernel.
 *
 * The runtime stores the index in @c paramv[-1] so that each kernel thread
 * can determine its unique work item.  Use only inside @c __global__ kernels.
 */
#define ARTS_GPU_INDEX() (*(paramv - 1))

/** @brief Convert a CUDA @c dim3 to an @c arts_dim3_t. */
static inline arts_dim3_t arts_from_dim3(dim3 d) {
  arts_dim3_t r;
  r.x = d.x;
  r.y = d.y;
  r.z = d.z;
  return r;
}
#endif

/** @} */ /* end gpu_kernel */

#ifdef __cplusplus
}
#endif
#endif /* ARTS_GPU_H */
