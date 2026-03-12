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
 * @file arts.h
 * @brief Public API for the ARTS (Abstract Runtime System).
 *
 * This header exposes every user-facing function in ARTS.  Include it as
 * @code
 * #include "arts.h"
 * @endcode
 *
 * @see arts_rt, arts_shutdown
 */
#ifndef ARTS_H
#define ARTS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ========================================================================= */
/** @defgroup core_types Core Types
 *  Fundamental typedefs shared by the entire runtime.
 *  @{ */

/** Globally Unique Identifier — 64-bit bitfield encoding type, rank, and key.
 */
typedef intptr_t arts_guid_t;

/** Sentinel value representing an invalid or absent GUID. */
#define NULL_GUID ((arts_guid_t)0x0)

/** @} */ /* end core_types */

/* ========================================================================= */
/** @defgroup type_enum Type / Access-Mode Enumeration
 *  Every GUID carries a type tag from this enum.
 *  @{ */

/**
 * @brief Runtime object type tag (stored in GUID bits 63–56).
 *
 * Identifies what kind of object a GUID refers to: EDT, event, datablock, etc.
 * All datablocks share the single @c ARTS_DB tag; the DB subtype is specified
 * via @c arts_db_types_t at creation time.
 * Access modes (read/write) are separate — see @c arts_db_access_mode_t.
 */
typedef enum {
  ARTS_NULL,     /**< Empty / untyped placeholder. */
  ARTS_EDT,      /**< Event-Driven Task (CPU and GPU share this tag). */
  ARTS_EVENT,    /**< Latch-based synchronization event. */
  ARTS_EPOCH,    /**< Termination-detection epoch. */
  ARTS_CALLBACK, /**< Inline event callback. */
  ARTS_BUFFER,   /**< Node-local buffer accessible by GUID. */
  ARTS_DB,       /**< DataBlock (all subtypes share this tag). */
  ARTS_LAST_TYPE /**< Sentinel — first invalid type value. */
} arts_type_t;

/**
 * @brief DataBlock access mode (per-dependency, stored in @c
 * arts_edt_dep_t.mode).
 *
 * Specifies how an EDT accesses a datablock dependency.  Set via
 * @c arts_add_dependence() / @c arts_signal_edt(), not at DB creation.
 */
typedef enum {
  DB_MODE_NULL = 0, /**< Unset / placeholder. */
  DB_MODE_RO,       /**< Read-Only (shared readers, no writeback). */
  DB_MODE_EW,    /**< Exclusive Write (single writer, frontier progression). */
  DB_MODE_RW,    /**< Read-Write, no ordering (LOCAL DBs only). */
  DB_MODE_VALUE, /**< Dependency carries a raw uint64 value (not a GUID). */
  DB_MODE_PTR,   /**< Dependency carries a copied pointer buffer. */
  DB_MODE_LC_SYNC,    /**< LC with synchronous GPU-to-CPU copy. */
  DB_MODE_LC_NO_COPY, /**< LC without data copy (just allocate on GPU). */
  DB_MODE_MEMSET,     /**< GPU zero-initialization. */
} arts_db_access_mode_t;

/**
 * @brief DataBlock subtype (stored in @c arts_db_s.db_type, NOT in the GUID).
 *
 * Specifies the storage and coherence class of a DataBlock.  All subtypes
 * share the same @c ARTS_DB tag in the GUID; the subtype is carried inside
 * the DB descriptor.
 */
typedef enum {
  ARTS_DB_DEFAULT = 0, /**< Distributed, CDAG-managed (OCR spec DB). */
  ARTS_DB_LOCAL,       /**< Node-pinned, no CDAG frontier. */
  ARTS_DB_GPU,         /**< GPU-pinned, CDAG-managed. */
  ARTS_DB_LC,          /**< Locality-class (CPU-GPU coherence). */
} arts_db_types_t;

/**
 * @brief EDT subtype (stored in @c arts_edt_s.edt_type, NOT in the GUID).
 *
 * All EDTs share the single @c ARTS_EDT tag in the GUID; the subtype is
 * carried inside the EDT descriptor.  CPU and GPU EDTs differ in scheduling
 * and execution but share the same GUID type.
 */
typedef enum {
  ARTS_EDT_DEFAULT = 0, /**< CPU EDT (standard). */
  ARTS_EDT_GPU = 1,     /**< GPU EDT (CUDA kernel or library host function). */
} arts_edt_types_t;

/** @} */ /* end type_enum */

/* ========================================================================= */
/** @defgroup hint_type Hint Type
 *  Advisory metadata for EDT and DataBlock creation.
 *  @{ */

/** Sentinel: use the node that is running the current EDT. */
#define ARTS_HINT_CURRENT_NODE ((unsigned int)-1)

/**
 * @brief Advisory metadata for EDT/DB creation.
 *
 * Pass a pointer to this struct as the last argument of creation functions.
 * NULL is always valid and selects default values (current node, no profiling).
 *
 * Initialize with compound literals:
 * @code
 * arts_edt_create(func, paramc, paramv, depc,
 *                 &(arts_hint_t){.route = 0});
 * arts_edt_create(func, paramc, paramv, depc,
 *                 &(arts_hint_t){.route = ARTS_HINT_CURRENT_NODE, .id = 42});
 * arts_db_create(&addr, len, NULL);
 * @endcode
 */
typedef struct {
  unsigned int route; /**< Target node rank. ARTS_HINT_CURRENT_NODE = current
                           node (default when NULL hint is passed). */
  uint64_t id;        /**< Compiler-assigned profiling ID. 0 = disabled. */
} arts_hint_t;

/** @} */ /* end hint_type */

/* ========================================================================= */
/** @defgroup dep_types Dependency Types
 *  Structures and function-pointer types used to wire EDT dependencies.
 *  @{ */

/**
 * @brief Describes a single dependency slot delivered to an EDT.
 *
 * Mode is set via @c arts_add_dependence() or @c arts_signal_edt().
 * User EDTs typically read @c guid / @c ptr and ignore @c mode.
 */
typedef struct {
  arts_guid_t guid;           /**< GUID of the DataBlock (or encoded value). */
  void *ptr;                  /**< Pointer to the DataBlock payload. */
  arts_db_access_mode_t mode; /**< Access mode for this dependency slot. */
} arts_edt_dep_t;

/**
 * @brief Function signature for Event-Driven Tasks (CPU and GPU).
 */
typedef void (*arts_edt_t)(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]);

/**
 * @brief Callback invoked inline when a latch event fires.
 */
typedef void (*event_callback_t)(arts_edt_dep_t data);

/** @} */ /* end dep_types */

/* ========================================================================= */
/** @defgroup user_callbacks User Callbacks
 *  Optional weak-symbol callbacks invoked by the runtime.
 *  Define any of these in your application to hook into the lifecycle.
 *  @{ */

/**
 * @brief Main entry-point EDT, scheduled on rank 0 after runtime init.
 *
 * If defined, the runtime creates this EDT on node 0 with:
 *   - @c paramv[0] = @c argc (cast to @c uint64_t)
 *   - @c paramv[1] = @c argv (cast to @c uint64_t)
 *
 * The EDT can call blocking operations like arts_wait_on_handle().
 *
 * @param paramc Number of static parameters (2 when called by the runtime).
 * @param paramv Parameter array: paramv[0]=argc, paramv[1]=(uint64_t)argv.
 * @param depc   Number of dependency slots (0 when called by the runtime).
 * @param depv   Dependency array (empty when called by the runtime).
 */
extern void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]);

/**
 * @brief Optional per-node startup callback invoked before worker parallel
 * startup.
 *
 * When defined by the application, ARTS calls this once per node on thread 0
 * after basic runtime initialization and before the parallel-start barrier.
 * CARTS uses this hook for distributed DB GUID reservation.
 *
 * @param node_id Rank of the current node.
 * @param argc    Original process argument count.
 * @param argv    Original process argument vector.
 */
extern void init_per_node(unsigned int node_id, int argc, char **argv);

/**
 * @brief Optional per-worker startup callback invoked after the parallel-start
 * barrier.
 *
 * When defined by the application, ARTS calls this on each worker thread
 * after the startup barrier and before normal scheduler execution. CARTS uses
 * this hook for owner-local distributed DB materialization.
 *
 * @param node_id   Rank of the current node.
 * @param worker_id Local worker index on the current node.
 * @param argc      Original process argument count.
 * @param argv      Original process argument vector.
 */
extern void init_per_worker(unsigned int node_id, unsigned int worker_id,
                            int argc, char **argv);

/** @} */ /* end user_callbacks */

/* ========================================================================= */
/** @defgroup event_slots Event Slot Types
 *  @{ */

/** Slot constants for latch-event signaling. */
typedef enum {
  ARTS_EVENT_LATCH_DECR_SLOT = 0, /**< Decrement the latch counter. */
  ARTS_EVENT_LATCH_INCR_SLOT = 1, /**< Increment the latch counter. */
  ARTS_EVENT_UPDATE = 2           /**< Update data (channel events only). */
} arts_latch_event_slot_t;

/** @} */ /* end event_slots */

/* ========================================================================= */
/** @defgroup event_behavior Event Behavior Types
 *  @{ */

/** Event behavior types (OCR-compatible).
 *
 *  All four behaviors share the same @c ARTS_EVENT GUID type and the same
 *  latch-counter mechanism.  The difference is in what happens after the
 *  counter reaches zero (fire) and on re-satisfy attempts.
 */
typedef enum {
  ARTS_EVENT_LATCH = 0, /**< N-counter, auto-destroy on fire (OCR LATCH_T). */
  ARTS_EVENT_ONCE,      /**< latch=1 shorthand, auto-destroy (OCR ONCE_T). */
  ARTS_EVENT_STICKY,  /**< latch=1, persist, error on re-satisfy (OCR STICKY_T).
                       */
  ARTS_EVENT_IDEM,    /**< latch=1, persist, ignore re-satisfy (OCR IDEM_T). */
  ARTS_EVENT_COUNTED, /**< N-counter, auto-destroy, no INCR (OCR-Vx). */
  ARTS_EVENT_CHANNEL, /**< Re-armable, version-based, DB-coupled (OCR-Vx). */
} arts_event_types_t;

/** @} */ /* end event_types */

/* ========================================================================= */

/**
 * @brief Thread-safe printf that serializes output across ARTS workers.
 * @return Number of characters written (excluding the rank prefix).
 */
int arts_printf(const char *format, ...);

/* ========================================================================= */
/** @defgroup runtime Runtime Lifecycle
 *  @{ */

/**
 * @brief Entry point to the ARTS runtime.
 *
 * Reads @c arts.cfg, initializes threading and networking, schedules
 * main_edt() on rank 0 (if defined), and blocks until
 * arts_shutdown() is called.
 *
 * @param argc Argument count from main().
 * @param argv Argument vector from main().
 * @return 0 on success.
 * @see arts_shutdown
 */
int arts_rt(int argc, char **argv);

/**
 * @brief Set the config file path before calling arts_rt().
 *
 * When set, arts_config_load() reads from this path instead of the
 * default ARTS_CONFIG env var / arts.cfg fallback.
 *
 * @param path  Null-terminated file path.  NULL or "" clears the override.
 */
void artsSetConfigPath(const char *path);

/**
 * @brief Inject config data as an in-memory string before calling arts_rt().
 *
 * When set, arts_config_load() parses this string (same INI format as
 * arts.cfg) instead of opening a file.  This enables self-contained binaries
 * that embed their runtime configuration at compile time.
 *
 * @param data  Null-terminated config string.  NULL or "" clears the override.
 */
void artsSetConfigData(const char *data);

/**
 * @brief Shut down the ARTS runtime.
 *
 * Triggers global termination detection.  It is possible to race to shutdown
 * if there are multiple concurrent calls.
 *
 * @see arts_rt
 */
void arts_shutdown();

/**
 * @brief Abort the ARTS runtime with an error code.
 *
 * Unlike arts_shutdown(), this function does not return.  It flushes
 * standard output streams and terminates the process immediately.
 * Remote nodes will detect the disconnection and shut down.
 *
 * @param error_code Process exit code (0-255).
 */
#ifdef __cplusplus
[[noreturn]]
#else
_Noreturn
#endif
void arts_abort(uint8_t error_code);

/** @} */ /* end runtime */

/* ========================================================================= */
/** @defgroup guid GUID Management
 *  Globally Unique Identifiers (GUIDs) and GUID ranges.
 *  @{ */

/**
 * @brief Reserve a GUID of the given @p type on node @p route.
 *
 * @param type  Type tag for the GUID (e.g. @c ARTS_EDT, @c ARTS_DB).
 * @param route Target node rank.
 * @return A new GUID.
 */
arts_guid_t arts_guid_reserve(arts_type_t type, unsigned int route);

/**
 * @brief Check whether @p guid is local to this node.
 *
 * @param guid GUID to test.
 * @return @c true if the GUID belongs to this node, @c false otherwise.
 */
bool arts_guid_is_local(arts_guid_t guid);

/**
 * @brief Return the rank of the node that owns @p guid.
 *
 * @param guid GUID to query.
 * @return Node rank.
 */
unsigned int arts_guid_get_rank(arts_guid_t guid);

/**
 * @brief Return the type tag encoded in @p guid.
 *
 * @param guid GUID to query.
 * @return The arts_type_t stored in the GUID.
 */
arts_type_t arts_guid_get_type(arts_guid_t guid);

/**
 * @brief Reserve a contiguous range of @p size GUIDs on node @p route.
 *
 * Returns the start GUID of the range.  Use @c arts_guid_from_index() to
 * access individual GUIDs within the range.
 *
 * @param type  Type tag for every GUID in the range.
 * @param size  Number of GUIDs to allocate.
 * @param route Target node rank.
 * @return The start GUID of the range, or @c NULL_GUID on failure.
 * @see arts_guid_from_index, arts_guid_index_from
 */
arts_guid_t arts_guid_reserve_range(arts_type_t type, unsigned int size,
                                    unsigned int route);

/**
 * @brief Get the GUID at @p idx offset from @p range_guid (no bounds check).
 *
 * This is the OCR-style index-based accessor.  The caller is responsible
 * for ensuring @p idx is within the range.
 *
 * @param range_guid The start GUID of a range.
 * @param idx        Zero-based offset.
 * @return The GUID at the requested offset.
 */
arts_guid_t arts_guid_from_index(arts_guid_t range_guid, unsigned int idx);

/**
 * @brief Compute the index of @p guid relative to @p range_guid.
 *
 * Returns -1 if the two GUIDs differ in type or rank, or if
 * @p guid precedes @p range_guid.
 *
 * @param range_guid The start GUID of a range.
 * @param guid       The GUID to look up.
 * @return Zero-based index, or -1 on mismatch.
 */
int arts_guid_index_from(arts_guid_t range_guid, arts_guid_t guid);

/**
 * @brief Reserve @p size GUIDs distributed round-robin across all nodes.
 *
 * @param size Number of GUIDs to allocate.
 * @param type Type tag for every GUID.
 * @return Array of GUIDs (caller must destroy with
 * arts_guid_round_robin_destroy).
 */
arts_guid_t *arts_guid_reserve_round_robin(unsigned int size, arts_type_t type);

/**
 * @brief Free an array returned by arts_guid_reserve_round_robin().
 */
void arts_guid_round_robin_destroy(arts_guid_t *guids);

/** @} */ /* end guid */

/* ========================================================================= */
/** @defgroup edt Event-Driven Tasks (EDT)
 *  Create, signal, and destroy asynchronous task units.
 *  @{ */

/**
 * @brief Create an EDT.
 *
 * The EDT will execute @p func_ptr once all @p depc dependency slots have
 * been satisfied via arts_signal_edt() or related functions.
 *
 * @param func_ptr Function to execute.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of @p paramc uint64_t values copied into the closure.
 * @param depc     Number of dependency slots.
 * @param hint     Advisory metadata (route, profiling id). NULL = defaults.
 * @return GUID of the newly created EDT.
 * @see arts_signal_edt, arts_edt_destroy
 */
arts_guid_t arts_edt_create(arts_edt_t func_ptr, uint32_t paramc,
                            const uint64_t *paramv, uint32_t depc,
                            const arts_hint_t *hint);

/**
 * @brief Create an EDT with a pre-reserved @p guid.
 *
 * The EDT runs on the home node of @p guid.
 *
 * @param func_ptr Function to execute.
 * @param guid     Pre-reserved GUID (determines target node).
 * @param paramc   Number of static parameters.
 * @param paramv   Array of parameters.
 * @param depc     Number of dependency slots.
 * @return The same @p guid, now associated with the EDT.
 * @see arts_guid_reserve
 */
arts_guid_t arts_edt_create_with_guid(arts_edt_t func_ptr, arts_guid_t guid,
                                      uint32_t paramc, const uint64_t *paramv,
                                      uint32_t depc);

/**
 * @brief Create an EDT in a specific epoch.
 *
 * The user must ensure the epoch is still live.
 *
 * @param func_ptr   Function to execute.
 * @param paramc     Number of static parameters.
 * @param paramv     Array of parameters.
 * @param depc       Number of dependency slots.
 * @param epoch_guid Epoch GUID (must still be live).
 * @param hint       Advisory metadata (route, profiling id). NULL = defaults.
 * @return GUID of the newly created EDT.
 * @see arts_initialize_and_start_epoch
 */
arts_guid_t arts_edt_create_with_epoch(arts_edt_t func_ptr, uint32_t paramc,
                                       const uint64_t *paramv, uint32_t depc,
                                       arts_guid_t epoch_guid,
                                       const arts_hint_t *hint);

/**
 * @brief Create an EDT with optional dependency-slot allocation.
 *
 * When @p has_depv is @c false the runtime does not allocate storage for the
 * dependency array, but still uses the @p depc counter.  Useful when an EDT
 * has many dependencies but does not need their result data.
 *
 * @param func_ptr Function to execute.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of parameters.
 * @param depc     Number of dependencies.
 * @param has_depv If @c false, skip depv allocation.
 * @param hint     Advisory metadata (route, profiling id). NULL = defaults.
 * @return GUID of the newly created EDT.
 */
arts_guid_t arts_edt_create_dep(arts_edt_t func_ptr, uint32_t paramc,
                                const uint64_t *paramv, uint32_t depc,
                                bool has_depv, const arts_hint_t *hint);

/**
 * @brief Create an EDT with a pre-reserved GUID and optional depv allocation.
 *
 * @param func_ptr Function to execute.
 * @param guid     Pre-reserved GUID (determines target node).
 * @param paramc   Number of static parameters.
 * @param paramv   Array of parameters.
 * @param depc     Number of dependencies.
 * @param has_depv If @c false, skip depv allocation.
 * @return The same @p guid, now associated with the EDT.
 */
arts_guid_t arts_edt_create_with_guid_dep(arts_edt_t func_ptr, arts_guid_t guid,
                                          uint32_t paramc,
                                          const uint64_t *paramv, uint32_t depc,
                                          bool has_depv);

/**
 * @brief Create an EDT in a specific epoch with optional depv allocation.
 *
 * @param func_ptr   Function to execute.
 * @param paramc     Number of static parameters.
 * @param paramv     Array of parameters.
 * @param depc       Number of dependencies.
 * @param epoch_guid Epoch GUID (must still be live).
 * @param has_depv   If @c false, skip depv allocation.
 * @param hint       Advisory metadata (route, profiling id). NULL = defaults.
 * @return GUID of the newly created EDT.
 */
arts_guid_t arts_edt_create_with_epoch_dep(
    arts_edt_t func_ptr, uint32_t paramc, const uint64_t *paramv, uint32_t depc,
    arts_guid_t epoch_guid, bool has_depv, const arts_hint_t *hint);

/**
 * @brief Destroy an EDT and remove its GUID from the routing table.
 *
 * EDTs are automatically destroyed after they finish running; call this only
 * to cancel an EDT that has not yet fired.
 *
 * @param guid GUID of the EDT to destroy.
 */
void arts_edt_destroy(arts_guid_t guid);

/**
 * @warning DEPRECATED — arts_signal_edt breaks DAG analyzability.
 *
 * arts_signal_edt is an imperative "push" operation that delivers data
 * to an EDT from within another EDT's body.  This makes the dependency
 * graph invisible to the runtime (edges are hidden inside EDT code),
 * preventing static analysis, scheduling optimization, and deadlock
 * detection.
 *
 * Use arts_add_dependence(source, destination, slot, mode) instead.
 * It is a declarative "this EDT depends on this data" statement that
 * builds a visible, analyzable DAG.
 *
 * arts_signal_edt remains available for internal runtime use and
 * backward compatibility with CARTS-generated code, but new user code
 * should exclusively use arts_add_dependence.
 */

/**
 * @brief Signal an EDT dependency slot with a DataBlock GUID.
 * @deprecated Use arts_add_dependence() instead.
 *
 * When all @c depc slots are satisfied the EDT is scheduled.  The
 * @c depv[slot] entry is filled with the GUID and a pointer to the DB data.
 *
 * @param edt_guid  GUID of the target EDT.
 * @param slot      Dependency slot index.
 * @param data_guid GUID of the DataBlock to deliver.
 * @param mode      Access mode (@c DB_MODE_RO or @c DB_MODE_EW).
 */
void arts_signal_edt(arts_guid_t edt_guid, uint32_t slot, arts_guid_t data_guid,
                     arts_db_access_mode_t mode);

/**
 * @brief Signal an EDT dependency slot with a plain 64-bit value.
 *
 * The value is stored in the @c guid field of @c depv[slot].
 *
 * @param edt_guid GUID of the target EDT.
 * @param slot     Dependency slot index.
 * @param value    Value to deliver.
 */
void arts_signal_edt_value(arts_guid_t edt_guid, uint32_t slot, uint64_t value);

/**
 * @brief Signal an EDT dependency slot with a data copy.
 *
 * @p size bytes starting at @p ptr are copied into the EDT's @c depv[slot].ptr.
 * The copy is freed automatically after the EDT runs.
 *
 * @param edt_guid GUID of the target EDT.
 * @param slot     Dependency slot index.
 * @param ptr      Source data pointer.
 * @param size     Number of bytes to copy.
 */
void arts_signal_edt_ptr(arts_guid_t edt_guid, uint32_t slot, void *ptr,
                         unsigned int size);

/**
 * @brief Signal an EDT slot with a pointer while preserving the DB GUID.
 *
 * Used for byte-slice dependencies where both the pointer
 * (@c db_ptr + byte_offset) and the original DB GUID are needed.
 *
 * @param edt_guid GUID of the target EDT.
 * @param slot     Dependency slot index.
 * @param db_guid  Original DataBlock GUID (stored in @c depv[slot].guid).
 * @param ptr      Computed pointer (stored in @c depv[slot].ptr).
 * @param size     Number of bytes.
 */
void arts_signal_edt_ptr_with_guid(arts_guid_t edt_guid, uint32_t slot,
                                   arts_guid_t db_guid, void *ptr,
                                   unsigned int size);

/**
 * @brief Signal an EDT slot as satisfied without any data.
 * @deprecated Use arts_add_dependence(NULL_GUID, dest, slot, mode) instead.
 *
 * Used for boundary conditions where a dependency should be skipped
 * (e.g. stencil edges).  The slot is marked @c ARTS_NULL.
 *
 * @param edt_guid GUID of the target EDT.
 * @param slot     Dependency slot index.
 */
void arts_signal_edt_null(arts_guid_t edt_guid, uint32_t slot);

/** @} */ /* end edt */

/* ========================================================================= */
/** @defgroup buffer Buffers
 *  Node-local buffers accessible by GUID.
 *  @{ */

/**
 * @brief Allocate a node-local buffer accessible by GUID.
 *
 * The buffer lives on the allocating node.  Remote nodes may write to it
 * via arts_set_buffer().  Each access decrements @p uses; when it reaches
 * zero the routing-table entry is freed.
 *
 * @param[out] buffer     Pointer to the allocated buffer.
 * @param      size       Buffer size in bytes.
 * @param      uses       Number of accesses before auto-free.
 * @param      epoch_guid Epoch for termination-detection accounting.
 * @return GUID of the buffer.
 * @see arts_set_buffer, arts_get_buffer
 */
arts_guid_t arts_allocate_local_buffer(void **buffer, unsigned int size,
                                       unsigned int uses,
                                       arts_guid_t epoch_guid);

/**
 * @brief Write data into a buffer identified by @p buffer_guid.
 *
 * @param buffer_guid Target buffer GUID.
 * @param buffer      Source data to copy.
 * @param size        Number of bytes.
 * @return Pointer to the buffer's internal storage.
 */
void *arts_set_buffer(arts_guid_t buffer_guid, void *buffer, unsigned int size);

/**
 * @brief Read the buffer identified by @p buffer_guid and decrement its use
 * count.
 *
 * If the use count reaches zero the routing-table entry is freed.
 * Only available on the node that allocated the buffer.
 *
 * @param buffer_guid Buffer GUID.
 * @return Pointer to the buffer data.
 */
void *arts_get_buffer(arts_guid_t buffer_guid);

/**
 * @brief Block until the buffer identified by @p buffer_guid is available.
 *
 * @param buffer_guid Buffer GUID.
 * @return Pointer to the buffer data.
 */
void *arts_block_for_buffer(arts_guid_t buffer_guid);

/** @} */ /* end buffer */

/* ========================================================================= */
/** @defgroup event Events
 *  Latch-based event synchronization primitives.
 *  @{ */

/**
 * @brief Create an event on node @p route.
 *
 * The @p type parameter selects the event behavior:
 *
 * - @c ARTS_EVENT_LATCH — N-counter, auto-destroy on fire.
 * - @c ARTS_EVENT_ONCE — latch=1, auto-destroy on fire.
 * - @c ARTS_EVENT_STICKY — latch=1, persists, error on re-satisfy.
 * - @c ARTS_EVENT_IDEM — latch=1, persists, ignore re-satisfy.
 * - @c ARTS_EVENT_COUNTED — N-counter, auto-destroy, rejects INCR_SLOT.
 * - @c ARTS_EVENT_CHANNEL — re-armable, version-based, DB-coupled.
 *
 * @param route       Target node rank (or @c ARTS_HINT_CURRENT_NODE).
 * @param type        Event behavior type.
 * @param latch_count Initial counter value.  Used for LATCH and COUNTED.
 *                    Ignored for ONCE/STICKY/IDEM (forced to 1) and
 *                    CHANNEL (forced to 0).
 * @param data_guid   DataBlock GUID for CHANNEL events.  Ignored for
 *                    all other types.  Pass @c NULL_GUID when not needed.
 * @return GUID of the new event.
 * @see arts_event_satisfy_slot, arts_add_dependence, arts_event_types_t
 */
arts_guid_t arts_event_create(unsigned int route, arts_event_types_t type,
                              unsigned int latch_count, arts_guid_t data_guid);

/**
 * @brief Create an event with a pre-reserved @p guid.
 *
 * The home node is determined by the rank encoded in @p guid.
 * See arts_event_create() for @p type, @p latch_count, @p data_guid.
 *
 * @param guid        Pre-reserved GUID (determines home node).
 * @param type        Event behavior type.
 * @param latch_count Initial counter value (see arts_event_create()).
 * @param data_guid   DataBlock GUID for CHANNEL events (see
 *                    arts_event_create()).
 * @return The same @p guid on success, @c NULL_GUID on failure.
 */
arts_guid_t arts_event_create_with_guid(arts_guid_t guid,
                                        arts_event_types_t type,
                                        unsigned int latch_count,
                                        arts_guid_t data_guid);

/**
 * @brief Check whether the event has already fired.
 *
 * @param event Event GUID.
 * @return @c true if fired, @c false otherwise.
 */
bool arts_is_event_fired(arts_guid_t event);

/**
 * @brief Destroy a local event.
 *
 * @param guid Event GUID.
 */
void arts_event_destroy(arts_guid_t guid);

/**
 * @brief Signal a latch-event slot.
 *
 * Use @c ARTS_EVENT_LATCH_INCR_SLOT to increment and
 * @c ARTS_EVENT_LATCH_DECR_SLOT to decrement the counter.  When the counter
 * reaches zero the event fires.
 *
 * @param event_guid Event GUID.
 * @param data_guid  DataBlock to broadcast when the event fires.
 * @param slot       Slot type (see arts_latch_event_slot_t).
 * @see arts_latch_event_slot_t
 */
void arts_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                             uint32_t slot);

/**
 * @brief Wire a source (event or DB) to a destination (EDT or event).
 *
 * Two-message pattern: sets @p mode on the destination's dep slot first,
 * then registers as a dependent on the source event.  The fire loop
 * delivers data with @c DB_MODE_NULL so mode is preserved.
 *
 * Accepts @c NULL_GUID as @p source — signals the slot immediately with
 * no data.
 *
 * @param source      Source event or DB GUID (or @c NULL_GUID).
 * @param destination Destination EDT or event GUID.
 * @param slot        Dependency slot on the destination.
 * @param mode        Access mode for DB data (@c DB_MODE_RO, @c DB_MODE_EW,
 *                    etc.).
 */
void arts_add_dependence(arts_guid_t source, arts_guid_t destination,
                         uint32_t slot, arts_db_access_mode_t mode);

/**
 * @brief Wire a DB source to a destination with byte-offset slicing.
 *
 * Same as @c arts_add_dependence, but delivers only a byte slice of the DB.
 *
 * @param source      Source DB or event GUID.
 * @param destination Destination EDT or event GUID.
 * @param slot        Dependency slot on the destination.
 * @param mode        Access mode.
 * @param byte_offset Byte offset into the DB payload.
 * @param len         Length in bytes of the slice.
 */
void arts_add_dependence_at(arts_guid_t source, arts_guid_t destination,
                            uint32_t slot, arts_db_access_mode_t mode,
                            uint64_t byte_offset, uint64_t len);

/**
 * @brief Register a callback to execute when @p source fires.
 *
 * Unlike an EDT, the callback runs inline on the thread that decrements the
 * counter to zero.
 *
 * @param source     Event GUID.
 * @param callback_t Function to invoke.
 */
void arts_add_local_event_callback(arts_guid_t source,
                                   event_callback_t callback_t);

/** @} */ /* end event */

/* ========================================================================= */
/** @defgroup db DataBlocks (DB)
 *  Fixed-size data objects shared between tasks via the CDAG memory model.
 *  @{ */

/**
 * @brief Create a DataBlock of @p len bytes.
 *
 * A DataBlock (DB) is the main memory abstraction used in ARTS to share data
 * between tasks.  Access mode is specified at dependency time via
 * arts_record_dep() or arts_signal_edt(), not at creation.
 *
 * @param[out] addr    Receives a pointer to the DB payload.  Set to @c NULL
 *                     when @c hint->route targets a remote node.
 * @param      len     Length in bytes.
 * @param      db_type Storage/coherence class (DEFAULT, LOCAL, GPU, LC).
 * @param      hint    Advisory metadata.  @c hint->route selects the target
 *                     node; NULL or ARTS_HINT_CURRENT_NODE = current node.
 * @return GUID of the created DB.
 * @see arts_signal_edt, arts_db_destroy
 */
arts_guid_t arts_db_create(void **addr, uint64_t len, arts_db_types_t db_type,
                           const arts_hint_t *hint);

/**
 * @brief Create a DataBlock with a pre-reserved @p guid.
 *
 * The route is encoded in the GUID.  If @p data is non-NULL it is copied
 * into the DB at creation time (avoids races with out-of-order EDTs).
 *
 * @param guid    Pre-reserved GUID (must be local).
 * @param len     Length in bytes.
 * @param db_type Storage/coherence class (DEFAULT, LOCAL, GPU, LC).
 * @param data    Optional source data to copy into the DB (NULL = uninit).
 * @param hint    Advisory metadata (profiling id). NULL = defaults.
 * @return Pointer to the DB payload.
 */
void *arts_db_create_with_guid(arts_guid_t guid, uint64_t len,
                               arts_db_types_t db_type, const void *data,
                               const arts_hint_t *hint);

/**
 * @brief Release the auto-acquired WRITE access for a DataBlock.
 *
 * When an EDT creates a local DB, the runtime automatically holds WRITE
 * access (OCR EW semantics).  Call this to release that access early —
 * before the EDT function returns — so that consumer EDTs waiting on
 * the DB can proceed.
 *
 * This is required when an EDT creates DBs and then blocks inside its
 * body (e.g. via arts_wait_on_handle), because the automatic release
 * in the EDT epilogue cannot run until the function returns.
 *
 * Calling this on a DB that was not auto-acquired (or was already
 * released) is a no-op.
 *
 * @param guid GUID of the DataBlock to release.
 */
void arts_db_release(arts_guid_t guid);

/**
 * @brief Temporarily release frontier locks for all DBs held by the current
 * EDT.
 *
 * Used inside arts_wait_on_handle to unblock consumer EDTs while the
 * creator EDT blocks on an epoch.  Only touches frontier locks -- does not
 * return route table entries or null tracking state.
 */
void arts_wait_release_dbs(void);

/**
 * @brief Re-acquire frontier locks for all DBs held by the current EDT.
 *
 * Used inside arts_wait_on_handle after the epoch completes.  Re-sets
 * WRITE_SET on each DB's current frontier head.
 */
void arts_wait_reacquire_dbs(void);

/**
 * @brief Destroy all copies of a DataBlock system-wide.
 *
 * If the calling EDT has acquired this DB (via creation auto-acquire or
 * dependency), the acquire is implicitly released first.  The route-table
 * entry is then marked for deletion; actual deallocation is deferred until
 * all outstanding route-table references are returned.
 *
 * @param guid DataBlock GUID.
 */
void arts_db_destroy(arts_guid_t guid);

/**
 * @brief Write data into a DataBlock on its home node and signal an EDT.
 *
 * @param ptr      Source data.
 * @param edt_guid EDT to signal upon completion.
 * @param db_guid  Target DataBlock.
 * @param slot     EDT dependency slot to satisfy.
 * @param offset   Byte offset within the DB.
 * @param len      Number of bytes to write.
 */
void arts_put_in_db(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                    unsigned int slot, unsigned int offset, unsigned int len);

/**
 * @brief Write data into a DataBlock on a specific node @p rank.
 *
 * @param ptr      Source data.
 * @param edt_guid EDT to signal upon completion.
 * @param db_guid  Target DataBlock.
 * @param slot     EDT dependency slot to satisfy.
 * @param offset   Byte offset within the DB.
 * @param len      Number of bytes to write.
 * @param rank     Node rank where the write is applied.
 */
void arts_put_in_db_at(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                       unsigned int slot, unsigned int offset, unsigned int len,
                       unsigned int rank);

/**
 * @brief Write data into a DataBlock within a specific epoch.
 *
 * @param ptr        Source data.
 * @param epoch_guid Epoch to associate the put with.
 * @param db_guid    Target DataBlock.
 * @param offset     Byte offset within the DB.
 * @param len        Number of bytes to write.
 */
void arts_put_in_db_epoch(void *ptr, arts_guid_t epoch_guid,
                          arts_guid_t db_guid, unsigned int offset,
                          unsigned int len);

/**
 * @brief Read data from a DataBlock on its home node.
 *
 * A copy of @p len bytes at @p offset is delivered to @p edt_guid via
 * arts_signal_edt_ptr().
 *
 * @param edt_guid Destination EDT.
 * @param db_guid  Source DataBlock.
 * @param slot     EDT dependency slot.
 * @param offset   Byte offset within the DB.
 * @param len      Number of bytes to read.
 */
void arts_get_from_db(arts_guid_t edt_guid, arts_guid_t db_guid,
                      unsigned int slot, unsigned int offset, unsigned int len);

/**
 * @brief Read data from a DataBlock on a specific node @p rank.
 *
 * @param edt_guid Destination EDT.
 * @param db_guid  Source DataBlock.
 * @param slot     EDT dependency slot.
 * @param offset   Byte offset within the DB.
 * @param len      Number of bytes to read.
 * @param rank     Node rank to read from.
 */
void arts_get_from_db_at(arts_guid_t edt_guid, arts_guid_t db_guid,
                         unsigned int slot, unsigned int offset,
                         unsigned int len, unsigned int rank);

/** @brief Rename a DataBlock, returning a new GUID pointing to the same data.
 */
arts_guid_t arts_db_rename(arts_guid_t guid);

/** @brief Rename a DataBlock to @p new_guid from @p old_guid. */
bool arts_db_rename_with_guid(arts_guid_t new_guid, arts_guid_t old_guid);

/** @brief Copy a DataBlock to a new GUID with a different type / access mode.
 */
arts_guid_t arts_db_copy_to_new_type(arts_guid_t old_guid,
                                     arts_db_types_t new_type);

/** @} */ /* end db */

/* ========================================================================= */
/** @defgroup epoch Epochs / Termination Detection
 *  Nested termination detection scopes.
 *  @{ */

/**
 * @brief Return the GUID of the currently active epoch.
 *
 * @return Current epoch GUID.
 */
arts_guid_t arts_get_current_epoch_guid();

/**
 * @brief Assign an EDT to a specific epoch.
 *
 * The caller must ensure the EDT has not yet run and the epoch is still live.
 *
 * @param edt_guid   EDT to assign.
 * @param epoch_guid Target epoch.
 */
void arts_add_edt_to_epoch(arts_guid_t edt_guid, arts_guid_t epoch_guid);

/**
 * @brief Create and immediately start a new epoch.
 *
 * Any EDTs created by the currently running EDT will belong to this epoch.
 * When the epoch completes, @p finish_edt_guid is signaled at @p slot with
 * the number of EDTs, buffer ops, get/puts, etc. executed.
 *
 * @param finish_edt_guid EDT to signal when the epoch finishes.
 * @param slot            Dependency slot to fill with the epoch summary.
 * @return GUID of the new epoch.
 * @see arts_wait_on_handle
 */
arts_guid_t arts_initialize_and_start_epoch(arts_guid_t finish_edt_guid,
                                            unsigned int slot);

/**
 * @brief Create an epoch without starting it.
 *
 * Use arts_start_epoch() to begin the epoch later.
 *
 * @param rank            Source node rank.
 * @param finish_edt_guid EDT to signal when the epoch finishes.
 * @param slot            Dependency slot for the epoch summary.
 * @return GUID of the new epoch.
 * @see arts_start_epoch
 */
arts_guid_t arts_initialize_epoch(unsigned int rank,
                                  arts_guid_t finish_edt_guid,
                                  unsigned int slot);

/**
 * @brief Start an epoch previously created with arts_initialize_epoch().
 *
 * @param epoch_guid Epoch GUID.
 */
void arts_start_epoch(arts_guid_t epoch_guid);

/**
 * @brief Block until @p epoch_guid finishes.
 *
 * The calling thread runs another scheduling round while waiting.
 * Only valid from the EDT that created the epoch.
 *
 * @param epoch_guid Epoch to wait for.
 * @return @c true on success.
 */
bool arts_wait_on_handle(arts_guid_t epoch_guid);

/**
 * @brief Yield the current EDT and run another scheduling round.
 */
void arts_yield();

/** @} */ /* end epoch */

/* ========================================================================= */
/** @defgroup util Utility Functions
 *  Query runtime state and miscellaneous helpers.
 *  @{ */

/** @brief Return the GUID of the currently executing EDT. */
arts_guid_t arts_get_current_guid();

/** @brief Return the rank of this node. */
unsigned int arts_get_current_node();

/** @brief Return the total number of nodes. */
unsigned int arts_get_total_nodes();

/** @brief Return the worker-thread id on this node. */
unsigned int arts_get_current_worker();

/**
 * @brief Return the total number of worker threads per node.
 *
 * Does not include network send/receive threads.
 */
unsigned int arts_get_total_workers();

/**
 * @brief Return the NUMA domain id of the current thread.
 * @return NUMA domain index. Returns 0 when HWLOC is not available.
 */
unsigned int arts_get_current_numa_domain();

/**
 * @brief Return the total number of NUMA domains.
 * @return Number of NUMA domains. Returns 1 when HWLOC is not available.
 */
unsigned int arts_get_total_numa_domains();

/** @brief Return the number of GPUs per node. */
unsigned int arts_get_total_gpus();

/** @brief Return a monotonic timestamp in nanoseconds. */
uint64_t arts_get_time_stamp();

/** @brief Return a thread-safe pseudo-random number. */
uint64_t arts_thread_safe_random();

/** @} */ /* end util */

#ifdef __cplusplus
}
#endif
#endif
