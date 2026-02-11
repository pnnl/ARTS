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

/** Ticket for context-switch wake-up signaling.
 *  @see arts_get_context_ticket, arts_signal_context */
typedef uint64_t arts_ticket_t;

/** @} */ /* end core_types */

/* ========================================================================= */
/** @defgroup type_enum Type / Access-Mode Enumeration
 *  Every GUID carries a type tag from this enum.
 *  @{ */

/**
 * @brief Type tag and DataBlock access-mode enumeration.
 *
 * Values below @c ARTS_DB_READ identify runtime object kinds (EDT, event,
 * epoch, …).  Values from @c ARTS_DB_READ through @c ARTS_DB_LC are
 * DataBlock access modes that control coherence, caching, and lifetime.
 */
typedef enum {
  ARTS_NULL = 0,         /**< Empty / untyped placeholder. */
  ARTS_EDT,              /**< Event-Driven Task (CPU). */
  ARTS_GPU_EDT,          /**< Event-Driven Task (GPU). */
  ARTS_EVENT,            /**< Latch-based synchronization event. */
  ARTS_PERSISTENT_EVENT, /**< Re-armable persistent event. */
  ARTS_EPOCH,            /**< Termination-detection epoch. */
  ARTS_CALLBACK,         /**< Inline event callback. */
  ARTS_BUFFER,           /**< Node-local buffer accessible by GUID. */

  /* ── DataBlock types ──────────────────────────────────────────────────── */

  ARTS_DB,       /**< Generic DataBlock (mode-less). */

  /* ── DataBlock access modes (used at dependency time) ─────────────────── */

  ARTS_DB_READ,
  ARTS_DB_WRITE,
  ARTS_DB_PIN,
  ARTS_DB_ONCE,
  ARTS_DB_ONCE_LOCAL,
  ARTS_DB_GPU_READ,  /**< GPU read-only DataBlock. */
  ARTS_DB_GPU_WRITE, /**< GPU exclusive-write DataBlock. */
  ARTS_DB_LC,        /**< Locality-class DataBlock. */

  /* ── Pseudo-types (not valid for allocation) ─────────────────────────── */

  ARTS_LAST_TYPE,     /**< Sentinel — first invalid type value. */
  ARTS_SINGLE_VALUE,  /**< Marker: dependency carries a uint64 value. */
  ARTS_PTR,           /**< Marker: dependency carries a pointer copy. */
  ARTS_DB_LC_SYNC,    /**< Locality-class with synchronous copy. */
  ARTS_DB_LC_NO_COPY, /**< Locality-class without data copy. */
  ARTS_DB_GPU_MEMSET  /**< GPU memset operation pseudo-type. */
} arts_type_t;

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
 * Access mode (READ/WRITE) is specified at EDT creation or signal time,
 * stored internally, and invisible to user code.
 */
typedef struct {
  arts_guid_t guid; /**< GUID of the DataBlock (or encoded value). */
  void *ptr;        /**< Pointer to the DataBlock payload. */
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

/**
 * @brief Handler function for arts_remote_send().
 */
typedef void (*send_handler_t)(void *args);

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
extern void arts_main_edt(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]);

/** @} */ /* end user_callbacks */

/* ========================================================================= */
/** @defgroup event_slots Event Slot Types
 *  @{ */

/** Slot constants for latch-event signaling. */
typedef enum {
  ARTS_EVENT_LATCH_DECR_SLOT = 0, /**< Decrement the latch counter. */
  ARTS_EVENT_LATCH_INCR_SLOT = 1, /**< Increment the latch counter. */
  ARTS_EVENT_UPDATE = 2           /**< Update data (persistent events only). */
} arts_latch_event_slot_t;

/** @} */ /* end event_slots */

/* ========================================================================= */
/** @defgroup range_array GUID Range and Array DB
 *  @{ */

/** Iterator over a contiguous range of GUIDs. */
struct arts_guid_range_s {
  unsigned int size;      /**< Total number of GUIDs in the range. */
  unsigned int index;     /**< Current iterator position. */
  arts_guid_t start_guid; /**< First GUID in the range. */
};
typedef struct arts_guid_range_s arts_guid_range_t;

/** Distributed array DataBlock spanning multiple nodes. */
struct arts_array_db_s {
  unsigned int element_size;       /**< Size of each element in bytes. */
  unsigned int elements_per_block; /**< Elements per node-local block. */
  unsigned int num_blocks;         /**< Total number of blocks. */
  char head[];                     /**< Flexible array of block GUIDs. */
};
typedef struct arts_array_db_s arts_array_db_t;

/** @} */ /* end range_array */

/**
 * @brief Thread-safe printf that serializes output across ARTS workers.
 */
void arts_printf(const char *format, ...);

/* ========================================================================= */
/** @defgroup runtime Runtime Lifecycle
 *  @{ */

/**
 * @brief Entry point to the ARTS runtime.
 *
 * Reads @c arts.cfg, initializes threading and networking, schedules
 * arts_main_edt() on rank 0 (if defined), and blocks until
 * arts_shutdown() is called.
 *
 * @param argc Argument count from main().
 * @param argv Argument vector from main().
 * @return 0 on success.
 * @see arts_shutdown
 */
int arts_rt(int argc, char **argv);

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
 * @param type  Type tag for the GUID (e.g. @c ARTS_EDT, @c ARTS_DB_READ).
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
 * @brief Allocate a contiguous range of @p size GUIDs on node @p route.
 *
 * Because GUIDs are formed by a bitfield with several fields, their raw
 * integer values may not be consecutive.  A GUID range provides an
 * iterator-style interface over a logically contiguous block.
 *
 * @param type  Type tag for every GUID in the range.
 * @param size  Number of GUIDs to allocate.
 * @param route Target node rank.
 * @return Pointer to a new GUID range, or @c NULL on failure.
 * @see arts_guid_range_get, arts_guid_range_next
 */
arts_guid_range_t *arts_guid_range_create(arts_type_t type, unsigned int size,
                                            unsigned int route);

/**
 * @brief Get the GUID at @p index within @p range.
 *
 * @param range Pointer to a GUID range.
 * @param index Zero-based offset from the start of the range.
 * @return The GUID at the requested position.
 */
arts_guid_t arts_guid_range_get(arts_guid_range_t *range, unsigned int index);

/**
 * @brief Advance the range iterator and return the next GUID.
 *
 * @note Not thread-safe.
 *
 * @param range Pointer to a GUID range.
 * @return The next GUID, or @c NULL_GUID if the end has been reached.
 */
arts_guid_t arts_guid_range_next(arts_guid_range_t *range);

/**
 * @brief Check whether the range iterator has more GUIDs.
 *
 * @note Not thread-safe.
 *
 * @param range Pointer to a GUID range.
 * @return @c true if GUIDs remain, @c false otherwise.
 */
bool arts_guid_range_has_next(arts_guid_range_t *range);

/**
 * @brief Reset the range iterator to the beginning.
 *
 * @note Not thread-safe.
 *
 * @param range Pointer to a GUID range.
 */
void arts_guid_range_reset_iter(arts_guid_range_t *range);

/**
 * @brief Reserve @p size GUIDs distributed round-robin across all nodes.
 *
 * @param size Number of GUIDs to allocate.
 * @param type Type tag for every GUID.
 * @return Array of GUIDs (caller must free).
 */
arts_guid_t *arts_guid_reserve_round_robin(unsigned int size,
                                            arts_type_t type);

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
arts_guid_t arts_edt_create_with_epoch_dep(arts_edt_t func_ptr,
                                           uint32_t paramc,
                                           const uint64_t *paramv,
                                           uint32_t depc,
                                           arts_guid_t epoch_guid,
                                           bool has_depv,
                                           const arts_hint_t *hint);

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
 * @brief Signal an EDT dependency slot with a DataBlock GUID.
 *
 * When all @c depc slots are satisfied the EDT is scheduled.  The
 * @c depv[slot] entry is filled with the GUID and a pointer to the DB data.
 * The acquire mode is determined by the type field of @p data_guid.
 *
 * @param edt_guid  GUID of the target EDT.
 * @param slot      Dependency slot index.
 * @param data_guid GUID of the DataBlock to deliver.
 */
void arts_signal_edt(arts_guid_t edt_guid, uint32_t slot,
                     arts_guid_t data_guid);

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
 * @brief Create a latch event on node @p route.
 *
 * A latch event maintains a counter that can be incremented/decremented via
 * arts_event_satisfy_slot().  When it reaches zero the event fires,
 * broadcasting its data to all registered dependents.
 *
 * @param route       Target node rank.
 * @param latch_count Initial counter value.
 * @return GUID of the new event.
 * @see arts_event_satisfy_slot, arts_add_dependence
 */
arts_guid_t arts_event_create(unsigned int route, unsigned int latch_count);

/**
 * @brief Create a latch event with a pre-reserved @p guid.
 *
 * @param guid        Pre-reserved GUID (determines home node).
 * @param latch_count Initial counter value.
 * @return The same @p guid, now associated with the event.
 */
arts_guid_t arts_event_create_with_guid(arts_guid_t guid,
                                        unsigned int latch_count);

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
 * @brief Wire an event to an EDT or another event.
 *
 * When @p source fires it will signal @p destination at @p slot.  If the
 * source has already fired the signal propagates immediately.
 *
 * @param source      Source event GUID.
 * @param destination Destination EDT or event GUID.
 * @param slot        Dependency slot on the destination.
 */
void arts_add_dependence(arts_guid_t source, arts_guid_t destination,
                         uint32_t slot);

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
/** @defgroup persistent_event Persistent Events
 *  Reusable synchronization points that can fire multiple times.
 *  @{ */

/**
 * @brief Create a persistent event on node @p route.
 *
 * A persistent event stays alive after firing and can be triggered again.
 *
 * @param route       Target node rank.
 * @param latch_count Initial counter value.
 * @param data_guid   DataBlock GUID to deliver when the event fires.
 * @return GUID of the new persistent event.
 */
arts_guid_t arts_persistent_event_create(unsigned int route,
                                         unsigned int latch_count,
                                         arts_guid_t data_guid);

/**
 * @brief Satisfy a persistent event.
 *
 * @param event_guid Persistent event GUID.
 * @param action     Slot / action type.
 * @param lock       Whether to acquire the event lock.
 */
void arts_persistent_event_satisfy(arts_guid_t event_guid, uint32_t action,
                                   bool lock);

/**
 * @brief Increment the latch count of a persistent event.
 *
 * Indicates that a new dependency has been added, allowing the event to fire
 * again after the counter drops back to zero.
 *
 * @param event_guid Persistent event GUID.
 */
void arts_persistent_event_increment_latch(arts_guid_t event_guid);

/**
 * @brief Decrement the latch count of a persistent event.
 *
 * If the counter reaches zero the event fires, signaling all dependents.
 *
 * @param event_guid Persistent event GUID.
 */
void arts_persistent_event_decrement_latch(arts_guid_t event_guid);

/**
 * @brief Add a dependence from a persistent event to an EDT slot.
 *
 * If the event's latch count is already zero the EDT is signaled immediately.
 *
 * @param event_source Source persistent event GUID.
 * @param edt_dest     Destination EDT GUID.
 * @param edt_slot     Dependency slot on the EDT.
 */
void arts_add_dependence_to_persistent_event(arts_guid_t event_source,
                                             arts_guid_t edt_dest,
                                             uint32_t edt_slot);

/**
 * @brief Add a dependence with a compiler-inferred acquire mode.
 *
 * @param event_source Source persistent event GUID.
 * @param edt_dest     Destination EDT GUID.
 * @param edt_slot     Dependency slot on the EDT.
 * @param mode Acquire mode override.
 */
void arts_add_dependence_to_persistent_event_with_mode(
    arts_guid_t event_source, arts_guid_t edt_dest, uint32_t edt_slot,
    arts_type_t mode);

/**
 * @brief Add a dependence with acquire mode override and diff tracking.
 *
 * @param event_source Source persistent event GUID.
 * @param edt_dest     Destination EDT GUID.
 * @param edt_slot     Dependency slot on the EDT.
 * @param mode Acquire mode override.
 */
void arts_add_dependence_to_persistent_event_with_mode_and_diff(
    arts_guid_t event_source, arts_guid_t edt_dest, uint32_t edt_slot,
    arts_type_t mode);

/**
 * @brief Add a dependence with byte offset for slice-based signaling.
 *
 * When @p byte_offset > 0 or @p len > 0, the persistent event will signal
 * with a pointer to (@c db_ptr + @p byte_offset) while preserving the DB GUID.
 *
 * @param event_source Source persistent event GUID.
 * @param edt_dest     Destination EDT GUID.
 * @param edt_slot     Dependency slot on the EDT.
 * @param mode Acquire mode.
 * @param byte_offset  Byte offset into the DataBlock.
 * @param len          Slice length in bytes.
 */
void arts_add_dependence_to_persistent_event_with_byte_offset(
    arts_guid_t event_source, arts_guid_t edt_dest, uint32_t edt_slot,
    arts_type_t mode, uint64_t byte_offset, uint64_t len);

/** @} */ /* end persistent_event */

/* ========================================================================= */
/** @defgroup db DataBlocks (DB)
 *  Fixed-size data objects shared between tasks via the CDAG memory model.
 *  @{ */

/**
 * @brief Create a local DataBlock of @p len bytes.
 *
 * A DataBlock (DB) is the main memory abstraction used in ARTS to share data
 * between tasks.  Access mode is specified at dependency time via
 * arts_record_dep() or arts_signal_edt(), not at creation.
 *
 * @param[out] addr Receives a pointer to the DB payload.
 * @param      len  Length in bytes.
 * @param      hint Advisory metadata (profiling id). NULL = defaults.
 * @return GUID of the created DB.
 * @see arts_signal_edt, arts_db_destroy
 */
arts_guid_t arts_db_create(void **addr, uint64_t len, const arts_hint_t *hint);

/**
 * @brief Create a DataBlock with a pre-reserved @p guid.
 *
 * The type and route are encoded in the GUID.
 *
 * @param guid Pre-reserved GUID (must be local).
 * @param len  Length in bytes.
 * @param hint Advisory metadata (profiling id). NULL = defaults.
 * @return Pointer to the DB payload.
 */
void *arts_db_create_with_guid(arts_guid_t guid, uint64_t len,
                               const arts_hint_t *hint);

/**
 * @brief Create a DataBlock with a pre-reserved @p guid and initial @p data.
 *
 * The data is copied into the DB at creation time.  This avoids a race
 * between user writes and out-of-order EDT acquisitions.
 *
 * @param guid Pre-reserved GUID (must be local).
 * @param data Source data to copy into the DB.
 * @param len  Length in bytes.
 * @return Pointer to the DB payload.
 */
void *arts_db_create_with_guid_and_data(arts_guid_t guid, void *data,
                                        uint64_t len);

/**
 * @brief Create an uninitialized DataBlock on remote node @p route.
 *
 * @param route Target node rank.
 * @param len   Length in bytes.
 * @return GUID of the created DB.
 */
arts_guid_t arts_db_create_remote(unsigned int route, uint64_t len);

/**
 * @brief Move a DataBlock to remote node @p rank.
 *
 * @warning The GUID does not change, so remote lookups still go to the
 *          original home node.  Local access from @p rank will succeed.
 *
 * @param db_guid GUID of the DataBlock to move.
 * @param rank    Destination node rank.
 */
void arts_db_move(arts_guid_t db_guid, unsigned int rank);

/**
 * @brief Destroy all copies of a DataBlock system-wide.
 *
 * @param guid DataBlock GUID.
 */
void arts_db_destroy(arts_guid_t guid);

/**
 * @brief Destroy the local copy of a DataBlock.
 *
 * If @p remote is @c true and the DB is not local, the request is forwarded
 * to the home node.
 *
 * @param guid   DataBlock GUID.
 * @param remote Whether to forward to the home node if not local.
 */
void arts_db_destroy_safe(arts_guid_t guid, bool remote);

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
                       unsigned int slot, unsigned int offset,
                       unsigned int len, unsigned int rank);

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
                      unsigned int slot, unsigned int offset,
                      unsigned int len);

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
                                     arts_type_t new_type);

/**
 * @brief Increment the latch on the persistent event associated with a DB.
 *
 * @param guid DataBlock GUID.
 */
void arts_db_increment_latch(arts_guid_t guid);

/**
 * @brief Decrement the latch on the persistent event associated with a DB.
 *
 * @param guid DataBlock GUID.
 */
void arts_db_decrement_latch(arts_guid_t guid);

/**
 * @brief Add a dependence from a DB's persistent event to an EDT slot.
 *
 * @param db_src   Source DataBlock GUID.
 * @param edt_dest Destination EDT GUID.
 * @param edt_slot EDT dependency slot.
 */
void arts_db_add_dependence(arts_guid_t db_src, arts_guid_t edt_dest,
                            uint32_t edt_slot);

/**
 * @brief Add a DB dependence with acquire mode override.
 *
 * @param db_src       Source DataBlock GUID.
 * @param edt_dest     Destination EDT GUID.
 * @param edt_slot     EDT dependency slot.
 * @param mode Acquire mode override.
 */
void arts_db_add_dependence_with_mode(arts_guid_t db_src, arts_guid_t edt_dest,
                                      uint32_t edt_slot,
                                      arts_type_t mode);

/**
 * @brief Add a DB dependence with acquire mode override and diff tracking.
 *
 * @param db_src       Source DataBlock GUID.
 * @param edt_dest     Destination EDT GUID.
 * @param edt_slot     EDT dependency slot.
 * @param mode Acquire mode override.
 */
void arts_db_add_dependence_with_mode_and_diff(arts_guid_t db_src,
                                               arts_guid_t edt_dest,
                                               uint32_t edt_slot,
                                               arts_type_t mode);

/**
 * @brief Record a dependency, auto-incrementing latch for @c ARTS_DB_WRITE.
 *
 * @param db_src       Source DataBlock GUID.
 * @param edt_dest     Destination EDT GUID.
 * @param edt_slot     EDT dependency slot.
 * @param mode Requested acquire mode.
 */
void arts_record_dep(arts_guid_t db_src, arts_guid_t edt_dest,
                     uint32_t edt_slot, arts_type_t mode);

/**
 * @brief Record a dependency at a byte offset within a DataBlock.
 *
 * When the DB is ready, the EDT receives a pointer to
 * (@c db_ptr + @p byte_offset) while the original DB GUID is preserved in
 * @c depv[slot].guid.  Used for stencil halo dependencies.
 *
 * @param db_src       Source DataBlock GUID.
 * @param edt_dest     Destination EDT GUID.
 * @param edt_slot     EDT dependency slot.
 * @param mode Requested acquire mode.
 * @param byte_offset  Byte offset into the DB.
 * @param len          Slice length in bytes.
 */
void arts_record_dep_at(arts_guid_t db_src, arts_guid_t edt_dest,
                        uint32_t edt_slot, arts_type_t mode,
                        uint64_t byte_offset, uint64_t len);

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

/**
 * @brief Obtain a context ticket for arts_signal_context().
 *
 * @return A new ticket value.
 * @see arts_context_switch, arts_signal_context
 */
arts_ticket_t arts_get_context_ticket();

/**
 * @brief Context-switch the current thread.
 *
 * Requires the @c tmt setting in @c arts.cfg.  The context sleeps until
 * @p wait_count signals are received via arts_signal_context().
 *
 * @param wait_count Number of signals required to wake up.
 * @return @c true on success.
 */
bool arts_context_switch(unsigned int wait_count);

/**
 * @brief Context-switch without blocking the current context.
 */
void arts_open_context_switch();

/** @brief Switch to the next available context. */
void arts_next_context();

/** @brief Return the id of the current context on this thread. */
unsigned int arts_get_context_id();

/**
 * @brief Wake a context that is sleeping from a context switch.
 *
 * @param ticket Ticket obtained from arts_get_context_ticket().
 * @return @c true if the signal was delivered.
 */
bool arts_signal_context(arts_ticket_t ticket);

/** @} */ /* end epoch */

/* ========================================================================= */
/** @defgroup arraydb Array DataBlocks
 *  Distributed arrays spanning all nodes.
 *  @{ */

/**
 * @brief Create a distributed array DB spread equally across all nodes.
 *
 * @param[out] addr         Receives the local portion pointer.
 * @param      element_size Size of each element in bytes.
 * @param      num_elements Total number of elements.
 * @return GUID for accessing the array DB.
 */
arts_guid_t arts_new_array_db(arts_array_db_t **addr, unsigned int element_size,
                              unsigned int num_elements);

/**
 * @brief Create a distributed array DB with a pre-reserved @p guid.
 *
 * The GUID can target any node but must be of type @c ARTS_DB_PIN.
 *
 * @param guid         Pre-reserved GUID.
 * @param element_size Size of each element in bytes.
 * @param num_elements Total number of elements.
 * @return Pointer to the local arts_array_db_t.
 */
arts_array_db_t *arts_new_array_db_with_guid(arts_guid_t guid,
                                             unsigned int element_size,
                                             unsigned int num_elements);

/**
 * @brief Create a node-local array DB (not shared across nodes).
 *
 * @param guid         Pre-reserved GUID.
 * @param element_size Size of each element in bytes.
 * @param num_elements Total number of elements.
 * @param data         Optional initial data (may be @c NULL).
 * @return Pointer to the local arts_array_db_t.
 */
arts_array_db_t *arts_new_local_array_db_with_guid(arts_guid_t guid,
                                                   unsigned int element_size,
                                                   unsigned int num_elements,
                                                   void *data);

/**
 * @brief Signal an EDT with all blocks of an array DB.
 *
 * @param array    Array DB.
 * @param edt_guid Target EDT.
 * @param slot     Starting dependency slot.
 */
void arts_signal_array_db(arts_array_db_t *array, arts_guid_t edt_guid,
                          unsigned int slot);

/**
 * @brief Read an element from an array DB at @p index.
 *
 * Delivered to @p edt_guid at @p slot via arts_signal_edt_ptr().
 *
 * @param edt_guid Destination EDT.
 * @param slot     Dependency slot.
 * @param array    Array DB.
 * @param index    Element index.
 */
void arts_get_from_array_db(arts_guid_t edt_guid, unsigned int slot,
                            arts_array_db_t *array, unsigned int index);

/**
 * @brief Write data into an array DB element and signal an EDT.
 *
 * The put falls within the current epoch.
 *
 * @param ptr      Source data.
 * @param edt_guid EDT to signal.
 * @param slot     Dependency slot.
 * @param array    Array DB.
 * @param index    Element index.
 */
void arts_put_in_array_db(void *ptr, arts_guid_t edt_guid, unsigned int slot,
                          arts_array_db_t *array, unsigned int index);

/**
 * @brief Launch an EDT for each element locally.
 *
 * Data is acquired read-only via arts_signal_edt_ptr().
 *
 * @param array    Array DB.
 * @param func_ptr Function to execute per element.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of parameters.
 */
void arts_for_each_in_array_db(arts_array_db_t *array, arts_edt_t func_ptr,
                               uint32_t paramc, const uint64_t *paramv);

/**
 * @brief Launch an EDT for each element across all nodes.
 *
 * Data is acquired read-only via arts_signal_edt_ptr().
 *
 * @param array    Array DB.
 * @param stride   Number of elements per EDT.
 * @param func_ptr Function to execute.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of parameters.
 */
void arts_for_each_in_array_db_at_data(arts_array_db_t *array,
                                       unsigned int stride, arts_edt_t func_ptr,
                                       uint32_t paramc, const uint64_t *paramv);

/**
 * @brief Gather all chunks of an array DB on one node and run an EDT.
 *
 * @param array    Array DB.
 * @param func_ptr Function to execute after gathering.
 * @param route    Node to gather on.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of parameters.
 * @param depc     Number of dependency slots (usually num_blocks).
 */
void arts_gather_array_db(arts_array_db_t *array, arts_edt_t func_ptr,
                          unsigned int route, uint32_t paramc,
                          const uint64_t *paramv, uint64_t depc);

/** @brief Gather array DB within a specific epoch. */
void arts_gather_array_db_epoch(arts_array_db_t *array, arts_edt_t func_ptr,
                                unsigned int route, uint32_t paramc,
                                const uint64_t *paramv, uint64_t depc,
                                arts_guid_t epoch_guid);

/** @brief Gather array DB chunks into an existing EDT. */
void arts_gather_array_db_in_edt(arts_array_db_t *array,
                                 arts_guid_t to_edt_guid, uint64_t slot_offset);

/**
 * @brief Atomic add on an array DB element.
 *
 * Signals @p edt_guid at @p slot upon completion.
 *
 * @param array    Array DB.
 * @param index    Element index.
 * @param to_add   Value to add.
 * @param edt_guid EDT to signal.
 * @param slot     Dependency slot.
 */
void arts_atomic_add_in_array_db(arts_array_db_t *array, unsigned int index,
                                 unsigned int to_add, arts_guid_t edt_guid,
                                 unsigned int slot);

/**
 * @brief Atomic compare-and-swap on an array DB element.
 *
 * Signals @p edt_guid at @p slot upon completion.
 *
 * @param array     Array DB.
 * @param index     Element index.
 * @param old_value Expected current value.
 * @param new_value Desired new value.
 * @param edt_guid  EDT to signal.
 * @param slot      Dependency slot.
 */
void arts_atomic_compare_and_swap_in_array_db(
    arts_array_db_t *array, unsigned int index, unsigned int old_value,
    unsigned int new_value, arts_guid_t edt_guid, unsigned int slot);

/** @} */ /* end arraydb */

/* ========================================================================= */
/** @defgroup util Utility Functions
 *  Query runtime state and miscellaneous helpers.
 *  @{ */

/** @brief Extract the GUID from an EDT dependency. */
inline arts_guid_t arts_get_guid_from_edt_dep(arts_edt_dep_t dep) {
  return dep.guid;
}

/** @brief Extract the data pointer from an EDT dependency. */
inline void *arts_get_ptr_from_edt_dep(arts_edt_dep_t dep) { return dep.ptr; }

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
 * @note Requires HWLOC.
 */
unsigned int arts_get_current_cluster();

/**
 * @brief Return the total number of NUMA domains.
 * @note Requires HWLOC.
 */
unsigned int arts_get_total_clusters();

/** @brief Return the number of GPUs per node. */
unsigned int arts_get_total_gpus();

/** @brief Return a monotonic timestamp in nanoseconds. */
uint64_t arts_get_time_stamp();

/** @brief Return a thread-safe pseudo-random number. */
uint64_t arts_thread_safe_random();

/**
 * @brief Send a function call to a specific node.
 *
 * If @p rank is the current node the function executes inline.  Otherwise
 * the arguments are serialized and sent over the network; the receiver
 * thread will invoke @p fun_ptr.
 *
 * @param rank    Target node rank.
 * @param fun_ptr Handler function.
 * @param args    Argument buffer.
 * @param size    Size of @p args in bytes.
 * @param free    Whether the runtime should free @p args after sending.
 */
void arts_remote_send(unsigned int rank, send_handler_t fun_ptr, void *args,
                      unsigned int size, bool free);

/** @} */ /* end util */

#ifdef __cplusplus
}
#endif
#endif
