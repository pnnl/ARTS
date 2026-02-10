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
#ifndef ARTS_RUNTIME_SYNC_RT_H
#define ARTS_RUNTIME_SYNC_RT_H
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file rt.h
 * @brief Core types, enumerations, and internal structures for the ARTS
 * runtime.
 *
 * This header defines the fundamental types used throughout ARTS: GUIDs,
 * type/access-mode enumerations, EDT and DataBlock structures, event
 * primitives, and termination-detection state.
 *
 * @note This is an internal header.  User code should include @c arts.h.
 */

#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>

#include "arts/arts_defs.h"
#include "arts/utils/link_list.h"

/* ========================================================================= */
/** @defgroup core_types Core Types
 *  Fundamental typedefs shared by the entire runtime.
 *  @{ */

/** Globally Unique Identifier — 64-bit bitfield encoding type, rank, and key.
 */
typedef intptr_t arts_guid_t;

/** Sentinel value representing an invalid or absent GUID. */
#define NULL_GUID ((arts_guid_t)0x0)

/** Opaque pointer type used in DataBlock creation variants. */
typedef uintptr_t arts_ptr_t;

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

  /* ── DataBlock access modes ──────────────────────────────────────────── */

  /** Write-once, read-many.
   *
   *  Create the DB in this mode and write data before signaling the GUID.
   *  The runtime aggregates requests and caches reads in the routing table. */
  ARTS_DB_READ,

  /** Exclusive write access.
   *
   *  Obtain by casting an @c ARTS_DB_READ DB to @c ARTS_DB_WRITE via
   *  arts_guid_cast() and signaling an EDT.
   *  @warning This mode is currently experimental. */
  ARTS_DB_WRITE,

  /** Pinned (node-local) mode — bypasses the CDAG memory model.
   *
   *  The DB is only accessible on the creating node.  Remote interaction
   *  is limited to explicit put/get operations. */
  ARTS_DB_PIN,

  /** Single-use mode — the DB is automatically freed after the first acquire.
   *
   *  Useful for one-shot data that is never reused. */
  ARTS_DB_ONCE,

  /** Local single-use mode — same as @c ARTS_DB_ONCE with the additional
   *  guarantee that the DB is co-located with the acquiring EDT
   *  (i.e. @c edt_guid and @c db_guid share the same route). */
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
/** @defgroup dep_types Dependency Types
 *  Structures and function-pointer types used to wire EDT dependencies.
 *  @{ */

/**
 * @brief Describes a single dependency slot delivered to an EDT.
 *
 * When an EDT fires, each satisfied dependency appears as an element of
 * the @c depv[] array passed to the EDT function.
 */
typedef struct {
  arts_guid_t guid;         /**< GUID of the DataBlock (or encoded value). */
  arts_type_t mode;         /**< Original type/mode of the DataBlock. */
  void *ptr;                /**< Pointer to the DataBlock payload. */
  arts_type_t acquire_mode; /**< Actual acquire mode used at delivery. */
} arts_edt_dep_t;

/**
 * @brief Function signature for Event-Driven Tasks (CPU and GPU).
 *
 * @param paramc Number of 64-bit static parameters.
 * @param paramv Array of @p paramc static parameter values.
 * @param depc   Number of dependency slots.
 * @param depv   Array of @p depc satisfied dependencies.
 */
typedef void (*arts_edt_t)(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                          arts_edt_dep_t depv[]);

/**
 * @brief Callback invoked inline when a latch event fires.
 *
 * @param data The dependency data from the event's satisfy call.
 */
typedef void (*event_callback_t)(arts_edt_dep_t data);

/**
 * @brief Handler function for arts_remote_send().
 *
 * @param args Pointer to the serialized argument buffer.
 */
typedef void (*send_handler_t)(void *args);

/** @} */ /* end dep_types */

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
/** @defgroup internal_structs Internal Runtime Structures
 *  These structures are exposed for layout visibility but are managed
 *  entirely by the runtime.  User code should not manipulate them directly.
 *  @{ */

/** Common header prepended to every runtime object (EDT, DB, event). */
struct arts_header_s {
  uint8_t type : 8;   /**< Type tag (@ref arts_type_t). */
  uint64_t size : 56; /**< Payload size in bytes. */
} __attribute__((aligned));

/** Internal DataBlock descriptor. */
struct arts_db_s {
  struct arts_header_s header;
  uint64_t arts_id;       /**< Compiler-assigned unique id (0 = unset). */
  arts_guid_t guid;       /**< GUID of this DataBlock. */
  arts_guid_t event_guid; /**< Associated persistent event GUID. */
  volatile unsigned int copyCount; /**< Number of outstanding copies. */
  volatile unsigned int reader;    /**< Active reader count. */
  volatile unsigned int writer;    /**< Active writer count. */
  volatile unsigned int version;   /**< Coherence version counter. */
  unsigned int time_stamp;         /**< Creation timestamp (relative). */
  void *db_list; /**< Node in the per-node DB tracking list. */
} __attribute__((aligned));

/** Internal EDT descriptor. */
struct arts_edt_s {
  struct arts_header_s header;
  uint64_t arts_id;          /**< Compiler-assigned unique id (0 = unset). */
  arts_edt_t func_ptr;       /**< User function to execute. */
  uint32_t paramc;           /**< Number of static parameters. */
  uint32_t depc;             /**< Number of dependency slots. */
  arts_guid_t current_edt;   /**< GUID of this EDT. */
  arts_guid_t output_buffer; /**< Optional output buffer GUID. */
  arts_guid_t epoch_guid;    /**< Enclosing epoch GUID. */
  unsigned int cluster;      /**< NUMA cluster assignment. */
  unsigned int node;         /**< Target node rank. */
  volatile unsigned int depcNeeded; /**< Remaining unsatisfied deps. */
  volatile unsigned int
      invalidateCount; /**< Outstanding cache invalidations. */
} __attribute__((aligned));

/** An individual dependent registered on an event or persistent event. */
struct arts_dependent_s {
  uint8_t type;                         /**< Dependent kind (EDT or event). */
  volatile unsigned int slot;           /**< Target dependency slot. */
  volatile arts_guid_t addr;            /**< GUID of the dependent EDT/event. */
  volatile event_callback_t callback_t; /**< Inline callback (if any). */
  volatile bool doneWriting;            /**< Write completion flag. */
  arts_type_t acquire_mode;             /**< Acquire mode for signaling. */
  uint64_t byte_offset; /**< Byte offset for slice dependencies. */
  uint64_t size;        /**< Slice size in bytes. */
};

/** Linked list node containing an array of dependents. */
struct arts_dependent_list_s {
  unsigned int size; /**< Number of dependents in this node. */
  struct arts_dependent_list_s *volatile next; /**< Next list node. */
  struct arts_dependent_s dependents[];        /**< Flexible array. */
};

/** Version record for a persistent event (one per re-arm cycle). */
struct arts_persistent_event_version_s {
  unsigned int version;                   /**< Version sequence number. */
  volatile unsigned int latch_count;      /**< Current latch counter. */
  volatile unsigned int dependent_count;  /**< Registered dependent count. */
  struct arts_dependent_list_s dependent; /**< Inline dependent list head. */
};

/** Internal persistent event descriptor. */
struct arts_persistent_event_s {
  volatile unsigned int lock; /**< Spin-lock for concurrent updates. */
  struct arts_header_s header;
  arts_guid_t data;                  /**< DataBlock GUID to deliver on fire. */
  struct arts_link_list_s *versions; /**< Version history list. */
} __attribute__((aligned));

/** Internal latch event descriptor. */
struct arts_event_s {
  struct arts_header_s header;
  volatile bool fired;                   /**< Whether the event has fired. */
  volatile unsigned int destroy_on_fire; /**< Auto-destroy flag. */
  volatile unsigned int latch_count;     /**< Current latch counter. */
  volatile unsigned int pos; /**< Allocation cursor for dependents. */
  volatile unsigned int dependent_count; /**< Registered dependent count. */
  arts_guid_t data; /**< DataBlock GUID to deliver on fire. */
  struct arts_dependent_list_s dependent; /**< Inline dependent list head. */
} __attribute__((aligned));

/** @} */ /* end internal_structs */

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

/* ========================================================================= */
/** @defgroup td_types Termination Detection
 *  @{ */

/** Three-phase termination detection state machine. */
typedef enum {
  PHASE_1, /**< Initial quiescence check. */
  PHASE_2, /**< Counter stabilization. */
  PHASE_3  /**< Termination confirmed. */
} termination_detection_phase_t;

/**
 * @brief Per-epoch termination detection state.
 *
 * Tracks active/finished task counts across the cluster to determine when
 * all work within the epoch has completed.
 */
typedef struct {
  termination_detection_phase_t phase;     /**< Current TD phase. */
  volatile unsigned int activeCount;       /**< Local active task count. */
  volatile unsigned int finishedCount;     /**< Local finished task count. */
  volatile unsigned int globalActiveCount; /**< Cluster-wide active count. */
  volatile unsigned int
      globalFinishedCount;               /**< Cluster-wide finished count. */
  volatile unsigned int lastActiveCount; /**< Previous-round active count. */
  volatile unsigned int
      lastFinishedCount;            /**< Previous-round finished count. */
  volatile uint64_t queued;         /**< Number of queued operations. */
  volatile uint64_t outstanding;    /**< Number of outstanding remote ops. */
  unsigned int terminationExitSlot; /**< EDT slot to signal on completion. */
  arts_guid_t terminationExitGuid;  /**< EDT to signal on completion. */
  arts_guid_t guid;                 /**< GUID of this epoch. */
  arts_guid_t pool_guid;            /**< Associated resource pool GUID. */
  volatile unsigned int *wait_ptr;  /**< Context-switch wait pointer. */
  volatile uint64_t ticket;         /**< Context-switch ticket. */
} arts_epoch_t;

/** @} */ /* end td_types */

/* ========================================================================= */
/** @defgroup buffer_type Buffer Type
 *  @{ */

/** Node-local buffer accessible by GUID. */
typedef struct {
  void *buffer;               /**< Pointer to the buffer data. */
  uint32_t *size_to_write;    /**< Pointer to the write-size counter. */
  unsigned int size;          /**< Buffer size in bytes. */
  arts_guid_t epoch_guid;     /**< Enclosing epoch GUID. */
  volatile unsigned int uses; /**< Remaining access count before auto-free. */
} arts_buffer_t;

/** @} */ /* end buffer_type */

/**
 * @brief Thread-safe printf that serializes output across ARTS workers.
 *
 * @param format printf-style format string.
 * @param ...    Format arguments.
 */
void arts_printf(const char *format, ...);

#ifdef __cplusplus
}
#endif

#endif /* ARTSRT_H */
