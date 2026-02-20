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
#ifndef ARTS_RUNTIME_RT_H
#define ARTS_RUNTIME_RT_H
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file rt.h
 * @brief Internal structures for the ARTS runtime.
 *
 * This header defines internal types used by the runtime implementation:
 * EDT and DataBlock descriptors, event structures, and termination-detection
 * state.  Public types (arts_guid_t, arts_type_t, etc.) live in arts.h.
 *
 * @note This is an internal header.  User code should include @c arts.h.
 */

#include "arts.h"

#include "arts/arts_defs.h"
#include "arts/utils/link_list.h"

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
  volatile unsigned int copy_count; /**< Number of outstanding copies. */
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
  unsigned int numa_domain;  /**< NUMA domain assignment. */
  unsigned int node;         /**< Target node rank. */
  volatile unsigned int depc_needed; /**< Remaining unsatisfied deps. */
  volatile unsigned int
      invalidate_count; /**< Outstanding cache invalidations. */
} __attribute__((aligned));

/** An individual dependent registered on an event or persistent event. */
struct arts_dependent_s {
  uint8_t type;                         /**< Dependent kind (EDT or event). */
  volatile unsigned int slot;           /**< Target dependency slot. */
  volatile arts_guid_t addr;            /**< GUID of the dependent EDT/event. */
  volatile event_callback_t callback_t; /**< Inline callback (if any). */
  volatile bool done_writing;            /**< Write completion flag. */
  arts_type_t mode;                     /**< Access mode for signaling. */
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
 * Tracks active/finished task counts across nodes to determine when
 * all work within the epoch has completed.
 */
typedef struct {
  termination_detection_phase_t phase;     /**< Current TD phase. */
  volatile unsigned int active_count;       /**< Local active task count. */
  volatile unsigned int finished_count;     /**< Local finished task count. */
  volatile unsigned int global_active_count; /**< Cluster-wide active count. */
  volatile unsigned int
      global_finished_count;               /**< Cluster-wide finished count. */
  volatile unsigned int last_active_count; /**< Previous-round active count. */
  volatile unsigned int
      last_finished_count;            /**< Previous-round finished count. */
  volatile uint64_t queued;         /**< Number of queued operations. */
  volatile uint64_t outstanding;    /**< Number of outstanding remote ops. */
  unsigned int termination_exit_slot; /**< EDT slot to signal on completion. */
  arts_guid_t termination_exit_guid;  /**< EDT to signal on completion. */
  arts_guid_t guid;                 /**< GUID of this epoch. */
  arts_guid_t pool_guid;            /**< Associated resource pool GUID. */
  volatile unsigned int *wait_ptr;  /**< Epoch-wait flag pointer. */
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

#ifdef __cplusplus
}
#endif

#endif /* ARTS_RUNTIME_RT_H */
