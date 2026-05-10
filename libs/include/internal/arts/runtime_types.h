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
#ifndef ARTS_RUNTIME_TYPES_H
#define ARTS_RUNTIME_TYPES_H
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

#include "arts/defs.h"
#include "arts/sync/shared.h"         /* arts_shared_t, ARTS_SHARED_FIELD */
#include "arts/utils/lockfree_lifo.h" /* arts_lf_stack_t */
#include "arts/utils/mpsc.h"          /* arts_mpsc_t */
#include <stdbool.h>
#ifndef __cplusplus
#include <stdatomic.h>
#endif

/* ========================================================================= */
/** @defgroup internal_structs Internal Runtime Structures
 *  These structures are exposed for layout visibility but are managed
 *  entirely by the runtime.  User code should not manipulate them directly.
 *  @{ */

/** Common header prepended to every runtime object (EDT, DB, event). */
struct arts_header_s {
  uint8_t type : 8;   /**< Type tag (@ref arts_type_t). */
  uint64_t size : 56; /**< Total allocation size in bytes (includes struct
                         metadata). */
} ARTS_ALIGNED_MAX;

/** Internal DataBlock descriptor. */
struct arts_db_s {
  ARTS_SHARED_FIELD; /* shared_t — first member, always */
  struct arts_header_s header;
  uint64_t arts_id; /**< Compiler-assigned unique id (0 = unset). */
  arts_guid_t guid; /**< GUID of this DataBlock. */
  volatile unsigned int copy_count; /**< Number of outstanding copies. */
  volatile unsigned int reader;     /**< Active reader count. */
  volatile unsigned int writer;     /**< Active writer count. */
  volatile unsigned int version;    /**< Coherence version counter. */
  unsigned int time_stamp;          /**< Creation timestamp (relative). */
  arts_db_types_t db_type; /**< Storage subtype (DEFAULT/LOCAL/GPU/LC). */
  void *db_list;           /**< Node in the per-node DB tracking list. */
  /* RC coherence per-DB state.  NULL when the DB has no DB-level
   * coherence (PIN/CXL) or before lazy install; otherwise points to a
   * struct arts_db_cache_s.  This field was added later so the
   * coherence_*.c sources compile under ARTS_COHERENCE_INTEGRATED. */
  void *coherence_cache;
} ARTS_ALIGNED_MAX;

/** Internal EDT descriptor. */
struct arts_edt_s {
  ARTS_SHARED_FIELD; /* shared_t — first member, always */
  struct arts_header_s header;
  uint64_t arts_id;          /**< Compiler-assigned unique id (0 = unset). */
  arts_edt_t func_ptr;       /**< User function to execute. */
  uint32_t paramc;           /**< Number of static parameters. */
  uint32_t depc;             /**< Number of dependency slots. */
  arts_guid_t current_edt;   /**< GUID of this EDT. */
  arts_guid_t epoch_guid;    /**< Enclosing epoch GUID. */
  arts_guid_t finish_event;  /**< LATCH event for finish-scope tracking.
                                  NULL_GUID = no finish-scope (legacy path). */
  unsigned int numa_domain;  /**< NUMA domain assignment. */
  unsigned int node;         /**< Target node rank. */
  arts_edt_types_t edt_type; /**< EDT subtype (DEFAULT=CPU, GPU). */
  volatile unsigned int depc_needed; /**< Remaining unsatisfied deps. */
  volatile unsigned int
      invalidate_count; /**< Outstanding cache invalidations. */
} ARTS_ALIGNED_MAX;

/** An individual dependent registered on an event (legacy structure;
 *  retained only for the legacy event.c body until follow-up). */
struct arts_dependent_s {
  uint8_t type;               /**< Dependent kind (EDT or event). */
  volatile unsigned int slot; /**< Target dependency slot. */
  volatile arts_guid_t addr;  /**< GUID of the dependent EDT/event. */
  volatile bool done_writing; /**< Write completion flag. */
  arts_db_access_mode_t mode; /**< Access mode for signaling. */
  uint64_t byte_offset;       /**< Byte offset for slice dependencies. */
  uint64_t size;              /**< Slice size in bytes. */
};

/** Linked list node containing an array of dependents. */
struct arts_dependent_list_s {
  unsigned int size; /**< Number of dependents in this node. */
  struct arts_dependent_list_s *volatile next; /**< Next list node. */
  struct arts_dependent_s dependents[];        /**< Flexible array. */
};

/** Forward-declared dep node (definition in arts/sync/event.h, Task 8). */
struct arts_event_dep_s;

/** Internal event descriptor — single generic type, hint-driven behavior.
 *  Configuration fields are immutable after init; dynamic fields are
 *  atomics.
 *
 *  Two declarations: the C path uses C11 `_Atomic`; the C++/nvcc path
 *  (this header is reached transitively from .cu files via arts_db_s)
 *  drops the qualifier so the layout is visible without requiring C11
 *  atomics — same approach as memory/coherence.h. */
#ifdef __cplusplus
struct arts_event_s {
  ARTS_SHARED_FIELD;           /* shared_t — first member, always */
  struct arts_header_s header; /* type=ARTS_EVENT */

  /* Configuration snapshot (immutable after arts_event_create_internal). */
  int32_t init_latch; /* signed; LATCH may start negative */
  uint32_t init_nb_deps_required;
  uint32_t max_nb_deps;
  uint8_t auto_destroy;
  uint8_t negative_latch_allowed;
  uint8_t multiple_fire;
  uint8_t _pad0;

  /* Dynamic counters. */
  int32_t curr_latch;
  int32_t nb_deps_left;
  uint32_t max_deps_left;
  bool fired;

  /* Discriminated storage by `multiple_fire` (see docs/event-refactor/spec.md
   * §2.1 / §3).  In C++ TUs the atomic / mpsc qualifiers are dropped so
   * struct layout is visible to nvcc — same trick used elsewhere. */
  union {
    struct {
      arts_guid_t data;
      arts_lf_stack_t deps_stack;
    } simple;
    struct {
      arts_mpsc_t data_queue;
      arts_mpsc_t dep_queue;
      uint8_t draining;
    } channel;
  };
} ARTS_ALIGNED_MAX;
#else
struct arts_event_s {
  ARTS_SHARED_FIELD;           /* shared_t — first member, always */
  struct arts_header_s header; /* type=ARTS_EVENT */

  /* Configuration snapshot (immutable after arts_event_create_internal). */
  int32_t init_latch;             /* signed; LATCH may start negative */
  uint32_t init_nb_deps_required; /* deps consumed per fire (CHANNEL spec=1) */
  uint32_t max_nb_deps;           /* lifetime cap on total registrations */
  uint8_t auto_destroy;
  uint8_t negative_latch_allowed;
  uint8_t multiple_fire;
  uint8_t _pad0;

  /* Dynamic counters (atomic).
   *   curr_latch  : satisfy decrements; non-CHANNEL fire trigger when
   *                 prev==1; CHANNEL fire trigger when curr_latch <= 0.
   *   nb_deps_left: addDep decrements; CHANNEL fire-pair counter (paired
   *                 with curr_latch).  Recharged by += init_nb_deps_required
   *                 inside CHANNEL drain.  For non-CHANNEL this is unused
   *                 (the non-CHANNEL fire condition is `prev==1` in
   *                 curr_latch alone).
   *   max_deps_left: monotonically decreasing total-lifetime cap.
   *                  When it reaches 0 and auto_destroy is set, the
   *                  event mark_deletes itself.
   *   fired       : non-CHANNEL single-fire CAS gate. */
  _Atomic(int32_t) curr_latch;
  _Atomic(int32_t) nb_deps_left;
  _Atomic(uint32_t) max_deps_left;
  _Atomic(bool) fired;

  /* Discriminated storage by `multiple_fire` — see
   * docs/event-refactor/spec.md §2.1 / §3.  union saves space on
   * CHANNEL events that don't need a single data slot, and on
   * non-CHANNEL events that don't need two FIFO queues. */
  union {
    struct {
      arts_guid_t data; /* last satisfy data; late binders read here */
      arts_lf_stack_t deps_stack; /* Treiber stack of pending consumers */
    } simple;
    struct {
      arts_mpsc_t data_queue;    /* satisfy FIFO */
      arts_mpsc_t dep_queue;     /* dep FIFO */
      _Atomic(uint8_t) draining; /* single-flight drainer gate */
    } channel;
  };
} ARTS_ALIGNED_MAX;
#endif

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
struct arts_epoch_s {
  ARTS_SHARED_FIELD;                  /* shared_t — first member, always. The
                                         route_table free_item dispatcher reads this
                                         offset-0 field to invoke the per-object deleter
                                         once the slot's lock count drops to 0 with
                                         DELETE set.  See libs/src/core/sync/epoch.c
                                         arts_epoch_deleter. */
  volatile unsigned int local_lock;   /**< Single-node active/finished lock. */
  volatile unsigned int phase;        /**< Current TD phase (PHASE_*). */
  volatile unsigned int active_count; /**< Local active task count. */
  volatile unsigned int finished_count;      /**< Local finished task count. */
  volatile unsigned int global_active_count; /**< Cluster-wide active count. */
  volatile unsigned int
      global_finished_count;               /**< Cluster-wide finished count. */
  volatile unsigned int last_active_count; /**< Previous-round active count. */
  volatile unsigned int
      last_finished_count;            /**< Previous-round finished count. */
  volatile uint64_t queued;           /**< Number of queued operations. */
  volatile uint64_t outstanding;      /**< Number of outstanding remote ops. */
  unsigned int termination_exit_slot; /**< EDT slot to signal on completion. */
  arts_guid_t termination_exit_guid;  /**< EDT to signal on completion. */
  arts_guid_t guid;                   /**< GUID of this epoch. */
  arts_guid_t pool_guid;              /**< Associated resource pool GUID. */
};
typedef struct arts_epoch_s arts_epoch_t;

/** @} */ /* end td_types */

/* ========================================================================= */
#ifdef __cplusplus
}
#endif

#endif /* ARTS_RUNTIME_TYPES_H */
