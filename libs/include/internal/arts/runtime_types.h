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
 * state.  Public types (arts_guid_t, arts_guid_kind_t, etc.) live in arts.h.
 *
 * @note This is an internal header.  User code should include @c arts.h.
 */

#include "arts.h"

#include "arts/defs.h"
/* DB-coherence layout types (arts_db_s / arts_db_cache_s / buffer / home,
 * plus the arts_coh_atomic_uint typedef) live in the coherence module's own
 * header; pulled in here so existing includers of runtime_types.h still see
 * the DB types unchanged. */
#include "arts/memory/coherence_types.h"
#include "arts/utils/lockfree_lifo.h" /* arts_lf_stack_t (event simple deps) */
#include "arts/utils/mpsc.h"          /* arts_mpsc_t (event channel) */
#include <stdbool.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdatomic.h>
#endif

/* ========================================================================= */
/** @defgroup internal_structs Internal Runtime Structures
 *  These structures are exposed for layout visibility but are managed
 *  entirely by the runtime.  User code should not manipulate them directly.
 *  @{ */

/** Internal EDT descriptor. */
struct arts_edt_s {
  uint64_t arts_id;    /**< Compiler-assigned unique id (0 = unset). */
  arts_edt_t func_ptr; /**< User function to execute. */
  uint32_t paramc;     /**< Number of static parameters. */
  uint32_t depc;       /**< Number of dependency slots. */
  /* The EDT's own GUID.  Unlike DB/Event/Epoch (always reached via a
   * route_table lookup whose key already IS the GUID), an EDT is dispatched
   * by raw pointer through the lock-free work-stealing deques — at dispatch
   * there is no key and no handler args, so the GUID must travel inside the
   * struct.  This is a load-bearing identity carrier, NOT a redundant
   * self-GUID: do not remove it. */
  arts_guid_t guid;
  arts_guid_t epoch_guid;    /**< Enclosing epoch GUID (NULL_GUID = none). */
  arts_guid_t finish_event;  /**< LATCH event for finish-scope tracking.
                                  NULL_GUID = no finish-scope (legacy path). */
  arts_edt_types_t edt_type; /**< EDT subtype (DEFAULT=CPU, GPU). */
  volatile unsigned int depc_needed; /**< Remaining unsatisfied deps (satisfy
                                          phase — driven to 0 by event/signal
                                          delivery before DB acquisition). */
  uint32_t resume_k; /**< Strict-sequential DB-acquire resume index: the
                          position in the GUID-sorted dep order up to which
                          DBs have been acquired.  0 at acquire start; advances
                          one dep at a time.  A parked (cross-rank) dep stops
                          the walk here until its wake resumes — never attempts
                          a higher-GUID lock while a lower one is outstanding,
                          which is what prevents cyclic cross-rank acquire. */
  volatile unsigned int
      invalidate_count; /**< Outstanding cache invalidations. */
} ARTS_ALIGNED_MAX;

/* Total allocation size of an EDT = struct + trailing [paramv | depv].
 * Computed from paramc/depc; no separate object header stores it. */
static inline uint64_t arts_edt_total_size(const struct arts_edt_s *edt) {
  return sizeof(struct arts_edt_s) +
         ((uint64_t)edt->paramc * sizeof(uint64_t)) +
         ((uint64_t)edt->depc * sizeof(arts_edt_dep_t));
}

/** Forward-declared dep node (definition in arts/sync/event.h, Task 8). */
struct arts_event_dep_s;

/** Internal event descriptor — single generic type, hint-driven behavior.
 *  The `is_channel` discriminator selects which union arm is active.
 *
 *  Non-CHANNEL semantics (`is_channel == 0`):
 *    - `latch` decrements per LATCH_DECR satisfy; fires at <= 0.
 *    - Fire is a pure state transition (sets `fired`, publishes `data`,
 *      drains `deps_stack`) and never destroys: the event lingers to serve
 *      late binders until an explicit `arts_event_destroy`.  Over-satisfy
 *      past the fire is silently absorbed.
 *
 *  CHANNEL semantics (`is_channel == 1`):
 *    - `nb_sat` increments per satisfy; `nb_deps` increments per add_dep.
 *    - Drainer pops one from each queue, decrements both counters, signals.
 *    - No auto-destroy; only explicit `arts_event_destroy`.
 *
 *  Two declarations: the C path uses C11 `_Atomic`; the C++/nvcc path
 *  drops the qualifier so the layout is visible without requiring C11
 *  atomics — same approach as memory/coherence.h. */
#ifdef __cplusplus
struct arts_event_s {
  uint8_t is_channel; /* discriminator: 0=simple, 1=channel */

  union {
    struct {
      int32_t latch;
      bool fired;
      arts_guid_t data;
      arts_lf_stack_t deps_stack;
    } simple;
    struct {
      uint32_t nb_sat;
      uint32_t nb_deps;
      arts_mpsc_t data_queue;
      arts_mpsc_t dep_queue;
      uint8_t draining;
    } channel;
  };
} ARTS_ALIGNED_MAX;
#else
struct arts_event_s {
  uint8_t is_channel; /* discriminator: 0=simple, 1=channel */

  union {
    struct {
      _Atomic(int32_t) latch;     /* fires at <= 0 */
      _Atomic(bool) fired;        /* single-fire CAS gate */
      arts_guid_t data;           /* last satisfy data; late binders read */
      arts_lf_stack_t deps_stack; /* Treiber stack of pending consumers */
    } simple;
    struct {
      _Atomic(uint32_t) nb_sat;  /* incremented per satisfy */
      _Atomic(uint32_t) nb_deps; /* incremented per add_dep */
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
