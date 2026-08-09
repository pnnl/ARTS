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
 *  Every GUID carries a kind tag from this enum.
 *  @{ */

/**
 * @brief GUID object kind tag.
 *
 * Identifies what kind of object a GUID refers to.  Encoded in the
 * GUID's 2-bit kind field (ARTS_GUID_TYPE_BITS = 2 → 4 valid values
 * + sentinel).  Bit encoding: RESERVED=00, DB=01, EVENT=10, EDT=11.
 * Zero-initialized GUIDs decode to RESERVED (never a valid object),
 * which is more correct than the legacy DB=0 assignment.
 *
 * All datablocks share the single ARTS_GUID_DB tag; the DB storage
 * subtype is specified via arts_db_types_t at creation time.
 * Access modes (read/write) are separate — see arts_db_access_mode_t.
 */
typedef enum {
  ARTS_GUID_RESERVED = 0, /**< Reserved sentinel; never produced.  NULL = 0. */
  ARTS_GUID_DB = 1,       /**< DataBlock (kind bits 01). */
  ARTS_GUID_EVENT = 2, /**< Latch-based synchronization event (kind bits 10). */
  ARTS_GUID_EDT = 3,   /**< Event-Driven Task (kind bits 11). */
  ARTS_GUID_LAST = 4   /**< Sentinel — first invalid kind value (= 4). */
} arts_guid_kind_t;

/**
 * @brief DataBlock access mode (per-dependency, stored in @c
 * arts_edt_dep_t.mode).
 *
 * Specifies how an EDT accesses a datablock dependency.  Set via
 * @c arts_add_dependence(), not at DB creation.
 */
typedef enum {
  DB_MODE_NULL = 0, /**< Unset / placeholder. */
  DB_MODE_RO,       /**< Read-Only (shared readers, no publish). */
  DB_MODE_RW,       /**< Read-Write (per-node exclusive, OCR RW semantics). */
  DB_MODE_VAL,      /**< Dependency carries a raw uint64 value (not a GUID). */
  /* Values >= DB_MODE_INTERNAL_BASE are reserved for runtime-internal
   * dispatch (PTR slices, GPU LC sync/alloc, GPU memset).  They never
   * appear in user-facing arts_add_dependence() arguments and are not
   * part of the public API surface. */
} arts_db_access_mode_t;

/**
 * @brief DataBlock subtype (stored in @c arts_db_s.db_type, NOT in the GUID).
 *
 * Specifies the storage class of a DataBlock.  All subtypes share the same
 * @c ARTS_GUID_DB kind tag; the subtype is carried inside the DB descriptor.
 * Naming convention: ARTS_DB_<storage>.  For ARTS_DB, consistency is
 * governed by the build-time memory model and coherence protocol; other
 * subtypes carry no DB-level coherence (hardware coherence only).
 * See @c docs/programming_model/memory_model.rst for the full model taxonomy.
 *
 *   ARTS_DB         — regular DRAM; contract/protocol selected at build time
 *   ARTS_DB_PIN     — regular DRAM, node-pinned, no DB-level coherence
 *   ARTS_DB_CXL     — CXL shared; HW cache coherence intra-node, app-ordered
 *                     (app-ordered, full DRF) across nodes
 *   ARTS_DB_GPU     — GPU staging; concurrent per-device replicas merged by
 *                     reduction at release (app-ordered)
 *   ARTS_DB_GPU_PIN — GPU staging, no DB-level coherence
 */
typedef enum {
  ARTS_DB = 0, /**< Regular DRAM; contract/protocol selected at build time. */
  ARTS_DB_PIN, /**< Node-pinned regular DRAM, no DB-level coherence. */
  ARTS_DB_CXL, /**< CXL shared; app-ordered, full DRF (compiled w/ CXL). */
  ARTS_DB_GPU, /**< GPU staging; per-device replicas merged at release (app-ordered
                  style). */
  ARTS_DB_GPU_PIN, /**< GPU staging (host pinned + per-device replica). */
} arts_db_types_t;

/* @c ARTS_DB_DEFAULT is the experiment-wide DB storage subtype every benchmark
 * uses.  Its concrete value is decided by the CMake build configuration.
 * Ships as ARTS_DB so the build works without special hardware; flip the cmake
 * option to retarget every benchmark in one place. */
#ifndef ARTS_DB_DEFAULT
#define ARTS_DB_DEFAULT ARTS_DB
#endif

/**
 * @brief DataBlock creation property bits (OCR-spec flags).
 *
 * Passed via the @c flags parameter of @c arts_db_create and
 * @c arts_db_create_with_guid.  Selects creator-side ownership semantics.
 *
 *   ARTS_DB_PROP_NONE        — Standard: creator EDT auto-acquires RW
 *                              (writer_count starts at 2 = sentinel + creator).
 *   ARTS_DB_PROP_NO_ACQUIRE  — Creator does NOT auto-acquire; the home rank
 *                              is the initial idle owner.  arts_db_create
 *                              returns *addr = NULL.  First consumer EDT
 *                              must perform a normal acquire to obtain
 *                              ownership.
 */
#define ARTS_DB_PROP_NONE 0x0000u
#define ARTS_DB_PROP_NO_ACQUIRE 0x0001u

/**
 * @brief EDT subtype (stored in @c arts_edt_s.edt_type, NOT in the GUID).
 *
 * All EDTs share the single @c ARTS_GUID_EDT kind tag in the GUID; the subtype
 * is carried inside the EDT descriptor.  CPU and GPU EDTs differ in scheduling
 * and execution but share the same GUID kind.
 */
typedef enum {
  ARTS_EDT_CPU = 0, /**< CPU EDT (standard). */
  ARTS_EDT_GPU = 1, /**< GPU EDT (CUDA kernel or library host function). */
} arts_edt_types_t;

/** @} */ /* end type_enum */

/* ========================================================================= */
/** @defgroup hint_type Hint Type
 *  Advisory metadata for EDT and DataBlock creation.
 *  @{ */

/** Sentinel: use the node that is running the current EDT. */
#define ARTS_HINT_CURRENT_RANK ((unsigned int)-1)

/** Sentinel route for @c arts_guid_reserve_range: the resulting range is
 *  distributed across all nodes round-robin by index — same `(range, idx)`
 *  on every rank yields the same GUID, but the home rank is `idx % nrank`.
 *  Used by the OCR shim's @c ocrGuidRangeCreate to honor OCR's "any EDT
 *  with same input → same GUID" invariant while keeping the range's homes
 *  distributed across ranks. */
#define ARTS_HINT_ROUND_ROBIN ((unsigned int)-2)

/** Sentinel: no placement preference — let the runtime's compile-time
 *  no-hint placement policy pick the rank (ROUNDROBIN or CREATOR; see
 *  arts_edt_create's hint==NULL behavior, which this is equivalent to).
 *  Distinct from @c ARTS_HINT_CURRENT_RANK (explicit self) and from passing
 *  a NULL hint pointer: usable by callers that must still populate other
 *  hint fields (finish_event, output_event, edt_id, flags) while leaving
 *  placement itself unpinned. */
#define ARTS_HINT_ANY_RANK ((unsigned int)-3)

/** @defgroup hint_structs Creation hints (purpose-specific)
 *
 *  Three independent hint structs so each creation API can grow features
 *  that do not apply to the others.
 *
 *  Migration from the legacy @c arts_hint_t :
 *    - @c arts_hint_t.rank → @c arts_edt_hint_t.rank or @c
 * arts_db_hint_t.rank
 *    - @c arts_hint_t.id    → @c arts_edt_hint_t.edt_id (DB hint has no id;
 *                              the DB profiling id field is dropped because
 *                              no functional code consumed it)
 *  @{ */

/** Bit flags for arts_edt_hint_t.flags.  Reserved for future EDT-create flags;
 *  finish scopes are no longer a flag (use ARTS_EVENT_HINT_FINISH +
 *  arts_edt_hint_t.finish_event instead). */
#define ARTS_EDT_FLAG_NONE 0x0000u

/** Hint passed to @c arts_edt_create.  Optional fields collapse the legacy
 *  six EDT-create variants into a single entry point:
 *    - @c rank   selects the home node (default current rank).
 *    - @c edt_id is the compiler-assigned profiling id (default 0).
 *    - @c guid   when non-NULL_GUID pre-reserves the EDT GUID; the home
 *                rank is then taken from that GUID and @c rank is ignored.
 *    - @c finish_event when non-NULL_GUID joins this EDT to that finish scope.
 *    - @c flags  bitfield of ARTS_EDT_FLAG_* (default ARTS_EDT_FLAG_NONE). */
typedef struct {
  /** Target node rank.  ARTS_HINT_CURRENT_RANK = current node (default) |
   *  ARTS_HINT_ANY_RANK = no preference (policy-selected, same as passing a
   *  NULL hint) | specific rank. */
  unsigned int rank;
  /** Compiler-assigned profiling identifier.  0 = disabled. */
  uint64_t edt_id;
  /** Pre-reserved GUID.  NULL_GUID = auto-allocate (default). */
  arts_guid_t guid;
  /** Finish event to join (bulk sync).  NULL_GUID = inherit the caller's
   *  ambient finish scope (default).  When set, this EDT (and its descendants)
   *  join that finish event: INCR at create, DECR at completion. */
  arts_guid_t finish_event;
  /** Output event (per-EDT result channel; OCR-style).  NULL_GUID = none
   *  (default).  When set, the runtime satisfies this event (DECR slot)
   *  after the EDT's data blocks have been released, carrying the result
   *  GUID the EDT body registered via @c arts_edt_set_result (NULL_GUID if
   *  it registered none).  Unlike @c finish_event this is never inherited —
   *  it belongs to this EDT only. */
  arts_guid_t output_event;
  /** Bitfield of ARTS_EDT_FLAG_*.  uint32_t for future flag growth.  Default
   * ARTS_EDT_FLAG_NONE (0). */
  uint32_t flags;
} arts_edt_hint_t;

#define ARTS_EDT_HINT_DEFAULTS                                                 \
  ((arts_edt_hint_t){.rank = ARTS_HINT_CURRENT_RANK,                           \
                     .edt_id = 0,                                              \
                     .guid = NULL_GUID,                                        \
                     .finish_event = NULL_GUID,                                \
                     .output_event = NULL_GUID,                                \
                     .flags = ARTS_EDT_FLAG_NONE})

/** Hint passed to @c arts_db_create.
 *
 *  @c access_offset and @c access_size are reserved for a future
 *  fine-grained slicing feature.  The current runtime accepts but
 *  ignores them; downstream code may already pass slice info that will
 *  become live in a follow-up patch. */
typedef struct {
  /** Target node rank.  ARTS_HINT_CURRENT_RANK | ARTS_HINT_ROUND_ROBIN |
   *  specific rank.  Default ARTS_HINT_CURRENT_RANK. */
  unsigned int rank;
  /** Reserved for future fine-grained slice access.  Default 0. */
  uint64_t access_offset;
  /** Reserved for future fine-grained slice access.  Default UINT64_MAX
   *  (= entire DB). */
  uint64_t access_size;
  /** Pre-reserved GUID.  NULL_GUID = auto-allocate (default).  When
   *  non-zero, the GUID's rank field is authoritative for routing and
   *  overrides @c rank above. */
  arts_guid_t guid;
  /** If true, a create at an already-occupied (home-local) GUID FAILS instead
   * of overwriting — OCR GUID_PROP_CHECK / rendezvous semantics.  Default
   * false = unconditional replace (a labeled-GUID reuse overwrites the prior
   * generation, releasing it). */
  bool check;
} arts_db_hint_t;

#define ARTS_DB_HINT_DEFAULTS                                                  \
  ((arts_db_hint_t){.rank = ARTS_HINT_CURRENT_RANK,                            \
                    .access_offset = 0,                                        \
                    .access_size = UINT64_MAX,                                 \
                    .guid = NULL_GUID})

/** @} */

/** @} */ /* end hint_type */

/* ========================================================================= */
/** @defgroup dep_types Dependency Types
 *  Structures and function-pointer types used to wire EDT dependencies.
 *  @{ */

/**
 * @brief Describes a single dependency slot delivered to an EDT.
 *
 * Mode is set via @c arts_add_dependence().
 * User EDTs typically read @c guid / @c ptr and ignore @c mode.
 */
typedef struct {
  arts_guid_t guid;           /**< GUID of the DataBlock (or encoded value). */
  void *ptr;                  /**< Pointer to the DataBlock payload. */
  arts_db_access_mode_t mode; /**< Access mode for this dependency slot. */
  /** Runtime-internal: DB storage class recorded at acquire time so release can
   *  route a coherent @c ARTS_DB (where @c ptr is the buffer payload) versus a
   *  pinned subtype (where @c ptr is the inline @c db+1) without recovering the
   *  subtype by pointer arithmetic — which is invalid for the buffer pointer
   *  and crashes once a concurrent destroy has removed the route entry.  User
   *  EDTs ignore this field. */
  arts_db_types_t subtype;
  /** Runtime-internal: set at acquire time when this serialized (RW) slot names
   *  a DB an earlier serialized slot of the same EDT already acquired.  Such a
   *  slot took a per-slot buffer ref but NOT a coherence (writer_count) hold —
   *  the earlier slot owns the single acquire/release for that DB.  Release
   *  drops the buffer ref but skips the coherence release for an alias, so the
   *  owner's writer_count is decremented exactly once per distinct DB. Recorded
   *  here (not re-derived by GUID scan at release) so a mid-EDT release that
   *  nulls the owning slot cannot make an alias masquerade as the owner.  User
   *  EDTs ignore this field. */
  bool alias;
  /** Runtime-internal: for a coherent @c ARTS_DB slot, the ref-counted handle
   *  on the owning DB descriptor (@c arts_db_s), taken once when this slot's
   *  buffer ref is secured at acquire and dropped LAST in @c release_one_dep
   *  (after the buffer-ref drop and the RW/RO coherence release).  Pinning the
   *  descriptor for the slot's whole acquire→release span keeps the descriptor
   *  — and the buffer slot + recycle pool embedded in it — alive while any
   *  buffer reference is outstanding, so a concurrent destroy cannot free the
   *  descriptor out from under a buffer's recycle-on-drop.  NULL when unset
   *  (non-coherent slot).  Typed @c void* here to avoid leaking the internal
   *  @c arts_shared_ptr_t into the public header; the runtime casts it.  User
   *  EDTs ignore this field. */
  void *db_pin;
} arts_edt_dep_t;

/**
 * @brief Function signature for Event-Driven Tasks (CPU and GPU).
 */
typedef void (*arts_edt_t)(uint32_t paramc, const uint64_t *paramv,
                           uint32_t depc, arts_edt_dep_t depv[]);

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
 * The EDT can call blocking operations like arts_event_wait().
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
} arts_latch_event_slot_t;

/** @} */ /* end event_slots */

/* ========================================================================= */
/**
 * @defgroup event_hint Event creation hint
 *
 * Every ARTS event is one of two kinds, selected by `channel`:
 *
 *   - simple (latch) — `latch` is the initial trigger counter; each
 *     LATCH_DECR satisfy decrements it and the event fires when it reaches
 *     <= 0.  Default 1 (single satisfy fires).  Firing is a pure state
 *     transition: the event lingers (`fire ≠ destroy`), serving every later
 *     `add_dependence` from the stored fire data until an explicit
 *     `arts_event_destroy`.  A satisfy past the fire is silently absorbed.
 *   - channel — multi-fire FIFO; each satisfy is paired with one
 *     `add_dependence` in arrival order.  All fields except `rank` and
 *     `guid` are ignored.
 *  @{ */
typedef struct {
  /** Target node rank.  ARTS_HINT_CURRENT_RANK = current node (default). */
  unsigned int rank;
  /** Initial latch counter.  Default 1.  Fires at <= 0 after LATCH_DECR
   *  satisfies (one decrement per satisfy). */
  int32_t latch;
  /** If true, this is a CHANNEL event: multi-fire FIFO with paired
   *  satisfy/dep queues.  All other hint fields except `rank` and `guid`
   *  are ignored.  Default false. */
  bool channel;
  /** Pre-reserved GUID.  NULL_GUID = auto-allocate (default).  When non-zero,
   *  the GUID's rank field is authoritative and overrides @c rank above. */
  arts_guid_t guid;
  /** If true, a create at an already-occupied (home-local) GUID FAILS (returns
   * NULL_GUID) instead of overwriting — OCR GUID_PROP_CHECK / rendezvous
   * semantics (the first creator wins; a later one observes the collision).
   * Default false = unconditional replace (a labeled-GUID reuse overwrites the
   * prior generation, releasing it). */
  bool check;
  /** If true, this is a FINISH event: a bulk-synchronization latch. All other
   *  fields (latch, channel, guid, check) are ignored — forced to a simple
   *  latch=1 (creator-token), current rank, auto-allocated GUID, auto_destroy.
   *  The runtime auto-chains it to the ambient finish scope and tracks its
   *  creator-token for cleanup. Wait on it with arts_event_wait. Default false.
   */
  bool finish;
  /** If true, this event self-destructs (route-slot detach) the instant it
   *  fires, instead of lingering for late binders.  Single-shot semantics for
   *  internal forwarder/proxy latches.  Implied by @c finish.  Immutable after
   *  creation — there is no runtime setter.  Default false (fire-and-linger).
   */
  bool auto_destroy;
} arts_event_hint_t;

/** OCR LATCH_T — counter event.  Argument is the initial counter value;
 *  fires when curr_latch reaches <= 0 via DECR satisfies. */
#define ARTS_EVENT_HINT_LATCH(counter_init)                                    \
  ((arts_event_hint_t){.rank = ARTS_HINT_CURRENT_RANK,                         \
                       .latch = (counter_init),                                \
                       .channel = false,                                       \
                       .guid = NULL_GUID})

/** Default values: single satisfy fires, then fire-and-linger. */
#define ARTS_EVENT_HINT_DEFAULTS ARTS_EVENT_HINT_LATCH(1)

/* Single-fire OCR event flavors all collapse to LATCH(1): the distinct
 * ONCE/IDEM/STICKY/COUNTED semantics (auto-destroy, over-satisfy error,
 * exact-N dep count) are subsumed by the unified fire-and-linger +
 * silent-over-satisfy model.  Aliases kept for source compatibility. */
#define ARTS_EVENT_HINT_ONCE ARTS_EVENT_HINT_LATCH(1)
#define ARTS_EVENT_HINT_IDEMPOTENT ARTS_EVENT_HINT_LATCH(1)
#define ARTS_EVENT_HINT_STICKY ARTS_EVENT_HINT_LATCH(1)
#define ARTS_EVENT_HINT_COUNTED(nb_deps) ARTS_EVENT_HINT_LATCH(1)

/** OCR CHANNEL_T — multi-fire FIFO event. */
#define ARTS_EVENT_HINT_CHANNEL                                                \
  ((arts_event_hint_t){                                                        \
      .rank = ARTS_HINT_CURRENT_RANK, .channel = true, .guid = NULL_GUID})
/** Bulk-synchronization finish event (latch=1 creator-token, auto_destroy). */
#define ARTS_EVENT_HINT_FINISH                                                 \
  ((arts_event_hint_t){.rank = ARTS_HINT_CURRENT_RANK, .finish = true})
/** @} */

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
 * @brief Reserve a GUID of the given @p kind on node @p rank.
 *
 * @param kind  Kind tag for the GUID (e.g. @c ARTS_GUID_EDT, @c ARTS_GUID_DB).
 * @param rank  Target node rank.
 * @return A new GUID.
 */
arts_guid_t arts_guid_reserve(arts_guid_kind_t kind, unsigned int rank);

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
 * @brief Return the kind tag encoded in @p guid.
 *
 * @param guid GUID to query.
 * @return The arts_guid_kind_t stored in the GUID.
 */
arts_guid_kind_t arts_guid_get_kind(arts_guid_t guid);

/**
 * @brief Reserve a contiguous range of @p size GUIDs on node @p route.
 *
 * Returns the start GUID of the range.  Use @c arts_guid_from_index() to
 * access individual GUIDs within the range.
 *
 * @param kind  Kind tag for every GUID in the range.
 * @param size  Number of GUIDs to allocate.
 * @param rank  Target node rank.  Special values:
 *              - @c ARTS_HINT_CURRENT_RANK — pin all GUIDs to the calling rank
 *              - @c ARTS_HINT_ROUND_ROBIN — distribute homes across ranks
 *                (`home = idx % nrank` in @c arts_guid_from_index).  The
 *                returned range GUID encodes a sentinel rank field; only
 *                @c arts_guid_from_index / @c arts_guid_index_from
 *                interpret it correctly.  Caller must broadcast this range
 *                GUID to every rank that will derive children.
 * @return The start GUID of the range, or @c NULL_GUID on failure.
 * @see arts_guid_from_index, arts_guid_index_from
 */
arts_guid_t arts_guid_reserve_range(arts_guid_kind_t kind, unsigned int size,
                                    unsigned int rank);

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

/** @} */ /* end guid */

/* ========================================================================= */
/** @defgroup edt Event-Driven Tasks (EDT)
 *  Create, signal, and destroy asynchronous task units.
 *  @{ */

/**
 * @brief Create an EDT.
 *
 * The EDT will execute @p func_ptr once all @p depc dependency slots have
 * been satisfied via arts_add_dependence().  All optional fields (target
 * rank, pre-reserved GUID, finish scope, profiling id) are carried in the
 * hint struct.  Pass @c NULL, or a hint with @c rank == ARTS_HINT_ANY_RANK,
 * for "no placement preference": the runtime's compile-time no-hint EDT
 * placement policy (ARTS_NOHINT_EDT_PLACEMENT: ROUNDROBIN default, or
 * CREATOR for the legacy pin-to-creator behavior) picks the execution rank.
 * A NULL hint also inherits the caller's ambient finish scope.
 *
 * @param func_ptr Function to execute.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of @p paramc uint64_t values copied into the closure.
 * @param depc     Number of dependency slots.
 * @param hint     Advisory metadata (rank, edt_id, guid, finish_event,
 *                 output_event).  NULL = defaults.
 * @return GUID of the newly created EDT.
 * @see arts_add_dependence, arts_edt_destroy
 */
arts_guid_t arts_edt_create(arts_edt_t func_ptr, uint32_t paramc,
                            const uint64_t *paramv, uint32_t depc,
                            const arts_edt_hint_t *hint);

/**
 * @brief Register the calling EDT's result GUID (output-event payload).
 *
 * Call from inside an EDT body.  The runtime delivers the registered GUID
 * by satisfying the EDT's output event (@c arts_edt_hint_t.output_event) —
 * strictly after the EDT's data blocks have been released, so a consumer
 * woken by the output event can never acquire one of this EDT's data
 * blocks before the writes are published.  A later call replaces the
 * value; without a call the output event fires with NULL_GUID.  No-op for
 * EDTs created without an output event, or outside a running EDT.
 */
void arts_edt_set_result(arts_guid_t result_guid);

/**
 * @brief Cancel a freshly-created EDT, removing its GUID from the routing
 * table.
 *
 * A pre-runnable EDT (unsatisfied dependency slots remain) is detached and
 * never runs.  Calling this on an EDT that has already become runnable is a
 * safe no-op — that EDT runs to completion (it travels by a held reference, so
 * the route-slot detach cannot free it mid-flight).  The EDT's run state is
 * therefore NOT the constraint.
 *
 * The caller MUST NOT access the EDT's GUID via arts_add_dependence (as source
 * or destination) or arts_edt_satisfy* concurrently with, or after, the
 * destroy: using a destroyed GUID is undefined, and a satisfy racing the
 * destroy is the one genuinely unsafe interleaving.  In practice the safe
 * pattern is to cancel a just-created EDT before wiring any dependence into it.
 *
 * EDTs are not auto-destroyed on completion; this explicit cancel path is
 * rarely needed.
 *
 * @param guid GUID of the EDT to cancel.
 */
void arts_edt_destroy(arts_guid_t guid);

/** @} */ /* end edt */

/* ========================================================================= */
/** @defgroup event Events
 *  Latch-based event synchronization primitives.
 *  @{ */

/**
 * @brief Create a generic event with the given hint.
 *
 * Behavior is fully determined by @p hint.  Pass @c NULL for OCR ONCE_T
 * defaults.  The runtime is kind-unaware: there is no event type tag.
 *
 * When @c hint->guid is non-zero the GUID is pre-reserved (labeled-GUID
 * path): the GUID's rank is the event home, cross-rank creates are
 * forwarded via MSG_EVENT_CREATE, and concurrent installs with
 * the same GUID are race-safe (first install wins, others are silent
 * no-ops per OCR labeled-event spec).
 *
 * @param hint Hint snapshot (NULL = ARTS_EVENT_HINT_DEFAULTS).
 * @return The event GUID (== hint->guid when pre-reserved), or NULL_GUID
 *         on failure or when a concurrent caller won the install race.
 * @see arts_event_satisfy_slot, arts_event_destroy
 */
arts_guid_t arts_event_create(const arts_event_hint_t *hint);

/**
 * @brief Satisfy an event with the conventional DECR slot.
 *
 * Convenience wrapper equivalent to
 * @c arts_event_satisfy_slot(event_guid, data_guid,
 * ARTS_EVENT_LATCH_DECR_SLOT). Aligned with OCR's @c ocrEventSatisfy(g, d). Use
 * this in the common case; only call @c arts_event_satisfy_slot directly when
 * you need INCR.
 */
void arts_event_satisfy(arts_guid_t event_guid, arts_guid_t data_guid);

/**
 * @brief Signal an event slot.
 *
 * Slots: ARTS_EVENT_LATCH_DECR_SLOT (decrement counter, optionally
 * carrying a data GUID), ARTS_EVENT_LATCH_INCR_SLOT (increment counter;
 * not allowed for events with @c multiple_fire=true).  Events fire when
 * curr_latch reaches 0.
 *
 * Cross-rank: the call is forwarded to the event's home rank via
 * MSG_EVENT_SATISFY_SLOT.  Callers MUST NOT attempt to
 * inspect fire state locally (no public API exposes it).
 */
void arts_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                             uint32_t slot);

/**
 * @brief Supply a dependency slot on an EDT directly (OCR-standard).
 *
 * Writes @p data_guid / @p mode into @p edt_guid's @p slot and decrements its
 * pending-dependency count, scheduling the EDT once the last dependency
 * lands.  Home-routed: the home rank's handler does the work (forwarded via
 * MSG_EDT_SATISFY_SLOT when @p edt_guid is remote).  @p ptr / @p size carry an
 * inline payload for @c DB_MODE_PTR delivery (otherwise NULL / 0).
 */
void arts_edt_satisfy_slot(arts_guid_t edt_guid, uint32_t slot,
                           arts_guid_t data_guid, arts_db_access_mode_t mode,
                           void *ptr, unsigned int size);

/** Deprecated alias of @c arts_edt_satisfy_slot (backward-compat). */
static inline void arts_signal_edt(arts_guid_t edt_guid, uint32_t slot,
                                   arts_guid_t db, arts_db_access_mode_t mode,
                                   void *ptr, unsigned int size) {
  arts_edt_satisfy_slot(edt_guid, slot, db, mode, ptr, size);
}

/**
 * @brief Register a dependent on an event source (OCR-standard).
 *
 * Entity-specific counterpart of @c arts_event_satisfy_slot: when @p source
 * (an event) fires, its data is delivered to @p destination's @p slot with
 * @p mode.  Home-routed (forwarded via MSG_EVENT_ADD_DEPENDENCE when @p source
 * is remote).  @c arts_add_dependence dispatches here for an event source.
 */
void arts_event_add_dependence(arts_guid_t source, arts_guid_t destination,
                               uint32_t slot, arts_db_access_mode_t mode);

/**
 * @brief Release a generic event.
 *
 * Removes the route_table entry via atomic claim-and-NULL.  The same GUID
 * may be re-created afterward (the destroyed slot is indistinguishable
 * from an uninitialized slot — a pattern shared with coherent ARTS_DB
 * DataBlocks). Any in-flight satisfies / add_dependences for this GUID are
 * routed through the OoO queue automatically.
 */
void arts_event_destroy(arts_guid_t guid);

/**
 * @brief Block the calling EDT until a FINISH event drains (bulk sync).
 *
 * Releases the finish event's creator-token, releases the caller's acquired
 * DBs, context-switches into the scheduler until the event fires and
 * auto-destroys, then reacquires DBs and resumes. Intended for the carts
 * compiler fallback; idiomatic code uses a continuation (arts_add_dependence
 * on the finish event) instead. Only meaningful for finish (auto_destroy)
 * events. Returns true on success.
 */
bool arts_event_wait(arts_guid_t event_guid);

/**
 * @brief Return the finish event the calling EDT currently belongs to.
 *
 * Returns the ambient finish-scope GUID inherited or joined by the running
 * EDT, or NULL_GUID if it belongs to no finish scope.
 */
arts_guid_t arts_current_finish_event(void);

/**
 * @brief Wire a source (event or DB) to a destination (EDT or event).
 *
 * OCR-standard convenience: a pure dispatcher over the entity-specific APIs,
 * branching on source/destination kind — no own wire or handler:
 *   - @c source is an event  → @c arts_event_add_dependence;
 *   - @c source is NULL / DB / a raw value, @c destination is an EDT
 *                            → @c arts_edt_satisfy_slot;
 *   - @c source is NULL / DB / a raw value, @c destination is an event
 *                            → @c arts_event_satisfy_slot.
 * The dep mode rides on the satisfy at fire time (stored in the event's
 * waiter metadata), so there is no separate up-front mode-set message.
 *
 * @param source      Source event or DB GUID (or @c NULL_GUID).
 * @param destination Destination EDT or event GUID.
 * @param slot        Dependency slot on the destination.
 * @param mode        Access mode for DB data (@c DB_MODE_RO, @c DB_MODE_RW,
 *                    etc.).
 */
void arts_add_dependence(arts_guid_t source, arts_guid_t destination,
                         uint32_t slot, arts_db_access_mode_t mode);

/** @} */ /* end event */

/* ========================================================================= */
/** @defgroup db DataBlocks (DB)
 *  Fixed-size data objects shared between tasks under the build-time memory
 *  model (see docs/programming_model/memory_model.rst).
 *  @{ */

/**
 * @brief Create a DataBlock of @p len bytes.
 *
 * A DataBlock (DB) is the main memory abstraction used in ARTS to share data
 * between tasks.  Access mode is specified at dependency time via
 * arts_add_dependence(), not at creation.
 *
 * @param[out] addr    Receives a pointer to the DB payload (uninitialized).
 *                     Set to @c NULL when @p flags includes
 *                     @c ARTS_DB_PROP_NO_ACQUIRE.
 * @param      len     Length in bytes.
 * @param      db_type Storage/coherence class (ARTS_DB, ARTS_DB_PIN,
 *                     ARTS_DB_GPU_PIN, ARTS_DB_GPU, ARTS_DB_CXL).
 * @param      flags   Property bits (@c ARTS_DB_PROP_NONE / @c
 *                     ARTS_DB_PROP_NO_ACQUIRE).
 * @param      hint    Advisory metadata.  @c hint->rank selects the home
 *                     node; ARTS_HINT_CURRENT_RANK = current node.  NULL =
 *                     no preference: the build's no-hint DB home policy
 *                     picks the rank (ARTS_NOHINT_DB_HOME: CREATOR default
 *                     — first-touch — or ROUNDROBIN).
 * @return GUID of the created DB.
 * @see arts_db_destroy
 */
arts_guid_t arts_db_create(void **addr, uint64_t len, arts_db_types_t db_type,
                           uint16_t flags, const arts_db_hint_t *hint);

/**
 * @brief Convenience wrapper: create a DataBlock with a pre-reserved GUID.
 *
 * Equivalent to setting @c hint->guid and calling @c arts_db_create; the
 * GUID must be local.  Returns the payload pointer directly (NULL when
 * @c ARTS_DB_PROP_NO_ACQUIRE is set).
 */
static inline void *arts_db_create_with_guid(arts_guid_t guid, uint64_t len,
                                             arts_db_types_t db_type,
                                             uint16_t flags,
                                             const arts_db_hint_t *hint) {
  arts_db_hint_t h = hint ? *hint : ARTS_DB_HINT_DEFAULTS;
  h.guid = guid;
  void *ptr = NULL;
  arts_db_create(&ptr, len, db_type, flags, &h);
  return ptr;
}

/**
 * @brief Release the auto-acquired WRITE access for a DataBlock.
 *
 * When an EDT creates a local DB, the runtime automatically holds WRITE
 * access (OCR RW semantics).  Call this to release that access early —
 * before the EDT function returns — so that consumer EDTs waiting on
 * the DB can proceed.
 *
 * This is required when an EDT creates DBs and then blocks inside its
 * body (e.g. via arts_event_wait), because the automatic release
 * in the EDT epilogue cannot run until the function returns.
 *
 * Calling this on a DB that was not auto-acquired (or was already
 * released) is a no-op.
 *
 * @param guid GUID of the DataBlock to release.
 * @param mode Access mode the DB was acquired/created with (DB_MODE_RW for
 *             created or written DBs, DB_MODE_RO for a read-only dep).
 */
void arts_db_release(arts_guid_t guid, arts_db_access_mode_t mode);

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

/** @} */ /* end db */

/* ========================================================================= */
/** @defgroup util Utility Functions
 *  Query runtime state and miscellaneous helpers.
 *  @{ */

/** @brief Return the GUID of the currently executing EDT. */
arts_guid_t arts_edt_get_current_guid();

/** @brief Return the rank of this node. */
unsigned int arts_get_current_rank();

/** @brief Return the total number of ranks. */
unsigned int arts_get_total_ranks();

/** @brief Return the worker-thread id on this node. */
unsigned int arts_get_current_worker();

/**
 * @brief Workers per rank on this node (excluding sender/receiver).
 *
 * Per-rank count.  Equivalent to the previous semantic of
 * @c arts_get_total_workers().
 */
unsigned int arts_get_workers_per_rank();

/**
 * @brief Total worker threads across all ranks.
 *
 * Equals @c arts_get_workers_per_rank() * @c arts_get_total_ranks().
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

/**
 * @brief GPUs visible to this rank (per-rank count).
 */
unsigned int arts_get_gpus_per_rank();

/** @brief Return a monotonic timestamp in nanoseconds. */
uint64_t arts_get_time_stamp();

/** @} */ /* end util */

#ifdef __cplusplus
}
#endif
#endif
