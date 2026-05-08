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
  ARTS_EDT = 0,  /**< Event-Driven Task (CPU and GPU share this tag). */
  ARTS_EVENT,    /**< Latch-based synchronization event. */
  ARTS_EPOCH,    /**< Termination-detection epoch. */
  ARTS_DB,       /**< DataBlock (all subtypes share this tag). */
  ARTS_LAST_TYPE /**< Sentinel — first invalid type value (= 4). */
} arts_type_t;

/**
 * @brief DataBlock access mode (per-dependency, stored in @c
 * arts_edt_dep_t.mode).
 *
 * Specifies how an EDT accesses a datablock dependency.  Set via
 * @c arts_add_dependence(), not at DB creation.
 */
typedef enum {
  DB_MODE_NULL = 0, /**< Unset / placeholder. */
  DB_MODE_RO,       /**< Read-Only (shared readers, no writeback). */
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
 * Specifies the storage and coherence class of a DataBlock.  All subtypes
 * share the same @c ARTS_DB tag in the GUID; the subtype is carried inside
 * the DB descriptor.  Naming convention: ARTS_DB_<storage>_<coherence>.
 * Coherence suffix = RC | LC | PIN.  Storage prefix omitted = regular DRAM.
 */
typedef enum {
  ARTS_DB_RC = 0, /**< Release Consistency (regular DRAM, distributed RC). */
  ARTS_DB_PIN,    /**< Node-pinned regular DRAM, no DB-level coherence. */
  ARTS_DB_GPU_PIN, /**< GPU staging (host pinned + per-device replica). */
  ARTS_DB_GPU_LC,  /**< GPU staging, Location Consistency (multi-GPU + reduce).
                    */
  ARTS_DB_CXL_LC,  /**< CXL shared, Location Consistency (compiled w/ CXL). */
} arts_db_types_t;

/* @c ARTS_DB_DEFAULT is the experiment-wide DB kind every benchmark uses
 * — its concrete value is decided by CMake (see ARTS_DB_DEFAULT_KIND in
 * the top-level CMakeLists.txt).  Locally we ship RC so the build works
 * without CXL hardware; flip the cmake variable to ARTS_DB_CXL_LC (or
 * any other kind) to retarget every benchmark in one place. */
#ifndef ARTS_DB_DEFAULT
#define ARTS_DB_DEFAULT ARTS_DB_RC
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
 * All EDTs share the single @c ARTS_EDT tag in the GUID; the subtype is
 * carried inside the EDT descriptor.  CPU and GPU EDTs differ in scheduling
 * and execution but share the same GUID type.
 */
typedef enum {
  ARTS_EDT_CPU = 0, /**< CPU EDT (standard). */
  ARTS_EDT_GPU = 1, /**< GPU EDT (CUDA kernel or library host function). */
} arts_edt_types_t;

/* @c ARTS_EDT_DEFAULT mirrors the @c ARTS_DB_DEFAULT pattern: the
 * experiment-wide EDT kind every benchmark assumes, decided by CMake (see
 * ARTS_EDT_DEFAULT_KIND in the top-level CMakeLists.txt).  Locally we ship
 * ARTS_EDT_CPU; flip the cmake variable to ARTS_EDT_GPU to retarget every
 * benchmark in one place. */
#ifndef ARTS_EDT_DEFAULT
#define ARTS_EDT_DEFAULT ARTS_EDT_CPU
#endif

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
 *  with same input → same GUID" invariant while preserving ARTS's
 *  round-robin home distribution policy. */
#define ARTS_HINT_ROUND_ROBIN ((unsigned int)-2)

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

/** Hint passed to @c arts_edt_create.  Optional fields collapse the legacy
 *  six EDT-create variants into a single entry point:
 *    - @c rank   selects the home node (default current rank).
 *    - @c edt_id is the compiler-assigned profiling id (default 0).
 *    - @c guid   when non-NULL_GUID pre-reserves the EDT GUID; the home
 *                rank is then taken from that GUID and @c rank is ignored.
 *    - @c epoch  when non-NULL_GUID assigns the EDT to that epoch; otherwise
 *                the runtime uses the caller's current epoch (if any). */
typedef struct {
  /** Target node rank.  ARTS_HINT_CURRENT_RANK = current node (default). */
  unsigned int rank;
  /** Compiler-assigned profiling identifier.  0 = disabled. */
  uint64_t edt_id;
  /** Pre-reserved GUID.  NULL_GUID = auto-allocate (default). */
  arts_guid_t guid;
  /** Owning epoch.  NULL_GUID = inherit caller's current epoch (default). */
  arts_guid_t epoch;
} arts_edt_hint_t;

#define ARTS_EDT_HINT_DEFAULTS                                                 \
  ((arts_edt_hint_t){.rank = ARTS_HINT_CURRENT_RANK,                           \
                     .edt_id = 0,                                              \
                     .guid = NULL_GUID,                                        \
                     .epoch = NULL_GUID})

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
} arts_db_hint_t;

#define ARTS_DB_HINT_DEFAULTS                                                  \
  ((arts_db_hint_t){.rank = ARTS_HINT_CURRENT_RANK,                            \
                    .access_offset = 0,                                        \
                    .access_size = UINT64_MAX,                                 \
                    .guid = NULL_GUID})

/** Hint passed to @c arts_db_put / @c arts_db_get.
 *
 *  Use @c ARTS_DB_OP_HINT_DEFAULTS to obtain a default-initialized value;
 *  pass @c NULL to the put/get call to use the defaults directly. */
typedef struct {
  /** Target rank for the operation.  ARTS_HINT_CURRENT_RANK = current node
   *  (default).  Other values: specific rank where the put/get is applied. */
  unsigned int rank;
  /** Epoch the operation belongs to.  NULL_GUID = current epoch (default).
   *  When set, the put is associated with the given epoch and signals
   *  via the epoch (no edt_guid is needed; pass NULL_GUID for edt_guid). */
  arts_guid_t epoch;
} arts_db_op_hint_t;

#define ARTS_DB_OP_HINT_DEFAULTS                                               \
  ((arts_db_op_hint_t){.rank = ARTS_HINT_CURRENT_RANK, .epoch = NULL_GUID})
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
 * The EDT can call blocking operations like arts_epoch_wait().
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
 * Every ARTS event is a single generic type whose behavior is determined by
 * the hint passed at create time.  Defaults compose into the OCR ONCE_T
 * semantic (latch=1, fire-and-destroy).  All other OCR flavors (IDEM,
 * STICKY, LATCH, COUNTED, CHANNEL) are realized by overriding individual
 * fields.  See docs/event-refactor/spec.md for the complete mapping.
 *  @{ */
typedef struct {
  /** Target node rank.  ARTS_HINT_CURRENT_RANK = current node (default). */
  unsigned int rank;
  /** Initial latch counter.  Default 1 (OCR ONCE/IDEM/STICKY); 0 for an
   *  immediately-firing event; LATCH events may use any signed integer
   *  (counter is decremented on DECR satisfies, incremented on INCR; fire
   *  occurs when the counter reaches zero). */
  int32_t latch;
  /** Deps consumed per fire round.  Default 1.  CHANNEL events spec-clamped
   *  to 1 (OCR 1.2 §B.5.2). */
  uint32_t nb_deps_required;
  /** Hard ceiling on total waiter registrations.  When the count reaches 0,
   *  the event is destroyed regardless of @c auto_destroy.  Default
   *  UINT32_MAX (effectively unlimited).  COUNTED uses params.nbDeps. */
  uint32_t max_nb_deps;
  /** If true, destroy the event after the terminal fire+drain.  Default
   *  true (OCR ONCE_T).  IDEM/STICKY set false. */
  bool auto_destroy;
  /** If false, a satisfy that would push curr_latch below zero raises
   *  ARTS_ERROR.  Default true.  STICKY sets false. */
  bool negative_latch_allowed;
  /** If true, the event re-fires whenever (latch<=0 && deps<=0); satisfies
   *  and deps are buffered in FIFO mpsc queues so producer-before-consumer
   *  ordering is preserved.  Default false.  CHANNEL sets true. */
  bool multiple_fire;
  /** Pre-reserved GUID.  NULL_GUID = auto-allocate (default).  When non-zero,
   *  the GUID's rank field is authoritative and overrides @c rank above. */
  arts_guid_t guid;
} arts_event_hint_t;

/** Internal helper — full-field designated initializer used by every
 *  specialization below.  Six positional args correspond to the fields
 *  that vary across event flavors (rank/guid stay at defaults). */
#define ARTS_EVENT_HINT_BASE(latch_, deps_, max_, ad_, neg_, mf_)              \
  ((arts_event_hint_t){.rank = ARTS_HINT_CURRENT_RANK,                         \
                       .latch = (latch_),                                      \
                       .nb_deps_required = (deps_),                            \
                       .max_nb_deps = (max_),                                  \
                       .auto_destroy = (ad_),                                  \
                       .negative_latch_allowed = (neg_),                       \
                       .multiple_fire = (mf_),                                 \
                       .guid = NULL_GUID})

/** OCR ONCE_T — fire-and-destroy.  Single satisfy fires + auto-destroys. */
#define ARTS_EVENT_HINT_ONCE                                                   \
  ARTS_EVENT_HINT_BASE(1, 1, UINT32_MAX, true, true, false)

/** Default hint == ONCE semantic. */
#define ARTS_EVENT_HINT_DEFAULTS ARTS_EVENT_HINT_ONCE

/** OCR IDEM_T — once-fire, persistent.  Late add_dependence delivers
 *  immediately from the stored data slot.  Subsequent satisfies are
 *  silent no-ops. */
#define ARTS_EVENT_HINT_IDEMPOTENT                                             \
  ARTS_EVENT_HINT_BASE(1, 1, UINT32_MAX, false, true, false)

/** OCR STICKY_T — like IDEM but over-satisfy (latch below zero) aborts. */
#define ARTS_EVENT_HINT_STICKY                                                 \
  ARTS_EVENT_HINT_BASE(1, 1, UINT32_MAX, false, false, false)

/** OCR LATCH_T — counter event.  Argument is the initial counter value
 *  (signed); negative values are legal and represent prefires.  Fire
 *  occurs when the counter reaches zero via DECR satisfies. */
#define ARTS_EVENT_HINT_LATCH(counter_init)                                    \
  ARTS_EVENT_HINT_BASE((counter_init), 1, UINT32_MAX, true, true, false)

/** OCR COUNTED_T — fire once after exactly @c nb_deps_max consumers have
 *  registered + satisfy has occurred.  Argument is the lifetime cap. */
#define ARTS_EVENT_HINT_COUNTED(nb_deps_max)                                   \
  ARTS_EVENT_HINT_BASE(1, 1, (nb_deps_max), true, true, false)

/** OCR CHANNEL_T — multi-fire FIFO event.  nbSat/nbDeps are spec-clamped
 *  to 1 (OCR 1.2 §B.5.2 limitation).  maxGen is implementation-driven —
 *  ARTS scales unbounded via mpsc linked lists. */
#define ARTS_EVENT_HINT_CHANNEL                                                \
  ARTS_EVENT_HINT_BASE(1, 1, UINT32_MAX, false, true, true)
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
 * @brief Set the config file path before calling arts_rt().
 *
 * When set, arts_config_load() reads from this path instead of the
 * default ARTS_CONFIG env var / arts.cfg fallback.
 *
 * @param path  Null-terminated file path.  NULL or "" clears the override.
 */
void arts_set_config_path(const char *path);

/**
 * @brief Inject config data as an in-memory string before calling arts_rt().
 *
 * When set, arts_config_load() parses this string (same INI format as
 * arts.cfg) instead of opening a file.  This enables self-contained binaries
 * that embed their runtime configuration at compile time.
 *
 * @param data  Null-terminated config string.  NULL or "" clears the override.
 */
void arts_set_config_data(const char *data);

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
arts_guid_t arts_guid_reserve(arts_type_t type, unsigned int rank);

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
 * @param route Target node rank.  Special values:
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
arts_guid_t arts_guid_reserve_range(arts_type_t type, unsigned int size,
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
 * rank, pre-reserved GUID, owning epoch, profiling id) are carried in the
 * hint struct.  Pass @c NULL for ARTS_EDT_HINT_DEFAULTS, which auto-allocates
 * a GUID on the current rank and inherits the caller's current epoch.
 *
 * @param func_ptr Function to execute.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of @p paramc uint64_t values copied into the closure.
 * @param depc     Number of dependency slots.
 * @param hint     Advisory metadata (rank, edt_id, guid, epoch).  NULL =
 *                 defaults.
 * @return GUID of the newly created EDT.
 * @see arts_add_dependence, arts_edt_destroy, arts_epoch_create
 */
arts_guid_t arts_edt_create(arts_edt_t func_ptr, uint32_t paramc,
                            const uint64_t *paramv, uint32_t depc,
                            const arts_edt_hint_t *hint);

/**
 * @brief Destroy an EDT and remove its GUID from the routing table.
 *
 * EDTs are automatically destroyed after they finish running; call this only
 * to cancel an EDT that has not yet fired.
 *
 * @param guid GUID of the EDT to destroy.
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
 * forwarded via ARTS_REMOTE_EVENT_MOVE_MSG, and concurrent installs with
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
 * ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG.  Callers MUST NOT attempt to
 * inspect fire state locally (no public API exposes it).
 */
void arts_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                             uint32_t slot);

/**
 * @brief Release a generic event.
 *
 * Removes the route_table entry via atomic claim-and-NULL.  The same GUID
 * may be re-created afterward (the destroyed slot is indistinguishable
 * from an uninitialized slot — a pattern shared with RC DataBlocks).
 * Any in-flight satisfies / add_dependences for this GUID are routed
 * through the OoO queue automatically.
 */
void arts_event_destroy(arts_guid_t guid);

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
 * @param mode        Access mode for DB data (@c DB_MODE_RO, @c DB_MODE_RW,
 *                    etc.).
 */
void arts_add_dependence(arts_guid_t source, arts_guid_t destination,
                         uint32_t slot, arts_db_access_mode_t mode);

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
 * arts_add_dependence(), not at creation.
 *
 * @param[out] addr    Receives a pointer to the DB payload (uninitialized).
 *                     Set to @c NULL when @p flags includes
 *                     @c ARTS_DB_PROP_NO_ACQUIRE.
 * @param      len     Length in bytes.
 * @param      db_type Storage/coherence class (RC, PIN, GPU_PIN, GPU_LC,
 *                     CXL_LC).
 * @param      flags   Property bits (@c ARTS_DB_PROP_NONE / @c
 *                     ARTS_DB_PROP_NO_ACQUIRE).
 * @param      hint    Advisory metadata.  @c hint->rank selects the target
 *                     node; NULL or ARTS_HINT_CURRENT_RANK = current node.
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
 * body (e.g. via arts_epoch_wait), because the automatic release
 * in the EDT epilogue cannot run until the function returns.
 *
 * Calling this on a DB that was not auto-acquired (or was already
 * released) is a no-op.
 *
 * @param guid GUID of the DataBlock to release.
 */
void arts_db_release(arts_guid_t guid);

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
 * @brief Write data into a DataBlock and signal an EDT (or epoch).
 *
 * Default behavior (@p hint == NULL or @c ARTS_DB_OP_HINT_DEFAULTS):
 * the write is routed to the DB's home rank (derived from @p db_guid) and
 * tracked under the current epoch.  Override via @p hint:
 *   - @c hint->rank: target a specific rank where the write is applied.
 *   - @c hint->epoch: associate the put with a non-current epoch; in this
 *     mode @p edt_guid may be @c NULL_GUID and @p slot is ignored — the
 *     operation signals via the epoch instead of an EDT slot.
 *
 * @param ptr      Source data.
 * @param edt_guid EDT to signal upon completion (or @c NULL_GUID when the
 *                 put is epoch-driven).
 * @param db_guid  Target DataBlock.
 * @param slot     EDT dependency slot to satisfy (ignored when epoch-driven).
 * @param offset   Byte offset within the DB.
 * @param len      Number of bytes to write.
 * @param hint     Optional hint; pass @c NULL for defaults.
 */
void arts_db_put(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                 unsigned int slot, unsigned int offset, unsigned int len,
                 const arts_db_op_hint_t *hint);

/**
 * @brief Read data from a DataBlock and deliver it to an EDT.
 *
 * A copy of @p len bytes at @p offset is delivered to @p edt_guid as a
 * pointer dependency.  Default behavior (@p hint == NULL): the read is
 * routed to the DB's home rank (derived from @p db_guid).  Override
 * @c hint->rank to read from a specific rank.  The @c epoch field of
 * the hint is currently unused for reads.
 *
 * @param edt_guid Destination EDT.
 * @param db_guid  Source DataBlock.
 * @param slot     EDT dependency slot.
 * @param offset   Byte offset within the DB.
 * @param len      Number of bytes to read.
 * @param hint     Optional hint; pass @c NULL for defaults.
 */
void arts_db_get(arts_guid_t edt_guid, arts_guid_t db_guid, unsigned int slot,
                 unsigned int offset, unsigned int len,
                 const arts_db_op_hint_t *hint);

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
arts_guid_t arts_epoch_get_current_guid();

/**
 * @brief Assign an EDT to a specific epoch.
 *
 * The caller must ensure the EDT has not yet run and the epoch is still live.
 *
 * @param edt_guid   EDT to assign.
 * @param epoch_guid Target epoch.
 */
void arts_epoch_add_edt(arts_guid_t edt_guid, arts_guid_t epoch_guid);

/**
 * @brief Create an epoch without starting it.
 *
 * Use arts_epoch_start() to begin the epoch later. Any EDTs created by the
 * currently running EDT (after the epoch is started) will belong to this
 * epoch. When the epoch completes, @p finish_edt_guid is signaled at @p slot
 * with the number of EDTs, buffer ops, get/puts, etc. executed.
 *
 * @param rank            Source node rank.
 * @param finish_edt_guid EDT to signal when the epoch finishes.
 * @param slot            Dependency slot for the epoch summary.
 * @return GUID of the new epoch.
 * @see arts_epoch_start, arts_epoch_wait
 */
arts_guid_t arts_epoch_create(unsigned int rank, arts_guid_t finish_edt_guid,
                              unsigned int slot);

/**
 * @brief Start an epoch previously created with arts_epoch_create().
 *
 * @param epoch_guid Epoch GUID.
 */
void arts_epoch_start(arts_guid_t epoch_guid);

/**
 * @brief Block until @p epoch_guid finishes.
 *
 * The calling thread runs another scheduling round while waiting.
 * Only valid from the EDT that created the epoch.
 *
 * @param epoch_guid Epoch to wait for.
 * @return @c true on success.
 */
bool arts_epoch_wait(arts_guid_t epoch_guid);

/** @} */ /* end epoch */

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

/** @brief Return the number of GPUs per node. */
unsigned int arts_get_total_gpus();

/** @brief Return a monotonic timestamp in nanoseconds. */
uint64_t arts_get_time_stamp();

/** @} */ /* end util */

#ifdef __cplusplus
}
#endif
#endif
