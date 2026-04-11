/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.                                          **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0                            **
******************************************************************************/
#ifndef ARTS_MEMORY_CDAG_LOCK_H
#define ARTS_MEMORY_CDAG_LOCK_H

/*
 * cdag_lock — async, fair, non-blocking reader-writer lock for ARTS DataBlocks.
 *
 * This module replaces the legacy "frontier" subsystem with a cleaner
 * realization of the exact same abstraction: an owner-serialized FIFO
 * generation queue with phase-fair alternation and queue-of-waiters
 * callback delivery.
 *
 * Design references:
 *   - Linux qrwlock (Waiman Long, 2013) — queued task-fair waiter list
 *   - Brandenburg & Anderson 2010 — phase-fair policy (B_w <= R+1 bound)
 *   - Tokio RwLock — waker-based async FIFO delivery
 *   - Legion/Realm event-based async regions
 *
 * Invariants:
 *   1. Exclusivity: at any instant, only the head generation delivers
 *      callbacks. Head is either RO (multiple concurrent) or EW (singleton).
 *   2. FIFO: head advances strictly in generation-creation order.
 *   3. Starvation-free: every submitted request eventually receives its
 *      callback because every generation eventually drains.
 *   4. Phase-fair: a writer arriving at an OPEN RO generation seals it, so
 *      no further reader can join — bounding the writer's wait to the
 *      current RO batch only.
 *   5. Non-blocking submit: cdag_lock_submit takes a short spinlock for
 *      O(1) pointer work; no callbacks, messages, or I/O happen inside the
 *      critical section.
 *
 * Separation of concerns:
 *   - This module manages ORDERING only. Physical data movement (route
 *     table, remote DB send) is the caller's responsibility via on_advance.
 */

#include "arts/memory/db.h"
#include "arts/runtime_types.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Generation mode — RO batch or EW singleton. Every arts_db_access_mode_t
 * maps to exactly one of these. */
typedef enum {
  CDAG_LOCK_MODE_RO,
  CDAG_LOCK_MODE_EW,
} cdag_lock_mode_t;

/* Per-request metadata. Caller fills this in at submit time; cdag_lock
 * stores a copy internally and returns it via on_advance callback when the
 * request becomes runnable. */
struct cdag_lock_request_s {
  struct arts_edt_s *edt;
  arts_guid_t edt_guid;
  unsigned int origin_rank;
  unsigned int slot;
  arts_db_access_mode_t mode;
};

/* Submit outcome. */
enum cdag_submit_result {
  /* Request is runnable NOW: it landed in the head generation.
   * Caller should invoke its immediate-delivery path for this request
   * (typically: decrement EDT depc_needed). Do NOT wait for a later
   * on_advance callback — it will not come for this request. */
  CDAG_SUBMIT_HEAD_IMMEDIATE,
  /* Request is queued in a non-head generation. Caller does nothing
   * further for this request. The on_advance callback will fire later
   * (from cdag_lock_release) when the request's generation becomes head. */
  CDAG_SUBMIT_QUEUED,
};

/* Opaque generation handle returned by submit and passed to release.
 * Internals are defined in cdag_lock.c. */
struct cdag_gen_s;

/* The lock itself. Fields are exposed so a cdag_lock_s can be embedded in
 * the DB descriptor, but callers should treat them as opaque. */
struct cdag_lock_s {
  volatile unsigned int queue_lock; /* test-and-set spinlock, bit 0 only */
  struct cdag_gen_s *head;
  struct cdag_gen_s *tail;
};

/* -------- Lifecycle -------- */

/* Initialize an embedded lock in place. */
void cdag_lock_init(struct cdag_lock_s *lock);

/* Tear down an embedded lock. Frees any leftover generations defensively;
 * asserts nothing (caller may know the lock is quiescent). */
void cdag_lock_destroy(struct cdag_lock_s *lock);

/* Allocate a lock on the heap. Used for the drop-in replacement of
 * arts_new_db_list() in db.c. */
struct cdag_lock_s *cdag_lock_new(void);

/* Free a heap-allocated lock. Calls cdag_lock_destroy internally. */
void cdag_lock_free(struct cdag_lock_s *lock);

/* -------- Core operations -------- */

/* Non-blocking submit. Stores a copy of *req in an internal waiter node.
 *
 * Returns CDAG_SUBMIT_HEAD_IMMEDIATE if this request is runnable now
 * (it joined or started the head generation). Returns CDAG_SUBMIT_QUEUED
 * if the request is waiting in a non-head generation.
 *
 * The caller does NOT track any per-request handle for release; see
 * cdag_lock_release for the matching contract. */
enum cdag_submit_result cdag_lock_submit(struct cdag_lock_s *lock,
                                         const struct cdag_lock_request_s *req);

/* Callback invoked by cdag_lock_release when a queued request becomes
 * runnable as its generation advances to head. The callback is invoked
 * once per newly-runnable request, outside the queue_lock, so it may
 * perform arbitrary work (message sends, EDT signaling, etc.). */
typedef void (*cdag_on_advance_cb)(const struct cdag_lock_request_s *req,
                                   void *ctx);

/* Release a slot from the current head generation. Every successful
 * submit (whether HEAD_IMMEDIATE or QUEUED) must be matched by exactly
 * one cdag_lock_release call, made from the context where that
 * request's work has finished. Because an EDT only runs after it has
 * been dispatched into the head generation (either via HEAD_IMMEDIATE
 * return or via a subsequent on_advance callback), the head at the time
 * of release is guaranteed to be the same generation the EDT was
 * dispatched into — so the caller does not need a per-request gen
 * handle.
 *
 * If this call decrements the head's active count to zero, the head
 * pointer advances to the next generation and on_advance is invoked
 * once per newly-runnable request in the new head. on_advance may be
 * NULL to drop callbacks silently. */
void cdag_lock_release(struct cdag_lock_s *lock, cdag_on_advance_cb on_advance,
                       void *ctx);

/* -------- Introspection helpers (used by remote signaling paths) -------- */

/* Visit every request's origin_rank in the head generation. Returns false
 * if head is empty. The visit callback is invoked with the internal lock
 * held, so it must not reenter cdag_lock APIs. */
bool cdag_lock_iter_head_ranks(struct cdag_lock_s *lock,
                               void (*visit)(unsigned int rank, void *ctx),
                               void *ctx);

/* Returns true if there is a head generation (i.e. at least one request
 * is currently delivered). */
bool cdag_lock_has_head(struct cdag_lock_s *lock);

/* Helper: classify an ARTS access mode as RO or EW for lock purposes. */
cdag_lock_mode_t cdag_lock_mode_of(arts_db_access_mode_t mode);

/* Shared dispatch callback used by both db.c and handler.c as the
 * on_advance hook for cdag_lock_release. Handles local EDT signaling
 * (decrement depc_needed, set depv[slot].ptr) and remote DB delivery
 * (arts_remote_db_full_send_now). ctx is the arts_db_s pointer. */
void arts_cdag_dispatch_cb(const struct cdag_lock_request_s *req, void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_CDAG_LOCK_H */
