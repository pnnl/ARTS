/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.                                          **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0                            **
******************************************************************************/

/*
 * cdag_lock — implementation of the async fair RW lock described in
 * libs/include/internal/arts/memory/cdag_lock.h.
 *
 * Algorithm (condensed):
 *
 *   submit(req):
 *     acquire queue_lock
 *     if head == NULL:
 *       head = tail = new_gen(req.mode) with [req] as the only waiter
 *       return HEAD_IMMEDIATE                       // runnable now
 *     g = tail
 *     if !g.sealed and g.mode == req.mode and g.mode == RO:
 *       append req to g.waiters; active++; return HEAD_IMMEDIATE if g==head
 * else QUEUED else: g.sealed = true tail = new_gen(req.mode) with [req] as the
 * only waiter g.next = tail return QUEUED release queue_lock
 *
 *   release(req):
 *     g = req.gen
 *     if --g.active > 0: return
 *     acquire queue_lock
 *     head = g.next; if head == NULL: tail = NULL
 *     snapshot new head's request list into a local array
 *     release queue_lock
 *     free g   (gen nodes only — waiter nodes of g were consumed and freed
 *               on submit already: we free them lazily in gen_free)
 *     for each snapshotted request: on_advance(req, ctx)
 *
 * The snapshot step is essential: once we release queue_lock, an EDT that
 * was just dispatched via on_advance can run, release, and advance head
 * past new_head — which would free new_head's waiter nodes out from under
 * our dispatch loop. By snapshotting into a local array before releasing
 * the lock, we decouple dispatch iteration from the live waiter list.
 */

#include "arts/memory/cdag_lock.h"
#include "arts/system/print.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------ */
/* Internal types                                                            */
/* ------------------------------------------------------------------------ */

struct cdag_waiter_node_s {
  struct cdag_lock_request_s req;
  struct cdag_waiter_node_s *next;
};

struct cdag_gen_s {
  cdag_lock_mode_t mode;
  bool sealed;                  /* no new joiners after this is set */
  volatile unsigned int active; /* atomic: outstanding-not-yet-released */
  struct cdag_waiter_node_s *waiters_head;
  struct cdag_waiter_node_s *waiters_tail;
  unsigned int waiter_count; /* cached size for snapshot allocation */
  struct cdag_gen_s *next;
};

/* ------------------------------------------------------------------------ */
/* Mode classification                                                       */
/* ------------------------------------------------------------------------ */

cdag_lock_mode_t cdag_lock_mode_of(arts_db_access_mode_t mode) {
  /* EW and MEMSET are exclusive writers; everything else (RO, RW, VALUE,
   * PTR, LC_SYNC, LC_NO_COPY) is treated as RO for ordering purposes.
   * This matches the current frontier's `write` flag computed in db.c. */
  switch (mode) {
  case DB_MODE_EW:
  case DB_MODE_MEMSET:
    return CDAG_LOCK_MODE_EW;
  default:
    return CDAG_LOCK_MODE_RO;
  }
}

/* ------------------------------------------------------------------------ */
/* Allocation helpers                                                        */
/* ------------------------------------------------------------------------ */

static struct cdag_gen_s *gen_alloc(cdag_lock_mode_t mode) {
  struct cdag_gen_s *g =
      (struct cdag_gen_s *)arts_calloc(1, sizeof(struct cdag_gen_s));
  if (!g) {
    ARTS_ERROR("cdag_lock: gen allocation failed");
    return NULL;
  }
  g->mode = mode;
  g->sealed = false;
  g->active = 0;
  g->waiters_head = NULL;
  g->waiters_tail = NULL;
  g->waiter_count = 0;
  g->next = NULL;
  return g;
}

static struct cdag_waiter_node_s *
waiter_alloc(const struct cdag_lock_request_s *req) {
  struct cdag_waiter_node_s *n = (struct cdag_waiter_node_s *)arts_calloc(
      1, sizeof(struct cdag_waiter_node_s));
  if (!n) {
    ARTS_ERROR("cdag_lock: waiter allocation failed");
    return NULL;
  }
  n->req = *req;
  n->next = NULL;
  return n;
}

static void gen_append_waiter(struct cdag_gen_s *g,
                              struct cdag_waiter_node_s *w) {
  if (g->waiters_tail) {
    g->waiters_tail->next = w;
  } else {
    g->waiters_head = w;
  }
  g->waiters_tail = w;
  g->waiter_count++;
}

static void gen_free(struct cdag_gen_s *g) {
  if (!g) {
    return;
  }
  struct cdag_waiter_node_s *w = g->waiters_head;
  while (w) {
    struct cdag_waiter_node_s *nxt = w->next;
    arts_free(w);
    w = nxt;
  }
  arts_free(g);
}

/* ------------------------------------------------------------------------ */
/* Lifecycle                                                                 */
/* ------------------------------------------------------------------------ */

void cdag_lock_init(struct cdag_lock_s *lock) {
  lock->queue_lock = 0;
  lock->head = NULL;
  lock->tail = NULL;
}

void cdag_lock_destroy(struct cdag_lock_s *lock) {
  struct cdag_gen_s *g = lock->head;
  while (g) {
    struct cdag_gen_s *nxt = g->next;
    gen_free(g);
    g = nxt;
  }
  lock->head = NULL;
  lock->tail = NULL;
}

struct cdag_lock_s *cdag_lock_new(void) {
  struct cdag_lock_s *lock =
      (struct cdag_lock_s *)arts_calloc(1, sizeof(struct cdag_lock_s));
  if (!lock) {
    ARTS_ERROR("cdag_lock: lock allocation failed");
    return NULL;
  }
  cdag_lock_init(lock);
  return lock;
}

void cdag_lock_free(struct cdag_lock_s *lock) {
  if (!lock) {
    return;
  }
  cdag_lock_destroy(lock);
  arts_free(lock);
}

/* ------------------------------------------------------------------------ */
/* Core operations                                                           */
/* ------------------------------------------------------------------------ */

enum cdag_submit_result
cdag_lock_submit(struct cdag_lock_s *lock,
                 const struct cdag_lock_request_s *req) {
  cdag_lock_mode_t mode = cdag_lock_mode_of(req->mode);

  /* Allocate waiter BEFORE taking the lock — never allocate under the
   * critical section. */
  struct cdag_waiter_node_s *w = waiter_alloc(req);
  if (!w) {
    return CDAG_SUBMIT_QUEUED;
  }

  /* Pre-allocate a fresh generation too; we'll free it if we end up
   * joining an existing one. This keeps allocation out of the critical
   * section. */
  struct cdag_gen_s *preallocated = gen_alloc(mode);
  if (!preallocated) {
    arts_free(w);
    return CDAG_SUBMIT_QUEUED;
  }

  arts_lock(&lock->queue_lock);

  enum cdag_submit_result result;

  if (lock->head == NULL) {
    /* Empty lock: first submit creates the head generation. */
    struct cdag_gen_s *g = preallocated;
    preallocated = NULL;
    gen_append_waiter(g, w);
    g->active = 1;
    lock->head = g;
    lock->tail = g;
    result = CDAG_SUBMIT_HEAD_IMMEDIATE;
  } else {
    struct cdag_gen_s *tail = lock->tail;

    /* Can the request join the tail generation?
     *   - tail must not be sealed
     *   - tail.mode must match req.mode
     *   - tail.mode must be RO (EW generations are singletons by design)
     *
     * Phase fairness: if a writer sealed an RO tail earlier and created
     * an EW tail after it, a new RO arriving now sees tail.mode == EW,
     * fails the match, and will create a new RO generation AFTER the EW
     * — bounding the writer's wait to the current RO batch only
     * (Brandenburg–Anderson B_w <= R+1).
     */
    bool can_join =
        !tail->sealed && tail->mode == mode && mode == CDAG_LOCK_MODE_RO;
    if (can_join) {
      gen_append_waiter(tail, w);
      arts_atomic_add(&tail->active, 1U);
      bool is_head = (tail == lock->head);
      result = is_head ? CDAG_SUBMIT_HEAD_IMMEDIATE : CDAG_SUBMIT_QUEUED;
    } else {
      /* Seal the current tail and create a new generation. */
      tail->sealed = true;
      struct cdag_gen_s *g = preallocated;
      preallocated = NULL;
      gen_append_waiter(g, w);
      g->active = 1;
      tail->next = g;
      lock->tail = g;
      result = CDAG_SUBMIT_QUEUED;
    }
  }

  arts_unlock(&lock->queue_lock);

  /* Free any unused preallocation outside the critical section. */
  if (preallocated) {
    /* preallocated never had waiters appended, so gen_free just frees it. */
    arts_free(preallocated);
  }

  return result;
}

void cdag_lock_release(struct cdag_lock_s *lock, cdag_on_advance_cb on_advance,
                       void *ctx) {
  /* The head's active count is decremented under queue_lock to serialize
   * with concurrent submits. This preserves the invariant that "active
   * reaches zero ⇒ no new joins can increment it back" — without this,
   * a concurrent submit between atomic-sub-to-0 and the advance step
   * could revive the generation and we'd free it with live waiters. */

  struct cdag_lock_request_s *snapshot = NULL;
  unsigned int snapshot_n = 0;

  arts_lock(&lock->queue_lock);

  struct cdag_gen_s *gen = lock->head;
  if (!gen) {
    /* Programming error: release with no head. */
    ARTS_ERROR("cdag_lock: release with empty queue");
    arts_unlock(&lock->queue_lock);
    return;
  }

  unsigned int remaining = arts_atomic_sub(&gen->active, 1U);
  if (remaining > 0) {
    /* More participants still in flight in this generation. */
    arts_unlock(&lock->queue_lock);
    return;
  }

  /* We are the last releaser for this generation — advance head. */
  lock->head = gen->next;
  if (lock->head == NULL) {
    lock->tail = NULL;
  }

  struct cdag_gen_s *new_head = lock->head;
  if (new_head && new_head->waiter_count > 0) {
    snapshot_n = new_head->waiter_count;
    snapshot = (struct cdag_lock_request_s *)arts_calloc(
        snapshot_n, sizeof(struct cdag_lock_request_s));
    if (!snapshot) {
      ARTS_ERROR("cdag_lock: snapshot allocation failed (n=%u)", snapshot_n);
      arts_unlock(&lock->queue_lock);
      gen_free(gen);
      return;
    }
    unsigned int i = 0;
    struct cdag_waiter_node_s *w = new_head->waiters_head;
    while (w && i < snapshot_n) {
      snapshot[i++] = w->req;
      w = w->next;
    }
    snapshot_n = i;
  }

  arts_unlock(&lock->queue_lock);

  /* Free the drained generation. */
  gen_free(gen);

  /* Dispatch callbacks for the new head's waiters, outside the lock. */
  if (on_advance && snapshot) {
    for (unsigned int i = 0; i < snapshot_n; i++) {
      on_advance(&snapshot[i], ctx);
    }
  }
  if (snapshot) {
    arts_free(snapshot);
  }
}

/* ------------------------------------------------------------------------ */
/* Introspection                                                             */
/* ------------------------------------------------------------------------ */

bool cdag_lock_iter_head_ranks(struct cdag_lock_s *lock,
                               void (*visit)(unsigned int rank, void *ctx),
                               void *ctx) {
  arts_lock(&lock->queue_lock);
  struct cdag_gen_s *h = lock->head;
  if (!h) {
    arts_unlock(&lock->queue_lock);
    return false;
  }
  struct cdag_waiter_node_s *w = h->waiters_head;
  while (w) {
    visit(w->req.origin_rank, ctx);
    w = w->next;
  }
  arts_unlock(&lock->queue_lock);
  return true;
}

bool cdag_lock_has_head(struct cdag_lock_s *lock) {
  arts_lock(&lock->queue_lock);
  bool has = (lock->head != NULL);
  arts_unlock(&lock->queue_lock);
  return has;
}
