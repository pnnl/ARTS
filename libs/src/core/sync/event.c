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

/*
 * arts_event_s redesign — see docs/event-refactor/spec.md.
 *
 * Single struct, hint-discriminated.  Two storage flavors share a union:
 *
 *   - simple (non-CHANNEL): one data slot + Treiber stack of pending deps.
 *     ONCE / IDEM / STICKY / COUNTED / LATCH all use this branch.  fire is
 *     gated by a fired flag CAS; late add_dependence after fire reads
 *     simple.data and delivers immediately.
 *
 *   - channel (CHANNEL only): two mpsc FIFO queues (data_queue, dep_queue)
 *     + a single-flight drainer sentinel.  satisfy / addDep push their
 *     payload onto the matching queue and decrement the matching counter;
 *     the drainer fires generations until either queue is empty.
 *
 * Race analysis and lock-freedom argument: spec §3.6 / §4.
 */

#include "arts/sync/event.h"

#include "arts.h"
#include "arts/compute/edt.h"
#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/remote/handler.h"
#include "arts/runtime_state.h" /* arts_node_info, event_dep_pool */
#include "arts/sync/shared.h"   /* arts_shared_init */
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/lockfree_lifo.h" /* arts_lf_stack_init / drain */
#include "arts/utils/malloc.h"
#include "arts/utils/mpsc.h"        /* arts_mpsc_t */
#include "arts/utils/tiered_pool.h" /* arts_tiered_pool_release */

#include <assert.h>
#include <time.h>

extern ARTS_THREAD_LOCAL struct arts_edt_s *current_edt;

/* --- Hint snapshot helpers ----------------------------------------------- */

static inline arts_event_hint_t hint_or_defaults(const arts_event_hint_t *h) {
  return h ? *h : ARTS_EVENT_HINT_DEFAULTS;
}

/* --- Per-flavor node free helpers ---------------------------------------- */

/* Free a single mpsc node back to the per-rank dep pool.  CHANNEL queue
 * payload is intrusive: the node IS arts_event_dep_s for both data and
 * dep entries (data uses target=guid, kind=ARTS_NULL marker). */
static inline void event_node_free(arts_lf_link_t *node) {
  arts_tiered_pool_release(arts_node_info.event_dep_pool,
                           (struct arts_event_dep_s *)node);
}

/* --- shared_t deleter (called by route_table.c free_item) ---------------- */

static void event_deleter(void *p) {
  struct arts_event_s *e = (struct arts_event_s *)p;
  /* Drain any leftover queued nodes and return them to the per-rank pool.
   * shared_t.count == 0 by precondition: no producer can push after this
   * point (every producer must hold a lookup ref while pushing). */
  if (e->multiple_fire) {
    arts_lf_link_t *chain = arts_mpsc_drain_remaining(&e->channel.data_queue);
    while (chain) {
      arts_lf_link_t *next =
          atomic_load_explicit(&chain->next, memory_order_relaxed);
      event_node_free(chain);
      chain = next;
    }
    chain = arts_mpsc_drain_remaining(&e->channel.dep_queue);
    while (chain) {
      arts_lf_link_t *next =
          atomic_load_explicit(&chain->next, memory_order_relaxed);
      event_node_free(chain);
      chain = next;
    }
  } else {
    arts_lf_link_t *chain = arts_lf_stack_drain(&e->simple.deps_stack);
    while (chain) {
      arts_lf_link_t *next =
          atomic_load_explicit(&chain->next, memory_order_relaxed);
      event_node_free(chain);
      chain = next;
    }
  }
  arts_free(e);
}

/* External forwarder — see event.h for rationale. */
void arts_event_free_internal(struct arts_event_s *e) { event_deleter(e); }

/* --- Internal allocation / install --------------------------------------- */

static struct arts_event_s *event_alloc(const arts_event_hint_t *h) {
  struct arts_event_s *e = arts_calloc(1, sizeof(struct arts_event_s));
  if (!e) {
    return NULL;
  }
  arts_shared_init(&e->shared, event_deleter);
  e->header.type = ARTS_GUID_EVENT;
  e->header.size = sizeof(*e);
  e->init_latch = h->latch;
  e->init_nb_deps_required = h->nb_deps_required;
  e->max_nb_deps = h->max_nb_deps;
  e->auto_destroy = h->auto_destroy ? 1 : 0;
  e->negative_latch_allowed = h->negative_latch_allowed ? 1 : 0;
  e->multiple_fire = h->multiple_fire ? 1 : 0;

  atomic_store_explicit(&e->curr_latch, h->latch, memory_order_relaxed);
  /* nb_deps_left starts at init_nb_deps_required for CHANNEL (counts down
   * to <=0 to fire, then recharges).  For non-CHANNEL it is unused — fire
   * is gated by `fired` CAS, not by deps counter. */
  atomic_store_explicit(&e->nb_deps_left, (int32_t)h->nb_deps_required,
                        memory_order_relaxed);
  atomic_store_explicit(&e->max_deps_left, h->max_nb_deps,
                        memory_order_relaxed);
  atomic_store_explicit(&e->fired, false, memory_order_relaxed);

  if (h->multiple_fire) {
    arts_mpsc_init(&e->channel.data_queue);
    arts_mpsc_init(&e->channel.dep_queue);
    atomic_store_explicit(&e->channel.draining, 0, memory_order_relaxed);
  } else {
    e->simple.data = NULL_GUID;
    arts_lf_stack_init(&e->simple.deps_stack);
  }
  return e;
}

bool arts_event_create_internal(arts_guid_t *guid,
                                const arts_event_hint_t *h_in) {
  arts_event_hint_t h = hint_or_defaults(h_in);
  unsigned int rank = h.rank;
  if (rank == ARTS_HINT_CURRENT_RANK) {
    rank = arts_global_rank_id;
  }

  struct arts_event_s *event = event_alloc(&h);
  if (!event) {
    return false;
  }

  if (rank == arts_global_rank_id) {
    if (*guid) {
      /* add_item_race: install only if slot is empty.  On success the
       * route_item lock starts at (gen<<32)|1 (Task 4e — install also
       * counts as one existence ref). */
      if (!arts_route_table_add_item_race(event, *guid, rank, false)) {
        /* Another caller won the race; silent no-op. */
        event_deleter(event);
        return false;
      }
      arts_route_table_fire_oo(*guid, arts_out_of_order_handler);
    } else {
      *guid = arts_guid_create_for_rank(rank, ARTS_GUID_EVENT);
      arts_route_table_add_item(event, *guid, rank, false);
    }
    return true;
  }
  /* Cross-rank: forward as a marshaled buffer.  Receiver
   * arts_remote_handle_event_move performs add_item_race.  Discard the
   * local allocation since the remote will materialise its own copy. */
  arts_remote_memory_move(rank, *guid, event, sizeof(*event),
                          ARTS_REMOTE_EVENT_MOVE_MSG, event_deleter);
  return true;
}

arts_guid_t arts_event_create(const arts_event_hint_t *hint) {
  TIME_EVENT_CREATE_START();
  INCREMENT_NUM_EVENT_CREATE_BY(1);
  arts_event_hint_t h = hint_or_defaults(hint);
  arts_guid_t g = h.guid;
  if (g != NULL_GUID) {
    h.rank = arts_guid_get_rank(g);
  }
  bool ok = arts_event_create_internal(&g, &h);
  TIME_EVENT_CREATE_STOP();
  if (h.guid != NULL_GUID) {
    return ok ? g : NULL_GUID;
  }
  return g;
}

/* ── Event destroy ──────────────────────────────────────────────────── */

void arts_event_destroy(arts_guid_t guid) {
  unsigned int rank = arts_guid_get_rank(guid);
  if (rank != arts_global_rank_id) {
    arts_remote_event_destroy(guid);
    return;
  }
  arts_route_table_mark_delete(guid);
}

/* ── Signal one queued dep ─────────────────────────────────────────────
 * For CHANNEL the data argument comes from the matching data_queue pop;
 * for non-CHANNEL it is e->simple.data. */
static void event_signal_one(struct arts_event_dep_s *d, arts_guid_t data) {
  if (d->kind == ARTS_GUID_EDT) {
    internal_signal_edt(d->target, d->slot, data, d->mode, NULL, 0);
  } else if (d->kind == ARTS_GUID_EVENT) {
    arts_event_satisfy_slot(d->target, data, d->slot);
  }
}

/* Allocate and populate an mpsc node from the per-rank pool. */
static struct arts_event_dep_s *event_node_alloc(arts_guid_kind_t kind,
                                                 arts_guid_t target,
                                                 uint32_t slot,
                                                 arts_db_access_mode_t mode) {
  struct arts_event_dep_s *node =
      (struct arts_event_dep_s *)arts_tiered_pool_alloc(
          arts_node_info.event_dep_pool);
  node->kind = kind;
  node->target = target;
  node->slot = slot;
  node->mode = mode;
  return node;
}

/* ── Non-CHANNEL drain ───────────────────────────────────────────────── */

/*
 * drain_simple_chain — idempotent, multi-caller drain of simple.deps_stack
 * after fired==true.  Invoked by:
 *   (a) the unique satisfy thread that just CAS-set fired (drain_simple).
 *   (b) any addDep thread that pushed onto the stack and then observed
 *       fired==true (race rescue for spec §4.1 R3-R4: addDep's push lands
 *       *after* satisfy's reverse_drain finished).
 *
 * Concurrent callers self-serialise inside arts_lf_stack_reverse_drain's
 * `atomic_exchange(&head, NULL)` — only one caller per chain, others see
 * NULL and exit.  The outer loop catches pushes that landed during a
 * caller's iteration.
 */
static void drain_simple_chain(struct arts_event_s *e, arts_guid_t event_guid) {
  arts_guid_t data = e->simple.data;
  for (;;) {
    arts_lf_link_t *fifo = arts_lf_stack_reverse_drain(&e->simple.deps_stack);
    if (!fifo) {
      return;
    }
    while (fifo) {
      arts_lf_link_t *next =
          atomic_load_explicit(&fifo->next, memory_order_relaxed);
      struct arts_event_dep_s *dep = (struct arts_event_dep_s *)fifo;
      event_signal_one(dep, data);
      event_node_free(fifo);
      if (atomic_fetch_sub_explicit(&e->max_deps_left, 1u,
                                    memory_order_acq_rel) == 1u) {
        if (e->auto_destroy) {
          /* Free the rest of this chain — they'd be delivered to a
           * destroyed event.  Other concurrent drainers see the stack
           * empty (or get a fresh chain that will also short-circuit
           * here). */
          fifo = next;
          while (fifo) {
            next = atomic_load_explicit(&fifo->next, memory_order_relaxed);
            event_node_free(fifo);
            fifo = next;
          }
          arts_route_table_mark_delete(event_guid);
          return;
        }
      }
      fifo = next;
    }
  }
}

/*
 * drain_simple — satisfy-side single-fire dispatcher.  CAS fired 0→1
 * gates the unique fire winner; the winner runs drain_simple_chain.
 * Caller has already written simple.data (if any); the CAS release
 * publishes the data store to late binders.
 */
static void drain_simple(struct arts_event_s *e, arts_guid_t event_guid) {
  int32_t latch = atomic_load_explicit(&e->curr_latch, memory_order_acquire);
  if (latch > 0) {
    return;
  }
  if (latch < 0 && !e->negative_latch_allowed) {
    ARTS_ERROR("negative latch on event with negative_latch_allowed=false");
  }
  bool fexp = false;
  if (!atomic_compare_exchange_strong_explicit(
          &e->fired, &fexp, true, memory_order_acq_rel, memory_order_acquire)) {
    return; /* another thread already won the single fire */
  }
  drain_simple_chain(e, event_guid);
}

/* ── CHANNEL drain (lock-free, single-flight via `draining` sentinel) ── */

static void try_drain_channel(struct arts_event_s *e, arts_guid_t event_guid) {
  /* Outer rescue loop: re-check the fire condition after we release
   * `draining` because a concurrent push may have arrived in the
   * window between our last queue-pop and the sentinel-clear. */
  for (;;) {
    int32_t latch = atomic_load_explicit(&e->curr_latch, memory_order_acquire);
    int32_t deps = atomic_load_explicit(&e->nb_deps_left, memory_order_acquire);
    if (latch > 0 || deps > 0) {
      return; /* condition not met */
    }
    if (atomic_load_explicit(&e->channel.draining, memory_order_acquire) == 1) {
      return; /* another thread already drains */
    }
    uint8_t expected = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &e->channel.draining, &expected, (uint8_t)1, memory_order_acq_rel,
            memory_order_acquire)) {
      return; /* lost the CAS race */
    }
    /* Inner fire loop: keep firing while both counters are <= 0. */
    while (atomic_load_explicit(&e->curr_latch, memory_order_acquire) <= 0 &&
           atomic_load_explicit(&e->nb_deps_left, memory_order_acquire) <= 0) {
      arts_lf_link_t *data_node = arts_mpsc_pop(&e->channel.data_queue);
      arts_lf_link_t *dep_node = arts_mpsc_pop(&e->channel.dep_queue);
      if (data_node == NULL || dep_node == NULL) {
        /* Queue empty mid-fire — should not happen if counters and
         * queues are in lockstep, but bail safely. */
        if (data_node) {
          event_node_free(data_node);
        }
        if (dep_node) {
          event_node_free(dep_node);
        }
        break;
      }
      /* Data node carries the satisfy data in `target` (kind == ARTS_NULL
       * marker). */
      arts_guid_t data = ((struct arts_event_dep_s *)data_node)->target;
      struct arts_event_dep_s *dep = (struct arts_event_dep_s *)dep_node;
      event_signal_one(dep, data);
      event_node_free(data_node);
      event_node_free(dep_node);
      /* Recharge counters: each fire consumed init_latch satisfies and
       * init_nb_deps_required deps. */
      atomic_fetch_add_explicit(&e->curr_latch, e->init_latch,
                                memory_order_acq_rel);
      atomic_fetch_add_explicit(&e->nb_deps_left,
                                (int32_t)e->init_nb_deps_required,
                                memory_order_acq_rel);
      /* Lifetime cap: max_deps_left counts total deliveries. */
      if (atomic_fetch_sub_explicit(&e->max_deps_left, 1u,
                                    memory_order_acq_rel) == 1u) {
        if (e->auto_destroy) {
          atomic_store_explicit(&e->channel.draining, 0, memory_order_release);
          arts_route_table_mark_delete(event_guid);
          return;
        }
      }
    }
    atomic_store_explicit(&e->channel.draining, 0, memory_order_release);
    /* Outer-while will re-check the fire condition for missed pushes. */
  }
}

/* ── arts_event_satisfy_slot ───────────────────────────────────────── */

void arts_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                             uint32_t slot) {
  TIME_EVENT_SIGNAL_START();
  INCREMENT_NUM_EVENT_SIGNAL_BY(1);

  if (current_edt && current_edt->invalidate_count > 0) {
    arts_out_of_order_event_satisfy_slot(current_edt->current_edt, event_guid,
                                         data_guid, slot, true);
    TIME_EVENT_SIGNAL_STOP();
    return;
  }

  struct arts_event_s *event = arts_route_table_lookup_event_safe(event_guid);
  if (!event) {
    unsigned int rank = arts_guid_get_rank(event_guid);
    if (rank != arts_global_rank_id) {
      arts_remote_event_satisfy_slot(event_guid, data_guid, slot);
    } else {
      arts_out_of_order_event_satisfy_slot(event_guid, event_guid, data_guid,
                                           slot, false);
    }
    TIME_EVENT_SIGNAL_STOP();
    return;
  }

  if (event->multiple_fire) {
    /* CHANNEL path: push data into FIFO, decrement latch counter, drain.
     * INCR_SLOT is non-sensical for CHANNEL (latch isn't a counter
     * threshold but a fire trigger; spec assertion). */
    if (slot != ARTS_EVENT_LATCH_DECR_SLOT) {
      ARTS_ERROR("CHANNEL: only DECR (slot 0) satisfy supported");
    }
    struct arts_event_dep_s *node =
        event_node_alloc(ARTS_GUID_LAST, data_guid, 0, DB_MODE_NULL);
    arts_mpsc_push(&event->channel.data_queue, &node->link);
    atomic_fetch_sub_explicit(&event->curr_latch, 1, memory_order_acq_rel);
    try_drain_channel(event, event_guid);
    arts_route_table_release(event_guid);
    TIME_EVENT_SIGNAL_STOP();
    return;
  }

  /* Non-CHANNEL path: ONCE / IDEM / STICKY / COUNTED / LATCH. */
  if (slot == ARTS_EVENT_LATCH_INCR_SLOT) {
    /* LATCH only: increment counter; no fire trigger here. */
    atomic_fetch_add_explicit(&event->curr_latch, 1, memory_order_acq_rel);
    arts_route_table_release(event_guid);
    TIME_EVENT_SIGNAL_STOP();
    return;
  }
  if (slot != ARTS_EVENT_LATCH_DECR_SLOT) {
    ARTS_ERROR("Event latch invalid slot %u", slot);
  }

  /* DECR satisfy: dec counter, check for unique fire trigger
   * (prev == 1).  Only that thread writes simple.data and runs drain. */
  int32_t prev =
      atomic_fetch_sub_explicit(&event->curr_latch, 1, memory_order_acq_rel);
  if (!event->negative_latch_allowed && prev <= 0) {
    arts_route_table_release(event_guid);
    ARTS_ERROR("over-satisfy on event with negative_latch_allowed=false");
  }
  if (prev == 1) {
    /* Unique fire trigger.  Write data BEFORE the fired CAS so the
     * release on the CAS publishes the data store to late binders. */
    if (data_guid != NULL_GUID) {
      event->simple.data = data_guid;
      atomic_thread_fence(memory_order_release);
    }
    /* CAS fired 0→1 here always succeeds (prev==1 is unique among
     * concurrent satisfies); race with addDep's drain_simple_chain
     * is benign because reverse_drain self-serialises. */
    bool fexp = false;
    (void)atomic_compare_exchange_strong_explicit(
        &event->fired, &fexp, true, memory_order_acq_rel, memory_order_acquire);
    drain_simple_chain(event, event_guid);
  }
  arts_route_table_release(event_guid);
  TIME_EVENT_SIGNAL_STOP();
}

/* OCR-aligned convenience wrapper: satisfy slot 0 (LATCH_DECR). */
void arts_event_satisfy(arts_guid_t event_guid, arts_guid_t data_guid) {
  arts_event_satisfy_slot(event_guid, data_guid, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* ── arts_add_dependence ──────────────────────────────────────────── */

void arts_add_dependence(arts_guid_t source, arts_guid_t destination,
                         uint32_t slot, arts_db_access_mode_t access_mode) {
  ARTS_INFO("Add Dependence from %lu to %lu at %u mode=%u", source, destination,
            slot, access_mode);

  /* DB_MODE_VAL: source is a raw 64-bit value. */
  if (access_mode == DB_MODE_VAL) {
    arts_guid_kind_t dest_type = arts_guid_get_kind(destination);
    if (dest_type == ARTS_GUID_EDT) {
      internal_signal_edt(destination, slot, source, DB_MODE_VAL, NULL, 0);
    } else if (dest_type == ARTS_GUID_EVENT) {
      arts_event_satisfy_slot(destination, source, slot);
    }
    return;
  }

  /* NULL source: signal immediately with no data. */
  if (source == NULL_GUID) {
    arts_guid_kind_t dest_type = arts_guid_get_kind(destination);
    if (dest_type == ARTS_GUID_EDT) {
      internal_signal_edt(destination, slot, NULL_GUID, access_mode, NULL, 0);
    } else if (dest_type == ARTS_GUID_EVENT) {
      arts_event_satisfy_slot(destination, NULL_GUID, slot);
    }
    return;
  }

  arts_guid_kind_t source_type = arts_guid_get_kind(source);

  /* DB source: immediate satisfy. */
  if (source_type == ARTS_GUID_DB) {
    arts_guid_kind_t dest_type = arts_guid_get_kind(destination);
    if (dest_type == ARTS_GUID_EDT) {
      internal_signal_edt(destination, slot, source, access_mode, NULL, 0);
    } else if (dest_type == ARTS_GUID_EVENT) {
      arts_event_satisfy_slot(destination, source, slot);
    }
    return;
  }

  /* Event source. */
  arts_guid_kind_t dest_type = arts_guid_get_kind(destination);

  /* Step 1: set mode on EDT dep slot. */
  if (dest_type == ARTS_GUID_EDT) {
    arts_set_dep_mode(destination, slot, access_mode);
  }

  /* Step 2: lookup + register. */
  struct arts_event_s *event = arts_route_table_lookup_event_safe(source);
  if (!event) {
    unsigned int rank = arts_guid_get_rank(source);
    if (rank != arts_global_rank_id) {
      arts_remote_add_dependence(source, destination, slot, rank, access_mode);
    } else {
      arts_out_of_order_add_dependence(source, destination, slot, access_mode,
                                       source);
    }
    return;
  }

  if (event->multiple_fire) {
    /* CHANNEL path: push dep into FIFO, decrement deps counter, drain. */
    struct arts_event_dep_s *node =
        event_node_alloc(dest_type, destination, slot, DB_MODE_NULL);
    arts_mpsc_push(&event->channel.dep_queue, &node->link);
    atomic_fetch_sub_explicit(&event->nb_deps_left, 1, memory_order_acq_rel);
    /* Lifetime cap check: max_deps_left tracks total deps registered. */
    if (atomic_fetch_sub_explicit(&event->max_deps_left, 1u,
                                  memory_order_acq_rel) == 0u) {
      ARTS_INFO("event dep count exceeds max_nb_deps — dropping");
    }
    try_drain_channel(event, source);
    arts_route_table_release(source);
    return;
  }

  /* Non-CHANNEL: already-fired ⇒ deliver immediately from simple.data. */
  if (atomic_load_explicit(&event->fired, memory_order_acquire)) {
    arts_guid_t data = event->simple.data;
    /* Decrement max_deps_left; if exhaustion + auto_destroy, mark_delete. */
    bool destroy_now = false;
    if (atomic_fetch_sub_explicit(&event->max_deps_left, 1u,
                                  memory_order_acq_rel) == 1u &&
        event->auto_destroy) {
      destroy_now = true;
    }
    arts_route_table_release(source);
    if (dest_type == ARTS_GUID_EDT) {
      internal_signal_edt(destination, slot, data, DB_MODE_NULL, NULL, 0);
    } else if (dest_type == ARTS_GUID_EVENT) {
      arts_event_satisfy_slot(destination, data, slot);
    }
    if (destroy_now) {
      arts_route_table_mark_delete(source);
    }
    return;
  }

  /* Enqueue dep onto Treiber stack. */
  struct arts_event_dep_s *dep =
      event_node_alloc(dest_type, destination, slot, DB_MODE_NULL);
  arts_lf_stack_push(&event->simple.deps_stack, &dep->link);

  /* Race rescue (spec §4.1 R3-R4): if the event fired between our
   * fired-check above and our push, the firing thread's drain may have
   * observed an empty stack and finished without our dep.  Re-load
   * fired with acquire and run drain_simple_chain to pick up the push.
   *
   * Critical: addDep MUST NOT call drain_simple (the CAS-fired path).
   * Only the unique satisfy thread that observed prev==1 may win CAS,
   * because only that thread has written simple.data.  An addDep
   * winning CAS would call drain_simple_chain → read simple.data
   * before the satisfier had published it, delivering NULL_GUID to
   * every consumer (race observed in event_once_storm @ iter=25). */
  if (atomic_load_explicit(&event->fired, memory_order_acquire)) {
    drain_simple_chain(event, source);
  }
  arts_route_table_release(source);
}
