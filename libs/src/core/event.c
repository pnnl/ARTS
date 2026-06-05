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
 * arts_event_s — single struct, is_channel-discriminated union.
 * See docs/superpowers/plans/2026-05-11-event-hint-redesign.md.
 *
 *   - simple (non-CHANNEL): latch + Treiber dep stack.  All single-fire
 *     OCR flavors (ONCE / IDEM / STICKY / COUNTED) and LATCH(N) use this
 *     branch.  Fire trigger: latch reaches <= 0 (unique winner observes
 *     prev==1 in the atomic_fetch_sub).  Fire is a pure state transition
 *     and never destroys (fire-and-linger) — the event stays addressable
 *     to serve late binders from the stored data until an explicit
 *     arts_event_destroy.  Over-satisfy past the fire is silently absorbed.
 *
 *   - channel (CHANNEL only): nb_sat / nb_deps monotonic counters +
 *     two mpsc FIFO queues (data_queue, dep_queue) + single-flight
 *     drainer sentinel.  Each satisfy push + nb_sat++; each addDep
 *     push + nb_deps++.  Drainer pops one from each queue per
 *     generation, decrements both counters.  No auto-destroy.
 */

#include "arts/event.h"

#include "arts.h"
#include "arts/edt.h"
#include "arts/edt_context.h" /* current_edt */
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/runtime_state.h" /* arts_node_info, event_dep_pool */
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/outbox.h"    /* outbound send helpers */
#include "arts/transport/protocol.h"  /* wire packet structs */
#include "arts/utils/lockfree_lifo.h" /* arts_lf_stack_init / drain */
#include "arts/utils/malloc.h"
#include "arts/utils/mpsc.h"        /* arts_mpsc_t */
#include "arts/utils/shared.h"      /* arts_shared_ptr_t, get/release */
#include "arts/utils/tiered_pool.h" /* arts_tiered_pool_release */

#include <assert.h>
#include <string.h>
#include <time.h>

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

/* --- cb deleter (route_table deleter-by-kind for ARTS_GUID_EVENT) --------- */

/* External linkage so route_table.c references it directly.  Runs once the
 * cb's strong refcount hits 0 (last holder released). */
void arts_event_deleter(void *p) {
  struct arts_event_s *e = (struct arts_event_s *)p;
  /* Drain any leftover queued nodes and return them to the per-rank pool.
   * strong == 0 by precondition: no producer can push after this point
   * (every producer must hold a lookup ref while pushing). */
  if (e->is_channel) {
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

/* Publish the event cb deleter into the route_table's per-kind table at startup
 * (decoupled registration — see arts_route_table_register_deleter). */
__attribute__((constructor)) static void arts_event_register_cb_deleter(void) {
  arts_route_table_register_deleter(ARTS_GUID_EVENT, arts_event_deleter);
}

/* External forwarder — see event.h for rationale. */
static void event_free_typed(struct arts_event_s *e) { arts_event_deleter(e); }

/* --- Internal allocation / install --------------------------------------- */

static struct arts_event_s *event_alloc(const arts_event_hint_t *h) {
  struct arts_event_s *e = arts_calloc(1, sizeof(struct arts_event_s));
  if (!e) {
    return NULL;
  }
  /* lifecycle/deleter handled by the route_table cb (deleter-by-kind) on
   * install — no per-object shared field to initialize. */
  e->is_channel = h->channel ? 1 : 0;

  if (e->is_channel) {
    atomic_store_explicit(&e->channel.nb_sat, 0u, memory_order_relaxed);
    atomic_store_explicit(&e->channel.nb_deps, 0u, memory_order_relaxed);
    arts_mpsc_init(&e->channel.data_queue);
    arts_mpsc_init(&e->channel.dep_queue);
    atomic_store_explicit(&e->channel.draining, 0, memory_order_relaxed);
  } else {
    atomic_store_explicit(&e->simple.latch, h->latch, memory_order_relaxed);
    atomic_store_explicit(&e->simple.fired, false, memory_order_relaxed);
    e->simple.data = NULL_GUID;
    arts_lf_stack_init(&e->simple.deps_stack);
  }
  return e;
}

static bool event_install(arts_guid_t *guid, const arts_event_hint_t *h_in) {
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
      if (h.check) {
        /* CHECK / rendezvous (e.g. OCR GUID_PROP_CHECK): fail if the GUID
         * already exists so the first creator wins and a later one observes
         * the collision.  add_item_race CAS-installs (firing the OoO list on
         * win); on loss the object stays ours, so we free it and return
         * NULL_GUID via the caller. */
        if (!arts_route_table_install_if_absent(event, *guid, rank, false)) {
          arts_event_deleter(event);
          return false;
        }
      } else {
        /* Default: unconditional install — a labeled-GUID reuse REPLACES the
         * prior generation (the displaced cb is released).  Drains the OoO
         * list internally. */
        arts_route_table_install(event, *guid, rank, false);
      }
    } else {
      *guid = arts_guid_create_for_rank(rank, ARTS_GUID_EVENT);
      arts_route_table_install(event, *guid, rank, false);
    }
    return true;
  }
  /* Cross-rank: forward as a marshaled buffer.  Receiver
   * arts_handler_event_create performs add_item_race.  Discard the
   * local allocation since the remote will materialise its own copy. */
  arts_send_memory_move(rank, *guid, event, sizeof(*event), MSG_EVENT_CREATE,
                        arts_event_deleter);
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
  bool ok = event_install(&g, &h);
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
    arts_send_event_destroy(guid);
    return;
  }
  arts_route_table_mark_delete(guid);
}

/* ── Signal one queued dep ─────────────────────────────────────────────
 * For CHANNEL the data argument comes from the matching data_queue pop;
 * for non-CHANNEL it is e->simple.data. */
static void event_signal_one(struct arts_event_dep_s *d, arts_guid_t data) {
  if (d->kind == ARTS_GUID_EDT) {
    arts_edt_satisfy_slot(d->target, d->slot, data, d->mode, NULL, 0);
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
 *   (a) the unique satisfy thread that just CAS-set fired.
 *   (b) any addDep thread that pushed onto the stack and then observed
 *       fired==true (race rescue: addDep's push lands *after* satisfy's
 *       reverse_drain finished).
 *
 * Concurrent callers self-serialise inside arts_lf_stack_reverse_drain's
 * `atomic_exchange(&head, NULL)` — only one caller per chain, others see
 * NULL and exit.  The outer loop catches pushes that landed during a
 * caller's iteration.  Fire never destroys (fire-and-linger): the event
 * stays addressable to serve late binders until an explicit destroy.
 */
static void drain_simple_chain(struct arts_event_s *e, arts_guid_t event_guid) {
  (void)event_guid; /* simple events never auto-destroy */
  arts_guid_t data = e->simple.data;
  for (;;) {
    arts_lf_link_t *fifo = arts_lf_stack_reverse_drain(&e->simple.deps_stack);
    if (!fifo) {
      return; /* drain complete */
    }
    while (fifo) {
      arts_lf_link_t *next =
          atomic_load_explicit(&fifo->next, memory_order_relaxed);
      struct arts_event_dep_s *dep = (struct arts_event_dep_s *)fifo;
      event_signal_one(dep, data);
      event_node_free(fifo);
      fifo = next;
    }
  }
}

/* ── CHANNEL drain (lock-free, single-flight via `draining` sentinel) ── */

static void try_drain_channel(struct arts_event_s *e, arts_guid_t event_guid) {
  (void)event_guid; /* CHANNEL never auto-destroys */
  /* Outer rescue loop: re-check the fire condition after we release
   * `draining` because a concurrent push may have arrived in the
   * window between our last queue-pop and the sentinel-clear. */
  for (;;) {
    uint32_t nb_sat =
        atomic_load_explicit(&e->channel.nb_sat, memory_order_acquire);
    uint32_t nb_deps =
        atomic_load_explicit(&e->channel.nb_deps, memory_order_acquire);
    if (nb_sat == 0 || nb_deps == 0) {
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
    /* Inner fire loop: keep firing while both counters > 0. */
    while (atomic_load_explicit(&e->channel.nb_sat, memory_order_acquire) > 0 &&
           atomic_load_explicit(&e->channel.nb_deps, memory_order_acquire) >
               0) {
      /* Both counters > 0 ⇒ a producer has fully committed one node to each
       * queue (the push links the node before its counter bump).  A *different*
       * concurrent producer that has swapped the queue head but not yet stored
       * its predecessor's next pointer makes the MPSC pop transiently return
       * NULL even though the committed node is present — the queue contract is
       * "retry later".  Spin on each pop until the in-flight link resolves;
       * never drop the already-popped partner (doing so desyncs the
       * counter/queue pair and strands the consumer EDT forever). */
      arts_lf_link_t *data_node;
      while ((data_node = arts_mpsc_pop(&e->channel.data_queue)) == NULL) {
      }
      arts_lf_link_t *dep_node;
      while ((dep_node = arts_mpsc_pop(&e->channel.dep_queue)) == NULL) {
      }
      /* Data node carries the satisfy data in `target` (kind == ARTS_NULL
       * marker). */
      arts_guid_t data = ((struct arts_event_dep_s *)data_node)->target;
      struct arts_event_dep_s *dep = (struct arts_event_dep_s *)dep_node;
      event_signal_one(dep, data);
      event_node_free(data_node);
      event_node_free(dep_node);
      atomic_fetch_sub_explicit(&e->channel.nb_sat, 1u, memory_order_acq_rel);
      atomic_fetch_sub_explicit(&e->channel.nb_deps, 1u, memory_order_acq_rel);
    }
    atomic_store_explicit(&e->channel.draining, 0, memory_order_release);
    /* Outer-while will re-check the fire condition for missed pushes. */
  }
}

/* ── arts_event_satisfy_slot ───────────────────────────────────────── */

/* Home-routed handler (OOO_EVENT_SATISFY_SLOT): item is the installed event,
 * ref-held by dispatch_or_defer.  Pure core — no lookup / acquire / release. */
void arts_handler_event_satisfy_slot(void *item, void *vargs) {
  struct arts_event_s *event = (struct arts_event_s *)item;
  struct arts_ooo_args_event_satisfy_s *a =
      (struct arts_ooo_args_event_satisfy_s *)vargs;
  arts_guid_t event_guid = a->event_guid;
  arts_guid_t data_guid = a->data_guid;
  uint32_t slot = a->slot;

  if (event->is_channel) {
    /* CHANNEL path: push data, increment nb_sat, drain. */
    if (slot != ARTS_EVENT_LATCH_DECR_SLOT) {
      ARTS_ERROR("CHANNEL: only DECR (slot 0) satisfy supported");
    }
    struct arts_event_dep_s *node =
        event_node_alloc(ARTS_GUID_LAST, data_guid, 0, DB_MODE_NULL);
    arts_mpsc_push(&event->channel.data_queue, &node->link);
    atomic_fetch_add_explicit(&event->channel.nb_sat, 1u, memory_order_acq_rel);
    try_drain_channel(event, event_guid);
    return;
  }

  /* Non-CHANNEL path: ONCE / IDEM / STICKY / COUNTED / LATCH. */
  if (slot == ARTS_EVENT_LATCH_INCR_SLOT) {
    /* LATCH only: increment counter; no fire trigger here. */
    atomic_fetch_add_explicit(&event->simple.latch, 1, memory_order_acq_rel);
    return;
  }
  if (slot != ARTS_EVENT_LATCH_DECR_SLOT) {
    ARTS_ERROR("Event latch invalid slot %u", slot);
  }

  /* DECR satisfy: dec counter, check for unique fire trigger
   * (prev == 1).  Only that thread writes simple.data and runs drain. */
  int32_t prev =
      atomic_fetch_sub_explicit(&event->simple.latch, 1, memory_order_acq_rel);
  if (prev <= 0) {
    return; /* over-satisfy past the unique fire: silently absorbed */
  }
  if (prev == 1) {
    /* Unique fire trigger.  Write data BEFORE the fired CAS so the
     * release on the CAS publishes the data store to late binders. */
    if (data_guid != NULL_GUID) {
      event->simple.data = data_guid;
      atomic_thread_fence(memory_order_release);
    }
    bool fexp = false;
    (void)atomic_compare_exchange_strong_explicit(&event->simple.fired, &fexp,
                                                  true, memory_order_acq_rel,
                                                  memory_order_acquire);
    drain_simple_chain(event, event_guid);
  }
}

/* arts_event_satisfy_slot — entity-specific API: satisfy an event's slot.
 *   home == self → dispatch_or_defer (acquire → arts_handler_event_satisfy_slot
 *                  or defer until the event installs);
 *   home != self → MSG_EVENT_SATISFY_SLOT wire;
 *   CDAG → force-defer on the GPU wrapper's slot. */
void arts_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                             uint32_t slot) {
  TIME_EVENT_SIGNAL_START();
  INCREMENT_NUM_EVENT_SIGNAL_BY(1);

  struct arts_ooo_args_event_satisfy_s a = {
      .event_guid = event_guid, .data_guid = data_guid, .slot = slot};
  if (current_edt && current_edt->invalidate_count > 0) {
    arts_ooo_push_guid(current_edt->guid, OOO_EVENT_SATISFY_SLOT, &a,
                       sizeof(a));
  } else if (arts_guid_get_rank(event_guid) != arts_global_rank_id) {
    arts_send_event_satisfy_slot(event_guid, data_guid, slot);
  } else {
    arts_ooo_dispatch_or_defer_guid(event_guid, OOO_EVENT_SATISFY_SLOT, &a,
                                    sizeof(a));
  }
  TIME_EVENT_SIGNAL_STOP();
}

/* OCR-aligned convenience wrapper: satisfy slot 0 (LATCH_DECR). */
void arts_event_satisfy(arts_guid_t event_guid, arts_guid_t data_guid) {
  arts_event_satisfy_slot(event_guid, data_guid, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* ── arts_event_add_dependence ─────────────────────────────────────── */

/* Home-routed handler (OOO_EVENT_ADD_DEPENDENCE): item is the installed source
 * event, ref-held by dispatch_or_defer.  Pure core — register the dependent on
 * the event (CHANNEL queue / fire-and-linger immediate deliver / Treiber
 * waiter).  No lookup / acquire / release. */
void arts_handler_event_add_dependence(void *item, void *vargs) {
  struct arts_event_s *event = (struct arts_event_s *)item;
  struct arts_ooo_args_event_add_dep_s *a =
      (struct arts_ooo_args_event_add_dep_s *)vargs;
  arts_guid_t source = a->source;
  arts_guid_t destination = a->destination;
  uint32_t slot = a->slot;
  arts_db_access_mode_t access_mode = a->mode;
  arts_guid_kind_t dest_type = arts_guid_get_kind(destination);

  if (event->is_channel) {
    /* CHANNEL path: push dep, increment nb_deps, drain. */
    struct arts_event_dep_s *node =
        event_node_alloc(dest_type, destination, slot, access_mode);
    arts_mpsc_push(&event->channel.dep_queue, &node->link);
    atomic_fetch_add_explicit(&event->channel.nb_deps, 1u,
                              memory_order_acq_rel);
    try_drain_channel(event, source);
    return;
  }

  /* Already-fired ⇒ deliver immediately from simple.data (fire-and-linger). */
  if (atomic_load_explicit(&event->simple.fired, memory_order_acquire)) {
    arts_guid_t data = event->simple.data;
    if (dest_type == ARTS_GUID_EDT) {
      arts_edt_satisfy_slot(destination, slot, data, access_mode, NULL, 0);
    } else if (dest_type == ARTS_GUID_EVENT) {
      arts_event_satisfy_slot(destination, data, slot);
    }
    return;
  }

  /* Not yet fired: enqueue dep onto the Treiber stack. */
  struct arts_event_dep_s *dep =
      event_node_alloc(dest_type, destination, slot, access_mode);
  arts_lf_stack_push(&event->simple.deps_stack, &dep->link);

  /* Race rescue: the event may have fired between our fired-check and our
   * push; re-load fired and drain so our dep is not stranded.  addDep MUST NOT
   * CAS `fired` — only the unique satisfy thread that wrote simple.data may. */
  if (atomic_load_explicit(&event->simple.fired, memory_order_acquire)) {
    drain_simple_chain(event, source);
  }
}

/* arts_event_add_dependence — entity-specific API: register a dependent on an
 * event source.  home==self → dispatch_or_defer (acquire → handler, or defer
 * until the event installs); home!=self → MSG_EVENT_ADD_DEPENDENCE wire. */
void arts_event_add_dependence(arts_guid_t source, arts_guid_t destination,
                               uint32_t slot, arts_db_access_mode_t mode) {
  unsigned int rank = arts_guid_get_rank(source);
  if (rank != arts_global_rank_id) {
    arts_send_event_add_dependence(source, destination, slot, rank, mode);
    return;
  }
  struct arts_ooo_args_event_add_dep_s a = {
      .source = source, .destination = destination, .slot = slot, .mode = mode};
  arts_ooo_dispatch_or_defer_guid(source, OOO_EVENT_ADD_DEPENDENCE, &a,
                                  sizeof(a));
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
      arts_edt_satisfy_slot(destination, slot, source, DB_MODE_VAL, NULL, 0);
    } else if (dest_type == ARTS_GUID_EVENT) {
      arts_event_satisfy_slot(destination, source, slot);
    }
    return;
  }

  /* NULL source: signal immediately with no data. */
  if (source == NULL_GUID) {
    arts_guid_kind_t dest_type = arts_guid_get_kind(destination);
    if (dest_type == ARTS_GUID_EDT) {
      arts_edt_satisfy_slot(destination, slot, NULL_GUID, access_mode, NULL, 0);
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
      arts_edt_satisfy_slot(destination, slot, source, access_mode, NULL, 0);
    } else if (dest_type == ARTS_GUID_EVENT) {
      arts_event_satisfy_slot(destination, source, slot);
    }
    return;
  }

  /* Event source — delegate to the entity-specific API.  The dep mode rides
   * on the satisfy at fire time (stored in the waiter node, replayed through
   * arts_edt_satisfy_slot), so no eager mode-set to the destination. */
  arts_event_add_dependence(source, destination, slot, access_mode);
}

static void send_remote_add_dependence_packet(unsigned int message_type,
                                              arts_guid_t source,
                                              arts_guid_t destination,
                                              uint32_t slot, unsigned int rank,
                                              arts_db_access_mode_t mode) {
  struct arts_remote_add_dependence_packet_s packet;
  packet.source = source;
  packet.destination = destination;
  packet.slot = slot;
  packet.mode = mode;
  arts_fill_packet_header(&packet.header, sizeof(packet), message_type);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_send_event_add_dependence(arts_guid_t source, arts_guid_t destination,
                                    uint32_t slot, unsigned int rank,
                                    arts_db_access_mode_t mode) {
  ARTS_DEBUG("Remote Add dependence sent %d", rank);
  send_remote_add_dependence_packet(MSG_EVENT_ADD_DEPENDENCE, source,
                                    destination, slot, rank, mode);
}

void arts_handler_event_create(void *ptr) {
  struct arts_remote_guid_only_packet_s *packet =
      (struct arts_remote_guid_only_packet_s *)ptr;
  uint64_t size =
      packet->header.size - sizeof(struct arts_remote_guid_only_packet_s);

  struct arts_event_s *mem_packet =
      (struct arts_event_s *)arts_malloc_align(size, 16);

  memcpy(mem_packet, packet + 1, size);
  /* Re-init local-only pointer state.  Event move only happens at create
   * time (queues / stack always empty at source), so re-initing to empty
   * is correct.  The sender-rank heap pointers in the wire image are
   * meaningless here. */
  if (mem_packet->is_channel) {
    arts_mpsc_init(&mem_packet->channel.data_queue);
    arts_mpsc_init(&mem_packet->channel.dep_queue);
    atomic_store_explicit(&mem_packet->channel.nb_sat, 0u,
                          memory_order_relaxed);
    atomic_store_explicit(&mem_packet->channel.nb_deps, 0u,
                          memory_order_relaxed);
    atomic_store_explicit(&mem_packet->channel.draining, 0,
                          memory_order_relaxed);
  } else {
    arts_lf_stack_init(&mem_packet->simple.deps_stack);
    /* latch / fired / data preserved from sender's post-init state. */
  }

  /* add_item_race installs the event under the route_table lock; on
   * success it also fires OoO replay internally, so no extra fire_oo
   * is required.  On rejection (another rank won the install race),
   * release the freshly-unmarshaled buffer through event_deleter (via
   * event_free_typed) — raw arts_free would skip the dep-stack
   * drain.  In practice the dep stack is empty at this point (nothing
   * has been pushed locally yet), but using the proper deleter keeps
   * lifecycle ownership symmetric with event_alloc. */
  if (!arts_route_table_install_if_absent(mem_packet, packet->guid,
                                          arts_global_rank_id, false)) {
    event_free_typed(mem_packet);
  }
}

void arts_send_event_destroy(arts_guid_t guid) {
  unsigned int rank = arts_guid_get_rank(guid);
  struct arts_remote_guid_only_packet_s packet;
  packet.guid = guid;
  arts_fill_packet_header(&packet.header, sizeof(packet), MSG_EVENT_DESTROY);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_handler_event_destroy(void *ptr) {
  struct arts_remote_guid_only_packet_s *packet =
      (struct arts_remote_guid_only_packet_s *)ptr;
  /* Before-create wire reorder (symmetric with arts_handler_db_destroy): a
   * DESTROY that reaches the event's home ahead of its CREATE must defer via
   * the OoO list (OOO_EVENT_DESTROY) and replay once the create handler
   * installs + drains, rather than mark_delete'ing an absent slot (which would
   * lose the destroy and leak the later-created event).  dispatch_or_defer's
   * hit path runs the replay (mark_delete) inline when the event already
   * exists; mark_delete is itself idempotent (exchange slot value -> NULL). */
  struct arts_ooo_args_event_destroy_s args = {.guid = packet->guid};
  arts_ooo_dispatch_or_defer_guid(packet->guid, OOO_EVENT_DESTROY, &args,
                                  sizeof(args));
}

void arts_send_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                                  uint32_t slot) {
  struct arts_remote_event_satisfy_slot_packet_s packet;
  packet.event = event_guid;
  packet.db = data_guid;
  packet.slot = slot;
  arts_fill_packet_header(&packet.header, sizeof(packet),
                          MSG_EVENT_SATISFY_SLOT);
  arts_remote_send_request_async((int)arts_guid_get_rank(event_guid),
                                 (char *)&packet, sizeof(packet));
}
