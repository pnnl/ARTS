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
#include "arts/epoch.h"

#include "arts.h"
#include "arts/edt.h"
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/db.h"
#include "arts/runtime_types.h"
#include "arts/edt_context.h" /* arts_edt_ctx_t + epoch stack accessors */
#include "arts/epoch_pool.h" /* arts_epoch_pool_get, arts_epoch_pool_clean */
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/outbox.h"   /* outbound send helpers */
#include "arts/transport/protocol.h" /* wire packet structs */
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h" /* arts_shared_ptr_t, get/release */

#define EPOCH_MASK 0x7FFFFFFFFFFFFFFF
#define EPOCH_BIT 0x8000000000000000

/*
 * arts_epoch_deleter — cb deleter for ARTS_GUID_EPOCH.
 *
 * Invoked by the cb once the last strong ref is dropped (install ref +
 * outstanding reader refs).  Mirrors the DB / EDT pattern.
 *
 * Two ownership models for epoch storage:
 *   - Stand-alone heap epochs (created by arts_epoch_alloc / shutdown epoch /
 *     arts_epoch_create on a cross-rank create): malloc'd on their own,
 *     so the deleter just frees the struct.
 *   - Pool-backed epochs (entries inside arts_epoch_pool_t.pool[]): the
 *     epoch_pool itself owns the storage and is freed when its outstanding
 *     count hits 0.  Individual pool entries must NOT be freed here — only
 *     the route_table slot is dropped via mark_delete.  pool_guid != 0
 *     distinguishes pool-backed entries.
 */
/* External linkage so route_table.c references it directly as the
 * deleter-by-kind for ARTS_GUID_EPOCH. */
void arts_epoch_deleter(void *self) {
  arts_epoch_t *epoch = (arts_epoch_t *)self;
  if (epoch->pool_guid != NULL_GUID) {
    /* Pool-backed: storage lives inside arts_epoch_pool_t.pool[].  The
     * pool itself is freed in arts_epoch_delete once outstanding hits 0.  No
     * per-entry free needed. */
    return;
  }
  arts_free(epoch);
}

/* Publish the epoch cb deleter into the route_table's per-kind table at startup
 * (decoupled registration — see arts_route_table_register_deleter). */
__attribute__((constructor)) static void arts_epoch_register_cb_deleter(void) {
  arts_route_table_register_deleter(ARTS_GUID_EPOCH, arts_epoch_deleter);
}

/*
 * Shutdown-epoch helpers.
 *
 * The shutdown epoch tracks all EDTs globally.  When the epoch completes
 * (active == finished, no outstanding), arts_shutdown_epoch_fire is called,
 * which triggers arts_shutdown().
 *
 * Three counters are incremented at different EDT lifecycle stages:
 *   inc_active  — when an EDT is created (globally visible).
 *   inc_queue   — when an EDT becomes ready (all deps satisfied).
 *   inc_finished — when an EDT completes execution.
 */
void arts_shutdown_epoch_inc_active() {
  if (arts_node_info.auto_shutdown_guid) {
    ARTS_DEBUG("shutdown_epoch: inc_active [Epoch:%lu]",
               arts_node_info.auto_shutdown_guid);
    arts_epoch_inc_active(arts_node_info.auto_shutdown_guid);
  }
}

void arts_shutdown_epoch_inc_queue() {
  if (arts_node_info.auto_shutdown_guid) {
    ARTS_DEBUG("shutdown_epoch: inc_queue [Epoch:%lu]",
               arts_node_info.auto_shutdown_guid);
    arts_epoch_inc_queue(arts_node_info.auto_shutdown_guid);
  }
}

void arts_shutdown_epoch_inc_finished() {
  if (arts_node_info.auto_shutdown_guid) {
    ARTS_DEBUG("shutdown_epoch: inc_finished [Epoch:%lu]",
               arts_node_info.auto_shutdown_guid);
    arts_epoch_inc_finished(arts_node_info.auto_shutdown_guid);
  }
}

void arts_shutdown_epoch_fire(arts_guid_t guid) {
  if (arts_node_info.auto_shutdown_guid == guid) {
    ARTS_INFO(
        "arts_shutdown_epoch_fire: Epoch[Guid:%lu] matched shutdown epoch — "
        "calling arts_shutdown()",
        guid);
    arts_shutdown();
  }
}

bool arts_epoch_dec_queue(arts_epoch_t *epoch) {
  uint64_t local;
  while (1) {
    local = epoch->queued;
    if (local == 1) {
      if (1 == arts_atomic_cswap_u64(&epoch->queued, 1, EPOCH_BIT)) {
        return true;
      }
    } else {
      if (local == arts_atomic_cswap_u64(&epoch->queued, local, local - 1)) {
        return false;
      }
    }
  }
}

/* Home-routed handlers (OOO_EPOCH_*) — item is the installed epoch, ref-held
 * by dispatch_or_defer.  Pure cores: no lookup / acquire / release.  Epoch is
 * broadcast-installed on every rank at create, so the local rank's copy is the
 * target; the OoO defer covers the create-broadcast-not-yet-arrived window. */
void arts_handler_epoch_inc_queue(void *item, void *vargs) {
  (void)vargs;
  arts_atomic_add_u64(&((arts_epoch_t *)item)->queued, 1);
}

void arts_handler_epoch_inc_active(void *item, void *vargs) {
  (void)vargs;
  arts_epoch_t *epoch = (arts_epoch_t *)item;
  if (arts_global_rank_count == 1) {
    arts_lock(&epoch->local_lock);
    epoch->active_count++;
    arts_unlock(&epoch->local_lock);
  } else {
    arts_atomic_add(&epoch->active_count, 1);
  }
}

void arts_epoch_inc_queue(arts_guid_t epoch_guid) {
  if (epoch_guid != NULL_GUID) {
    struct arts_ooo_args_epoch_s a = {.epoch_guid = epoch_guid};
    arts_ooo_dispatch_or_defer_guid(epoch_guid, OOO_EPOCH_INC_QUEUE, &a,
                                    sizeof(a));
  }
}

void arts_epoch_inc_active(arts_guid_t epoch_guid) {
  struct arts_ooo_args_epoch_s a = {.epoch_guid = epoch_guid};
  arts_ooo_dispatch_or_defer_guid(epoch_guid, OOO_EPOCH_INC_ACTIVE, &a,
                                  sizeof(a));
}

/*
 * arts_epoch_inc_finished — Called when an EDT finishes execution.
 *
 * Single-node fast path: local_lock serializes active_count/finished_count
 * updates so a finishing EDT cannot observe active==finished while another
 * thread is concurrently creating new work in the same epoch.
 *
 * Multi-node: owner rank collects responses; non-owner ranks decrement
 * their queued counter and, when it hits 1, send their active/finished
 * counts to the owner for global reduction.
 */
/* Home-routed handler (OOO_EPOCH_INC_FINISHED) — pure core on the acquired
 * epoch.  The single-node fire path calls arts_epoch_delete (mark_delete drops
 * the install ref); dispatch_or_defer's ref keeps the epoch alive until this
 * returns, so the deleter runs as a deferred free afterward. */
void arts_handler_epoch_inc_finished(void *item, void *vargs) {
  arts_epoch_t *epoch = (arts_epoch_t *)item;
  arts_guid_t epoch_guid = ((struct arts_ooo_args_epoch_s *)vargs)->epoch_guid;
  if (arts_global_rank_count == 1) {
    bool fire_epoch = false;
    arts_lock(&epoch->local_lock);
    epoch->finished_count++;
    if (epoch->finished_count > 0 &&
        epoch->finished_count == epoch->active_count &&
        epoch->phase == (unsigned int)PHASE_1) {
      epoch->phase = (unsigned int)PHASE_3;
      fire_epoch = true;
    }
    arts_unlock(&epoch->local_lock);

    if (fire_epoch) {
      if (epoch->termination_exit_guid) {
        arts_edt_satisfy_slot(
            epoch->termination_exit_guid, epoch->termination_exit_slot,
            (arts_guid_t)(epoch->finished_count), DB_MODE_VAL, NULL, 0);
      } else {
        arts_shutdown_epoch_fire(epoch->guid);
      }
      arts_epoch_delete(epoch_guid, NULL);
    }
    return;
  }
  arts_atomic_add(&epoch->finished_count, 1);
  unsigned int rank = arts_guid_get_rank(epoch_guid);
  if (rank == arts_global_rank_id) {
    if (!arts_atomic_sub_u64(&epoch->queued, 1)) {
      uint64_t old_out =
          arts_atomic_cswap_u64(&epoch->outstanding, 0, arts_global_rank_count);
      if (old_out == 0) {
        arts_epoch_request_broadcast(epoch_guid);
      }
    }
  } else {
    if (arts_epoch_dec_queue(epoch)) {
      arts_send_epoch_send(rank, epoch_guid, epoch->active_count,
                           epoch->finished_count);
    }
  }
}

void arts_epoch_inc_finished(arts_guid_t epoch_guid) {
  if (epoch_guid != NULL_GUID) {
    struct arts_ooo_args_epoch_s a = {.epoch_guid = epoch_guid};
    arts_ooo_dispatch_or_defer_guid(epoch_guid, OOO_EPOCH_INC_FINISHED, &a,
                                    sizeof(a));
  }
}

/* Home-routed reply core (OOO_EPOCH_REQUEST): a non-home rank, having received
 * an epoch query, forwards its local active/finished counts to dest.  Pure
 * core on the dispatch-acquired epoch; the OoO defer covers the window where
 * the self-rank epoch broadcast-install has not yet arrived. */
void arts_handler_epoch_request(void *item, void *vargs) {
  arts_epoch_t *epoch = (arts_epoch_t *)item;
  struct arts_ooo_args_epoch_request_s *a = vargs;
  arts_atomic_fetch_and_u64(&epoch->queued, EPOCH_MASK);
  if (!arts_atomic_cswap_u64(&epoch->queued, 0, EPOCH_BIT)) {
    arts_send_epoch_send(a->dest, a->epoch_guid, epoch->active_count,
                         epoch->finished_count);
  }
}

void arts_epoch_reply(arts_guid_t epoch_guid, unsigned int source,
                      unsigned int dest) {
  struct arts_ooo_args_epoch_request_s a = {
      .epoch_guid = epoch_guid, .source = source, .dest = dest};
  arts_ooo_dispatch_or_defer_guid(epoch_guid, OOO_EPOCH_REQUEST, &a, sizeof(a));
}

arts_epoch_t *arts_epoch_alloc(arts_guid_t *guid, arts_guid_t edt_guid,
                               unsigned int slot) {
  INCREMENT_NUM_EPOCH_CREATE_BY(1);
  if (*guid == NULL_GUID) {
    *guid = arts_guid_create_for_rank(arts_global_rank_id, ARTS_GUID_EPOCH);
  }

  arts_epoch_t *epoch = (arts_epoch_t *)arts_calloc(1, sizeof(arts_epoch_t));
  epoch->phase = PHASE_1;
  epoch->termination_exit_guid = edt_guid;
  epoch->termination_exit_slot = slot;
  epoch->guid = *guid;
  epoch->pool_guid = NULL_GUID;
  epoch->queued = (arts_guid_is_local(*guid)) ? 0 : EPOCH_BIT;
  /* add_item_race wraps epoch in a cb (deleter-by-kind) and fires the
   * OoO list internally on a successful install. */
  arts_route_table_install_if_absent(epoch, *guid, arts_global_rank_id, false);
  return epoch;
}

/*
 * arts_shutdown_epoch_create — Initialize the global termination epoch.
 *
 * Pre-seeds active_count and queued with the total number of workers,
 * since each worker thread will call arts_epoch_inc_finished when it
 * finishes its initialization sequence.
 */
bool arts_shutdown_epoch_create() {
  if (arts_node_info.auto_shutdown_guid) {
    arts_node_info.auto_shutdown_guid =
        arts_guid_create_for_rank(0, ARTS_GUID_EPOCH);
    arts_epoch_t *epoch =
        arts_epoch_alloc(&arts_node_info.auto_shutdown_guid, NULL_GUID, 0);
    unsigned int total_workers = arts_get_workers_per_rank();
    arts_atomic_add(&epoch->active_count, total_workers);
    arts_atomic_add_u64(&epoch->queued, total_workers);
    ARTS_INFO(
        "arts_shutdown_epoch_create: Epoch[Guid:%lu] created with %u workers",
        arts_node_info.auto_shutdown_guid, total_workers);
    return true;
  }
  return false;
}

void arts_epoch_add_edt(arts_guid_t edt_guid, arts_guid_t epoch_guid) {
  /* lookup_edt pairs with release on the success path. */
  arts_shared_ptr_t edt_h = arts_route_table_lookup_edt(edt_guid);
  struct arts_edt_s *edt = (struct arts_edt_s *)arts_shared_get(edt_h);
  if (edt) {
    edt->epoch_guid = epoch_guid;
    arts_epoch_inc_active(epoch_guid);
    arts_shared_release(&edt_h);
    return;
  }
}

void arts_epoch_request_broadcast(arts_guid_t epoch_guid) {
  unsigned int origin_rank = arts_guid_get_rank(epoch_guid);
  for (unsigned int i = 0; i < arts_global_rank_count; i++) {
    if (i != origin_rank) {
      arts_send_epoch_request(i, epoch_guid);
    }
  }
}

arts_guid_t arts_epoch_create(unsigned int rank, arts_guid_t finish_edt_guid,
                              unsigned int slot) {
  arts_guid_t guid = NULL_GUID;
  // I think the idea is this is that during parallel start
  // (arts_node_info.ready_to_execute > 0) This means that the epoch will be
  // created on all nodes assuming that each node goes through the
  // initializeEpoch code path. Pool assume the current host...
  if (!arts_node_info.ready_to_execute || rank != arts_global_rank_id) {
    guid = arts_guid_create_for_rank(rank, ARTS_GUID_EPOCH);
    arts_epoch_alloc(&guid, finish_edt_guid, slot);
    if (!arts_node_info.ready_to_execute) {
      for (unsigned int i = 0; i < arts_global_rank_count; i++) {
        if (i != arts_global_rank_id) {
          arts_send_epoch_create(i, guid, finish_edt_guid, slot);
        }
      }
    }
  } else // Lets get it from the pool...
  {
    arts_epoch_t *epoch = arts_epoch_pool_get(finish_edt_guid, slot);
    guid = epoch->guid;
  }
  return guid;
}

void arts_epoch_start(arts_guid_t epoch_guid) {
  /* lookup_epoch pairs with release on the success path. */
  arts_shared_ptr_t epoch_h = arts_route_table_lookup_epoch(epoch_guid);
  arts_epoch_t *epoch = (arts_epoch_t *)arts_shared_get(epoch_h);
  if (epoch) {
    arts_set_current_epoch_guid(epoch->guid);
    arts_atomic_add(&epoch->active_count, 1);
    arts_atomic_add_u64(&epoch->queued, 1);
    arts_shared_release(&epoch_h);
  } else {
    ARTS_WARN("Epoch [Guid:%lu] doesn't exist in the Route table", epoch_guid);
  }
}

bool arts_epoch_check(arts_epoch_t *epoch, unsigned int total_active,
                      unsigned int total_finish) {
  unsigned int diff = total_active - total_finish;
  ARTS_INFO("Checking Epoch [Guid:%lu, TotalActive:%u, TotalFinish:%u, "
            "Diff:%u, Phase:%u, LastActive:%u, LastFinished:%u]",
            epoch->guid, total_active, total_finish, diff, epoch->phase,
            epoch->last_active_count, epoch->last_finished_count);

  if (total_finish && !diff) {
    if (arts_global_rank_count == 1) {
      /*
       * Single-node: skip the two-phase confirmation protocol (no network
       * round needed).  CAS ensures exactly one thread transitions to
       * PHASE_3 and fires the epoch, even under concurrent callers.
       */
      unsigned int old_phase = arts_atomic_cswap(
          &epoch->phase, (unsigned int)PHASE_1, (unsigned int)PHASE_3);
      if (old_phase == (unsigned int)PHASE_1) {
        ARTS_DEBUG("arts_epoch_check: CAS won PHASE_1->PHASE_3, firing epoch "
                   "[Guid:%lu]",
                   epoch->guid);
        if (epoch->termination_exit_guid) {
          arts_edt_satisfy_slot(
              epoch->termination_exit_guid, epoch->termination_exit_slot,
              (arts_guid_t)(total_finish), DB_MODE_VAL, NULL, 0);
        } else {
          arts_shutdown_epoch_fire(epoch->guid);
        }
      }
      return false;
    }

    /*
     * Multi-node: two-phase confirmation protocol.
     * Protected by the outstanding gate in arts_epoch_reduce_submit (single
     * entrant), but CAS on the PHASE_2 -> PHASE_3 transition provides
     * defense-in-depth.
     */
    if (epoch->phase == (unsigned int)PHASE_2 &&
        epoch->last_active_count == total_active &&
        epoch->last_finished_count == total_finish) {
      unsigned int old_phase = arts_atomic_cswap(
          &epoch->phase, (unsigned int)PHASE_2, (unsigned int)PHASE_3);
      if (old_phase == (unsigned int)PHASE_2) {
        ARTS_DEBUG("arts_epoch_check: CAS won PHASE_2->PHASE_3, firing epoch "
                   "[Guid:%lu]",
                   epoch->guid);
        if (epoch->termination_exit_guid) {
          arts_edt_satisfy_slot(
              epoch->termination_exit_guid, epoch->termination_exit_slot,
              (arts_guid_t)(total_finish), DB_MODE_VAL, NULL, 0);
        } else {
          arts_shutdown_epoch_fire(epoch->guid);
        }
      }
      return false;
    }
    // Counts differ from last round or first time: record and request another.
    epoch->last_active_count = total_active;
    epoch->last_finished_count = total_finish;
    epoch->phase = (unsigned int)PHASE_2;
    return true;
  }
  epoch->phase = (unsigned int)PHASE_1;
  return (epoch->queued == 0);
}

/* Home-routed reduce core (OOO_EPOCH_SEND): fold a remote rank's reported
 * active/finished counts into the home epoch's global tally and drive the
 * Mattern two-phase termination check.  Pure core on the dispatch-acquired
 * epoch — the engine's ref keeps the epoch alive across the PHASE_3
 * arts_epoch_delete (which re-acquires + mark_delete; the deferred free runs
 * after this returns). */
void arts_handler_epoch_send(void *item, void *vargs) {
  arts_epoch_t *epoch = (arts_epoch_t *)item;
  struct arts_ooo_args_epoch_send_s *a = vargs;
  arts_guid_t epoch_guid = a->epoch_guid;
  unsigned int total_active =
      arts_atomic_add(&epoch->global_active_count, a->active);
  unsigned int total_finish =
      arts_atomic_add(&epoch->global_finished_count, a->finish);
  uint64_t outstanding_before = epoch->outstanding;
  if (arts_atomic_sub_u64(&epoch->outstanding, 1) == 1) {
    total_active += epoch->active_count;
    total_finish += epoch->finished_count;

    ARTS_DEBUG("arts_epoch_reduce_submit [Guid:%lu]: total_active=%u, "
               "total_finish=%u, "
               "phase=%u, outstanding_before=%lu, queued=%lu",
               epoch_guid, total_active, total_finish, epoch->phase,
               outstanding_before, epoch->queued);

    // Reset for the next round
    epoch->global_active_count = 0;
    epoch->global_finished_count = 0;

    if (arts_epoch_check(epoch, total_active, total_finish)) {
      ARTS_DEBUG("  arts_epoch_check returned TRUE - broadcasting new request");
      arts_atomic_add_u64(&epoch->outstanding, arts_global_rank_count - 1);
      arts_epoch_request_broadcast(epoch_guid);
      // A better idea will be to know when to kick off a new round
      // the checkinCount == 0 indicates there is a new round can be kicked
      // off
      //                arts_atomic_sub(&epoch->checkinCount, 1);
    } else {
      ARTS_DEBUG(
          "  arts_epoch_check returned FALSE - epoch completed or advancing "
          "to phase %u",
          epoch->phase);
      arts_atomic_sub_u64(&epoch->outstanding, 1);
      /* Race fix: arts_epoch_inc_finished (home rank) sets queued→0
       * and tries CAS outstanding 0→rank_count.  If outstanding was
       * still 1 at that moment the CAS fails and nobody restarts the
       * broadcast.  Re-check here after we brought outstanding to 0. */
      uint64_t queued_now = epoch->queued;
      ARTS_DEBUG("arts_epoch_reduce_submit FALSE [Guid:%lu]: inner-sub done, "
                 "outstanding now 0, queued=%lu, phase=%u",
                 epoch_guid, queued_now, epoch->phase);
      if (queued_now == 0 && epoch->phase != (unsigned int)PHASE_3) {
        uint64_t old_out = arts_atomic_cswap_u64(&epoch->outstanding, 0,
                                                 arts_global_rank_count);
        if (old_out == 0) {
          ARTS_DEBUG(
              "arts_epoch_reduce_submit: restarting broadcast — queued=0 race "
              "caught, CAS SUCCESS");
          arts_epoch_request_broadcast(epoch_guid);
        } else {
          ARTS_DEBUG("arts_epoch_reduce_submit: queued=0 but CAS FAIL "
                     "(outstanding=%lu) "
                     "— arts_epoch_inc_finished already restarted",
                     old_out);
        }
      }
    }

    if (epoch->phase == PHASE_3) {
      ARTS_DEBUG("  Deleting epoch [Guid:%lu] - termination complete",
                 epoch_guid);
      arts_epoch_delete(epoch_guid, NULL);
      return;
    }
  } else {
    ARTS_DEBUG("arts_epoch_reduce_submit [Guid:%lu]: outstanding=%lu (still "
               "waiting for "
               "more responses)",
               epoch_guid, outstanding_before - 1);
  }
}

void arts_epoch_reduce_submit(arts_guid_t epoch_guid, unsigned int active,
                              unsigned int finish) {
  struct arts_ooo_args_epoch_send_s a = {
      .epoch_guid = epoch_guid, .active = active, .finish = finish};
  arts_ooo_dispatch_or_defer_guid(epoch_guid, OOO_EPOCH_SEND, &a, sizeof(a));
}

void arts_epoch_delete(arts_guid_t epoch_guid, arts_epoch_t *epoch) {
  // Can't call delete unless we already hit two barriers thus it must exit
  /* re-acquire via lookup_epoch whenever the caller did not
   * pass an already-owned pointer.  The epoch parameter is now treated as
   * advisory only — we always lookup to read pool_guid safely.  Mark_delete
   * (below) is the canonical drop path; the cb deleter (arts_epoch_deleter)
   * runs once the last ref is released.  For pool-backed entries the deleter
   * is a no-op (storage lives inside arts_epoch_pool_t), and the pool struct
   * itself is freed explicitly here when its outstanding count reaches 0. */
  (void)epoch;
  arts_shared_ptr_t e_h = arts_route_table_lookup_epoch(epoch_guid);
  arts_epoch_t *e = (arts_epoch_t *)arts_shared_get(e_h);
  if (!e) {
    return;
  }
  arts_guid_t pool_guid = e->pool_guid;
  arts_shared_release(&e_h);

  if (pool_guid) {
    /* Pool wrapper still uses ARTS_GUID_EDT type tag (arts_epoch_pool_t is a
     * different struct that does NOT embed ARTS_SHARED_FIELD), so the
     * type-aware lookup_*_safe variants do not apply.  Documented
     * exception: pool storage is freed explicitly here / arts_epoch_pool_clean
     * rather than via the route_table free_item dispatcher.  Migration
     * would require also embedding shared_t in arts_epoch_pool_t; deferred
     * since the pool object has no concurrent destroy race. */
    arts_epoch_pool_t *pool =
        (arts_epoch_pool_t *)arts_route_table_lookup_data(pool_guid);
    /* Drop the route_table slot for this individual epoch instance so
     * arts_epoch_wait's lookup-poll exits.  mark_delete consumes the
     * install-existence ref injected by add_item_race. */
    arts_route_table_mark_delete(epoch_guid);
    if (arts_guid_is_local(pool_guid)) {
      if (pool && !arts_atomic_sub(&pool->outstanding, 1)) {
        /* Pool's last outstanding entry is gone; detach its slot (NULL cb
         * deleter ⇒ mark_delete does not free the pool).  The
         * arts_epoch_pool_t storage itself is freed in the next
         * arts_epoch_pool_get / arts_epoch_pool_clean sweep. */
        arts_route_table_mark_delete(pool_guid);
        for (unsigned int i = 0; i < arts_global_rank_count; i++) {
          if (i != arts_global_rank_id) {
            arts_send_epoch_delete(i, epoch_guid);
          }
        }
      }
    } else if (pool) {
      for (unsigned int i = 0; i < pool->size; i++) {
        arts_route_table_mark_delete(pool->pool[i].guid);
      }
      /* Detach the pool's own route_table slot (NULL cb deleter ⇒ no free
       * here) and free the storage explicitly. */
      arts_route_table_mark_delete(pool_guid);
      arts_free(pool);
    }
  } else {
    /* Stand-alone heap epoch: mark_delete drops the install-existence ref;
     * once count hits 0 the dispatcher calls arts_epoch_deleter which
     * frees the struct. */
    arts_route_table_mark_delete(epoch_guid);

    if (arts_guid_is_local(epoch_guid)) {
      for (unsigned int i = 0; i < arts_global_rank_count; i++) {
        if (i != arts_global_rank_id) {
          arts_send_epoch_delete(i, epoch_guid);
        }
      }
    }
  }
}

void arts_yield() {
  TIME_EDT_EXEC_STOP();
  INCREMENT_NUM_YIELD_BY(1);
  arts_edt_ctx_t tl;
  arts_edt_ctx_save(&tl);
  TIME_YIELD_START();
  arts_node_info.scheduler();
  TIME_YIELD_STOP();
  arts_edt_ctx_restore(&tl);
  TIME_EDT_EXEC_START();
}

/*
 * arts_epoch_wait — Block current EDT until the given epoch completes.
 *
 * Increments the epoch's finished counter (this EDT is now "done" from the
 * epoch's perspective), then spin-polls the scheduler loop until the epoch
 * is removed from the route table (by arts_epoch_delete).  Also breaks out if
 * the thread's alive flag becomes false (e.g., arts_shutdown was called).
 *
 * Uses route-table presence instead of an in-struct completion flag so the
 * wait loop never dereferences an epoch after arts_epoch_delete frees it.
 */
bool arts_epoch_wait(arts_guid_t epoch_guid) {
  TIME_EDT_EXEC_STOP();
  arts_guid_t *guid = arts_check_epoch_is_root(epoch_guid);
  ARTS_INFO("arts_epoch_wait: Waiting on epoch [Guid:%lu]", epoch_guid);
  // For now lets leave this rule here
  if (guid) {
    arts_guid_t local = *guid;
    *guid = NULL_GUID; // Unset
    /* lookup_epoch + paired release.  Spin briefly while the slot is still
     * absent (epoch GUID was promised but add_item_race has not yet stored
     * the cb).  Release immediately — we only need confirmation that the
     * epoch exists; subsequent loops poll presence via lookup_epoch again. */
    arts_shared_ptr_t epoch_h = arts_route_table_lookup_epoch(local);
    arts_epoch_t *epoch = (arts_epoch_t *)arts_shared_get(epoch_h);
    if (!epoch) {
      // Epoch may still be absent in route table; spin briefly
      for (int retries = 0; !epoch && retries < 1000; retries++) {
        epoch_h = arts_route_table_lookup_epoch(local);
        epoch = (arts_epoch_t *)arts_shared_get(epoch_h);
      }
      if (!epoch) {
        ARTS_WARN("arts_epoch_wait: Epoch [Guid:%lu] not found in route table",
                  local);
        TIME_EDT_EXEC_START();
        return false;
      }
    }
    arts_shared_release(&epoch_h);
    arts_epoch_inc_finished(local);

    // Release all acquired DB dependencies before blocking so consumer EDTs
    // can proceed while this EDT waits on the epoch.
    arts_wait_release_dbs();

    INCREMENT_NUM_YIELD_BY(1);
    arts_edt_ctx_t tl;
    arts_edt_ctx_save(&tl);
    TIME_YIELD_START();
    while (arts_thread_info.alive) {
      /* Poll: when arts_epoch_delete -> mark_delete fires, the cb is detached
       * from the slot; lookup_epoch returns NULL and we exit.  Use
       * lookup_epoch (acquire/release pair) so we never read freed memory. */
      arts_shared_ptr_t e_h = arts_route_table_lookup_epoch(local);
      arts_epoch_t *e = (arts_epoch_t *)arts_shared_get(e_h);
      if (!e) {
        break;
      }
      arts_shared_release(&e_h);
      arts_node_info.scheduler();
    }
    // Continue running until the scheduler reports no more ready work
    while (arts_node_info.scheduler()) {
      ;
    }
    TIME_YIELD_STOP();
    arts_edt_ctx_restore(&tl);

    // Re-acquire all DB dependencies after the epoch completes.
    arts_wait_reacquire_dbs();

    arts_epoch_pool_clean();

    TIME_EDT_EXEC_START();
    return true;
  }
  TIME_EDT_EXEC_START();
  return false;
}

void arts_send_epoch_create(unsigned int rank, arts_guid_t epoch_guid,
                            arts_guid_t edt_guid, unsigned int slot) {
  struct arts_remote_epoch_init_packet_s packet;
  packet.epoch_guid = epoch_guid;
  packet.edt_guid = edt_guid;
  packet.slot = slot;
  arts_fill_packet_header(&packet.header, sizeof(packet), MSG_EPOCH_CREATE);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_handler_epoch_create(void *pack) {
  ARTS_DEBUG("Net Epoch Init Rec");
  struct arts_remote_epoch_init_packet_s *packet =
      (struct arts_remote_epoch_init_packet_s *)pack;
  arts_guid_t local_epoch_guid = packet->epoch_guid;
  arts_epoch_alloc(&local_epoch_guid, packet->edt_guid, packet->slot);
  packet->epoch_guid = local_epoch_guid;
}

void arts_send_epoch_init_pool(unsigned int rank, unsigned int pool_size,
                               arts_guid_t start_guid, arts_guid_t pool_guid) {
  //    ARTS_INFO("Net Epoch Init Pool Send: %u %lu %lu", rank, start_guid,
  //    pool_guid);
  struct arts_remote_epoch_init_pool_packet_s packet;
  packet.pool_size = pool_size;
  packet.start_guid = start_guid;
  packet.pool_guid = pool_guid;
  arts_fill_packet_header(&packet.header, sizeof(packet), MSG_EPOCH_INIT_POOL);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_handler_epoch_init_pool(void *pack) {
  struct arts_remote_epoch_init_pool_packet_s *packet =
      (struct arts_remote_epoch_init_pool_packet_s *)pack;
  arts_guid_t local_pool_guid = packet->pool_guid;
  arts_guid_t local_start_guid = packet->start_guid;
  arts_epoch_pool_t *pool = arts_epoch_pool_create(
      &local_pool_guid, packet->pool_size, &local_start_guid);
  arts_link_epoch_pool_to_tls(pool);
  packet->pool_guid = local_pool_guid;
  packet->start_guid = local_start_guid;
}

void arts_send_epoch_request(unsigned int rank, arts_guid_t guid) {
  struct arts_remote_guid_only_packet_s packet;
  packet.guid = guid;
  arts_fill_packet_header(&packet.header, sizeof(packet), MSG_EPOCH_REQUEST);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

/* MSG_EPOCH_REQUEST RX is inline-decoded in the dispatcher → arts_epoch_reply.
 */

void arts_send_epoch_send(unsigned int rank, arts_guid_t guid,
                          unsigned int active, unsigned int finish) {
  struct arts_remote_epoch_send_packet_s packet;
  packet.epoch_guid = guid;
  packet.active = active;
  packet.finish = finish;
  arts_fill_packet_header(&packet.header, sizeof(packet), MSG_EPOCH_SEND);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

/* MSG_EPOCH_SEND RX is inline-decoded in the dispatcher →
 * arts_epoch_reduce_submit. */

void arts_send_epoch_delete(unsigned int rank, arts_guid_t epoch_guid) {
  struct arts_remote_guid_only_packet_s packet;
  packet.guid = epoch_guid;
  arts_fill_packet_header(&packet.header, sizeof(packet), MSG_EPOCH_DELETE);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_handler_epoch_delete(void *pack) {
  struct arts_remote_guid_only_packet_s *packet =
      (struct arts_remote_guid_only_packet_s *)pack;
  arts_epoch_delete(packet->guid, NULL);
}
