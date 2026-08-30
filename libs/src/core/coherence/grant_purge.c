/* SPDX-License-Identifier: Apache-2.0
 *
 * The PURGE release policy's half of the migrating grant — the home hub.
 *
 * A holder gives the write right back the moment its last local writer
 * finishes, unasked.  The home therefore sends a holder nothing, ever: it
 * queues demand locally and hands the right out again from its own idle
 * state.  There is no revocation round, no notice to the current holder, and
 * no second decrementer of the holder's word — every transition of it is
 * committed by a single CAS on the rank that owns it.
 *
 * That leaves the home with two things to arbitrate and one place to do it.
 * The two are the request FIFO and a single-slot return latch; the one place
 * is THE TAIL below, entered under the transfer baton from five points — a
 * return accepted, a round closing, a request arriving at an idle home, the
 * home's own last writer finishing, and the create-time transition that makes
 * the home the idle holder.  The tail serves AT MOST ONE round and then holds
 * the baton until that round's CONFIRM: a serve loop drives a second grant
 * out of a home that has already given it away, and a baton dropped mid-round
 * lets a return be accepted underneath the round whose CONFIRM then names a
 * rank that has already handed the right back.
 *
 * Everything hands off by CAS or by ending in that tail.  Nothing here
 * publishes a fact and assumes a consumer: a queued request re-checked after
 * a baton release, a parked return consumed from either side, and the word's
 * own possession edge are each somebody's obligation, discharged exactly
 * once.
 */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/directory.h"
#include "arts/coherence/handlers.h"
#include "arts/db.h"
#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/print.h"
#include "arts/system/schedfuzz.h"
#include "arts/system/threads.h"
#include "arts/transport/net.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/counter/Preamble.h"

/* ===== The word's count-dropping edge ==================================== */

unsigned int arts_grant_purge_release_next(unsigned int cur, bool purgeable,
                                           unsigned int *out_act) {
  if (!ARTS_GRANT_OWN_OF(cur)) {
    /* Nothing is held, so nothing is owed: a double or spurious release. */
    *out_act = ARTS_GRANT_ACT_NONE;
    return cur;
  }
  unsigned int count = ARTS_GRANT_COUNT_OF(cur);
  if (count == 0u) {
    /* Possession with no hold under it.  A decrement here would borrow the
     * possession bit and leave a count of 2^30-1 that nothing can ever drain,
     * so the edge is refused rather than committed. */
    *out_act = ARTS_GRANT_ACT_ILLEGAL;
    return cur;
  }
  if (count > 1u) {
    *out_act = ARTS_GRANT_ACT_NONE;
    return ARTS_GRANT_OWN | (count - 1u);
  }
  if (purgeable) {
    *out_act = ARTS_GRANT_ACT_RETURN;
    return 0u;
  }
  /* The home's own idle edge: it keeps possession — there is nobody to hand
   * it back to — and becomes the server for whatever queued while it wrote. */
  *out_act = ARTS_GRANT_ACT_TAIL;
  return ARTS_GRANT_OWN;
}

/* ===== The return wire =================================================== */

/* Holder → home: "I am done; the right is yours again."  Data-less by
 * construction — the write-through publish that preceded this release already
 * put the bytes at the home, and a sentinel block has none. */
static void send_grant_return(struct arts_db_cache_s *cache) {
  arts_guid_t db_guid = cache->db_guid;
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  INCREMENT_NUM_GRANT_PURGE_RETURN_BY(1);
  struct arts_msg_grant_return_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_GRANT_RETURN);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    /* Unreachable in a consistent state (the home never gives the right back
     * to itself), but routed like the wire path rather than special-cased. */
    struct arts_ooo_args_db_grant_return_s args = {
        .returner = p.header.rank,
        .db_guid = db_guid,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_GRANT_RETURN, &args,
                                    sizeof(args));
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

/* ===== THE TAIL ========================================================== */

enum grant_tail_entry_e {
  GRANT_TAIL_SERVE,   /* the baton was just won with the home idle */
  GRANT_TAIL_RECHECK, /* a round closed; put the baton down and look again */
  GRANT_TAIL_ACCEPT,  /* a return was latched and consumed by this caller */
};

/* Try to become the server.  Only a won baton may enter the tail: entering
 * unconditionally puts two consumers on a single-consumer FIFO, and on a loss
 * the current holder's own re-check picks the newly published state up. */
static void grant_tail(struct arts_db_s *db, enum grant_tail_entry_e entry,
                       unsigned int returner);

static void grant_tail_try_serve(struct arts_db_s *db) {
  unsigned int expected = 0u;
  if (atomic_compare_exchange_strong_explicit(&db->invalidate_in_flight,
                                              &expected, 1u,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
    grant_tail(db, GRANT_TAIL_SERVE, ARTS_GRANT_NO_RETURNER);
  }
}

static void grant_tail(struct arts_db_s *db, enum grant_tail_entry_e entry,
                       unsigned int returner) {
  struct arts_db_cache_s *cache = &db->cache;
  const unsigned int home = arts_global_rank_id;
  unsigned int latched = returner;

  if (entry == GRANT_TAIL_ACCEPT) {
    goto accept;
  }
  if (entry == GRANT_TAIL_RECHECK) {
    goto release_recheck;
  }

serve: {
  unsigned int front;
  struct arts_rdzv_landing_s front_rdzv;
  /* The front's RANK is read before any claim: a self front must never see
   * possession transiently cleared, because every concurrent home-local
   * acquirer would read that window as "no grant" and open its own request.
   * The peek is stable — popping is the baton holder's alone. */
  if (arts_home_grantreq_queue_peek(&db->pending_rw, &front, &front_rdzv,
                                    NULL) &&
      atomic_load_explicit(&db->rw_holder, memory_order_acquire) == home) {
    if (front == home) {
      /* Nothing travels and nothing installs: the home already holds both the
       * right and the bytes, so the round completes here, synchronously. */
      unsigned int self_rank;
      struct arts_rdzv_landing_s self_rdzv;
      (void)arts_home_grantreq_queue_pop(&db->pending_rw, &self_rank,
                                         &self_rdzv, NULL);
      if (self_rdzv.cookie != 0) {
        arts_db_buf_landing_recycle(
            cache, (struct arts_db_buffer_s *)(uintptr_t)self_rdzv.cookie);
      }
      /* Clear the coalescing flag BEFORE draining, and with a full barrier.
       * The producer is push-then-flag-CAS, so a waiter this drain misses
       * finds the flag already clear and opens its own request; the other
       * order strands it behind a home whose fast path never re-requests.
       * Both halves of that Dekker are RMW-or-seq_cst on purpose. */
      __atomic_store_n(&cache->grant_req_in_flight, 0u, __ATOMIC_SEQ_CST);
      arts_db_drain_pending_rw_after_grant(cache, /*version=*/0,
                                           /*has_next=*/false);
      arts_ooo_drain_guid(cache->db_guid);
      goto release_recheck;
    }
    /* Remote front: claim the right away from the home in ONE CAS.  It is
     * what separates "the home can grant" from "already granted this round"
     * (both leave rw_holder naming the home with the queue non-empty) and
     * from "home writers are live", where it fails and the home's own release
     * inherits the duty. */
    if (arts_atomic_cswap(&cache->writer_count, ARTS_GRANT_SEED_IDLE, 0u) ==
        ARTS_GRANT_SEED_IDLE) {
      unsigned int target;
      struct arts_rdzv_landing_s target_rdzv;
      uint64_t target_has = ARTS_GRANT_VERSION_NONE;
      (void)arts_home_grantreq_queue_pop(&db->pending_rw, &target,
                                         &target_rdzv, &target_has);
      db->pending_install_owner = target;
      /* Judge the requester's copy against the canonical axis, and read that
       * axis AFTER the claim above.  The claim is what holds it still: the
       * home may only serve while it possesses the right, which the previous
       * holder gave back — and it gave it back only once its own publish had
       * landed and been stamped.  So no publish can be in flight here, and an
       * equality found now cannot go stale under the send.  Reading before
       * the claim would compare against an axis a landing publish could still
       * advance, and promise a freshness the bytes no longer have.
       *
       * A requester that reported nothing is never suppressed: it has no copy
       * to keep, so "equal" would be equal to nothing. */
      bool data_less = (target_has != ARTS_GRANT_VERSION_NONE &&
                        target_has == arts_db_grant_serve_version(db));
      arts_db_send_grant_response(cache, target, &target_rdzv, data_less);
      return; /* baton held through this round's CONFIRM */
    }
    /* Home writers are live.  Put the baton down and stop — re-arming here
     * spins until they finish and, on a progress thread, spins against the
     * very round a writer's release is waiting on.  One re-check closes the
     * window where the last of them reached its idle edge just before our
     * release made the baton available. */
    atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
    atomic_thread_fence(memory_order_seq_cst);
    arts_sched_fuzz_point(); /* widen the release<->re-check window */
    if (arts_atomic_read(&cache->writer_count) == ARTS_GRANT_SEED_IDLE) {
      unsigned int expected = 0u;
      if (atomic_compare_exchange_strong_explicit(
              &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
              memory_order_acquire)) {
        goto serve;
      }
    }
    return;
  }
}

release_recheck:
  atomic_store_explicit(&db->invalidate_in_flight, 0u, memory_order_release);
  /* Dekker-style publication: the baton-release store must be globally
   * visible BEFORE the loads below, or a requester that pushed and lost its
   * baton CAS inside the window is missed — a plain release-store followed by
   * loads permits exactly that StoreLoad reordering. */
  atomic_thread_fence(memory_order_seq_cst);
  arts_sched_fuzz_point(); /* widen the release<->re-check window */
  {
    unsigned int expected = 0u;
    if (arts_home_grantreq_queue_pending(&db->pending_rw) &&
        atomic_load_explicit(&db->rw_holder, memory_order_acquire) == home &&
        atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      goto serve;
    }
  }
  {
    unsigned int parked =
        atomic_load_explicit(&db->pending_return_from, memory_order_acquire);
    unsigned int holder =
        atomic_load_explicit(&db->rw_holder, memory_order_acquire);
    unsigned int expected = 0u;
    if (parked != ARTS_GRANT_NO_RETURNER &&
        (holder == parked || holder == home) &&
        atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      unsigned int want = parked;
      if (atomic_compare_exchange_strong_explicit(
              &db->pending_return_from, &want, ARTS_GRANT_NO_RETURNER,
              memory_order_acq_rel, memory_order_acquire)) {
        latched = parked;
        goto accept;
      }
      /* The returner's own handler took the slot.  Never leave holding the
       * baton: go round again so it is released and re-examined. */
      goto release_recheck;
    }
  }
  return;

accept: {
  /* The ex-holder keeps the bytes it wrote, so its copy must be on the sharer
   * roster BEFORE any next owner can be served — a serve that overtook this
   * would leave that copy outside the round that retires it. */
  arts_db_grant_note_ex_holder(db, latched);
  atomic_store_explicit(&db->rw_holder, home, memory_order_seq_cst);
  /* Take possession back.  The write right is issued from here one rank at a
   * time, so the home provably holds nothing at this point and the CAS
   * commits.  The one state that could have made it fail — two ranks each
   * seeded as holder of one block, each handing back what it believed it
   * held — is a create pattern the programming model no longer admits, so
   * this stays a strict precondition rather than a branch kept alive for a
   * case that cannot arise. */
  unsigned int prev =
      arts_atomic_cswap(&cache->writer_count, 0u, ARTS_GRANT_SEED_IDLE);
  assert(prev == 0u &&
         "a returned write right lands on a home that holds nothing");
  (void)prev;
  INCREMENT_NUM_GRANT_PURGE_ACCEPT_BY(1);
  arts_sched_fuzz_point(); /* widen the accept<->serve window */
  goto serve;
}
}

/* ===== Home handler: a return arrived ==================================== */

/* Cat-B pure body (OoO g_ooo_table[OOO_DB_GRANT_RETURN]): the engine has
 * acquired the home db_s and pinned a ref across this call.  A return is
 * never rejected — it is accepted now or parked for the round close already
 * inbound — so the slot is handled publish-then-check on both sides and
 * consumed by CAS: exactly one of the two actors takes any given return. */
void arts_db_grant_return_arrived(struct arts_db_s *db,
                                  unsigned int returner) {
  /* The latch is a home field, past the size of a cache-only stub.  A return
   * that overtook the block's home CREATE is deferred by the engine and
   * replayed on the install's drain, so reaching this body with no home
   * directory would be a lost return — a wedge, never a drop. */
  assert(db->home_initialized &&
         "a return reaches the block's home, which owns the directory");

  /* Into an EMPTY slot, never over an occupant.  Grants are serialized, so at
   * most one hand-back can be outstanding and the slot is always free here;
   * writing unconditionally would silently drop the occupant — and with it
   * the roster registration its acceptance owes.  Loud, because after the
   * create contract narrowed there is no legitimate way to reach it. */
  unsigned int slot_empty = ARTS_GRANT_NO_RETURNER;
  bool latched = atomic_compare_exchange_strong_explicit(
      &db->pending_return_from, &slot_empty, returner, memory_order_seq_cst,
      memory_order_seq_cst);
  assert(latched && "one write right yields one outstanding hand-back");
  (void)latched;
  atomic_thread_fence(memory_order_seq_cst);
  unsigned int holder =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  if (holder == returner || holder == arts_global_rank_id) {
    unsigned int expected = 0u;
    if (atomic_compare_exchange_strong_explicit(
            &db->invalidate_in_flight, &expected, 1u, memory_order_acq_rel,
            memory_order_acquire)) {
      unsigned int want = returner;
      if (atomic_compare_exchange_strong_explicit(
              &db->pending_return_from, &want, ARTS_GRANT_NO_RETURNER,
              memory_order_acq_rel, memory_order_acquire)) {
        grant_tail(db, GRANT_TAIL_ACCEPT, returner);
        return;
      }
      /* The round close took it first; we hold the baton and must not keep
       * it — hand it back through the same re-check every exit uses. */
      grant_tail(db, GRANT_TAIL_RECHECK, ARTS_GRANT_NO_RETURNER);
      return;
    }
  }
  /* Parked: the baton holder's tail re-examines the slot when its round
   * closes.  Losing the baton race is not a failure — it is the hand-off. */
  INCREMENT_NUM_GRANT_PURGE_LATCHED_BY(1);
}

/* The standalone vehicle's arrival.  A hand-back that rode a publish reaches
 * arts_db_grant_return_arrived from that publish's landing instead; there is
 * one acceptance, and the vehicle is not part of it. */
void arts_handler_db_grant_return(void *item_v, void *args_v) {
  struct arts_ooo_args_db_grant_return_s *a =
      (struct arts_ooo_args_db_grant_return_s *)args_v;
  arts_db_grant_return_arrived((struct arts_db_s *)item_v, a->returner);
}

/* ===== Release-policy seams ============================================== */

void arts_db_grant_home_idle_transition(struct arts_db_s *db) {
  atomic_store_explicit(&db->rw_holder, arts_global_rank_id,
                        memory_order_seq_cst);
  /* That store and the tail's first queue read are a StoreLoad pair, ordered
   * only by the fence the tail's release re-check carries.  So this enters
   * through the baton like every other entry: on a win we serve whatever
   * queued behind the holder we just replaced, and on a loss the rank holding
   * the baton re-reads the directory after its own release and finds us. */
  grant_tail_try_serve(db);
}

void arts_db_grant_request_arrived(struct arts_db_cache_s *cache,
                                   struct arts_db_s *db,
                                   unsigned int requester) {
  (void)cache;
  (void)requester;
  /* Publish-then-check: the requester is already on the FIFO, and the fence
   * orders that push against this load, so a home that reads a holder other
   * than itself is a home whose eventual return will find the request.  There
   * is no notice to that holder — it hands the right back on its own. */
  atomic_thread_fence(memory_order_seq_cst);
  if (atomic_load_explicit(&db->rw_holder, memory_order_acquire) !=
      arts_global_rank_id) {
    return;
  }
  grant_tail_try_serve(db);
}

void arts_db_grant_round_close(struct arts_db_cache_s *cache,
                               struct arts_db_s *db) {
  (void)cache;
  /* The caller has already flipped rw_holder to the rank that just confirmed.
   * Nothing is revoked here; the tail only puts the baton down and re-examines
   * the queue and the latch behind the fence. */
  grant_tail(db, GRANT_TAIL_RECHECK, ARTS_GRANT_NO_RETURNER);
}

bool arts_db_grant_release_skip(struct arts_db_cache_s *cache) {
  /* Possession is not a hold here, so the guard tests the count field alone:
   * the home's idle state carries possession with no writer under it, and
   * decrementing that word would borrow the possession bit. */
  return ARTS_GRANT_COUNT_OF(arts_atomic_read(&cache->writer_count)) == 0u;
}

/* The two count-dropping edges share one committer: one CAS carries the
 * decrement and whatever possession change it implies, so no window exists in
 * which the count has fallen but the right has not yet moved. */
static void grant_commit_count_edge(struct arts_db_cache_s *cache) {
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  bool purgeable = !is_home && arts_global_rank_count > 1;
  unsigned int act = ARTS_GRANT_ACT_NONE;
  for (;;) {
    unsigned int cur = arts_atomic_read(&cache->writer_count);
    unsigned int next = arts_grant_purge_release_next(cur, purgeable, &act);
    if (act == ARTS_GRANT_ACT_ILLEGAL) {
      assert(false && "a count-dropping edge found possession with no hold");
      return;
    }
    if (next == cur) {
      return; /* nothing to commit */
    }
    if (arts_atomic_cswap(&cache->writer_count, cur, next) == cur) {
      break;
    }
  }
  if (act == ARTS_GRANT_ACT_RETURN) {
    send_grant_return(cache);
    return;
  }
  if (act == ARTS_GRANT_ACT_TAIL) {
    grant_tail_try_serve(arts_db_of_cache(cache));
  }
}

void arts_db_grant_release_commit(struct arts_db_cache_s *cache) {
  grant_commit_count_edge(cache);
}

/* ===== The hand-back as a rider on this release's own publish =========== */

bool arts_db_grant_release_claim(struct arts_db_cache_s *cache,
                                 bool will_publish) {
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  if (!will_publish || is_home || arts_global_rank_count <= 1) {
    /* Nothing to ride, or nothing to hand back.  The home is excluded on
     * purpose and not only because it has no one to hand the right to: its
     * idle edge SERVES, and serving before the publish has stamped the
     * canonical axis would hand the next owner a version this release has
     * already moved past. */
    return false;
  }
  /* ONE attempt, on the exact word that says "this rank holds it and I am its
   * only writer".  A retry loop would be wrong rather than merely slow: any
   * other word means a concurrent writer, whose own release must take the
   * count-dropping edge, and committing an ordinary decrement here would then
   * be followed by a second one after the publish. */
  if (arts_atomic_cswap(&cache->writer_count, ARTS_GRANT_SEED_HOLDING, 0u) !=
      ARTS_GRANT_SEED_HOLDING) {
    return false;
  }
  /* Armed, and idle for as long as it stays armed: the local fast path
   * refuses a word with no possession, and a new grant is only ever issued
   * from a directory naming the home — which this hand-back is what changes.
   * So nothing can raise the count under an armed obligation. */
  assert(ARTS_GRANT_COUNT_OF(arts_atomic_read(&cache->writer_count)) == 0u &&
         "an armed hand-back rests on a word with no holds");
  (void)arts_atomic_swap(&cache->pending_grant_return, 1u);
  return true;
}

bool arts_db_grant_return_claim_leg(struct arts_db_cache_s *cache) {
  if (arts_atomic_cswap(&cache->pending_grant_return, 1u, 0u) != 1u) {
    return false;
  }
  assert(ARTS_GRANT_COUNT_OF(arts_atomic_read(&cache->writer_count)) == 0u &&
         "a hand-back is discharged from a word with no holds");
  INCREMENT_NUM_GRANT_PURGE_FLAG_BY(1);
  return true;
}

void arts_db_grant_release_settle(struct arts_db_cache_s *cache) {
  /* Still armed after the publish returned: the commit leg this obligation
   * meant to ride had already been sent when it was armed — the release
   * joined a flight instead of driving one — so no leg will carry it and it
   * converts to the message of its own.  A lost CAS means a leg took it. */
  if (arts_atomic_cswap(&cache->pending_grant_return, 1u, 0u) != 1u) {
    return;
  }
  assert(ARTS_GRANT_COUNT_OF(arts_atomic_read(&cache->writer_count)) == 0u &&
         "a hand-back is discharged from a word with no holds");
  send_grant_return(cache);
}

/* Wake a drained waiter WITHOUT adding a hold for it: its hold was installed
 * before the walk began, together with everyone else's. */
static void commit_wake_cb(arts_guid_t edt_guid, unsigned int slot,
                           void *vctx) {
  (void)vctx;
  mark_edt_secured_by_guid(edt_guid, slot);
  mark_edt_ready_by_guid(edt_guid, slot);
}

unsigned int arts_db_grant_commit_drain(struct arts_db_cache_s *cache,
                                        uint64_t version) {
  (void)version;
  /* Take the whole chain and COUNT it before waking any of it, then convert
   * the commit's single hold into one hold per waiter in a single step.  The
   * count has to precede the waking: a waiter woken while the walk is still
   * going could release, reach the edge and hand the right back, and the next
   * waiter's hold would then land on a word that possesses nothing — a count
   * under no owner, and a write turn on a block this rank no longer holds.
   *
   * Nothing else can touch the count in between: this policy has no external
   * decrementer, and none of these waiters is runnable yet.  The point of
   * doing it this way is what the LAST of them then sees — a word carrying
   * exactly its own hold, which is the one state whose release can claim the
   * whole right and hand it back on its own publish instead of paying for a
   * message of its own.
   *
   * The chain is taken AFTER the coalescing flag was cleared, so the Dekker
   * with the producer is unchanged: a waiter pushed past this take finds the
   * flag already clear and opens its own request. */
  arts_lf_link_t *chain = arts_pending_rw_queue_take(&cache->pending_rw);
  unsigned int n = arts_pending_rw_chain_count(chain);
  if (n > 1u) {
    arts_atomic_add(&cache->writer_count, n - 1u);
  }
  arts_pending_rw_chain_wake(chain, commit_wake_cb, NULL);
  return n;
}

void arts_db_grant_commit_finish(struct arts_db_cache_s *cache,
                                 unsigned int drained) {
  if (drained > 0u) {
    /* The commit's hold became those waiters' holds; the last of them to
     * release closes the edge, and its hand-back rides that release's own
     * publish.  It may get there before this rank's install has even been
     * confirmed at the home — that arrival finds the directory still naming
     * the previous holder, parks in the home's single-slot latch, and is taken
     * by the confirmation itself, which is exactly what the latch is for. */
    return;
  }
  /* Nobody wanted it after all: there is no release to carry anything, so the
   * commit closes its own edge and hands the right straight back. */
  grant_commit_count_edge(cache);
}

void arts_db_grant_install(struct arts_db_cache_s *cache) {
  /* Possession plus the drain guard, in one transition from "holds nothing".
   * An add cannot state that precondition, and possession carried by an add
   * would leave the bit's meaning depending on what the word happened to
   * hold.  Grants are baton-serialized and the serve clears possession before
   * it ships, so the precondition is met by construction. */
  unsigned int prev =
      arts_atomic_cswap(&cache->writer_count, 0u, ARTS_GRANT_SEED_HOLDING);
  assert(prev == 0u && "a grant installs only on a rank that holds nothing");
  (void)prev;
}
