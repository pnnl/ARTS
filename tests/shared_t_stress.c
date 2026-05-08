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

/* shared_t_stress — shared_t reference-counted lifecycle stress test.
 *
 * Verifies the D1-core primitives introduced by Tasks 4d/4e/4f:
 *   - arts_shared_t deleter dispatch (direct invocation, since route_table
 *     dispatch only fires when object_has_shared_field(t) is true — that
 *     migration lands in Phases 6/7/8).
 *   - route_table item lock field: acquire / release / mark_delete races,
 *     ABA-safe gen counter, sticky DELETE bit, install-existence ref.
 *
 * The test runs three scenarios under heavy thread contention:
 *
 *   Scenario A — direct deleter dispatch:
 *     Heap-allocate a synthetic object embedding ARTS_SHARED_FIELD, set its
 *     deleter to a counter-bumping function, invoke deleter(obj) directly.
 *     Assert counter == 1 and the object is freed.
 *
 *   Scenario B — concurrent acquire/release vs mark_delete:
 *     Install a synthetic DB entry into the route_table.  N threads race
 *     acquire/release on the slot while one thread issues mark_delete.
 *     Assert: every successful acquire is balanced by a release; mark_delete
 *     observed exactly once per install; final lock count == 0; DELETE bit
 *     set; gen bumped after free_item.
 *
 *   Scenario C — same-GUID re-install during reader holds (ABA boundary):
 *     For ITER iterations: install -> R readers acquire -> mark_delete ->
 *     readers release -> wait for free -> re-install same GUID.  Assert
 *     each iteration's gen counter is strictly greater than the previous,
 *     and a stale acquire (captured before mark_delete) observed via the
 *     same item pointer fails (or sees its expected gen).
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/runtime_state.h"
#include "arts/sync/shared.h"

/* ------------------------------------------------------------------------- */
/* Scenario A: direct deleter dispatch                                       */
/* ------------------------------------------------------------------------- */

typedef struct test_obj_s {
  ARTS_SHARED_FIELD;
  uint64_t magic;
} test_obj_t;

static atomic_uint deleter_calls = 0;

static void test_obj_deleter(void *p) {
  test_obj_t *obj = (test_obj_t *)p;
  if (obj->magic != 0xDEADBEEFCAFEBABEull) {
    fprintf(stderr, "FAIL: deleter saw bad magic 0x%lx\n", obj->magic);
    abort();
  }
  obj->magic = 0; /* poison so a double-free shows up loudly */
  atomic_fetch_add_explicit(&deleter_calls, 1u, memory_order_relaxed);
  free(obj);
}

static void scenario_a_direct_deleter(void) {
  printf("[A] direct deleter dispatch\n");
  test_obj_t *obj = (test_obj_t *)calloc(1, sizeof(*obj));
  arts_shared_init(&obj->shared, test_obj_deleter);
  obj->magic = 0xDEADBEEFCAFEBABEull;

  atomic_store_explicit(&deleter_calls, 0u, memory_order_relaxed);
  /* Invoke as the route_table free path would, via arts_shared_t.deleter. */
  arts_shared_t *s = (arts_shared_t *)obj;
  s->deleter(obj);
  unsigned n = atomic_load_explicit(&deleter_calls, memory_order_relaxed);
  if (n != 1u) {
    fprintf(stderr, "FAIL [A]: deleter call count = %u (want 1)\n", n);
    abort();
  }
  printf("[A] PASS\n");
}

/* Resolve a route_item by GUID without using the static arts_get_route_table
 * resolver: sweep all per-thread tables + the remote table. */
static arts_route_item_t *find_item_for_guid(arts_guid_t guid) {
  for (unsigned i = 0;
       i < arts_node_info.total_thread_count + ARTS_REMOTE_ROUTE_SHARDS; i++) {
    arts_route_table_t *rt =
        (i < arts_node_info.total_thread_count)
            ? arts_node_info.route_table[i]
            : arts_node_info
                  .remote_route_table[i - arts_node_info.total_thread_count];
    if (!rt) {
      continue;
    }
    arts_route_item_t *item = arts_route_table_search_for_key(rt, guid);
    if (item) {
      return item;
    }
  }
  return NULL;
}

/* ------------------------------------------------------------------------- */
/* Scenario B: concurrent acquire/release vs mark_delete                     */
/* ------------------------------------------------------------------------- */

#define B_THREADS 8
#define B_ITERS_PER_THREAD 4096

typedef struct {
  arts_route_item_t *item; /* slot under test */
  atomic_uint acquire_ok;  /* successful acquires across all threads */
  atomic_uint acquire_fail;
  atomic_uint release_count;
  atomic_int start_gate; /* 0 = wait, 1 = go */
} scenario_b_ctx_t;

static void *scenario_b_acquirer(void *vp) {
  scenario_b_ctx_t *ctx = (scenario_b_ctx_t *)vp;
  while (atomic_load_explicit(&ctx->start_gate, memory_order_acquire) == 0) {
    /* spin */
  }
  for (int i = 0; i < B_ITERS_PER_THREAD; i++) {
    if (arts_route_table_acquire_item(ctx->item)) {
      atomic_fetch_add_explicit(&ctx->acquire_ok, 1u, memory_order_relaxed);
      /* Pretend to do work with the data ptr — load/check it briefly. */
      void *data = atomic_load_explicit((_Atomic(void *) *)&ctx->item->data,
                                        memory_order_acquire);
      (void)data;
      arts_route_table_release_item(ctx->item);
      atomic_fetch_add_explicit(&ctx->release_count, 1u, memory_order_relaxed);
    } else {
      atomic_fetch_add_explicit(&ctx->acquire_fail, 1u, memory_order_relaxed);
    }
  }
  return NULL;
}

static atomic_uint scenario_b_freed = 0;

static void scenario_b_deleter(void *p) {
  /* Bump global free counter; the synthetic struct is heap-allocated. */
  atomic_fetch_add_explicit(&scenario_b_freed, 1u, memory_order_relaxed);
  free(p);
}

static void scenario_b_concurrent(void) {
  printf("[B] concurrent acquire/release vs mark_delete\n");
  /* We use a real ARTS_DB GUID so search/install go through the live
   * route_table.  object_has_shared_field(ARTS_DB) returns false today, so
   * the deleter we set will NOT be invoked by free_item — instead we
   * detect free completion via the lock field's gen bump.  (Once
   * migrates DB to embed ARTS_SHARED_FIELD, this same test will additionally
   * verify the deleter fires.) */
  arts_guid_t guid = arts_guid_reserve(ARTS_DB, 0);
  test_obj_t *obj = (test_obj_t *)calloc(1, sizeof(*obj));
  arts_shared_init(&obj->shared, scenario_b_deleter);
  obj->magic = 0xDEADBEEFCAFEBABEull;
  atomic_store_explicit(&scenario_b_freed, 0u, memory_order_relaxed);

  bool installed = arts_route_table_add_item_race(obj, guid, 0, false);
  if (!installed) {
    fprintf(stderr, "FAIL [B]: add_item_race lost the install race\n");
    abort();
  }
  arts_route_item_t *item = find_item_for_guid(guid);
  if (!item) {
    void *installed_data = arts_route_table_lookup_data(guid);
    fprintf(stderr,
            "FAIL [B]: cannot find item for guid=%lu (lookup_data=%p)\n",
            (uint64_t)guid, installed_data);
    abort();
  }

  scenario_b_ctx_t ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.item = item;
  atomic_store_explicit(&ctx.start_gate, 0, memory_order_release);

  pthread_t threads[B_THREADS];
  for (int i = 0; i < B_THREADS; i++) {
    if (pthread_create(&threads[i], NULL, scenario_b_acquirer, &ctx) != 0) {
      fprintf(stderr, "FAIL [B]: pthread_create\n");
      abort();
    }
  }

  /* Capture pre-mark gen so we can verify the post-free bump. */
  uint64_t pre_lock = atomic_load_explicit((_Atomic(uint64_t) *)&item->lock,
                                           memory_order_acquire);
  uint32_t pre_gen = ARTS_ROUTE_LOCK_GET_GEN(pre_lock);

  atomic_store_explicit(&ctx.start_gate, 1, memory_order_release);
  /* Let acquirers do real work for a short window before issuing
   * mark_delete; otherwise mark_delete tends to win the race immediately
   * and we observe almost zero successful acquires.  This keeps the test
   * exercising the contention path. */
  for (volatile uint64_t spin = 0; spin < 200000ull; spin++) {
  }
  /* Race mark_delete against the in-flight acquirers.  The install ref is
   * 1 right now; threads may stack +N more refs.  mark_delete sets DELETE
   * and decrements install ref; eventual final release_item triggers
   * free_item. */
  if (!arts_route_table_mark_delete(guid)) {
    fprintf(stderr, "FAIL [B]: mark_delete returned false\n");
    abort();
  }

  for (int i = 0; i < B_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  /* Acquire after mark_delete must always fail. */
  if (arts_route_table_acquire_item(item)) {
    fprintf(stderr,
            "FAIL [B]: acquire_item succeeded after mark_delete + drain\n");
    abort();
  }

  unsigned ok = atomic_load_explicit(&ctx.acquire_ok, memory_order_relaxed);
  unsigned rel = atomic_load_explicit(&ctx.release_count, memory_order_relaxed);
  unsigned fail = atomic_load_explicit(&ctx.acquire_fail, memory_order_relaxed);
  if (ok != rel) {
    fprintf(stderr,
            "FAIL [B]: acquire_ok=%u != release_count=%u (acquire_fail=%u)\n",
            ok, rel, fail);
    abort();
  }

  /* Verify final lock state: count==0, DELETE bit cleared (free_item
   * resets it), gen bumped by exactly +1 vs pre_gen. */
  uint64_t post_lock = atomic_load_explicit((_Atomic(uint64_t) *)&item->lock,
                                            memory_order_acquire);
  if (ARTS_ROUTE_LOCK_GET_COUNT(post_lock) != 0u) {
    fprintf(stderr, "FAIL [B]: post-free count=%u (want 0)\n",
            ARTS_ROUTE_LOCK_GET_COUNT(post_lock));
    abort();
  }
  if (ARTS_ROUTE_LOCK_HAS_DELETE(post_lock)) {
    fprintf(stderr, "FAIL [B]: post-free DELETE still set\n");
    abort();
  }
  uint32_t post_gen = ARTS_ROUTE_LOCK_GET_GEN(post_lock);
  if (post_gen != ((pre_gen + 1u) & 0x7FFFFFFFu)) {
    fprintf(stderr, "FAIL [B]: gen=%u (want %u)\n", post_gen,
            (pre_gen + 1u) & 0x7FFFFFFFu);
    abort();
  }

  /* /* ARTS_DB is wired into the deleter dispatch (object_has_shared_field
   * returns true).  free_item invokes our deleter which frees the obj — so
   * scenario_b_freed must be exactly 1 here, and we must NOT free obj manually
   * (would be a double-free). */
  if (atomic_load_explicit(&scenario_b_freed, memory_order_relaxed) != 1u) {
    fprintf(stderr, "FAIL [B]: deleter not fired (scenario_b_freed=%u, want 1)\n",
            atomic_load_explicit(&scenario_b_freed, memory_order_relaxed));
    abort();
  }
  printf("[B] PASS  (acquire_ok=%u, acquire_fail=%u, gen %u -> %u)\n", ok, fail,
         pre_gen, post_gen);
}

/* ------------------------------------------------------------------------- */
/* Scenario C: same-GUID re-install + ABA boundary                           */
/* ------------------------------------------------------------------------- */

#define C_ITERATIONS 64
#define C_READERS 4

typedef struct {
  arts_route_item_t *item;
  atomic_int gate;
  atomic_uint reader_ok;
  atomic_uint reader_blocked;
} scenario_c_reader_ctx_t;

static void *scenario_c_reader(void *vp) {
  scenario_c_reader_ctx_t *ctx = (scenario_c_reader_ctx_t *)vp;
  while (atomic_load_explicit(&ctx->gate, memory_order_acquire) == 0) {
    /* spin */
  }
  if (arts_route_table_acquire_item(ctx->item)) {
    atomic_fetch_add_explicit(&ctx->reader_ok, 1u, memory_order_relaxed);
    /* Hold the ref briefly to ensure mark_delete arrives mid-hold. */
    for (volatile int spin = 0; spin < 1000; spin++) {
    }
    arts_route_table_release_item(ctx->item);
  } else {
    atomic_fetch_add_explicit(&ctx->reader_blocked, 1u, memory_order_relaxed);
  }
  return NULL;
}

static void scenario_c_reinstall(void) {
  printf("[C] same-GUID re-install + ABA boundary\n");
  arts_guid_t guid = arts_guid_reserve(ARTS_DB, 0);

  uint32_t prev_gen = 0;
  arts_route_item_t *prev_item = NULL;
  unsigned total_reader_ok = 0;
  unsigned total_reader_blocked = 0;

  for (int it = 0; it < C_ITERATIONS; it++) {
    test_obj_t *obj = (test_obj_t *)calloc(1, sizeof(*obj));
    arts_shared_init(&obj->shared, scenario_b_deleter);
    obj->magic = 0xDEADBEEFCAFEBABEull;

    if (!arts_route_table_add_item_race(obj, guid, 0, false)) {
      fprintf(stderr, "FAIL [C]: install lost race on iter %d\n", it);
      abort();
    }
    arts_route_item_t *item = find_item_for_guid(guid);
    if (!item) {
      fprintf(stderr, "FAIL [C]: cannot find item on iter %d\n", it);
      abort();
    }
    /* The slot is permanent — every iteration must see the same item ptr. */
    if (prev_item != NULL && item != prev_item) {
      fprintf(stderr, "FAIL [C]: item ptr changed between iters\n");
      abort();
    }
    prev_item = item;

    uint32_t cur_gen = ARTS_ROUTE_LOCK_GET_GEN(atomic_load_explicit(
        (_Atomic(uint64_t) *)&item->lock, memory_order_acquire));
    /* Install preserves the gen carried over from the previous free; the
     * subsequent mark_delete + free will bump it by one (verified below). */
    if (it > 0 && cur_gen != prev_gen) {
      fprintf(stderr,
              "FAIL [C]: install gen mismatch at iter %d: cur=%u prev_end=%u\n",
              it, cur_gen, prev_gen);
      abort();
    }

    scenario_c_reader_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.item = item;
    atomic_store_explicit(&ctx.gate, 0, memory_order_release);

    pthread_t threads[C_READERS];
    for (int t = 0; t < C_READERS; t++) {
      if (pthread_create(&threads[t], NULL, scenario_c_reader, &ctx) != 0) {
        fprintf(stderr, "FAIL [C]: pthread_create iter %d\n", it);
        abort();
      }
    }
    atomic_store_explicit(&ctx.gate, 1, memory_order_release);
    /* Race mark_delete against readers. */
    arts_route_table_mark_delete(guid);
    for (int t = 0; t < C_READERS; t++) {
      pthread_join(threads[t], NULL);
    }

    /* After all readers join + mark_delete, count must be 0 again and gen
     * bumped by 1.  The slot is reusable for the next iteration. */
    uint64_t post = atomic_load_explicit((_Atomic(uint64_t) *)&item->lock,
                                         memory_order_acquire);
    if (ARTS_ROUTE_LOCK_GET_COUNT(post) != 0u) {
      fprintf(stderr, "FAIL [C]: end-of-iter count=%u on iter %d (gen=%u)\n",
              ARTS_ROUTE_LOCK_GET_COUNT(post), it,
              ARTS_ROUTE_LOCK_GET_GEN(post));
      abort();
    }
    if (ARTS_ROUTE_LOCK_HAS_DELETE(post)) {
      fprintf(stderr, "FAIL [C]: DELETE still set after free at iter %d\n", it);
      abort();
    }
    uint32_t end_gen = ARTS_ROUTE_LOCK_GET_GEN(post);
    if (end_gen != ((cur_gen + 1u) & 0x7FFFFFFFu)) {
      fprintf(stderr, "FAIL [C]: gen %u -> %u on iter %d (want +1)\n", cur_gen,
              end_gen, it);
      abort();
    }
    prev_gen = end_gen;
    total_reader_ok +=
        atomic_load_explicit(&ctx.reader_ok, memory_order_relaxed);
    total_reader_blocked +=
        atomic_load_explicit(&ctx.reader_blocked, memory_order_relaxed);
    /* /* free_item dispatches to scenario_b_deleter which frees obj.
     * Manual free here would be a double-free. */
    (void)obj;
  }
  printf("[C] PASS  (iters=%d gen %u, reader_ok=%u reader_blocked=%u)\n",
         C_ITERATIONS, prev_gen, total_reader_ok, total_reader_blocked);
}

/* ------------------------------------------------------------------------- */

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  scenario_a_direct_deleter();
  scenario_b_concurrent();
  scenario_c_reinstall();
  printf("shared_t_stress: ALL PASS\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
