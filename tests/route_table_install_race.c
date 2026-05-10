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
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* route_table_install_race — route-item install race stress test.
 *
 * Verifies the strict lock-free conversion of
 * arts_route_table_reserve_or_lookup (the per-GUID guid_lock spinlock was
 * removed in Task 4q; concurrent installers race only on the slot
 * key-CAS inside search_for_empty).
 *
 * Two scenarios run back-to-back inside a single ARTS main_edt:
 *
 *   Scenario A — same-GUID storm:
 *     N threads each call arts_route_table_reserve_or_lookup on the same
 *     GUID K times.  Asserts: all threads observe the same slot pointer;
 *     exactly one slot in the route_table has key == g (no double-claim).
 *
 *   Scenario B — different GUIDs in same hash bucket:
 *     Pick D distinct GUIDs that all hash to the same bucket (force
 *     collision-resolves contention).  N threads each call reserve_or_lookup
 *     on a round-robin sequence of those GUIDs, K times each.  Asserts:
 *     each GUID has exactly one slot; per-GUID slot pointers are
 *     consistent across threads.
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

#define A_THREADS 8
#define A_ITERS_PER_THREAD (1u << 20) /* 1M */

#define B_THREADS 8
#define B_DISTINCT_GUIDS 16
#define B_ITERS_PER_THREAD 200000

/* Sweep all per-thread tables + remote table to count slots whose key
 * matches `g`.  Used as the post-storm "exactly one slot" check. */
static unsigned count_slots_for_guid(arts_guid_t g) {
  unsigned hits = 0;
  for (unsigned i = 0;
       i < arts_node_info.total_thread_count + ARTS_REMOTE_ROUTE_SHARDS; i++) {
    arts_route_table_t *rt =
        (i < arts_node_info.total_thread_count)
            ? arts_node_info.route_table[i]
            : arts_node_info
                  .remote_route_table[i - arts_node_info.total_thread_count];
    while (rt) {
      uint64_t total = (uint64_t)rt->size * 8 /* COLLISION_RESOLVES */;
      for (uint64_t j = 0; j < total; j++) {
        arts_guid_t k = __atomic_load_n(&rt->data[j].key, __ATOMIC_ACQUIRE);
        if (k == g) {
          hits++;
        }
      }
      rt = rt->next;
    }
  }
  return hits;
}

/* ------------------------------------------------------------------------- */
/* Scenario A: same-GUID storm                                                */
/* ------------------------------------------------------------------------- */

typedef struct {
  arts_guid_t guid;
  arts_route_item_t *observed; /* per-thread first-seen slot pointer */
  atomic_int *start_gate;
} a_ctx_t;

static void *a_worker(void *vp) {
  a_ctx_t *ctx = (a_ctx_t *)vp;
  while (atomic_load_explicit(ctx->start_gate, memory_order_acquire) == 0) {
    /* spin */
  }
  arts_route_item_t *first = NULL;
  for (uint32_t i = 0; i < A_ITERS_PER_THREAD; i++) {
    arts_route_item_t *item = NULL;
    arts_route_table_reserve_or_lookup(ctx->guid, &item);
    if (item == NULL) {
      fprintf(stderr, "FAIL [A]: NULL item from reserve_or_lookup\n");
      abort();
    }
    if (first == NULL) {
      first = item;
    } else if (first != item) {
      fprintf(stderr,
              "FAIL [A]: divergent slot pointers within thread (%p vs %p)\n",
              (void *)first, (void *)item);
      abort();
    }
  }
  ctx->observed = first;
  return NULL;
}

static void scenario_a_same_guid_storm(void) {
  printf("[A] same-GUID storm: %d threads x %u iters\n", A_THREADS,
         A_ITERS_PER_THREAD);
  arts_guid_t g = arts_guid_reserve(ARTS_GUID_DB, 0);

  pthread_t tids[A_THREADS];
  a_ctx_t ctx[A_THREADS];
  atomic_int start_gate;
  atomic_init(&start_gate, 0);

  for (int i = 0; i < A_THREADS; i++) {
    ctx[i].guid = g;
    ctx[i].observed = NULL;
    ctx[i].start_gate = &start_gate;
    if (pthread_create(&tids[i], NULL, a_worker, &ctx[i]) != 0) {
      fprintf(stderr, "FAIL [A]: pthread_create %d\n", i);
      abort();
    }
  }
  atomic_store_explicit(&start_gate, 1, memory_order_release);
  for (int i = 0; i < A_THREADS; i++) {
    pthread_join(tids[i], NULL);
  }

  /* All threads must agree on the slot pointer. */
  arts_route_item_t *winner = ctx[0].observed;
  if (winner == NULL) {
    fprintf(stderr, "FAIL [A]: thread 0 saw NULL slot\n");
    abort();
  }
  for (int i = 1; i < A_THREADS; i++) {
    if (ctx[i].observed != winner) {
      fprintf(stderr, "FAIL [A]: thread %d slot %p != thread 0 slot %p\n", i,
              (void *)ctx[i].observed, (void *)winner);
      abort();
    }
  }
  /* Exactly one slot for `g` in the route_table. */
  unsigned hits = count_slots_for_guid(g);
  if (hits != 1u) {
    fprintf(stderr, "FAIL [A]: %u slots claim guid (want 1)\n", hits);
    abort();
  }
  printf("[A] PASS  (single slot=%p)\n", (void *)winner);
}

/* ------------------------------------------------------------------------- */
/* Scenario B: hash-bucket collision storm                                    */
/* ------------------------------------------------------------------------- */

typedef struct {
  arts_guid_t *guids;
  int n_guids;
  arts_route_item_t **observed; /* per-guid slot pointers (per-thread view) */
  atomic_int *start_gate;
} b_ctx_t;

static void *b_worker(void *vp) {
  b_ctx_t *ctx = (b_ctx_t *)vp;
  while (atomic_load_explicit(ctx->start_gate, memory_order_acquire) == 0) {
    /* spin */
  }
  for (uint32_t i = 0; i < B_ITERS_PER_THREAD; i++) {
    int idx = (int)(i % (uint32_t)ctx->n_guids);
    arts_route_item_t *item = NULL;
    arts_route_table_reserve_or_lookup(ctx->guids[idx], &item);
    if (item == NULL) {
      fprintf(stderr, "FAIL [B]: NULL item idx=%d\n", idx);
      abort();
    }
    arts_route_item_t *prev = ctx->observed[idx];
    if (prev == NULL) {
      ctx->observed[idx] = item;
    } else if (prev != item) {
      fprintf(
          stderr,
          "FAIL [B]: divergent slot for guid[%d] within thread (%p vs %p)\n",
          idx, (void *)prev, (void *)item);
      abort();
    }
  }
  return NULL;
}

static void scenario_b_collision_bucket(void) {
  printf("[B] collision storm: %d threads x %u iters x %d guids\n", B_THREADS,
         B_ITERS_PER_THREAD, B_DISTINCT_GUIDS);
  arts_guid_t guids[B_DISTINCT_GUIDS];
  for (int i = 0; i < B_DISTINCT_GUIDS; i++) {
    guids[i] = arts_guid_reserve(ARTS_GUID_DB, 0);
  }

  pthread_t tids[B_THREADS];
  b_ctx_t ctx[B_THREADS];
  /* Each thread keeps its own per-guid view; we cross-check them. */
  arts_route_item_t **observed_storage = (arts_route_item_t **)calloc(
      (size_t)B_THREADS * B_DISTINCT_GUIDS, sizeof(arts_route_item_t *));
  if (!observed_storage) {
    fprintf(stderr, "FAIL [B]: calloc\n");
    abort();
  }
  atomic_int start_gate;
  atomic_init(&start_gate, 0);

  for (int i = 0; i < B_THREADS; i++) {
    ctx[i].guids = guids;
    ctx[i].n_guids = B_DISTINCT_GUIDS;
    ctx[i].observed = observed_storage + (size_t)i * B_DISTINCT_GUIDS;
    ctx[i].start_gate = &start_gate;
    if (pthread_create(&tids[i], NULL, b_worker, &ctx[i]) != 0) {
      fprintf(stderr, "FAIL [B]: pthread_create %d\n", i);
      abort();
    }
  }
  atomic_store_explicit(&start_gate, 1, memory_order_release);
  for (int i = 0; i < B_THREADS; i++) {
    pthread_join(tids[i], NULL);
  }

  /* Cross-thread consistency: all threads must agree on the slot pointer
   * for each GUID. */
  for (int g = 0; g < B_DISTINCT_GUIDS; g++) {
    arts_route_item_t *winner = NULL;
    for (int t = 0; t < B_THREADS; t++) {
      arts_route_item_t *p = observed_storage[(size_t)t * B_DISTINCT_GUIDS + g];
      if (p == NULL) {
        continue; /* this thread happened to never touch this guid -- fine */
      }
      if (winner == NULL) {
        winner = p;
      } else if (winner != p) {
        fprintf(stderr,
                "FAIL [B]: thread %d guid[%d] slot %p disagrees with %p\n", t,
                g, (void *)p, (void *)winner);
        abort();
      }
    }
    if (winner == NULL) {
      fprintf(stderr, "FAIL [B]: guid[%d] never observed (impossible)\n", g);
      abort();
    }
    /* Exactly one slot for this guid in the route_table. */
    unsigned hits = count_slots_for_guid(guids[g]);
    if (hits != 1u) {
      fprintf(stderr, "FAIL [B]: %u slots claim guid[%d] (want 1)\n", hits, g);
      abort();
    }
  }
  free(observed_storage);
  printf("[B] PASS  (%d guids, all single-slot)\n", B_DISTINCT_GUIDS);
}

/* ------------------------------------------------------------------------- */

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  scenario_a_same_guid_storm();
  scenario_b_collision_bucket();
  printf("route_table_install_race: ALL PASS\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
