/* SPDX-License-Identifier: Apache-2.0
 *
 * T008 — arts_tiered_pool_numa_id tier-1 NUMA sharding (tiered_pool.h).
 *
 * arts_tiered_pool_numa_id is a STUB today: it unconditionally returns 0, so
 * even with num_numa_nodes>1 every alloc/release routes to shard 0 and shards
 * 1..N-1 stay permanently empty (census 29.md §4: "Tier 1 currently collapses
 * to 1 shard").  This test PINS that current behavior so a future wiring of
 * real NUMA domains is a deliberate, test-visible change rather than a silent
 * drift:
 *
 *  (1) numa_id stub: arts_tiered_pool_numa_id(tid) == 0 for every tid.
 *  (2) Collapse: with num_numa_nodes=4, drive a tier-1 spill (release past
 *      H_local so a batch lands in a NUMA shard) and assert that shard 0 holds
 *      the spilled nodes while shards 1..3 remain empty (count==0).
 *
 * If/when numa_id becomes a real sharding function, assertion (2)'s "shards
 * 1..3 empty" will fail — exactly the signal that the stub was replaced.
 *
 * Standalone: same runtime-global + libc-shim setup as T007.
 */

#include "arts/runtime_state.h"
#include "arts/utils/lockfree_lifo.h"
#include "arts/utils/lockfree_pool.h"
#include "arts/utils/tiered_pool.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void *arts_calloc_align(size_t nmemb, size_t size, size_t align) {
  void *p = NULL;
  if (posix_memalign(&p, align, nmemb * size) != 0) {
    return NULL;
  }
  return p;
}
void arts_free(void *ptr) { free(ptr); }

#define NODE_SIZE 64
#define NUM_NUMA 4

typedef struct {
  arts_lf_link_t link;
  uint32_t id;
} node_t;

int main(void) {
  /* (1) numa_id stub returns 0 for every input. */
  for (uint32_t tid = 0; tid < 64; tid++) {
    if (arts_tiered_pool_numa_id(tid) != 0) {
      (void)fprintf(stderr,
                    "FAIL tiered_pool_numa: numa_id(%u) != 0 (stub changed?)\n",
                    tid);
      return 1;
    }
  }

  /* (2) Collapse: 4 NUMA shards configured, but everything routes to shard 0.
   */
  arts_tiered_pool_cfg_t cfg = {
      .H_local = 4, .B_local = 4, .H_numa = 1000, .B_numa = 64};
  arts_tiered_pool_t p;
  arts_tiered_pool_init_explicit(&p, NODE_SIZE, /*num_threads*/ 4, NUM_NUMA,
                                 cfg);
  arts_thread_info.thread_id = 1; /* numa_id(1) is still 0 (stub) */
  arts_thread_info.numa_domain_id = 1;

  /* Allocate a batch of nodes, then release more than H_local so a B_local
   * batch spills from tcache into the (stub) NUMA shard. */
  enum { N = 16 };
  void *held[N];
  for (int i = 0; i < N; i++) {
    held[i] = arts_tiered_pool_alloc(&p);
    if (!held[i]) {
      return 1;
    }
  }
  for (int i = 0; i < N; i++) {
    arts_tiered_pool_release(&p, held[i]); /* drives tcache past H_local=4 */
  }

  uint32_t shard0 =
      atomic_load_explicit(&p.numa[0].pool.count, memory_order_relaxed);
  if (shard0 == 0) {
    (void)fprintf(stderr,
                  "FAIL tiered_pool_numa: shard 0 empty after spill (expected "
                  "the stub to route the spill here)\n");
    return 1;
  }
  for (int s = 1; s < NUM_NUMA; s++) {
    uint32_t cnt =
        atomic_load_explicit(&p.numa[s].pool.count, memory_order_relaxed);
    if (cnt != 0) {
      (void)fprintf(stderr,
                    "FAIL tiered_pool_numa: shard %d has %u nodes — numa_id is "
                    "no longer a 0-stub; update this test for real sharding\n",
                    s, cnt);
      return 1;
    }
  }

  arts_tiered_pool_destroy(&p);
  printf("PASS tiered_pool_numa: numa_id is a 0-stub; %d-shard pool collapses "
         "to shard 0 (shards 1..%d empty), tier-1 sharding not yet wired\n",
         NUM_NUMA, NUM_NUMA - 1);
  return 0;
}
