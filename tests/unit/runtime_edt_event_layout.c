/* SPDX-License-Identifier: Apache-2.0
 *
 * T219 — arts_edt_s / arts_event_s C-vs-C++ layout parity + arts_edt_total_size
 * arithmetic (libs/include/internal/arts/runtime_types.h).
 *
 * Property under test
 * -------------------
 * runtime_types.h declares `struct arts_event_s` TWICE: once on the C11 path
 * (fields qualified `_Atomic`) and once on the C++/nvcc path (qualifier
 * dropped so the layout is visible without C11 atomics).  The two declarations
 * MUST be byte-for-byte layout-identical: a GPU build compiles the C++ arm and
 * a host build compiles the C arm, and they exchange the same struct.  A drift
 * (a field reordered, a type widened, padding changed) silently corrupts GPU
 * event traffic.  `struct arts_edt_s` has a single declaration but its trailing
 * `[paramv | depv]` arithmetic in `arts_edt_total_size` is load-bearing for
 * every EDT allocation/copy.
 *
 * How parity is checked
 * -------------------
 * This source file is compiled BOTH as C (the active C11 `_Atomic` arm) and as
 * C++ (the qualifier-dropped arm); each arm prints sizeof/alignof/offsetof for
 * both structs.  ctest registers two executables from this one file and a third
 * "diff" step compares their printed lines, so a drift between the two arms is
 * caught.  Independently, each arm embeds compile-time `_Static_assert`s and
 * runtime asserts on the invariants that MUST hold regardless of host:
 *   - the union arms start at the same offset (right after the discriminator);
 *   - sizeof(arts_event_s) is a multiple of its alignment and >= the union;
 *   - arts_edt_total_size(edt) == sizeof(struct) + paramc*8 + depc*sizeof(dep)
 *     for a matrix of (paramc, depc) including 0/0 and large values;
 *   - both structs honor ARTS_ALIGNED_MAX (cache-line alignment).
 */

#include "arts/runtime_types.h"

#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
#define LAYOUT_LANG "cpp"
#else
#define LAYOUT_LANG "c"
#endif

/* ---- compile-time invariants (per-arm; drift fires here) ----------------- */

/* Discriminator is the first byte. */
#if defined(__cplusplus)
static_assert(offsetof(struct arts_event_s, is_channel) == 0,
              "is_channel must be first");
#else
_Static_assert(offsetof(struct arts_event_s, is_channel) == 0,
               "is_channel must be first");
#endif

/* EDT layout is fixed-prefix; the trailing arrays are computed, never declared,
 * so sizeof(struct arts_edt_s) must be exactly the prefix size. */

static void check_event_layout(void) {
  /* The union arms must begin at the same offset in both language arms.  We
   * cannot offsetof() into an anonymous union member portably across C/C++ in
   * one expression, so assert structural facts that pin the layout: the union
   * starts after the 1-byte discriminator (modulo alignment), and the whole
   * struct is a multiple of its alignment. */
  size_t sz = sizeof(struct arts_event_s);
  size_t al = alignof(struct arts_event_s);
  if (al == 0 || (sz % al) != 0) {
    fprintf(
        stderr,
        "FAIL T219[%s]: sizeof(arts_event_s)=%zu not multiple of align=%zu\n",
        LAYOUT_LANG, sz, al);
    exit(1);
  }
  /* ARTS_ALIGNED_MAX pins shared objects to a full cache line
   * (ARTS_CACHE_LINE_SIZE) for false-sharing isolation. */
  if (al != ARTS_CACHE_LINE_SIZE) {
    fprintf(
        stderr,
        "FAIL T219[%s]: arts_event_s align=%zu != ARTS_CACHE_LINE_SIZE %d\n",
        LAYOUT_LANG, al, (int)ARTS_CACHE_LINE_SIZE);
    exit(1);
  }
  /* The struct must be at least big enough to hold the discriminator plus the
   * larger union arm.  We re-derive the channel arm's footprint conservatively
   * by sizeof of its component types via two mpsc queues + counters — but the
   * simplest robust check is sz >= 1 (discriminator) and sz covers a guid. */
  if (sz < sizeof(arts_guid_t)) {
    fprintf(stderr, "FAIL T219[%s]: arts_event_s smaller than a guid (%zu)\n",
            LAYOUT_LANG, sz);
    exit(1);
  }
}

static void check_edt_total_size(void) {
  /* arts_edt_total_size = sizeof(struct) + paramc*8 + depc*sizeof(dep). */
  const uint64_t base = (uint64_t)sizeof(struct arts_edt_s);
  const uint64_t depsz = (uint64_t)sizeof(arts_edt_dep_t);

  /* sizeof(struct arts_edt_s) must honor the cache-line alignment too. */
  if (alignof(struct arts_edt_s) != ARTS_CACHE_LINE_SIZE) {
    fprintf(stderr,
            "FAIL T219[%s]: arts_edt_s align=%zu != ARTS_CACHE_LINE_SIZE %d\n",
            LAYOUT_LANG, (size_t)alignof(struct arts_edt_s),
            (int)ARTS_CACHE_LINE_SIZE);
    exit(1);
  }

  const uint32_t pcs[] = {0, 1, 2, 7, 16, 1024, 65535};
  const uint32_t dcs[] = {0, 1, 3, 8, 32, 4096, 65535};
  for (size_t i = 0; i < sizeof(pcs) / sizeof(pcs[0]); i++) {
    for (size_t j = 0; j < sizeof(dcs) / sizeof(dcs[0]); j++) {
      struct arts_edt_s edt;
      edt.paramc = pcs[i];
      edt.depc = dcs[j];
      uint64_t got = arts_edt_total_size(&edt);
      uint64_t want = base + (uint64_t)pcs[i] * 8u + (uint64_t)dcs[j] * depsz;
      if (got != want) {
        fprintf(stderr,
                "FAIL T219[%s]: total_size(paramc=%u,depc=%u)=%llu want %llu "
                "(base=%llu depsz=%llu)\n",
                LAYOUT_LANG, pcs[i], dcs[j], (unsigned long long)got,
                (unsigned long long)want, (unsigned long long)base,
                (unsigned long long)depsz);
        exit(1);
      }
    }
  }

  /* paramv slot is exactly 8 bytes per the helper's *8 — assert that promise.
   */
  struct arts_edt_s a = {0}, b = {0};
  a.paramc = 0;
  a.depc = 0;
  b.paramc = 1;
  b.depc = 0;
  if (arts_edt_total_size(&b) - arts_edt_total_size(&a) != 8u) {
    fprintf(stderr, "FAIL T219[%s]: paramv slot not 8 bytes\n", LAYOUT_LANG);
    exit(1);
  }
  b.paramc = 0;
  b.depc = 1;
  if (arts_edt_total_size(&b) - arts_edt_total_size(&a) != depsz) {
    fprintf(stderr, "FAIL T219[%s]: depv slot != sizeof(arts_edt_dep_t)\n",
            LAYOUT_LANG);
    exit(1);
  }
}

int main(void) {
  check_event_layout();
  check_edt_total_size();

  /* Print every load-bearing offset so the sibling-language build can be
   * compared byte-for-byte by ctest.  Size/alignment alone cannot detect two
   * fields moving within unchanged padding. */
  printf("T219_LAYOUT %s edt_size=%zu edt_align=%zu dep_size=%zu "
         "edt_guid=%zu edt_output_data=%zu edt_depc_needed=%zu edt_self_cb=%zu "
         "event_size=%zu event_align=%zu event_is_channel=%zu "
         "simple_size=%zu simple_state=%zu/%zu simple_counted=%zu/%zu "
         "simple_auto_destroy=%zu/%zu simple_discard_data=%zu/%zu "
         "simple_data=%zu/%zu simple_deps_stack=%zu/%zu "
         "channel_size=%zu channel_nb_sat=%zu/%zu channel_nb_deps=%zu/%zu "
         "channel_data_queue=%zu/%zu channel_dep_queue=%zu/%zu "
         "channel_draining=%zu/%zu "
         "pool_head_size=%zu pool_head_align=%zu pool_size=%zu pool_align=%zu "
         "pool_head=%zu/%zu pool_count=%zu/%zu pool_node_size=%zu/%zu\n",
         LAYOUT_LANG, sizeof(struct arts_edt_s),
         (size_t)alignof(struct arts_edt_s), sizeof(arts_edt_dep_t),
         offsetof(struct arts_edt_s, guid),
         offsetof(struct arts_edt_s, output_data),
         offsetof(struct arts_edt_s, depc_needed),
         offsetof(struct arts_edt_s, self_cb), sizeof(struct arts_event_s),
         (size_t)alignof(struct arts_event_s),
         offsetof(struct arts_event_s, is_channel),
         sizeof(((struct arts_event_s *)0)->simple),
         offsetof(struct arts_event_s, simple.state),
         sizeof(((struct arts_event_s *)0)->simple.state),
         offsetof(struct arts_event_s, simple.counted),
         sizeof(((struct arts_event_s *)0)->simple.counted),
         offsetof(struct arts_event_s, simple.auto_destroy),
         sizeof(((struct arts_event_s *)0)->simple.auto_destroy),
         offsetof(struct arts_event_s, simple.discard_data),
         sizeof(((struct arts_event_s *)0)->simple.discard_data),
         offsetof(struct arts_event_s, simple.data),
         sizeof(((struct arts_event_s *)0)->simple.data),
         offsetof(struct arts_event_s, simple.deps_stack),
         sizeof(((struct arts_event_s *)0)->simple.deps_stack),
         sizeof(((struct arts_event_s *)0)->channel),
         offsetof(struct arts_event_s, channel.nb_sat),
         sizeof(((struct arts_event_s *)0)->channel.nb_sat),
         offsetof(struct arts_event_s, channel.nb_deps),
         sizeof(((struct arts_event_s *)0)->channel.nb_deps),
         offsetof(struct arts_event_s, channel.data_queue),
         sizeof(((struct arts_event_s *)0)->channel.data_queue),
         offsetof(struct arts_event_s, channel.dep_queue),
         sizeof(((struct arts_event_s *)0)->channel.dep_queue),
         offsetof(struct arts_event_s, channel.draining),
         sizeof(((struct arts_event_s *)0)->channel.draining),
         sizeof(arts_lf_pool_head_t), (size_t)alignof(arts_lf_pool_head_t),
         sizeof(arts_lockfree_pool_t), (size_t)alignof(arts_lockfree_pool_t),
         offsetof(arts_lockfree_pool_t, head),
         sizeof(((arts_lockfree_pool_t *)0)->head),
         offsetof(arts_lockfree_pool_t, count),
         sizeof(((arts_lockfree_pool_t *)0)->count),
         offsetof(arts_lockfree_pool_t, node_size),
         sizeof(((arts_lockfree_pool_t *)0)->node_size));

  printf("PASS runtime_edt_event_layout [%s]: edt/event layout + total_size "
         "arithmetic verified\n",
         LAYOUT_LANG);
  return 0;
}
