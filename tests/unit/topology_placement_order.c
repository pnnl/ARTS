/* SPDX-License-Identifier: Apache-2.0
 *
 * T227 — topology.c PU-placement comparator ordering (pu_entry_compare /
 * pu_entry_by_core).  These are file-static in topology.c, so per the test
 * convention this TU #includes topology.c directly to pull the statics in;
 * the runtime symbols that get_thread_mask/print_mask reference (ARTS_ERROR
 * -> arts_abort, ARTS_INFO -> arts_thread_info / arts_global_rank_id) are
 * stubbed here so the unit links standalone without the ARTS runtime.
 *
 * Property under test
 * -------------------
 * pu_entry_compare implements the THREAD-PLACEMENT POLICY: spread across
 * physical cores before using hyperthread siblings, NUMA-0 first within a
 * round.  Sort key = (pu_rank_in_core ASC, numa_id ASC, pu_os_index ASC).
 * Consequence: position 0..(ncores-1) of the sorted array is "round 0" — one
 * PU per core, the first PU of each core, ordered NUMA-0 first — and only AFTER
 * every core's round-0 PU does round 1 (the first HT sibling of each core)
 * appear.  pu_entry_by_core implements the Phase-2 helper sort used to compute
 * pu_rank_in_core: (core_os_index ASC, pu_os_index ASC).
 *
 * Interleaving / determinism
 * --------------------------
 * Pure comparators — no threads.  We feed synthetic pu_entry_s[] arrays that
 * model a 2-package / 2-NUMA / SMT-2 machine deliberately given to qsort in a
 * SCRAMBLED order, then assert the EXACT post-sort sequence.  We also assert
 * the comparators are a strict total order (antisymmetry, transitivity-by-key,
 * irreflexivity) so qsort behavior is well-defined.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- standalone stubs for the runtime symbols topology.c references -------
 * Only get_thread_mask / print_mask use these; the comparators do not.  We
 * provide minimal definitions so the included TU links. */
#include <stdbool.h>

struct _stub_priv {
  char pad[64];
  unsigned int group_pos;
};
/* arts_thread_info is declared `extern ARTS_THREAD_LOCAL struct
 * arts_runtime_private_s` in runtime_state.h, but that header is pulled in by
 * topology.c's include of print.h, so we must NOT redefine the type.  Instead
 * we supply the storage for the real symbol below, after the include. */
unsigned int arts_global_rank_id = 0;
#include <stdint.h>
_Noreturn void arts_abort(uint8_t code);
_Noreturn void arts_abort(uint8_t code) {
  fprintf(stderr, "arts_abort(%u)\n", (unsigned)code);
  exit(code);
}

/* Pull in the static comparators + struct pu_entry_s. */
#include "../../libs/src/core/system/topology.c"

/* Provide storage for the thread-local arts_thread_info that print.h's inline
 * macros reference (declared extern in runtime_state.h via the include above).
 */
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

/* ------------------------------------------------------------------------- */

static const char *role_unused(void) { return role_char(ARTS_ROLE_WORKER); }

static void fail(const char *msg) {
  fprintf(stderr, "FAIL topology_placement_order: %s\n", msg);
  exit(1);
}

/* Recompute pu_rank_in_core exactly as get_thread_mask Phase 2 does, so the
 * synthetic input mirrors what feeds pu_entry_compare in production. */
static void assign_rank_in_core(struct pu_entry_s *a, unsigned n) {
  qsort(a, n, sizeof(*a), pu_entry_by_core);
  unsigned rank = 0;
  for (unsigned i = 0; i < n; i++) {
    if (i > 0 && a[i].core_os_index != a[i - 1].core_os_index) {
      rank = 0;
    }
    a[i].pu_rank_in_core = rank++;
  }
}

int main(void) {
  (void)role_unused; /* silence unused-static from the included TU */

  /* Model: 2 packages, 2 NUMA nodes (pkg==numa), 2 cores per package, SMT-2.
   * Cores: 0,1 on numa 0 ; 2,3 on numa 1.  PUs are os-indexed core-major:
   *   core0 -> pu 0 (rank0), pu 4 (rank1)
   *   core1 -> pu 1 (rank0), pu 5 (rank1)
   *   core2 -> pu 2 (rank0), pu 6 (rank1)
   *   core3 -> pu 3 (rank0), pu 7 (rank1)
   * numa(core0)=0 numa(core1)=0 numa(core2)=1 numa(core3)=1.
   *
   * Expected pu_entry_compare order (round0 numa0-first, then round0 numa1,
   * then round1 numa0, then round1 numa1):
   *   pu0(c0,n0,r0), pu1(c1,n0,r0), pu2(c2,n1,r0), pu3(c3,n1,r0),
   *   pu4(c0,n0,r1), pu5(c1,n0,r1), pu6(c2,n1,r1), pu7(c3,n1,r1)
   */
  struct pu_entry_s in[8];
  /* Fill core/pu/numa, leave pu_rank_in_core to be computed.  Deliberately
   * insert in scrambled order to prove the sort, not the insertion order. */
  struct {
    unsigned pu, core, numa;
  } spec[8] = {
      {7, 3, 1}, {0, 0, 0}, {5, 1, 0}, {2, 2, 1},
      {4, 0, 0}, {3, 3, 1}, {1, 1, 0}, {6, 2, 1},
  };
  for (unsigned i = 0; i < 8; i++) {
    in[i].pu_os_index = spec[i].pu;
    in[i].core_os_index = spec[i].core;
    in[i].pkg_os_index = spec[i].numa; /* pkg == numa in this model */
    in[i].numa_id = spec[i].numa;
    in[i].pu_rank_in_core = 0;
  }

  /* --- pu_entry_by_core: assigns rank_in_core, groups by core --- */
  assign_rank_in_core(in, 8);
  /* After by_core sort: core-major, pu ascending within core. */
  for (unsigned i = 1; i < 8; i++) {
    if (in[i].core_os_index < in[i - 1].core_os_index) {
      fail("pu_entry_by_core: core_os_index not non-decreasing");
    }
    if (in[i].core_os_index == in[i - 1].core_os_index &&
        in[i].pu_os_index < in[i - 1].pu_os_index) {
      fail("pu_entry_by_core: pu_os_index not ascending within core");
    }
  }
  /* rank_in_core must be 0 for the first PU of each core, 1 for its sibling. */
  for (unsigned i = 0; i < 8; i++) {
    unsigned expect_rank = (in[i].pu_os_index >= 4) ? 1u : 0u;
    if (in[i].pu_rank_in_core != expect_rank) {
      fail("pu_rank_in_core mis-assigned");
    }
  }

  /* --- pu_entry_compare: round-0 (one PU/core, NUMA-0 first) before HT --- */
  qsort(in, 8, sizeof(*in), pu_entry_compare);
  unsigned expect_pu[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  for (unsigned i = 0; i < 8; i++) {
    if (in[i].pu_os_index != expect_pu[i]) {
      fprintf(stderr,
              "  at pos %u: got pu=%u (core=%u numa=%u rank=%u) want pu=%u\n",
              i, in[i].pu_os_index, in[i].core_os_index, in[i].numa_id,
              in[i].pu_rank_in_core, expect_pu[i]);
      fail("pu_entry_compare ordering wrong");
    }
  }
  /* Explicitly assert the round structure: first 4 are rank 0, last 4 rank 1.
   */
  for (unsigned i = 0; i < 4; i++) {
    if (in[i].pu_rank_in_core != 0) {
      fail("round-0 region contains an HT sibling");
    }
  }
  for (unsigned i = 4; i < 8; i++) {
    if (in[i].pu_rank_in_core != 1) {
      fail("round-1 region contains a round-0 PU");
    }
  }
  /* Within round 0, NUMA-0 PUs precede NUMA-1 PUs. */
  if (!(in[0].numa_id == 0 && in[1].numa_id == 0 && in[2].numa_id == 1 &&
        in[3].numa_id == 1)) {
    fail("round-0 not NUMA-0-first");
  }

  /* --- strict-total-order properties of pu_entry_compare --- */
  for (unsigned i = 0; i < 8; i++) {
    if (pu_entry_compare(&in[i], &in[i]) != 0) {
      fail("pu_entry_compare not reflexive-zero");
    }
    for (unsigned j = 0; j < 8; j++) {
      int ij = pu_entry_compare(&in[i], &in[j]);
      int ji = pu_entry_compare(&in[j], &in[i]);
      if ((ij < 0 && ji <= 0) || (ij > 0 && ji >= 0) || (ij == 0 && ji != 0)) {
        fail("pu_entry_compare not antisymmetric");
      }
    }
  }

  /* --- a single-NUMA UMA machine: ordering reduces to (rank, pu) --- */
  struct pu_entry_s uma[4] = {
      {.pu_os_index = 2,
       .core_os_index = 1,
       .numa_id = 0,
       .pu_rank_in_core = 0},
      {.pu_os_index = 0,
       .core_os_index = 0,
       .numa_id = 0,
       .pu_rank_in_core = 0},
      {.pu_os_index = 3,
       .core_os_index = 1,
       .numa_id = 0,
       .pu_rank_in_core = 1},
      {.pu_os_index = 1,
       .core_os_index = 0,
       .numa_id = 0,
       .pu_rank_in_core = 1},
  };
  qsort(uma, 4, sizeof(*uma), pu_entry_compare);
  unsigned uma_expect[4] = {0, 2, 1, 3}; /* rank0: pu0,pu2 ; rank1: pu1,pu3 */
  for (unsigned i = 0; i < 4; i++) {
    if (uma[i].pu_os_index != uma_expect[i]) {
      fail("UMA round-robin ordering wrong");
    }
  }

  printf("PASS topology_placement_order: spread-before-HT + NUMA-0-first + "
         "by-core grouping verified\n");
  return 0;
}
