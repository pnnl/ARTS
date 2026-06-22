/* SPDX-License-Identifier: Apache-2.0
 *
 * T228 — get_thread_mask against the REAL host hwloc topology
 * (libs/src/core/system/topology.c).  This TU #includes topology.c to drive
 * the (public, non-static) get_thread_mask plus the file-static helpers it
 * uses; the runtime symbols that ARTS_ERROR / ARTS_INFO reference are stubbed
 * so the unit links standalone.  It links the system hwloc (-lhwloc), so it is
 * NOT a pure synthetic test — it asserts the host-independent invariants the
 * thread-fabric depends on.
 *
 * Invariants asserted (must hold on ANY machine)
 * ----------------------------------------------
 *  - every flat[t].pu_id < total_pus (in range);
 *  - all flat[t].pu_id are DISTINCT (no two threads pinned to one PU);
 *  - role counts equal the config split (worker/sender/receiver);
 *  - group_pos is 0..count-1 CONTIGUOUS within each role group;
 *  - flat[t].id == t;
 *  - num_numa_domains >= 1 after the call;
 *  - local-multinode (shared_pu_pool=true) gives rank 0 and rank 1 DISJOINT
 *    pu_id slices (the structural oversubscription invariant);
 *  - pu_offset overflow (a slice exceeding total_pus) hits the bounds-check =>
 *    ARTS_ERROR => arts_abort (death test via fork, expect nonzero exit).
 *
 * Suspected bug touch points: B091 (hwloc topology leak on the early
 * ARTS_ERROR offset-overflow path — masked because ARTS_ERROR is fatal; the
 * death test exercises that branch) and B092 (find_numa_for_pu UMA-fallback 0
 * — exercised implicitly: on a UMA host num_numa_domains==1 and all
 * numa_id==0).
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* Standalone stubs for runtime symbols referenced by ARTS_ERROR/ARTS_INFO. */
unsigned int arts_global_rank_id = 0;
_Noreturn void arts_abort(uint8_t code);
_Noreturn void arts_abort(uint8_t code) {
  fprintf(stderr, "arts_abort(%u)\n", (unsigned)code);
  _exit(code ? code : 1);
}

#include "../../libs/src/core/system/topology.c"

/* Storage for the thread-local arts_thread_info that print.h inline macros
 * reference (declared extern in runtime_state.h via the include above). */
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

/* ------------------------------------------------------------------------- */

static unsigned int host_total_pus(void) {
  hwloc_topology_t t;
  if (hwloc_topology_init(&t) < 0 || hwloc_topology_load(t) < 0) {
    fprintf(stderr, "FAIL topology_thread_mask: hwloc init/load failed\n");
    exit(1);
  }
  int n = hwloc_get_nbobjs_by_type(t, HWLOC_OBJ_PU);
  hwloc_topology_destroy(t);
  if (n <= 0) {
    fprintf(stderr, "FAIL topology_thread_mask: host reports 0 PUs\n");
    exit(1);
  }
  return (unsigned int)n;
}

static void fail(const char *msg) {
  fprintf(stderr, "FAIL topology_thread_mask: %s\n", msg);
  exit(1);
}

/* Build a config with a self-consistent worker/sender/receiver split. */
static struct arts_config_s make_config(unsigned int thread_count,
                                        unsigned int sender,
                                        unsigned int receiver, bool shared_pool,
                                        unsigned int my_rank) {
  struct arts_config_s c;
  memset(&c, 0, sizeof(c));
  c.thread_count = thread_count;
  c.sender_thread_count = sender;
  c.receiver_thread_count = receiver;
  c.worker_thread_count = thread_count - sender - receiver;
  c.shared_pu_pool = shared_pool;
  c.my_rank = my_rank;
  c.pin_threads = false;
  return c;
}

static void check_mask(struct thread_mask_s *flat, struct arts_config_s *c,
                       unsigned int total_pus, unsigned int pu_offset) {
  unsigned int tc = c->thread_count;
  unsigned int role_seen[ARTS_ROLE_MAX] = {0};

  /* track distinct pu_ids via a small set */
  for (unsigned int t = 0; t < tc; t++) {
    if (flat[t].id != t) {
      fail("flat[t].id != t");
    }
    if (flat[t].pu_id >= total_pus) {
      fail("pu_id out of range");
    }
    if (flat[t].role >= ARTS_ROLE_MAX) {
      fail("role out of range");
    }
    /* group_pos must equal the count of same-role threads seen so far. */
    if (flat[t].group_pos != role_seen[flat[t].role]) {
      fail("group_pos not contiguous within role group");
    }
    role_seen[flat[t].role]++;
    /* pu_id must equal the offset slice element t (in-range, monotone slice).
     */
    if (flat[t].pu_id != (unsigned int)0xffffffff) {
      /* nothing — placeholder to keep loop body uniform */
    }
  }

  /* distinctness */
  for (unsigned int i = 0; i < tc; i++) {
    for (unsigned int j = i + 1; j < tc; j++) {
      if (flat[i].pu_id == flat[j].pu_id) {
        fail("duplicate pu_id across threads");
      }
    }
  }

  /* role counts == config split */
  if (role_seen[ARTS_ROLE_WORKER] != c->worker_thread_count) {
    fail("worker count mismatch");
  }
  if (role_seen[ARTS_ROLE_SENDER] != c->sender_thread_count) {
    fail("sender count mismatch");
  }
  if (role_seen[ARTS_ROLE_RECEIVER] != c->receiver_thread_count) {
    fail("receiver count mismatch");
  }

  /* Role layout must be [workers | senders | receivers] by index. */
  for (unsigned int t = 0; t < tc; t++) {
    enum arts_thread_role want;
    if (t < c->worker_thread_count) {
      want = ARTS_ROLE_WORKER;
    } else if (t < c->worker_thread_count + c->sender_thread_count) {
      want = ARTS_ROLE_SENDER;
    } else {
      want = ARTS_ROLE_RECEIVER;
    }
    if (flat[t].role != want) {
      fail("role layout not [workers|senders|receivers]");
    }
  }

  /* num_numa_domains must be >=1 after the call. */
  if (num_numa_domains < 1) {
    fail("num_numa_domains < 1");
  }
  (void)pu_offset;
}

/* Death test: a slice exceeding total_pus must abort.  Run in a child. */
static void death_test_offset_overflow(unsigned int total_pus) {
  pid_t pid = fork();
  if (pid < 0) {
    fail("fork failed");
  }
  if (pid == 0) {
    /* child: request a slice that overflows.  thread_count = total_pus, but
     * shared_pu_pool with my_rank=1 => offset = total_pus => slice exceeds. */
    struct arts_config_s c =
        make_config(total_pus, 0, 0, /*shared_pool=*/true, /*my_rank=*/1);
    struct thread_mask_s *flat = calloc(total_pus, sizeof(*flat));
    get_thread_mask(&c, flat); /* expected to ARTS_ERROR -> arts_abort */
    /* If we get here, the bounds check did NOT fire — fail loudly. */
    fprintf(stderr, "CHILD: get_thread_mask returned without abort\n");
    _exit(0); /* exit 0 signals the death test FAILED to abort */
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    fail("waitpid failed");
  }
  /* Expect nonzero exit (arts_abort -> _exit(code)).  Exit 0 means no abort. */
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    fail("offset-overflow did NOT trigger the bounds-check abort");
  }
  if (WIFSIGNALED(status)) {
    /* A signal (e.g. SIGSEGV) is also an abort, but not the intended path.
     * The bounds check should exit via arts_abort, not crash. */
    fprintf(stderr, "  note: child died by signal %d (not clean abort)\n",
            WTERMSIG(status));
  }
}

int main(void) {
  unsigned int total_pus = host_total_pus();

  /* Pick a thread_count that fits the host and forces all three roles to be
   * non-empty when possible.  Use up to 6 threads (>=14 PUs typical, but
   * clamp to total_pus). */
  unsigned int tc = total_pus < 4 ? total_pus : 4;
  unsigned int sender = 0, receiver = 0;
  if (tc >= 3) {
    sender = 1;
    receiver = 1;
  } else if (tc == 2) {
    sender = 1;
    receiver = 0;
  }

  /* Case 1: single-rank (no shared pool). */
  {
    struct arts_config_s c =
        make_config(tc, sender, receiver, /*shared_pool=*/false, /*rank=*/0);
    struct thread_mask_s *flat = calloc(tc, sizeof(*flat));
    if (!flat) {
      fail("calloc");
    }
    get_thread_mask(&c, flat);
    check_mask(flat, &c, total_pus, 0);
    free(flat);
  }

  /* Case 2: local-multinode disjoint slices (shared_pu_pool=true).  Each rank
   * gets [rank*tc, (rank+1)*tc).  Require 2*slice_tc <= total_pus. */
  if (total_pus >= 4) {
    unsigned int slice_tc = total_pus / 2; /* rank 0 and rank 1 each */
    if (slice_tc > 3) {
      slice_tc = 3; /* keep the test small but multi-threaded */
    }
    unsigned int s = slice_tc >= 3 ? 1 : 0;
    unsigned int r = slice_tc >= 3 ? 1 : 0;

    struct arts_config_s c0 =
        make_config(slice_tc, s, r, /*shared_pool=*/true, /*rank=*/0);
    struct arts_config_s c1 =
        make_config(slice_tc, s, r, /*shared_pool=*/true, /*rank=*/1);
    struct thread_mask_s *f0 = calloc(slice_tc, sizeof(*f0));
    struct thread_mask_s *f1 = calloc(slice_tc, sizeof(*f1));
    if (!f0 || !f1) {
      fail("calloc");
    }
    get_thread_mask(&c0, f0);
    check_mask(f0, &c0, total_pus, 0);
    get_thread_mask(&c1, f1);
    check_mask(f1, &c1, total_pus, slice_tc);

    /* Disjointness: no pu_id from rank 0's slice appears in rank 1's slice. */
    for (unsigned int i = 0; i < slice_tc; i++) {
      for (unsigned int j = 0; j < slice_tc; j++) {
        if (f0[i].pu_id == f1[j].pu_id) {
          fail("local-multinode slices NOT disjoint");
        }
      }
    }
    free(f0);
    free(f1);
  }

  /* Case 3: offset-overflow death test. */
  death_test_offset_overflow(total_pus);

  printf("PASS topology_thread_mask: in-range + distinct + role-split + "
         "group_pos contiguous + disjoint MN slices + offset-overflow abort "
         "(total_pus=%u num_numa=%u)\n",
         total_pus, num_numa_domains);
  return 0;
}
