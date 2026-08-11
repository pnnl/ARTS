/* SPDX-License-Identifier: Apache-2.0
 *
 * Whitebox rendezvous-PUT exercise for the libfabric transport core
 * (transport/net.c).  Brings the fabric up standalone — NO ARTS runtime — and
 * drives the one-sided data plane at itself: an 8 MiB payload moves by
 * fi_writedata into a pre-registered landing buffer, with the write-completion
 * immediate (txid) pairing against a registered expectation.
 *
 * Coverage:
 *   1. txid allocation: never 0 (0 is the "no landing advertised" sentinel),
 *      rank bits in the top 16.
 *   2. landing advertisement: arts_net_rdzv_local resolves a regpool pointer
 *      to the {raddr, rkey} the provider's negotiated mr_mode requires.
 *   3. order A (data before expect): PUT 8 MiB, wait for the remote-write
 *      completion to be processed, THEN register the expectation — the
 *      callback must fire inline and the landing bytes must equal the source
 *      ("imm seen => landing buffer valid").
 *   4. order B (expect before data): register the expectation FIRST, then PUT
 *      — the callback must fire from the progress path once the immediate
 *      arrives, bytes verified again.
 *   5. local completion: the on_local_done callback runs and tx_outstanding
 *      returns to zero (source-buffer release gate).
 *
 * Standalone: #includes net.c (its statics) and links the real regpool object
 * + libfabric; runtime externs are libc-backed shims.  FI_PROVIDER=tcp is
 * pinned by the test's environment.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "arts/memory/regpool.h"
#include "arts/runtime_state.h"
#include "arts/system/threads.h"
#include "arts/transport/dispatcher.h"
#include "arts/transport/protocol.h"

/* ---- runtime externs the module (and regpool) reference ------------------ */
unsigned int arts_global_rank_id = 0;
unsigned int arts_global_rank_count = 1; /* self-only fabric */
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;
struct arts_runtime_shared_s arts_node_info;
void arts_abort(uint8_t code) { exit(code ? code : 1); }

/* ---- dispatch stub (no fi_send messages are exercised here) --------------- */
void arts_transport_dispatch_packet(struct arts_msg_header_s *packet) {
  (void)packet;
}
void arts_transport_loopback_post(const void *packet, unsigned int size) {
  (void)packet;
  (void)size;
}

/* Compile the module under test (its statics: g_net, reap paths, rdzv table). */
#include "../../libs/src/core/transport/net.c"

/* counter stubs: net.c's introspection increments resolve here (this harness
 * links no counter TU). */
ARTS_THREAD_LOCAL arts_counter_t arts_thread_local_counters[NUM_COUNTER_TYPES];
void arts_counter_increment_by(arts_counter_t *counter, uint64_t num) {
  (void)counter;
  (void)num;
}

/* ------------------------------------------------------------------------- */

#define PUT_LEN ((size_t)8 * 1024 * 1024) /* 8 MiB — 4x the control ceiling */

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static _Atomic int g_local_done;
static void on_local_done(void *arg) {
  (void)arg;
  atomic_fetch_add_explicit(&g_local_done, 1, memory_order_acq_rel);
}

static _Atomic int g_paired;
static void on_data(void *arg) {
  atomic_fetch_add_explicit(&g_paired, 1, memory_order_acq_rel);
  if (arg != NULL) {
    *(int *)arg = atomic_load_explicit(&g_paired, memory_order_acquire);
  }
}

/* Spin progress until `*ctr` reaches `want` or 5 s. */
static bool wait_ctr(_Atomic int *ctr, int want) {
  uint64_t deadline = now_ns() + 5000000000ULL;
  while (atomic_load_explicit(ctr, memory_order_acquire) < want) {
    arts_net_progress();
    if (now_ns() > deadline) {
      return false;
    }
  }
  return true;
}

static void fill_pattern(unsigned char *p, size_t len, unsigned seed) {
  for (size_t i = 0; i < len; i++) {
    p[i] = (unsigned char)((i * 131u + seed) ^ (i >> 12));
  }
}

static bool check_pattern(const unsigned char *p, size_t len, unsigned seed) {
  for (size_t i = 0; i < len; i++) {
    if (p[i] != (unsigned char)((i * 131u + seed) ^ (i >> 12))) {
      fprintf(stderr, "  byte mismatch at %zu\n", i);
      return false;
    }
  }
  return true;
}

int main(void) {
  setenv("FI_PROVIDER", "tcp", 1);

  arts_net_init(NULL, NULL, NULL); /* auto -- FI_PROVIDER above still applies */
  if (!arts_regpool_init(arts_net_domain(), (size_t)64 * 1024 * 1024, 0)) {
    fprintf(stderr, "FAIL net_rendezvous: regpool init failed\n");
    return 1;
  }
  arts_net_rx_arm();

  /* av-insert our own EP address at index 0 -> fi_addr_t 0 == self. */
  unsigned char addr[ARTS_NET_ADDR_MAX];
  unsigned alen = arts_net_own_address(addr, sizeof(addr));
  arts_net_av_insert_table(addr, alen, 1);

  int failures = 0;

  /* --- (1) txid allocation ------------------------------------------- */
  uint64_t t1 = arts_net_rdzv_txid_next();
  uint64_t t2 = arts_net_rdzv_txid_next();
  if (t1 == 0 || t2 == 0 || t1 == t2 || (t1 >> 48) != arts_global_rank_id) {
    fprintf(stderr, "FAIL net_rendezvous: txid allocation (t1=%llx t2=%llx)\n",
            (unsigned long long)t1, (unsigned long long)t2);
    ++failures;
  } else {
    printf("PASS net_rendezvous: txid allocation (nonzero, unique, rank bits)\n");
  }

  /* --- (2) landing advertisement -------------------------------------- */
  unsigned char *src = (unsigned char *)arts_regpool_alloc_aligned(PUT_LEN, 64);
  unsigned char *dst = (unsigned char *)arts_regpool_alloc_aligned(PUT_LEN, 64);
  if (src == NULL || dst == NULL) {
    fprintf(stderr, "FAIL net_rendezvous: regpool alloc failed\n");
    return 1;
  }
  fill_pattern(src, PUT_LEN, 7);
  memset(dst, 0, PUT_LEN);

  uint64_t raddr = 0, rkey = 0;
  if (!arts_net_rdzv_local(dst, PUT_LEN, &raddr, &rkey)) {
    fprintf(stderr, "FAIL net_rendezvous: rdzv_local failed to resolve dst\n");
    return 1;
  }
  printf("PASS net_rendezvous: landing advert raddr=0x%llx rkey=0x%llx "
         "(mr_mode=0x%x)\n",
         (unsigned long long)raddr, (unsigned long long)rkey, g_net.mr_mode);

  /* --- (3) order A: data arrives BEFORE the expectation --------------- */
  uint64_t before_arrived =
      atomic_load_explicit(&g_rdzv_arrived_count, memory_order_acquire);
  arts_net_put_payload(0, raddr, rkey, t1, src, PUT_LEN, on_local_done, NULL);

  /* Wait until the remote-write completion has been PROCESSED (the arrival
   * either paired or parked a marker) — g_rdzv_arrived_count counts that. */
  uint64_t deadline = now_ns() + 5000000000ULL;
  while (atomic_load_explicit(&g_rdzv_arrived_count, memory_order_acquire) ==
         before_arrived) {
    arts_net_progress();
    if (now_ns() > deadline) {
      fprintf(stderr,
              "FAIL net_rendezvous: no remote-write completion within 5 s "
              "(FI_REMOTE_CQ_DATA not delivered?)\n");
      return 1;
    }
  }
  /* Data has arrived and no expectation was parked — registering it now must
   * fire the callback inline (order A). */
  int cb_seq = 0;
  arts_net_rdzv_expect(t1, on_data, &cb_seq);
  if (atomic_load_explicit(&g_paired, memory_order_acquire) != 1) {
    fprintf(stderr,
            "FAIL net_rendezvous: order-A expect did not fire inline\n");
    ++failures;
  } else if (!check_pattern(dst, PUT_LEN, 7)) {
    fprintf(stderr, "FAIL net_rendezvous: order-A landing bytes corrupt\n");
    ++failures;
  } else {
    printf("PASS net_rendezvous: order A (data->expect) 8 MiB landed intact\n");
  }
  if (!wait_ctr(&g_local_done, 1)) {
    fprintf(stderr, "FAIL net_rendezvous: local-done callback missing\n");
    ++failures;
  }

  /* --- (4) order B: expectation registered BEFORE the PUT ------------- */
  fill_pattern(src, PUT_LEN, 42);
  memset(dst, 0, PUT_LEN);
  arts_net_rdzv_expect(t2, on_data, NULL); /* parks (no data yet) */
  if (atomic_load_explicit(&g_paired, memory_order_acquire) != 1) {
    fprintf(stderr, "FAIL net_rendezvous: order-B expect fired early\n");
    ++failures;
  }
  arts_net_put_payload(0, raddr, rkey, t2, src, PUT_LEN, on_local_done, NULL);
  if (!wait_ctr(&g_paired, 2)) {
    fprintf(stderr, "FAIL net_rendezvous: order-B pairing never fired\n");
    ++failures;
  } else if (!check_pattern(dst, PUT_LEN, 42)) {
    fprintf(stderr, "FAIL net_rendezvous: order-B landing bytes corrupt\n");
    ++failures;
  } else {
    printf("PASS net_rendezvous: order B (expect->data) 8 MiB landed intact\n");
  }

  /* --- (5) local completions drain ------------------------------------ */
  if (!wait_ctr(&g_local_done, 2)) {
    fprintf(stderr, "FAIL net_rendezvous: second local-done missing\n");
    ++failures;
  }
  for (int i = 0; i < 2000; i++) {
    if (atomic_load_explicit(&g_net.tx_outstanding, memory_order_relaxed) == 0) {
      break;
    }
    arts_net_progress();
  }
  if (atomic_load_explicit(&g_net.tx_outstanding, memory_order_relaxed) != 0) {
    fprintf(stderr, "FAIL net_rendezvous: TX still outstanding after drain\n");
    ++failures;
  }

  arts_regpool_free(src);
  arts_regpool_free(dst);

  arts_net_quiesce();
  arts_regpool_cleanup();
  arts_net_teardown();

  if (failures) {
    fprintf(stderr, "FAIL net_rendezvous: %d failure(s)\n", failures);
    return 1;
  }
  printf("PASS net_rendezvous: all checks OK\n");
  return 0;
}
