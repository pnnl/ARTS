/* SPDX-License-Identifier: Apache-2.0
 *
 * Whitebox self-send exercise for the libfabric transport core
 * (transport/net.c).  Brings the fabric up standalone — NO ARTS runtime, no
 * sockets, no worker threads — fakes the minimal runtime globals the module
 * touches, av-inserts this process's own EP address at index 0, and drives the
 * send path straight at the internal core (net_send_core, which skips the
 * production self-send-to-loopback guard) so the message rides the fabric and
 * loops back to our own receive buffers.  A stubbed dispatch symbol captures
 * the delivered bytes.
 *
 * Coverage:
 *   1. small (<= inject_size) header-only send -> fi_inject -> RX -> exact bytes
 *   2. one > inject_size send -> registered bounce + fi_send -> TX completion
 *      frees the bounce (tx_outstanding returns to 0)
 *   3. 200 rapid sends -> all delivered; report any EAGAIN-ring passes (soft:
 *      the tcp provider rarely back-pressures a loopback, so this only logs)
 *
 * Standalone: #includes net.c (its statics) and links the real regpool object
 * (registered memory for the bounce path) + libfabric; runtime externs are
 * libc-backed shims.  FI_PROVIDER=tcp is pinned by the test's environment.
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

/* ---- dispatch capture (the stubbed RX sink) ------------------------------ */
static _Atomic int g_recv_count;
static _Atomic unsigned g_last_len;
static unsigned char g_last[8192];

void arts_transport_dispatch_packet(struct arts_msg_header_s *packet) {
  unsigned len = (unsigned)packet->size;
  if (len <= sizeof(g_last)) {
    memcpy(g_last, packet, len);
    atomic_store_explicit(&g_last_len, len, memory_order_release);
  }
  atomic_fetch_add_explicit(&g_recv_count, 1, memory_order_acq_rel);
}

/* net.c references the self-loopback post in the public send wrapper; the test
 * drives net_send_core directly and never hits it, but it must link. */
void arts_transport_loopback_post(const void *packet, unsigned int size) {
  (void)packet;
  (void)size;
}

/* Compile the module under test (its statics: net_send_core, g_net, ...). */
#include "../../libs/src/core/transport/net.c"

/* counter stubs: net.c's introspection increments resolve here (this harness
 * links no counter TU). */
ARTS_THREAD_LOCAL arts_counter_t arts_thread_local_counters[NUM_COUNTER_TYPES];
void arts_counter_increment_by(arts_counter_t *counter, uint64_t num) {
  (void)counter;
  (void)num;
}

/* ------------------------------------------------------------------------- */

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Spin progress until the cumulative delivered count reaches `want` or 5 s. */
static bool wait_for(int want) {
  uint64_t deadline = now_ns() + 5000000000ULL;
  while (atomic_load_explicit(&g_recv_count, memory_order_acquire) < want) {
    arts_net_progress();
    if (now_ns() > deadline) {
      return false;
    }
  }
  return true;
}

int main(void) {
  setenv("FI_PROVIDER", "tcp", 1);

  arts_net_init(NULL, NULL, NULL); /* auto -- FI_PROVIDER above still applies */
  /* Registered pool for the bounce/send path AND the recv landing buffers
   * (needs net_init's domain). */
  if (!arts_regpool_init(arts_net_domain(), arts_net_mr_endpoint(), (size_t)32 * 1024 * 1024, 0)) {
    fprintf(stderr, "FAIL net_selfsend: regpool init failed\n");
    return 1;
  }
  /* RX is armed from the pool, so it must follow regpool_init (mirrors the
   * runtime's node_init ordering). */
  arts_net_rx_arm();

  /* Capture the negotiated facts before teardown frees fi_info. */
  char cap_prov[64];
  snprintf(cap_prov, sizeof(cap_prov), "%s",
           g_net.info->fabric_attr->prov_name);
  size_t cap_inject = g_net.inject_size;
  uint32_t cap_mrmode = g_net.mr_mode;

  /* av-insert our own EP address at index 0 -> fi_addr_t 0 == self. */
  unsigned char addr[ARTS_NET_ADDR_MAX];
  unsigned alen = arts_net_own_address(addr, sizeof(addr));
  arts_net_av_insert_table(addr, alen, 1);

  int failures = 0;
  int expected = 0;

  /* --- (1) small header-only send: fi_inject path --------------------- */
  struct arts_msg_time_sync_req_packet_s pk;
  memset(&pk, 0, sizeof(pk));
  arts_fill_packet_header(&pk.header, sizeof(pk), MSG_TIME_SYNC_REQUEST);
  pk.worker_send_time = 0xdeadbeefcafef00dULL;

  if (sizeof(pk) > g_net.inject_size) {
    fprintf(stderr,
            "NOTE net_selfsend: tsreq (%zu B) exceeds inject_size (%zu) — "
            "message 1 uses the bounce path\n",
            sizeof(pk), g_net.inject_size);
  }
  arts_net_send_core(0,(char *)&pk, (unsigned)sizeof(pk), NULL, 0, 0, NULL);
  expected += 1;
  if (!wait_for(expected)) {
    fprintf(stderr, "FAIL net_selfsend: message 1 not delivered within 5 s\n");
    return 1;
  }
  if (atomic_load_explicit(&g_last_len, memory_order_acquire) != sizeof(pk) ||
      memcmp(g_last, &pk, sizeof(pk)) != 0) {
    fprintf(stderr, "FAIL net_selfsend: message 1 byte mismatch\n");
    ++failures;
  } else {
    printf("PASS net_selfsend: message 1 (inject) round-tripped %zu bytes\n",
           sizeof(pk));
  }

  /* --- (2) > inject_size send: bounce + fi_send + completion free ----- */
  const unsigned big_len = 4096;
  unsigned char *big = (unsigned char *)malloc(big_len);
  memset(big, 0, big_len);
  struct arts_msg_header_s *bh = (struct arts_msg_header_s *)big;
  arts_fill_packet_header(bh, big_len, MSG_TIME_SYNC_REQUEST);
  for (unsigned i = sizeof(*bh); i < big_len; i++) {
    big[i] = (unsigned char)(i * 31u + 7u);
  }
  arts_net_send_core(0,(char *)big, big_len, NULL, 0, 0, NULL);
  expected += 1;
  if (!wait_for(expected)) {
    fprintf(stderr, "FAIL net_selfsend: message 2 not delivered within 5 s\n");
    return 1;
  }
  if (atomic_load_explicit(&g_last_len, memory_order_acquire) != big_len ||
      memcmp(g_last, big, big_len) != 0) {
    fprintf(stderr, "FAIL net_selfsend: message 2 byte mismatch\n");
    ++failures;
  } else {
    printf("PASS net_selfsend: message 2 (bounce %u B) round-tripped\n",
           big_len);
  }
  free(big);

  /* Let any straggling TX completion reap; the bounce path must return
   * tx_outstanding to zero once completions drain. */
  for (int i = 0; i < 1000; i++) {
    if (atomic_load_explicit(&g_net.tx_outstanding, memory_order_relaxed) == 0) {
      break;
    }
    arts_net_progress();
  }
  uint64_t outstanding =
      atomic_load_explicit(&g_net.tx_outstanding, memory_order_relaxed);
  if (outstanding != 0) {
    fprintf(stderr, "FAIL net_selfsend: %llu TX still outstanding after drain\n",
            (unsigned long long)outstanding);
    ++failures;
  }

  /* --- (3) 200 rapid sends: no loss; observe (soft) EAGAIN rings ------ */
  uint64_t eagain_before =
      atomic_load_explicit(&g_net.eagain_count, memory_order_relaxed);
  const int burst = 200;
  for (int i = 0; i < burst; i++) {
    struct arts_msg_time_sync_req_packet_s q;
    memset(&q, 0, sizeof(q));
    arts_fill_packet_header(&q.header, sizeof(q), MSG_TIME_SYNC_REQUEST);
    q.worker_send_time = (uint64_t)i;
    arts_net_send_core(0,(char *)&q, (unsigned)sizeof(q), NULL, 0, 0, NULL);
    /* Interleave a little progress so completions/recv keep flowing. */
    if ((i & 15) == 0) {
      arts_net_progress();
    }
  }
  expected += burst;
  if (!wait_for(expected)) {
    fprintf(stderr,
            "FAIL net_selfsend: burst delivered %d of %d within 5 s\n",
            atomic_load_explicit(&g_recv_count, memory_order_acquire), expected);
    ++failures;
  } else {
    printf("PASS net_selfsend: burst of %d messages all delivered\n", burst);
  }
  uint64_t eagain_after =
      atomic_load_explicit(&g_net.eagain_count, memory_order_relaxed);
  printf("INFO net_selfsend: EAGAIN-ring passes during burst = %llu "
         "(soft; provider back-pressure is opportunistic)\n",
         (unsigned long long)(eagain_after - eagain_before));

  /* Final drain so no TX is left outstanding before teardown. */
  for (int i = 0; i < 2000; i++) {
    if (atomic_load_explicit(&g_net.tx_outstanding, memory_order_relaxed) == 0) {
      break;
    }
    arts_net_progress();
  }

  /* Two-phase teardown mirrors the runtime: quiesce the fabric (close ep/cq/av,
   * return recv buffers) while the pool is still live, then free the pool, then
   * close the domain/fabric. */
  arts_net_quiesce();
  arts_regpool_cleanup();
  arts_net_teardown();

  if (failures) {
    fprintf(stderr, "FAIL net_selfsend: %d failure(s)\n", failures);
    return 1;
  }
  printf("PASS net_selfsend: provider=%s inject_size=%zu mr_mode=0x%x "
         "multi_recv=yes — all checks OK\n",
         cap_prov, cap_inject, cap_mrmode);
  return 0;
}
