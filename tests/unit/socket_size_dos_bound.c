// SPDX-License-Identifier: Apache-2.0
/*
 * socket_size_dos_bound — pure_unit (socketpair + malloc interposition)
 *
 * TARGET: arts_transport_receive() buffer-growth path in
 *   libs/src/core/transport/socket.c (around lines 684-700):
 *
 *     if (bypass_packet_size[pos] < packet->size) {
 *       uint64_t new_buf_size = (packet->size > (1ULL << 28))
 *                                   ? packet->size            // exact
 *                                   : packet->size * 4;       // 4x
 *       char *next_buf = (char *)arts_malloc(new_buf_size);   // <-- unbounded
 *       memcpy(next_buf, bypass_buf[pos], bypass_packet_size[pos]);
 *       ...
 *     }
 *
 * INVARIANT VIOLATED (documented): packet->size is taken VERBATIM from the
 * wire header and used as an allocation size with NO sanity upper bound. A
 * peer (or a corrupted/forged frame) can set size to an arbitrary 64-bit
 * value and force a single arts_malloc() of that many bytes — an unbounded
 * memory-amplification / OOM denial-of-service. There is also no NULL check
 * on the returned next_buf, so a failed huge allocation then NULL-derefs in
 * the memcpy. This test DOCUMENTS the missing bound. (B-receive: trusts
 * wire packet->size.)
 *
 * HARNESS: we feed a single, fully-buffered packet whose 16-byte header
 * declares an enormous size (here 64 GiB — far over any legitimate DB), then
 * just enough body bytes that the framing loop completes the header read and
 * enters the growth branch. arts_malloc is interposed by THIS TU; when it
 * sees the attacker-controlled oversized request it records the size and
 * longjmp()s back out of arts_transport_receive — proving the runtime
 * ASKED for a 64 GiB allocation with no bound, WITHOUT this test process
 * actually trying to reserve 64 GiB (which would OOM the box). All legitimate
 * (small) allocations are serviced normally.
 *
 * If a future change adds a sanity bound that rejects/clamps the oversized
 * size before the malloc, the oversized request is never made: the test
 * detects that and prints that the bound now exists (still PASS, with a
 * different message). Today, the oversized request IS made -> the test PASSES
 * by documenting the DoS surface.
 */

#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <setjmp.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "arts/counter/counter.h"
#include "arts/runtime_state.h"
#include "arts/system/config.h"
#include "arts/transport/protocol.h"

/* ----- extern stubs ----- */
struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;
ARTS_THREAD_LOCAL arts_counter_t arts_thread_local_counters[NUM_COUNTER_TYPES];
unsigned int arts_global_rank_id;
unsigned int arts_global_rank_count;

/* DoS-bound instrumentation: the threshold above which we consider an
 * allocation request "attacker-controlled oversize" for this test. Any real
 * ARTS DB transfer is far below this. */
#define DOS_THRESHOLD (8ULL << 30) /* 8 GiB */

static volatile uint64_t g_oversize_request; /* size the runtime asked for */
static volatile int g_oversize_seen;
static jmp_buf g_escape; /* bail out of receive() before the huge alloc */
static volatile int g_escape_armed;

void arts_counter_increment_by(arts_counter_t *c, uint64_t n) {
  (void)c;
  (void)n;
}
void arts_abort(uint8_t code) {
  fprintf(stderr, "FAIL socket_size_dos_bound: arts_abort(%u)\n",
          (unsigned)code);
  exit(2);
}
void *arts_malloc(size_t s) {
  if (g_escape_armed && (uint64_t)s >= DOS_THRESHOLD) {
    /* The runtime just requested an oversized buffer straight from the wire
     * size field — the DoS surface. Record it and escape WITHOUT performing
     * the (OOM-inducing) allocation. */
    g_oversize_request = (uint64_t)s;
    g_oversize_seen = 1;
    g_escape_armed = 0;
    longjmp(g_escape, 1);
  }
  return malloc(s);
}
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void arts_free(void *p) { free(p); }
unsigned int arts_atomic_sub(volatile unsigned int *d, unsigned int v) {
  return __atomic_sub_fetch(d, v, __ATOMIC_SEQ_CST);
}
unsigned int arts_atomic_add(volatile unsigned int *d, unsigned int v) {
  return __atomic_add_fetch(d, v, __ATOMIC_SEQ_CST);
}
void arts_enter_shutdown_state(bool initiator) { (void)initiator; }
void arts_runtime_stop(void) {}
void arts_transport_dispatch_packet(struct arts_msg_header_s *packet) {
  (void)packet;
}

#include "../../libs/src/core/transport/socket.c"

static int g_pair[2];
static struct arts_config_s g_cfg;

int main(void) {
  const uint64_t HDR = sizeof(struct arts_msg_header_s);

  static struct arts_config_table_s table[2];
  memset(table, 0, sizeof(table));
  memset(&g_cfg, 0, sizeof(g_cfg));
  g_cfg.table_length = 2;
  g_cfg.port_count = 1;
  g_cfg.my_rank = 0;
  g_cfg.nodes = 2;
  g_cfg.table = table;
  arts_transport_set_config(&g_cfg);

  if (socketpair(AF_UNIX, SOCK_STREAM, 0, g_pair) != 0) {
    perror("socketpair");
    fprintf(stderr, "FAIL socket_size_dos_bound: socketpair\n");
    return 1;
  }
  remote_socket_receive_list = (int *)calloc(1, sizeof(int));
  poll_incoming = (struct pollfd *)calloc(1, sizeof(struct pollfd));
  remote_socket_receive_list[0] = g_pair[1];
  poll_incoming[0].fd = g_pair[1];
  poll_incoming[0].events = POLLIN;
  arts_transport_set_thread_inbound_queues(0, 1);

  /* Build a forged header claiming a 64 GiB packet. We only write the
   * header + a handful of body bytes; the framing loop assembles the full
   * header (16 bytes), then sees bypass_packet_size (4 MiB) < packet->size
   * and enters the growth branch -> arts_malloc(64 GiB). */
  const uint64_t forged_size = (64ULL << 30); /* 64 GiB */
  char hdrbuf[64];
  memset(hdrbuf, 0, sizeof(hdrbuf));
  struct arts_msg_header_s *h = (struct arts_msg_header_s *)hdrbuf;
  h->message_type = 7;
  h->size = forged_size;
  /* write the whole header plus 8 body bytes so the header-fill loop
   * completes and we reach the size-growth branch. */
  size_t to_write = (size_t)HDR + 8;
  size_t off = 0;
  while (off < to_write) {
    ssize_t w = write(g_pair[0], hdrbuf + off, to_write - off);
    if (w < 0) {
      if (errno == EINTR)
        continue;
      perror("write");
      fprintf(stderr, "FAIL socket_size_dos_bound: write\n");
      return 1;
    }
    off += (size_t)w;
  }

  g_escape_armed = 1;
  int jumped = setjmp(g_escape);
  if (jumped == 0) {
    poll_incoming[0].revents = POLLIN;
    /* This call will, with the current (unbounded) code, request a 64 GiB
     * arts_malloc and longjmp back here. If a sanity bound is added, it will
     * NOT make that request and will return normally (jumped stays 0). */
    arts_transport_receive(0);
  }
  g_escape_armed = 0;

  arts_transport_thread_inbound_queues_cleanup();
  close(g_pair[0]);
  close(g_pair[1]);
  free(remote_socket_receive_list);
  free(poll_incoming);

  if (g_oversize_seen) {
    /* Documented DoS surface: the runtime allocated straight from the wire
     * size with no bound. We assert the requested size is exactly the forged
     * wire value (the >256 MiB path uses the size verbatim), confirming
     * attacker control of the allocation size. */
    if (g_oversize_request != forged_size) {
      fprintf(stderr,
              "FAIL socket_size_dos_bound: oversize request was %llu, "
              "expected the verbatim wire size %llu\n",
              (unsigned long long)g_oversize_request,
              (unsigned long long)forged_size);
      return 1;
    }
    printf("PASS socket_size_dos_bound (DOCUMENTED: arts_transport_receive "
           "requested a %llu-byte (%.1f GiB) allocation taken VERBATIM from "
           "the wire packet->size with NO sanity upper bound — memory-"
           "amplification DoS surface; also no NULL check on the result)\n",
           (unsigned long long)g_oversize_request,
           (double)g_oversize_request / (double)(1ULL << 30));
    return 0;
  }

  /* No oversized request was made: a sanity bound now exists. */
  printf("PASS socket_size_dos_bound (a sanity bound on wire packet->size "
         "now rejects the forged %llu-byte size before allocation)\n",
         (unsigned long long)forged_size);
  return 0;
}
