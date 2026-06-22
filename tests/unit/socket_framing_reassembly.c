// SPDX-License-Identifier: Apache-2.0
/*
 * socket_framing_reassembly — pure_unit (socketpair + extern stubs)
 *
 * TARGET: arts_transport_receive() reassembly loop in
 *   libs/src/core/transport/socket.c  (the cross-rank RX hot path).
 *
 * The framing loop is normally inseparable from globals + real sockets +
 * the dispatcher + counters + arts_node_info.  Following the project
 * precedent of compiling a runtime .c directly into a standalone test TU
 * (tests build edt_gpu.cu by #include'ing edt.c), this test
 * `#include "socket.c"` and supplies stub definitions for every extern it
 * references (arts_malloc/free, arts_atomic_*, counters, arts_node_info,
 * arts_thread_info, arts_runtime_stop, arts_enter_shutdown_state, and the
 * dispatch hook).  arts_transport_dispatch_packet is captured so the test
 * can ASSERT exactly which packets were framed, in which order, with which
 * sizes — the property the loop must preserve regardless of how the TCP
 * stream is chopped.
 *
 * A connected AF_UNIX socketpair() stands in for one peer's receive
 * socket.  We write a controlled byte stream into one end and let the
 * runtime's receive() drain the other, slicing the stream so we hit each
 * untested resume/grow/coalesce window:
 *
 *   1. PARTIAL-HEADER RESUME: deliver fewer than sizeof(arts_msg_header_s)
 *      bytes, then the rest on a later poll. The loop must save
 *      re_receive_res, return, and on the next call resume from the saved
 *      partial and frame the complete packet.
 *   2. MID-SIZE RESUME: deliver a full header + part of the body, then the
 *      remaining body later. The size-fill loop must save/restore and
 *      complete the packet.
 *   3. COALESCED PACKETS IN ORDER: deliver several whole packets in one
 *      recv; all must be dispatched, in stream order, each with its exact
 *      size (this exercises the memmove compaction + packet-pointer
 *      advance — targets B-receive-memmove).
 *   4. COALESCED + TRAILING PARTIAL: whole packet(s) followed by a partial
 *      next packet in the same recv; the whole ones dispatch now, the
 *      partial is compacted-to-front + saved and completed on the next poll.
 *   5. BUFFER GROWTH ×4: a packet larger than the 4 MiB initial buffer but
 *      <=256 MiB must trigger the *4 growth path and frame correctly.
 *   6. BUFFER GROWTH EXACT (>256 MiB): a packet over the 256 MiB threshold
 *      must use the exact-size growth path (no 4x over-alloc) and frame.
 *
 * Because all reassembly state is thread-local and per-socket, a single
 * test thread driving one socket exercises the exact code path a receiver
 * thread runs.  Success => "PASS socket_framing_reassembly".  A framing
 * defect (wrong order, wrong size, lost/duplicated packet, resume reading
 * stale offset) makes an assertion fail => the test reports a real runtime
 * bug rather than being weakened.
 */

#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* ----- extern stubs the included socket.c needs (resolved here, no libarts)
 * ----- */
#include "arts/counter/counter.h" /* arts_counter_t, arts_thread_local_counters */
#include "arts/runtime_state.h" /* arts_runtime_shared_s/private_s, arts_node_info */
#include "arts/system/config.h"      /* struct arts_config_s */
#include "arts/transport/protocol.h" /* struct arts_msg_header_s */

/* runtime globals socket.c reads/writes */
struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;
ARTS_THREAD_LOCAL arts_counter_t arts_thread_local_counters[NUM_COUNTER_TYPES];
unsigned int arts_global_rank_id;
unsigned int arts_global_rank_count;

/* capture of dispatch + shutdown signalling */
#define MAX_CAP 4096
static unsigned int g_cap_type[MAX_CAP];
static uint64_t g_cap_size[MAX_CAP];
static int g_cap_n;
static int g_runtime_stop_calls;
static int g_shutdown_calls;

void arts_counter_increment_by(arts_counter_t *c, uint64_t n) {
  (void)c;
  (void)n;
}
void arts_abort(uint8_t code) {
  fprintf(stderr, "FAIL socket_framing_reassembly: arts_abort(%u) called\n",
          (unsigned)code);
  exit(2);
}
void *arts_malloc(size_t s) { return malloc(s); }
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void arts_free(void *p) { free(p); }
unsigned int arts_atomic_sub(volatile unsigned int *d, unsigned int v) {
  return __atomic_sub_fetch(d, v, __ATOMIC_SEQ_CST);
}
unsigned int arts_atomic_add(volatile unsigned int *d, unsigned int v) {
  return __atomic_add_fetch(d, v, __ATOMIC_SEQ_CST);
}
void arts_enter_shutdown_state(bool initiator) {
  (void)initiator;
  g_shutdown_calls++;
}
void arts_runtime_stop(void) { g_runtime_stop_calls++; }

/* the framing loop hands each fully-assembled packet here */
void arts_transport_dispatch_packet(struct arts_msg_header_s *packet) {
  if (g_cap_n < MAX_CAP) {
    g_cap_type[g_cap_n] = packet->message_type;
    g_cap_size[g_cap_n] = packet->size;
    /* sanity: header size must be at least the header itself */
    if (packet->size < sizeof(struct arts_msg_header_s)) {
      fprintf(stderr, "FAIL: dispatched packet->size %llu < header size %zu\n",
              (unsigned long long)packet->size,
              sizeof(struct arts_msg_header_s));
      exit(1);
    }
  }
  g_cap_n++;
}

/* now pull in the real framing implementation */
#include "../../libs/src/core/transport/socket.c"

/* ---------------- test harness ---------------- */

static int g_pair[2]; /* g_pair[0] = writer (peer) ; g_pair[1] = receiver fd */
static struct arts_config_s g_cfg;

/* Build a single packet: header{message_type,size} then (size - hdr) body
 * bytes filled with a recognizable pattern. Returns malloc'd buffer of
 * `size` bytes. */
static char *make_packet(unsigned int type, uint64_t size) {
  assert(size >= sizeof(struct arts_msg_header_s));
  char *buf = (char *)malloc(size);
  assert(buf);
  struct arts_msg_header_s *h = (struct arts_msg_header_s *)buf;
  memset(h, 0, sizeof(*h));
  h->message_type = type;
  h->size = size;
  for (uint64_t k = sizeof(*h); k < size; k++) {
    buf[k] = (char)(type + k);
  }
  return buf;
}

/* run one receive() pass over our single owned socket */
static bool drive_receive(void) {
  poll_incoming[0].revents = POLLIN;
  bool r = arts_transport_receive(0);
  poll_incoming[0].revents = 0;
  return r;
}

/* write `n` bytes from buf into the peer end. The peer fd is non-blocking;
 * when the kernel socket buffer fills (EAGAIN) we drain the receiver side
 * via drive_receive() to make room, so this never deadlocks even for
 * payloads far larger than the socket buffer. */
static void peer_write(const char *buf, size_t n) {
  size_t off = 0;
  while (off < n) {
    ssize_t w = write(g_pair[0], buf + off, n - off);
    if (w < 0) {
      if (errno == EINTR)
        continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        drive_receive();
        continue;
      }
      perror("peer_write");
      fprintf(stderr, "FAIL socket_framing_reassembly: peer_write\n");
      exit(1);
    }
    off += (size_t)w;
  }
}

static void reset_capture(void) {
  g_cap_n = 0;
  g_runtime_stop_calls = 0;
  g_shutdown_calls = 0;
}

static void setup(void) {
  /* table_length = 2 => count = table_length-1 = 1 owned receive socket */
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
    fprintf(stderr, "FAIL socket_framing_reassembly: socketpair\n");
    exit(1);
  }

  /* The framing loop indexes remote_socket_receive_list[i] and
   * poll_incoming[i] for i in [thread_start, thread_stop). Allocate one
   * slot and wire it to the receiver end of the pair. */
  remote_socket_receive_list = (int *)calloc(1, sizeof(int));
  poll_incoming = (struct pollfd *)calloc(1, sizeof(struct pollfd));
  remote_socket_receive_list[0] = g_pair[1];
  poll_incoming[0].fd = g_pair[1];
  poll_incoming[0].events = POLLIN;

  /* Make the peer (writer) end non-blocking so peer_write() can detect a
   * full socket buffer and drain the receiver to make room (see
   * peer_write). The receiver fd is read with MSG_DONTWAIT by the runtime
   * code itself, so its blocking mode is irrelevant. */
  int fl = fcntl(g_pair[0], F_GETFL, 0);
  fcntl(g_pair[0], F_SETFL, fl | O_NONBLOCK);

  /* per-thread reassembly buffers for one owned socket [0,1) */
  arts_transport_set_thread_inbound_queues(0, 1);
}

static void teardown(void) {
  arts_transport_thread_inbound_queues_cleanup();
  close(g_pair[0]);
  close(g_pair[1]);
  free(remote_socket_receive_list);
  free(poll_incoming);
  remote_socket_receive_list = NULL;
  poll_incoming = NULL;
}

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL socket_framing_reassembly: %s (line %d)\n", msg,   \
              __LINE__);                                                       \
      teardown();                                                              \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(void) {
  const uint64_t HDR = sizeof(struct arts_msg_header_s);
  setup();

  /* ---- Scenario 1: PARTIAL-HEADER RESUME ----
   * deliver fewer than HDR bytes; receive() must save partial and dispatch
   * nothing; then deliver the rest + body; the completed packet dispatches. */
  reset_capture();
  {
    uint64_t size = HDR + 16;
    char *p = make_packet(101, size);
    /* split inside the header */
    size_t first = HDR - 1;
    peer_write(p, first);
    drive_receive();
    CHECK(g_cap_n == 0, "partial-header: dispatched before complete");
    /* remaining bytes (rest of header + whole body) */
    peer_write(p + first, (size_t)(size - first));
    drive_receive();
    CHECK(g_cap_n == 1, "partial-header: expected exactly 1 packet");
    CHECK(g_cap_type[0] == 101, "partial-header: wrong type");
    CHECK(g_cap_size[0] == size, "partial-header: wrong size");
    free(p);
  }

  /* ---- Scenario 2: MID-SIZE RESUME ----
   * deliver full header + partial body, then the rest. */
  reset_capture();
  {
    uint64_t size = HDR + 4096;
    char *p = make_packet(102, size);
    size_t first = HDR + 100; /* header complete, body partial */
    peer_write(p, first);
    drive_receive();
    CHECK(g_cap_n == 0, "mid-size: dispatched before body complete");
    peer_write(p + first, (size_t)(size - first));
    drive_receive();
    CHECK(g_cap_n == 1, "mid-size: expected exactly 1 packet");
    CHECK(g_cap_type[0] == 102, "mid-size: wrong type");
    CHECK(g_cap_size[0] == size, "mid-size: wrong size");
    free(p);
  }

  /* ---- Scenario 3: COALESCED PACKETS IN ORDER ----
   * three whole packets of different sizes, written back-to-back, must all
   * dispatch in stream order with exact sizes (memmove compaction +
   * packet-pointer advance). */
  reset_capture();
  {
    uint64_t s0 = HDR + 8, s1 = HDR + 200, s2 = HDR + 17;
    char *a = make_packet(201, s0);
    char *b = make_packet(202, s1);
    char *c = make_packet(203, s2);
    peer_write(a, s0);
    peer_write(b, s1);
    peer_write(c, s2);
    /* one drive may need a few passes to drain everything that arrived;
     * loop until no more progress (bounded). */
    int passes = 0;
    while (g_cap_n < 3 && passes < 8) {
      drive_receive();
      passes++;
    }
    CHECK(g_cap_n == 3, "coalesced: expected exactly 3 packets");
    CHECK(g_cap_type[0] == 201 && g_cap_size[0] == s0, "coalesced: pkt0");
    CHECK(g_cap_type[1] == 202 && g_cap_size[1] == s1, "coalesced: pkt1");
    CHECK(g_cap_type[2] == 203 && g_cap_size[2] == s2, "coalesced: pkt2");
    free(a);
    free(b);
    free(c);
  }

  /* ---- Scenario 4: COALESCED + TRAILING PARTIAL ----
   * one whole packet followed by a partial second packet (only part of its
   * body) in the same byte burst; the whole one dispatches now, the partial
   * must be compacted-to-front + saved and completed on the next poll. */
  reset_capture();
  {
    uint64_t s0 = HDR + 32;
    uint64_t s1 = HDR + 500;
    char *a = make_packet(211, s0);
    char *b = make_packet(212, s1);
    /* write whole a + header of b + 50 body bytes of b */
    peer_write(a, s0);
    size_t b_first = HDR + 50;
    peer_write(b, b_first);
    int passes = 0;
    while (g_cap_n < 1 && passes < 8) {
      drive_receive();
      passes++;
    }
    CHECK(g_cap_n == 1, "coalesced+partial: first packet must dispatch");
    CHECK(g_cap_type[0] == 211 && g_cap_size[0] == s0,
          "coalesced+partial: pkt0 wrong");
    /* deliver the rest of b */
    peer_write(b + b_first, (size_t)(s1 - b_first));
    passes = 0;
    while (g_cap_n < 2 && passes < 8) {
      drive_receive();
      passes++;
    }
    CHECK(g_cap_n == 2, "coalesced+partial: second packet must complete");
    CHECK(g_cap_type[1] == 212 && g_cap_size[1] == s1,
          "coalesced+partial: pkt1 wrong (stale-offset resume?)");
    free(a);
    free(b);
  }

  /* ---- Scenario 5: BUFFER GROWTH ×4 (>4 MiB, <=256 MiB) ----
   * the initial per-socket buffer is PACKET_SIZE (4 MiB); a packet just
   * over that must trigger the *4 growth and still frame correctly. */
  reset_capture();
  {
    uint64_t size = (uint64_t)PACKET_SIZE + 4096; /* ~4 MiB + a bit */
    char *p = make_packet(221, size);
    peer_write(p, size);
    int passes = 0;
    while (g_cap_n < 1 && passes < 64) {
      drive_receive();
      passes++;
    }
    CHECK(g_cap_n == 1, "growth-4x: expected exactly 1 packet");
    CHECK(g_cap_type[0] == 221, "growth-4x: wrong type");
    CHECK(g_cap_size[0] == size, "growth-4x: wrong size");
    CHECK(bypass_packet_size[0] >= size, "growth-4x: buffer not grown");
    free(p);
  }

  /* ---- Scenario 6: BUFFER GROWTH EXACT (>256 MiB) ----
   * a packet over the 256 MiB threshold must use the exact-size growth
   * path. This allocates a large (>256 MiB) buffer twice (peer copy + the
   * runtime buffer) — kept to the minimum over-threshold size to bound RAM
   * (~256 MiB + a page each). */
  reset_capture();
  {
    uint64_t size = (1ULL << 28) + 4096; /* just over 256 MiB */
    char *p = make_packet(231, size);
    /* peer_write drains the receiver on EAGAIN, so a payload far larger
     * than the socket buffer streams through without deadlock. */
    peer_write(p, size);
    int passes = 0;
    while (g_cap_n < 1 && passes < 4096) {
      drive_receive();
      passes++;
    }
    CHECK(g_cap_n == 1, "growth-exact: expected exactly 1 packet");
    CHECK(g_cap_type[0] == 231, "growth-exact: wrong type");
    CHECK(g_cap_size[0] == size, "growth-exact: wrong size");
    /* exact path: buffer grown to exactly packet->size (no 4x) */
    CHECK(bypass_packet_size[0] == size,
          "growth-exact: buffer not exact-sized (>256MiB path)");
    free(p);
  }

  teardown();
  printf("PASS socket_framing_reassembly (6 scenarios: partial-header, "
         "mid-size, coalesced, coalesced+partial, growth-4x, growth-exact)\n");
  return 0;
}
