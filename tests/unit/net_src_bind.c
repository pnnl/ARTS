/* SPDX-License-Identifier: Apache-2.0
 *
 * Whitebox check of arts_net_init's interface/domain steering hints
 * (transport/net.c) — standalone fabric, no ARTS runtime.
 *
 * Coverage:
 *   1. net_interface bind ("lo"): the endpoint's fi_getname address must be
 *      that interface's IPv4 address (127.0.0.1), proving the src_addr hint
 *      constrains the provider instead of leaving it on the default route.
 *   2. A bound endpoint still moves bytes: av-insert self + one send through
 *      the production send path must deliver.
 *   3. fabric_domain pin: re-initializing with the domain name the first
 *      bring-up negotiated must succeed (domain_attr->name plumbing).
 *   4. A net_interface that matches no interface is fatal (forked child must
 *      exit nonzero), never a silent fallback to the default route.
 *
 * Same shim pattern as net_selfsend.c: #includes net.c, fakes the runtime
 * globals, stubs dispatch as the RX sink.  tcp provider throughout — the only
 * IP provider guaranteed present in CI.
 */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
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

static _Atomic int g_recv_count;

void arts_transport_dispatch_packet(struct arts_msg_header_s *packet) {
  (void)packet;
  atomic_fetch_add_explicit(&g_recv_count, 1, memory_order_acq_rel);
}

void arts_transport_loopback_post(const void *packet, unsigned int size) {
  (void)packet;
  (void)size;
}

/* Compile the module under test (its statics). */
#include "../../libs/src/core/transport/net.c"

/* counter stubs: net.c's introspection increments resolve here (this harness
 * links no counter TU). */
ARTS_THREAD_LOCAL arts_counter_t arts_thread_local_counters[NUM_COUNTER_TYPES];
void arts_counter_increment_by(arts_counter_t *counter, uint64_t num) {
  (void)counter;
  (void)num;
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int main(void) {
  int failures = 0;

  /* --- (4) missing interface is fatal: probe in a forked child first, so a
   * regression to silent-fallback cannot take down the whole test. ------- */
  pid_t pid = fork();
  if (pid == 0) {
    freopen("/dev/null", "w", stderr); /* silence the expected ERROR line */
    arts_net_init("tcp", NULL, "no-such-interface-000");
    _exit(0); /* reaching here means the fatal contract was dropped */
  }
  int status = 0;
  waitpid(pid, &status, 0);
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    fprintf(stderr, "FAIL net_src_bind: bogus interface did not fail\n");
    failures++;
  }

  /* --- (1) bind to loopback and verify the endpoint's own address ------- */
  arts_net_init("tcp", NULL, "lo");

  unsigned char addr[ARTS_NET_ADDR_MAX];
  unsigned alen = arts_net_own_address(addr, sizeof(addr));
  if (alen < sizeof(struct sockaddr_in)) {
    fprintf(stderr, "FAIL net_src_bind: fi_getname len %u too short\n", alen);
    return 1;
  }
  struct sockaddr_in sin;
  memcpy(&sin, addr, sizeof(sin));
  char ip[64] = {0};
  inet_ntop(AF_INET, &sin.sin_addr, ip, sizeof(ip));
  if (sin.sin_family != AF_INET || strcmp(ip, "127.0.0.1") != 0) {
    fprintf(stderr,
            "FAIL net_src_bind: bound address %s (family %d), want 127.0.0.1\n",
            ip, (int)sin.sin_family);
    failures++;
  }

  /* Remember the negotiated domain name for check (3). */
  char domain_name[128];
  snprintf(domain_name, sizeof(domain_name), "%s",
           g_net.info->domain_attr->name);

  /* --- (2) the bound endpoint moves bytes ------------------------------- */
  if (!arts_regpool_init(arts_net_domain(), (size_t)32 * 1024 * 1024, 0)) {
    fprintf(stderr, "FAIL net_src_bind: regpool init failed\n");
    return 1;
  }
  arts_net_rx_arm();
  arts_net_av_insert_table(addr, alen, 1);

  struct arts_msg_header_s pk;
  arts_fill_packet_header(&pk, sizeof(pk), MSG_TIME_SYNC_REQUEST);
  arts_net_send_core(0, (char *)&pk, (unsigned)sizeof(pk), NULL, 0, 0, NULL);
  uint64_t deadline = now_ns() + 5000000000ULL;
  while (atomic_load_explicit(&g_recv_count, memory_order_acquire) < 1) {
    arts_net_progress();
    if (now_ns() > deadline) {
      fprintf(stderr, "FAIL net_src_bind: bound-EP send not delivered\n");
      failures++;
      break;
    }
  }

  /* --- (3) pin the negotiated domain name in a fresh child -------------- */
  pid = fork();
  if (pid == 0) {
    arts_net_init("tcp", domain_name, NULL);
    _exit(arts_net_domain() != NULL ? 0 : 1);
  }
  waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "FAIL net_src_bind: domain pin '%s' rejected\n",
            domain_name);
    failures++;
  }

  if (failures) {
    return 1;
  }
  printf("PASS net_src_bind: lo-bound EP (127.0.0.1) delivered, domain pin "
         "'%s' ok, bogus iface fatal\n",
         domain_name);
  return 0;
}
