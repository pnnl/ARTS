// SPDX-License-Identifier: Apache-2.0
/*
 * socket_set_ip_null_ifa — pure_unit (getifaddrs interposition + fork)
 *
 * TARGET: arts_transport_set_ip() in libs/src/core/transport/socket.c,
 *   the rank-determination IP-match loop (around line 229):
 *
 *       for (ifa = ifap; ifa && !found; ifa = ifa->ifa_next) {
 *         if (ifa->ifa_addr->sa_family == AF_INET) {   // <-- NO NULL GUARD
 *
 * INVARIANT UNDER TEST: getifaddrs(3) is documented to return entries whose
 * ifa_addr field MAY be NULL (e.g. an interface that has no address, certain
 * tunnel/bonding devices). The earlier net_interface remap loop guards with
 * `ifa->ifa_addr && ...` (line 158); this rank-match loop does NOT. An entry
 * with ifa_addr==NULL therefore dereferences NULL -> SIGSEGV. This maps to
 * suspected bug B-set-ip-null.
 *
 * HARNESS: we interpose getifaddrs/freeifaddrs (our definitions win over
 * libc at link time) to return a deterministic list whose FIRST entry has
 * ifa_addr == NULL, followed by a well-formed AF_INET entry that does NOT
 * match any table IP (so `found` stays false and the loop visits the NULL
 * entry). getaddrinfo over numeric table IPs resolves offline. ARTS_RANK
 * and SLURM_PROCID are cleared so control reaches the ifa loop.
 *
 * arts_transport_set_ip is invoked in a forked child; the parent classifies
 * the child's exit:
 *   - child crashes with SIGSEGV  => the unguarded deref fired. This is the
 *     real runtime defect: we report FAIL (exposes_runtime_bug). The test is
 *     left correct-and-failing; it is NOT weakened to pass.
 *   - child returns normally      => a NULL guard exists; PASS.
 *
 * Note: a separate NULL-skip assertion would also pass once a guard is added,
 * but until then the SIGSEGV is the observable property.
 */

#define _GNU_SOURCE
#include <ifaddrs.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "arts/counter/counter.h"
#include "arts/runtime_state.h"
#include "arts/system/config.h"
#include "arts/transport/protocol.h"

/* ----- extern stubs the included socket.c needs ----- */
struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;
ARTS_THREAD_LOCAL arts_counter_t arts_thread_local_counters[NUM_COUNTER_TYPES];
unsigned int arts_global_rank_id;
unsigned int arts_global_rank_count;

void arts_counter_increment_by(arts_counter_t *c, uint64_t n) {
  (void)c;
  (void)n;
}
void arts_abort(uint8_t code) { _exit(70 + code); }
void *arts_malloc(size_t s) { return malloc(s); }
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

/* ----- getifaddrs interposition: NULL-ifa_addr first entry ----- */
static struct ifaddrs g_ifa_null; /* ifa_addr == NULL */
static struct ifaddrs g_ifa_inet; /* well-formed AF_INET, non-matching IP */
static struct sockaddr_in g_inet_addr;

int getifaddrs(struct ifaddrs **ifap) {
  memset(&g_ifa_null, 0, sizeof(g_ifa_null));
  memset(&g_ifa_inet, 0, sizeof(g_ifa_inet));
  memset(&g_inet_addr, 0, sizeof(g_inet_addr));

  /* Entry 0: an interface with NO address — ifa_addr is NULL.
   * This is the exact shape getifaddrs() can legitimately return. */
  static char name0[] = "nulldev";
  g_ifa_null.ifa_name = name0;
  g_ifa_null.ifa_addr = NULL; /* <-- the hazard */
  g_ifa_null.ifa_next = &g_ifa_inet;

  /* Entry 1: a normal AF_INET interface whose IP (203.0.113.7, TEST-NET-3)
   * is guaranteed NOT to be in our config table, so the match loop never
   * sets `found` and must keep iterating past the NULL entry. */
  static char name1[] = "fakeeth0";
  g_inet_addr.sin_family = AF_INET;
  g_inet_addr.sin_addr.s_addr = htonl(0xCB007107u); /* 203.0.113.7 */
  g_ifa_inet.ifa_name = name1;
  g_ifa_inet.ifa_addr = (struct sockaddr *)&g_inet_addr;
  g_ifa_inet.ifa_next = NULL;

  *ifap = &g_ifa_null;
  return 0;
}

void freeifaddrs(struct ifaddrs *ifa) { (void)ifa; /* static storage */ }

/* pull in the real implementation */
#include "../../libs/src/core/transport/socket.c"

/* write end of the survive-sentinel pipe; the child writes one byte AFTER
 * arts_transport_set_ip returns. If the child crashes (bare SIGSEGV, or a
 * sanitizer that intercepts the SEGV and _exit()s) the byte is never written,
 * so the parent reliably distinguishes "survived" from "crashed" regardless
 * of how the crash terminates the process. */
static int g_survive_fd = -1;

static void run_child(void) {
  /* Ensure the env rank shortcuts are not taken so we reach the ifa loop. */
  unsetenv("ARTS_RANK");
  unsetenv("SLURM_PROCID");

  static struct arts_config_table_s table[2];
  static unsigned int p0[1] = {25000};
  static unsigned int p1[1] = {25001};
  memset(table, 0, sizeof(table));
  /* numeric IPs resolve offline via getaddrinfo(AF_INET) */
  table[0].rank = 0;
  table[0].ip_address = (char *)"10.99.0.1";
  table[0].ports = p0;
  table[1].rank = 1;
  table[1].ip_address = (char *)"10.99.0.2";
  table[1].ports = p1;

  static struct arts_config_s cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.table_length = 2;
  cfg.port_count = 1;
  cfg.my_rank = 0;
  cfg.nodes = 2;
  cfg.net_interface = NULL; /* skip the (guarded) remap loop */
  cfg.table = table;

  arts_transport_set_config(&cfg);
  /* If a NULL guard exists, this returns (found stays false). If not, the
   * deref of g_ifa_null.ifa_addr->sa_family crashes before we get here. */
  bool found = arts_transport_set_ip(&cfg);

  /* Survived the loop -> emit the sentinel byte (1 if found, else 0). */
  unsigned char ok = found ? 1 : 2;
  ssize_t w;
  do {
    w = write(g_survive_fd, &ok, 1);
  } while (w < 0 && errno == EINTR);
  _exit(0);
}

int main(void) {
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    perror("pipe");
    fprintf(stderr, "FAIL socket_set_ip_null_ifa: pipe\n");
    return 1;
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    fprintf(stderr, "FAIL socket_set_ip_null_ifa: fork\n");
    return 1;
  }
  if (pid == 0) {
    close(pipefd[0]);
    g_survive_fd = pipefd[1];
    run_child();
    _exit(99); /* unreachable */
  }

  close(pipefd[1]);

  /* Read the survive sentinel. EOF (0 bytes) => child died before writing
   * => crash. */
  unsigned char sentinel = 0;
  ssize_t r;
  do {
    r = read(pipefd[0], &sentinel, 1);
  } while (r < 0 && errno == EINTR);
  bool survived = (r == 1);
  close(pipefd[0]);

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    perror("waitpid");
    fprintf(stderr, "FAIL socket_set_ip_null_ifa: waitpid\n");
    return 1;
  }

  if (!survived) {
    /* Child crashed before emitting the sentinel. Report what we can about
     * how it died (a bare build shows SIGSEGV; a sanitized build typically
     * shows a sanitizer-driven exit). Either way the unguarded NULL deref
     * fired. This is the real defect — the test stays correct-and-failing. */
    if (WIFSIGNALED(status)) {
      fprintf(stderr,
              "FAIL socket_set_ip_null_ifa: arts_transport_set_ip "
              "crashed (signal %d) on a NULL ifa_addr — unguarded "
              "ifa->ifa_addr->sa_family at the rank-match loop "
              "(socket.c:229). EXPOSES RUNTIME BUG B-set-ip-null.\n",
              WTERMSIG(status));
    } else {
      fprintf(stderr,
              "FAIL socket_set_ip_null_ifa: arts_transport_set_ip aborted "
              "on a NULL ifa_addr (child exit %d, no survive sentinel) — "
              "unguarded ifa->ifa_addr->sa_family at the rank-match loop "
              "(socket.c:229). EXPOSES RUNTIME BUG B-set-ip-null.\n",
              WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    }
    return 1;
  }

  /* Survived: a NULL guard skipped the NULL entry. The sentinel value is
   * incidental (found vs not-found); the property under test is no-crash. */
  printf("PASS socket_set_ip_null_ifa (NULL ifa_addr handled without crash; "
         "sentinel=%u)\n",
         (unsigned)sentinel);
  return 0;
}
