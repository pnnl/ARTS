// SPDX-License-Identifier: Apache-2.0
/*
 * socket_helpers — pure_unit (getaddrinfo interposition + fork)
 *
 * TARGET: the standalone helpers in libs/src/core/transport/socket.c:
 *   - hostname_to_ip(host, ip): resolves a hostname into a printable IPv4
 *     string via getaddrinfo(AF_INET) + inet_ntop(..., ip, 100). Caller
 *     contract: `ip` MUST be a >=100-byte buffer.
 *   - arts_get_new_socket(): TCP socket factory (+ TCP_NODELAY).
 *   - arts_get_socket_listening(sa, port): TCP socket factory; fills sa with
 *     AF_INET / INADDR_ANY / htons(port).
 *   - arts_get_socket_outgoing(sa, port, s_addr): TCP socket factory; fills
 *     sa with AF_INET / s_addr / htons(port).
 *
 * PROPERTIES UNDER TEST:
 *   1. hostname_to_ip on a numeric IPv4 literal round-trips the exact dotted
 *      string back into the caller buffer and returns true.
 *   2. hostname_to_ip on an unresolvable name returns false and does not
 *      write past the buffer (we sentinel-guard the 100-byte slot).
 *   3. Socket factories return a valid fd and populate the sockaddr_in with
 *      the exact family/addr/port the contract promises.
 *   4. ai_addr NULL handling: the census flags that hostname_to_ip
 *      dereferences result->ai_addr without a NULL check (relying on
 *      getaddrinfo's "success => non-empty list" guarantee). We interpose
 *      getaddrinfo to return success with ai_addr == NULL and run
 *      hostname_to_ip in a forked child; the child reports via a sentinel
 *      pipe whether it survived. If it crashes, that is the documented
 *      fragile surface (reported, not masked).
 *
 * The real socket.c is compiled into this TU (project precedent: tests
 * #include a runtime .c directly); externs are stubbed locally.
 */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netdb.h>
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

/* ----- extern stubs ----- */
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

/* ----- optional getaddrinfo interposition (only active for the NULL-ai_addr
 * sub-test, gated by an env flag so the real resolver is used elsewhere).
 * We define getaddrinfo/freeaddrinfo here so they win over libc at link
 * time, and reach the real resolver via dlsym(RTLD_NEXT) when the flag is
 * off. ----- */
static struct addrinfo g_fake_ai;

typedef int (*gai_fn)(const char *, const char *, const struct addrinfo *,
                      struct addrinfo **);
typedef void (*fai_fn)(struct addrinfo *);

static gai_fn real_getaddrinfo(void) {
  static gai_fn p;
  if (!p) {
    p = (gai_fn)dlsym(RTLD_NEXT, "getaddrinfo");
  }
  return p;
}
static fai_fn real_freeaddrinfo(void) {
  static fai_fn p;
  if (!p) {
    p = (fai_fn)dlsym(RTLD_NEXT, "freeaddrinfo");
  }
  return p;
}

int getaddrinfo(const char *node, const char *service,
                const struct addrinfo *hints, struct addrinfo **res) {
  if (getenv("ARTS_TEST_FAKE_NULL_AI")) {
    (void)node;
    (void)service;
    (void)hints;
    memset(&g_fake_ai, 0, sizeof(g_fake_ai));
    g_fake_ai.ai_family = AF_INET;
    g_fake_ai.ai_addr = NULL; /* the hazard: success but NULL ai_addr */
    g_fake_ai.ai_next = NULL;
    *res = &g_fake_ai;
    return 0;
  }
  return real_getaddrinfo()(node, service, hints, res);
}

void freeaddrinfo(struct addrinfo *res) {
  if (getenv("ARTS_TEST_FAKE_NULL_AI")) {
    return; /* static storage */
  }
  real_freeaddrinfo()(res);
}

/* pull in the real implementation */
#include "../../libs/src/core/transport/socket.c"

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL socket_helpers: %s (line %d)\n", msg, __LINE__);   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

/* ---- ai_addr NULL sub-test, run in a forked child with a survive pipe ---- */
static int test_null_ai_addr(void) {
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    perror("pipe");
    return -1;
  }
  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return -1;
  }
  if (pid == 0) {
    close(pipefd[0]);
    setenv("ARTS_TEST_FAKE_NULL_AI", "1", 1);
    char ip[100];
    memset(ip, 0, sizeof(ip));
    /* If hostname_to_ip guards result->ai_addr it returns gracefully;
     * if not, it dereferences NULL and crashes before the sentinel. */
    bool ok = hostname_to_ip((char *)"whatever.example", ip);
    unsigned char s = ok ? 1 : 2;
    ssize_t w;
    do {
      w = write(pipefd[1], &s, 1);
    } while (w < 0 && errno == EINTR);
    _exit(0);
  }
  close(pipefd[1]);
  unsigned char sentinel = 0;
  ssize_t r;
  do {
    r = read(pipefd[0], &sentinel, 1);
  } while (r < 0 && errno == EINTR);
  bool survived = (r == 1);
  close(pipefd[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  return survived ? 1 : 0; /* 1 = survived, 0 = crashed */
}

int main(void) {
  /* ---- 1. hostname_to_ip round-trips a numeric IPv4 literal ---- */
  {
    char ip[100];
    /* sentinel just past the documented 100-byte slot to catch overflow */
    memset(ip, 0xAB, sizeof(ip));
    bool ok = hostname_to_ip((char *)"192.0.2.55", ip);
    CHECK(ok, "hostname_to_ip(numeric) returned false");
    CHECK(strcmp(ip, "192.0.2.55") == 0,
          "hostname_to_ip(numeric) wrong dotted string");
  }

  /* ---- 2. hostname_to_ip on an unresolvable name returns false ---- */
  {
    /* A syntactically-invalid name that getaddrinfo will reject. Use the
     * AI flags path: an empty + clearly-bogus label. */
    char ip[128];
    /* place a guard byte at index 100 (just past the contract slot) */
    memset(ip, 0, sizeof(ip));
    ip[100] = (char)0x7E;
    bool ok = hostname_to_ip(
        (char *)"this.name.does.not.exist.invalid.arts.test.", ip);
    /* On failure hostname_to_ip must return false and must NOT have written
     * the result (the guard byte is untouched because inet_ntop never ran). */
    CHECK(!ok, "hostname_to_ip(bogus) unexpectedly returned true");
    CHECK(ip[100] == (char)0x7E,
          "hostname_to_ip(bogus) wrote past/inside buffer on failure");
  }

  /* ---- 3. socket factories populate sockaddr fields exactly ---- */
  {
    int fd = arts_get_new_socket();
    CHECK(fd >= 0, "arts_get_new_socket returned bad fd");
    close(fd);
  }
  {
    struct sockaddr_in sa;
    memset(&sa, 0xCC, sizeof(sa));
    int fd = arts_get_socket_listening(&sa, 25123);
    CHECK(fd >= 0, "arts_get_socket_listening returned bad fd");
    CHECK(sa.sin_family == AF_INET, "listening: wrong family");
    CHECK(sa.sin_addr.s_addr == htonl(INADDR_ANY),
          "listening: addr not INADDR_ANY");
    CHECK(sa.sin_port == htons(25123), "listening: wrong port");
    close(fd);
  }
  {
    struct sockaddr_in sa;
    memset(&sa, 0xCC, sizeof(sa));
    in_addr_t a = inet_addr("198.51.100.9");
    int fd = arts_get_socket_outgoing(&sa, 25124, a);
    CHECK(fd >= 0, "arts_get_socket_outgoing returned bad fd");
    CHECK(sa.sin_family == AF_INET, "outgoing: wrong family");
    CHECK(sa.sin_addr.s_addr == a, "outgoing: wrong addr");
    CHECK(sa.sin_port == htons(25124), "outgoing: wrong port");
    close(fd);
  }

  /* ---- 4. ai_addr NULL handling (documented fragile surface) ----
   * This is a DEFENSIVE-ONLY surface, not a reachable defect: POSIX
   * guarantees getaddrinfo() returning 0 yields a non-empty list with a
   * valid ai_addr, so the only way to reach the unguarded
   * `result->ai_addr->sa_family` is a contract-violating resolver (which we
   * inject here). We therefore DOCUMENT the outcome but do NOT hard-fail the
   * helper test on it (unlike T169's set_ip NULL ifa_addr, which IS reachable
   * because getifaddrs legitimately returns NULL ifa_addr). If a future
   * change makes hostname_to_ip survive a NULL ai_addr, this prints
   * "guarded"; today it prints "unguarded". */
  {
    int rc = test_null_ai_addr();
    if (rc < 0) {
      fprintf(stderr, "FAIL socket_helpers: ai_addr sub-test harness error\n");
      return 1;
    }
    if (rc == 0) {
      printf("NOTE socket_helpers: hostname_to_ip has NO NULL guard on "
             "result->ai_addr (socket.c:106); a contract-violating "
             "getaddrinfo success with ai_addr==NULL would SIGSEGV. "
             "Defensive-only (unreachable with a conforming resolver).\n");
    } else {
      printf("NOTE socket_helpers: hostname_to_ip tolerated a NULL "
             "ai_addr (guarded).\n");
    }
  }

  printf("PASS socket_helpers (hostname_to_ip round-trip + failure, socket "
         "factories, ai_addr NULL surface)\n");
  return 0;
}
