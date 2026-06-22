/* SPDX-License-Identifier: Apache-2.0
 *
 * T233 — async-signal-safe formatters in signals.c (write_uint, signal_name)
 * plus arts_atomic_print truncation (print.h).  write_uint and signal_name are
 * file-static in signals.c, so this TU #includes signals.c to pull them in; the
 * two runtime symbols signals.c references (arts_global_rank_id,
 * arts_enter_shutdown_state) are stubbed so the unit links standalone.
 *
 * Properties under test
 * -------------------
 * write_uint(val): fills a 16-byte stack buffer back-to-front and write()s the
 *   populated tail to STDERR.  Invariants:
 *     - 0 prints "0" (the special case);
 *     - a single digit prints that digit;
 *     - UINT_MAX (4294967295) prints all 10 digits, no overflow of buf[16];
 *     - powers of ten and boundary values print exactly;
 *     - the emitted byte count is in [1,16] and equals the decimal length.
 *   We capture STDERR via a pipe and compare to a reference snprintf("%u").
 *
 * signal_name(sig): async-signal-safe switch; MUST enumerate exactly the
 *   signals the handlers install (crash {SEGV,BUS,FPE} + term {TERM,INT,ALRM,
 *   HUP}); any other -> "UNKNOWN".  We assert each installed signal maps to its
 *   literal and a sampling of un-installed signals map to "UNKNOWN".
 *
 * arts_atomic_print(fmt,...): vsnprintf into a 4096 stack buffer + single
 *   write(STDERR).  A formatted length >= 4096 must TRUNCATE to size-1 bytes in
 *   ONE write (no overflow, no second write).  We feed an oversized format and
 *   assert the captured output length is exactly 4095 and the tail is the
 *   truncated content (no NUL written, no trailing garbage).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ---- standalone stubs for symbols signals.c references ------------------- */
/* arts_global_rank_id is declared extern in identity.h (pulled in via the
 * include below); define storage here. */
unsigned int arts_global_rank_id = 0;
void arts_enter_shutdown_state(bool initiator) { (void)initiator; }

/* Pull in the real signals.c (static write_uint + signal_name). */
#include "../../libs/src/core/system/signals.c"

/* signals.c also references the thread-local arts_thread_info (declared extern
 * in runtime_state.h, now visible via signals.c's include chain). Provide
 * standalone storage so the TU links without the full runtime. */
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

/* print.h (arts_atomic_print) is reachable via signals.c's includes. */
#include "arts/system/print.h"

/* ------------------------------------------------------------------------- */

static void fail(const char *msg) {
  /* write directly: stderr may be redirected mid-test. */
  char line[256];
  int n = snprintf(line, sizeof(line), "FAIL signals_write_uint: %s\n", msg);
  (void)write(STDOUT_FILENO, line, (size_t)n);
  exit(1);
}

/* Capture whatever the callback writes to STDERR_FILENO into out (NUL-term). */
static size_t capture_stderr(void (*body)(void *), void *arg, char *out,
                             size_t out_cap) {
  int saved = dup(STDERR_FILENO);
  if (saved < 0) {
    fail("dup(stderr)");
  }
  int pfd[2];
  if (pipe(pfd) != 0) {
    fail("pipe");
  }
  /* Make read end non-blocking so we never hang if body writes nothing. */
  fflush(stderr);
  if (dup2(pfd[1], STDERR_FILENO) < 0) {
    fail("dup2");
  }
  close(pfd[1]);

  body(arg);

  fflush(stderr);
  /* Restore stderr BEFORE reading, so the write end is fully closed. */
  if (dup2(saved, STDERR_FILENO) < 0) {
    fail("dup2 restore");
  }
  close(saved);

  /* Drain the pipe. */
  size_t total = 0;
  for (;;) {
    if (total >= out_cap - 1) {
      break;
    }
    ssize_t r = read(pfd[0], out + total, out_cap - 1 - total);
    if (r > 0) {
      total += (size_t)r;
      continue;
    }
    break; /* r==0 (EOF, write end closed) or r<0 */
  }
  close(pfd[0]);
  out[total] = '\0';
  return total;
}

struct uint_arg {
  unsigned int v;
};
static void call_write_uint(void *a) { write_uint(((struct uint_arg *)a)->v); }

static void check_write_uint(unsigned int v) {
  char got[64];
  struct uint_arg a = {v};
  size_t n = capture_stderr(call_write_uint, &a, got, sizeof(got));
  char want[32];
  int wn = snprintf(want, sizeof(want), "%u", v);
  if (n != (size_t)wn || memcmp(got, want, n) != 0) {
    char msg[160];
    snprintf(msg, sizeof(msg),
             "write_uint(%u): got \"%s\" (%zu bytes) want \"%s\" (%d bytes)", v,
             got, n, want, wn);
    fail(msg);
  }
  /* buf[16] invariant: at most 10 digits for any unsigned int (32-bit). */
  if (n < 1 || n > 16) {
    fail("write_uint byte count outside [1,16]");
  }
}

struct print_arg {
  const char *fmt;
  const char *blob;
};
static void call_atomic_print(void *a) {
  struct print_arg *p = (struct print_arg *)a;
  arts_atomic_print(p->fmt, p->blob);
}

int main(void) {
  /* ---- write_uint: 0, single digits, UINT_MAX, powers of ten, boundaries --
   */
  check_write_uint(0);
  for (unsigned int d = 1; d <= 9; d++) {
    check_write_uint(d);
  }
  check_write_uint(10);
  check_write_uint(99);
  check_write_uint(100);
  check_write_uint(999);
  check_write_uint(1000);
  check_write_uint(10000);
  check_write_uint(100000);
  check_write_uint(1000000);
  check_write_uint(10000000);
  check_write_uint(100000000);
  check_write_uint(1000000000);
  check_write_uint(4294967294u);       /* UINT_MAX - 1 */
  check_write_uint(4294967295u);       /* UINT_MAX, 10 digits */
  check_write_uint((unsigned)INT_MAX); /* 2147483647 */
  check_write_uint(2147483648u);       /* INT_MAX + 1 */

  /* ---- signal_name: every installed signal -> literal; others -> UNKNOWN ---
   */
  struct {
    int sig;
    const char *name;
  } installed[] = {
      {SIGSEGV, "SIGSEGV"}, {SIGBUS, "SIGBUS"}, {SIGFPE, "SIGFPE"},
      {SIGTERM, "SIGTERM"}, {SIGINT, "SIGINT"}, {SIGALRM, "SIGALRM"},
      {SIGHUP, "SIGHUP"},
  };
  for (size_t i = 0; i < sizeof(installed) / sizeof(installed[0]); i++) {
    const char *got = signal_name(installed[i].sig);
    if (strcmp(got, installed[i].name) != 0) {
      char msg[128];
      snprintf(msg, sizeof(msg), "signal_name(%d) = \"%s\" want \"%s\"",
               installed[i].sig, got, installed[i].name);
      fail(msg);
    }
  }
  /* Signals NOT in the table must map to UNKNOWN. */
  int unknowns[] = {SIGUSR1, SIGUSR2, SIGCHLD, SIGQUIT, SIGKILL, 0, 9999};
  for (size_t i = 0; i < sizeof(unknowns) / sizeof(unknowns[0]); i++) {
    if (strcmp(signal_name(unknowns[i]), "UNKNOWN") != 0) {
      char msg[128];
      snprintf(msg, sizeof(msg), "signal_name(%d) != UNKNOWN", unknowns[i]);
      fail(msg);
    }
  }
  /* Consistency with the actual install set: the watcher's arts_term_sigs[]
   * must each have a non-UNKNOWN name (kept in sync invariant). */
  for (unsigned i = 0; i < arts_term_sig_count; i++) {
    if (strcmp(signal_name(arts_term_sigs[i]), "UNKNOWN") == 0) {
      fail("an installed term signal maps to UNKNOWN (name table drift)");
    }
  }

  /* ---- arts_atomic_print truncation: >=4096 formatted -> single 4095 write -
   */
  {
    /* Build a blob longer than the 4096 buffer so vsnprintf truncates. */
    size_t blobsz = 8000;
    char *blob = malloc(blobsz + 1);
    if (!blob) {
      fail("malloc blob");
    }
    memset(blob, 'X', blobsz);
    blob[blobsz] = '\0';

    struct print_arg pa = {"%s", blob};
    char *got = malloc(8192);
    if (!got) {
      fail("malloc got");
    }
    size_t n = capture_stderr(call_atomic_print, &pa, got, 8192);
    /* Single write of exactly size-1 == 4095 bytes (the truncated content),
     * no embedded NUL written. */
    if (n != 4095) {
      char msg[128];
      snprintf(msg, sizeof(msg),
               "arts_atomic_print truncation wrote %zu bytes, want 4095", n);
      fail(msg);
    }
    for (size_t i = 0; i < n; i++) {
      if (got[i] != 'X') {
        fail("arts_atomic_print truncated output corrupted");
      }
    }
    free(blob);
    free(got);
  }

  /* A short print must pass through unchanged (no spurious truncation). */
  {
    struct print_arg pa = {"%s", "hello-short"};
    char got[64];
    size_t n = capture_stderr(call_atomic_print, &pa, got, sizeof(got));
    if (n != strlen("hello-short") || memcmp(got, "hello-short", n) != 0) {
      fail("arts_atomic_print short string altered");
    }
  }

  printf("PASS signals_write_uint: write_uint digits + signal_name table + "
         "arts_atomic_print truncation verified\n");
  return 0;
}
