/* SPDX-License-Identifier: Apache-2.0
 *
 * T198 — shutdown_all idempotency.
 *
 * shutdown_all detaches the whole Treiber stack with one atomic exchange to
 * NULL, then joins+frees.  A second call exchanges out an already-NULL head and
 * the drain loop runs zero times — it must be a clean no-op: no crash, no
 * double-join of an already-joined/freed thread, no double-free.
 *
 * Build a handful of real forwarders, write a little data, close the
 * write-ends (so readers hit EOF and exit — the documented precondition), then
 * call shutdown_all() TWICE.  ASan/LSan would catch a double-free or
 * use-after-free on the second pass; a double pthread_join of a freed handle
 * would crash.
 *
 * #include's stdio_forward.c only for symmetry with the cluster (uses the
 * public API; no static needed here).
 */
#define ARTS_SYSTEM_PRINT_H
#define ARTS_WARN(...) ((void)0)
#include "../../libs/src/core/transport/stdio_forward.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { N = 5 };

int main(void) {
  FILE *sink = tmpfile();
  if (!sink) {
    (void)fprintf(stderr, "FAIL idempotent: tmpfile\n");
    return 1;
  }

  int wfd[N];
  for (int i = 0; i < N; i++) {
    wfd[i] = arts_stdio_forwarder_make_pipe((unsigned int)i, "stdout", sink);
    if (wfd[i] < 0) {
      (void)fprintf(stderr, "FAIL idempotent: make_pipe %d -1\n", i);
      return 1;
    }
    const char *msg = "hello\n";
    (void)write(wfd[i], msg, strlen(msg));
  }

  /* Close every write-end -> readers see EOF and leave their loops. */
  for (int i = 0; i < N; i++) {
    close(wfd[i]);
  }

  /* First shutdown: joins + frees all N nodes, leaves head == NULL. */
  arts_stdio_forwarder_shutdown_all();

  /* Second shutdown: must be a clean no-op (exchange NULL, zero iterations). */
  arts_stdio_forwarder_shutdown_all();

  /* Third for good measure. */
  arts_stdio_forwarder_shutdown_all();

  if (__atomic_load_n(&g_forwarders, __ATOMIC_ACQUIRE) != NULL) {
    (void)fprintf(stderr, "FAIL idempotent: g_forwarders not NULL after "
                          "shutdown\n");
    return 1;
  }

  fclose(sink);
  printf("PASS stdio_forward_shutdown_idempotent: %d nodes, 3x shutdown_all "
         "no crash/double-free\n",
         N);
  return 0;
}
