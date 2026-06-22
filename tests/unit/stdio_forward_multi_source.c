/* SPDX-License-Identifier: Apache-2.0
 *
 * T202 — concurrent multi-reader to ONE shared sink fd: per-source byte count
 * preserved (no byte loss), WITHOUT assuming line/chunk order.
 *
 * The launcher points many forwarders' sinks at the same master FILE* (e.g.
 * stdout): several reader threads write(2) to the same underlying fd
 * concurrently.  write(2) to a shared fd is only atomic up to a bound, so bytes
 * from different sources can interleave at sub-write granularity — line order
 * is NOT a promise (suspected-bug #3).  The contract that MUST hold is: no byte
 * is lost; every source's bytes all arrive.
 *
 * Each of N forwarders writes COUNT bytes all equal to its own distinct byte
 * value s (s = 1..N; value 0 is reserved as "absent").  After shutdown the
 * single shared sink is read back and bytes are tallied BY VALUE — order
 * irrelevant.  Each value must appear exactly COUNT times and no other value
 * may appear.  A loss in the copy loop shows up as a short count for that
 * source.
 *
 * #include's stdio_forward.c for cluster symmetry (public API used here).
 */
#define ARTS_SYSTEM_PRINT_H
#define ARTS_WARN(...) ((void)0)
#include "../../libs/src/core/transport/stdio_forward.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { N = 6, COUNT = 50000 }; /* > 4096 per source -> multiple read chunks */

int main(void) {
  /* One shared sink for all N forwarders. */
  FILE *sink = tmpfile();
  if (!sink) {
    (void)fprintf(stderr, "FAIL multi_source: tmpfile\n");
    return 1;
  }

  int wfd[N];
  for (int s = 0; s < N; s++) {
    wfd[s] = arts_stdio_forwarder_make_pipe((unsigned int)s, "stdout", sink);
    if (wfd[s] < 0) {
      (void)fprintf(stderr, "FAIL multi_source: make_pipe %d -1\n", s);
      return 1;
    }
  }

  /* Feed each source COUNT bytes of its distinct value (s+1). */
  unsigned char *chunk = (unsigned char *)malloc(COUNT);
  if (!chunk) {
    (void)fprintf(stderr, "FAIL multi_source: malloc\n");
    return 1;
  }
  for (int s = 0; s < N; s++) {
    unsigned char val = (unsigned char)(s + 1);
    memset(chunk, val, COUNT);
    size_t off = 0;
    while (off < COUNT) {
      ssize_t w = write(wfd[s], chunk + off, COUNT - off);
      if (w < 0) {
        (void)fprintf(stderr, "FAIL multi_source: feed write src %d\n", s);
        return 1;
      }
      off += (size_t)w;
    }
  }
  free(chunk);

  /* Close all write-ends -> EOF -> readers drain and exit. */
  for (int s = 0; s < N; s++) {
    close(wfd[s]);
  }
  arts_stdio_forwarder_shutdown_all();

  /* Tally the shared sink by byte value. */
  if (fflush(sink) != 0 || fseek(sink, 0, SEEK_SET) != 0) {
    (void)fprintf(stderr, "FAIL multi_source: flush/seek\n");
    return 1;
  }
  size_t tally[256] = {0};
  unsigned char rbuf[8192];
  size_t r;
  while ((r = fread(rbuf, 1, sizeof(rbuf), sink)) > 0) {
    for (size_t i = 0; i < r; i++) {
      tally[rbuf[i]]++;
    }
  }

  int ok = 1;
  for (int s = 0; s < N; s++) {
    unsigned char val = (unsigned char)(s + 1);
    if (tally[val] != COUNT) {
      (void)fprintf(stderr,
                    "FAIL multi_source: source %d (byte 0x%02x) count=%zu, "
                    "expected %d (byte loss under shared-fd interleave)\n",
                    s, val, tally[val], COUNT);
      ok = 0;
    }
  }
  /* No foreign byte values should appear. */
  for (int v = 0; v < 256; v++) {
    if (v >= 1 && v <= N) {
      continue;
    }
    if (tally[v] != 0) {
      (void)fprintf(stderr,
                    "FAIL multi_source: unexpected byte 0x%02x x%zu in sink\n",
                    v, tally[v]);
      ok = 0;
    }
  }
  if (!ok) {
    return 1;
  }

  fclose(sink);
  printf("PASS stdio_forward_multi_source: %d sources x %d bytes each, "
         "per-source count preserved (order-agnostic)\n",
         N, COUNT);
  return 0;
}
