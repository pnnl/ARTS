/* SPDX-License-Identifier: Apache-2.0
 *
 * T200 — stdio_forward stderr-sink fidelity (symmetric with T194).
 *
 * The launcher forwards BOTH stdout and stderr for every non-master rank, so
 * the copy loop must be sink-agnostic.  This pins the stderr path: a multi-KB
 * deterministic payload written into the write-end, data-then-EOF close,
 * shutdown_all, then byte-exact compare of the sink contents.  The sink is a
 * temp FILE* (a real backing fd, as stderr would be) so fileno(sink) yields a
 * valid fd to write to.
 *
 * #include's stdio_forward.c for the static reader-thread body / globals.
 */
#define ARTS_SYSTEM_PRINT_H
#define ARTS_WARN(...) ((void)0)
#include "../../libs/src/core/transport/stdio_forward.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { PAYLOAD = 70001 };

static unsigned char pattern_byte(size_t i) {
  return (unsigned char)((i * 17u + 3u) & 0xFFu);
}

int main(void) {
  FILE *sink = tmpfile();
  if (!sink) {
    (void)fprintf(stderr, "FAIL stderr: tmpfile() failed\n");
    return 1;
  }

  int wfd = arts_stdio_forwarder_make_pipe(1, "stderr", sink);
  if (wfd < 0) {
    (void)fprintf(stderr, "FAIL stderr: make_pipe returned -1\n");
    return 1;
  }

  unsigned char *src = (unsigned char *)malloc(PAYLOAD);
  if (!src) {
    (void)fprintf(stderr, "FAIL stderr: malloc\n");
    return 1;
  }
  for (size_t i = 0; i < PAYLOAD; i++) {
    src[i] = pattern_byte(i);
  }

  size_t off = 0;
  while (off < PAYLOAD) {
    ssize_t w = write(wfd, src + off, PAYLOAD - off);
    if (w < 0) {
      (void)fprintf(stderr, "FAIL stderr: write to pipe failed\n");
      return 1;
    }
    off += (size_t)w;
  }
  close(wfd);

  arts_stdio_forwarder_shutdown_all();

  if (fflush(sink) != 0 || fseek(sink, 0, SEEK_SET) != 0) {
    (void)fprintf(stderr, "FAIL stderr: flush/seek sink\n");
    return 1;
  }
  unsigned char *got = (unsigned char *)malloc(PAYLOAD + 16);
  if (!got) {
    (void)fprintf(stderr, "FAIL stderr: malloc got\n");
    return 1;
  }
  size_t total = fread(got, 1, PAYLOAD + 16, sink);
  if (total != PAYLOAD) {
    (void)fprintf(stderr, "FAIL stderr: sink has %zu bytes, expected %d\n",
                  total, PAYLOAD);
    return 1;
  }
  if (memcmp(got, src, PAYLOAD) != 0) {
    (void)fprintf(stderr, "FAIL stderr: payload mismatch\n");
    return 1;
  }

  free(src);
  free(got);
  fclose(sink);
  printf("PASS stdio_forward_stderr: %d bytes byte-exact via stderr sink\n",
         PAYLOAD);
  return 0;
}
