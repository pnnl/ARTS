/* SPDX-License-Identifier: Apache-2.0
 *
 * T194 — stdio_forward forwarder_thread_main copy-loop byte fidelity.
 *
 * Property under test: the reader thread spawned by make_pipe copies bytes
 * VERBATIM from its pipe read-end to the sink fd, with no loss and no
 * truncation, across:
 *   - a payload strictly larger than the internal buf[4096] (forces multiple
 *     read(2) iterations and crosses the 4096-chunk boundary repeatedly), and
 *   - the data-then-EOF ordering: bytes written, then the write-end closed.
 *     Pipe semantics guarantee a pending read returns the buffered tail before
 *     the next read returns 0 (EOF); a regression that broke on the first short
 *     read would silently drop the last chunk (suspected-bug #1, tail-flush).
 *
 * Harness: redirect the sink to a temp file (the FILE* given to make_pipe is
 * backed by a real fd, so the forwarder's fileno(sink) write lands there).
 * Write a deterministic multi-KB pattern into the returned write-end, close it,
 * shutdown_all (joins the reader), then read the temp file back and assert
 * byte-exact equality including the tail.
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

/* > 4096 and not a multiple of 4096 so the final read returns a short chunk. */
enum { PAYLOAD = 100003 };

static unsigned char pattern_byte(size_t i) {
  /* Deterministic, non-trivial pattern: every byte value cycles, so a dropped
   * or reordered chunk changes the compared bytes. */
  return (unsigned char)((i * 31u + 7u) & 0xFFu);
}

int main(void) {
  FILE *sink = tmpfile();
  if (!sink) {
    (void)fprintf(stderr, "FAIL fidelity: tmpfile() failed\n");
    return 1;
  }

  int wfd = arts_stdio_forwarder_make_pipe(0, "stdout", sink);
  if (wfd < 0) {
    (void)fprintf(stderr, "FAIL fidelity: make_pipe returned -1\n");
    return 1;
  }

  unsigned char *src = (unsigned char *)malloc(PAYLOAD);
  if (!src) {
    (void)fprintf(stderr, "FAIL fidelity: malloc\n");
    return 1;
  }
  for (size_t i = 0; i < PAYLOAD; i++) {
    src[i] = pattern_byte(i);
  }

  /* Write the whole payload to the write-end, looping on partial writes. */
  size_t off = 0;
  while (off < PAYLOAD) {
    ssize_t w = write(wfd, src + off, PAYLOAD - off);
    if (w < 0) {
      (void)fprintf(stderr, "FAIL fidelity: write to pipe failed\n");
      return 1;
    }
    off += (size_t)w;
  }

  /* data-then-EOF: close the write-end AFTER all bytes are in the pipe. */
  close(wfd);

  /* Join the reader (caller-guarantees-EOF contract satisfied above). */
  arts_stdio_forwarder_shutdown_all();

  /* Read back what the forwarder wrote into the sink. */
  if (fflush(sink) != 0) {
    (void)fprintf(stderr, "FAIL fidelity: fflush sink\n");
    return 1;
  }
  if (fseek(sink, 0, SEEK_SET) != 0) {
    (void)fprintf(stderr, "FAIL fidelity: fseek sink\n");
    return 1;
  }

  unsigned char *got = (unsigned char *)malloc(PAYLOAD + 16);
  if (!got) {
    (void)fprintf(stderr, "FAIL fidelity: malloc got\n");
    return 1;
  }
  size_t total = fread(got, 1, PAYLOAD + 16, sink);
  if (total != PAYLOAD) {
    (void)fprintf(stderr,
                  "FAIL fidelity: sink has %zu bytes, expected %d "
                  "(tail-flush / truncation regression)\n",
                  total, PAYLOAD);
    return 1;
  }
  for (size_t i = 0; i < PAYLOAD; i++) {
    if (got[i] != src[i]) {
      (void)fprintf(
          stderr, "FAIL fidelity: byte %zu mismatch: got 0x%02x want 0x%02x\n",
          i, got[i], src[i]);
      return 1;
    }
  }

  free(src);
  free(got);
  fclose(sink);
  printf("PASS stdio_forward_fidelity: %d bytes byte-exact incl tail\n",
         PAYLOAD);
  return 0;
}
