/* SPDX-License-Identifier: Apache-2.0
 *
 * pure_unit test for the SSH launcher command-string construction inside
 *   void arts_launcher_ssh_startup_processes(struct arts_launcher_s *)
 *
 * Target bug: the command builder's final_length OOB write, now FIXED.
 * -----------------------------------------------------------------------
 * The builder used to accumulate the RAW return values of snprintf (the number
 * of bytes that WOULD have been written, NOT the truncated count).  Once the
 * running sum exceeded sizeof(command)==4096, two things broke:
 *   (a) `sizeof(command) - final_length` underflowed size_t (became a huge
 *       value) on every subsequent snprintf, defeating the bound, and
 *   (b) the final `command[final_length] = '\0'` wrote at an index past the end
 *       of the 4096-byte buffer — a stack OOB write.
 * A long CWD (getcwd into cwd[1024]) and/or many/long argv tokens drive the sum
 * over 4096.
 *
 * The fix introduces a bounded-append helper `arts_cmd_appendf` that every
 * append now goes through.  It:
 *   - returns immediately (clamping *length to buf_size-1) once *length has
 *     already reached buf_size, so no further write happens past the end;
 *   - bounds each vsnprintf by `remaining = buf_size - *length` (never
 *     underflowing, since *length < buf_size here);
 *   - on truncation (`written >= remaining`) caps *length at buf_size-1.
 * With *length always <= buf_size-1, the terminating `command[final_length]`
 * is in bounds even when the composed string is truncated.
 *
 * Quoting: the builder interpolates cwd/argv raw; the WHOLE command is
 * single-quoted once by arts_shell_quote() for the outer `sh -c`, which is
 * the runtime's quoting strategy.  The builder under test is the
 * inner, pre-quote string; this test asserts the builder is overflow-safe and
 * produces the expected pre-quote bytes (the outer layer handles shell safety).
 *
 * The command builder is inlined inside a routine that also getcwd()s,
 * readlink()s /proc/self/exe, fork()s and execlp()s ssh, so it cannot be
 * called standalone.  This test reproduces the EXACT builder snippet verbatim
 * (arts_cmd_appendf + build_launch_command / build_kill_command) so it is a
 * faithful model of the runtime path; keep it line-aligned with launcher.c if
 * that file changes.
 *
 * With the fix in place the test PASSES (exit 0): the bounded-append model
 * truncates safely and command[final_length] stays in bounds, even when driven
 * with the ARTS_T192_TRIGGER_OOB block (which now writes into an exact-size
 * heap buffer and must NOT trip AddressSanitizer).
 *
 * Build knobs:
 *   -DARTS_T192_TRIGGER_OOB  : additionally perform the real terminating
 *                              `command[final_length]='\0'` into an exact-size
 *                              heap buffer.  With the fix, final_length is
 *                              clamped to COMMAND_SIZE-1, so this write is in
 *                              bounds and ASan stays silent.  (Pre-fix this is
 *                              where the heap-buffer-overflow fired.)
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COMMAND_SIZE 4096 /* matches launcher.c `char command[4096];` */

/* ------------------------------------------------------------------ *
 * Verbatim mirror of arts_cmd_appendf: bounded printf-append that     *
 * clamps *length so the buffer stays null-terminatable at             *
 * buf[*length].                                                       *
 * ------------------------------------------------------------------ */
static void arts_cmd_appendf(char *buf, size_t buf_size, size_t *length,
                             const char *fmt, ...) {
  if (buf_size == 0 || *length >= buf_size) {
    if (buf_size != 0) {
      *length = buf_size - 1;
    }
    return;
  }

  size_t remaining = buf_size - *length;
  va_list args;
  va_start(args, fmt);
  int written = vsnprintf(buf + *length, remaining, fmt, args);
  va_end(args);

  if (written < 0) {
    return;
  }
  if ((size_t)written >= remaining) {
    /* Truncated: the buffer is now full up to its last writable byte. */
    *length = buf_size - 1;
  } else {
    *length += (size_t)written;
  }
}

/* ------------------------------------------------------------------ *
 * Verbatim mirror of the launch-mode command build (self_exe present), *
 * now routed through arts_cmd_appendf.                                 *
 * Writes into `command` (caller-owned, COMMAND_SIZE bytes) and returns *
 * the accumulated final_length — clamped to COMMAND_SIZE-1, so the    *
 * runtime's `command[final_length]='\0'` is always in bounds.         *
 * No env passthrough here (ARTS_CONFIG/LD_LIBRARY_PATH optional in the *
 * runtime); ARTS_RANK + self_exe + argv mirror is exact.              *
 * ------------------------------------------------------------------ */
static size_t build_launch_command(char *command, const char *cwd,
                                   const char *self_exe, int rank,
                                   unsigned int argc, char **argv) {
  size_t final_length = 0;

  arts_cmd_appendf(command, COMMAND_SIZE, &final_length, "cd %s && ", cwd);
  arts_cmd_appendf(command, COMMAND_SIZE, &final_length, "ARTS_RANK=%d ", rank);
  arts_cmd_appendf(command, COMMAND_SIZE, &final_length, "%s", self_exe);
  for (unsigned int j = 1; j < argc; j++) {
    arts_cmd_appendf(command, COMMAND_SIZE, &final_length, " %s", argv[j]);
  }
  return final_length;
}

/* Verbatim mirror of the kill_mode basename branch, routed through
 * arts_cmd_appendf. */
static size_t build_kill_command(char *command, const char *binary_name) {
  size_t final_length = 0;
  arts_cmd_appendf(command, COMMAND_SIZE, &final_length, "pkill %s",
                   binary_name);
  return final_length;
}

static int g_fail = 0;
#define REPORT(...)                                                            \
  do {                                                                         \
    fprintf(stderr, "FAIL launcher_ssh_command_build: " __VA_ARGS__);          \
    fprintf(stderr, "\n");                                                     \
    g_fail = 1;                                                                \
  } while (0)

int main(void) {
  /* ============================================================ *
   * Overflow safety of the bounded-append builder.                *
   * Construct an argv whose untruncated total would far exceed   *
   * COMMAND_SIZE.  The fixed builder must clamp final_length to  *
   * COMMAND_SIZE-1 so the terminating write is in bounds, and    *
   * must keep the produced string null-terminated within the     *
   * buffer.                                                      *
   * ============================================================ */
  {
    /* A realistic-ish but pathological CWD + argv. getcwd() can return up to
     * 1023 chars; a few long argv tokens easily exceed 4096 total. */
    char cwd[1024];
    memset(cwd, 'd', sizeof(cwd) - 1);
    cwd[sizeof(cwd) - 1] = '\0';

    char self_exe[256];
    memset(self_exe, 'e', sizeof(self_exe) - 1);
    self_exe[sizeof(self_exe) - 1] = '\0';

    /* Build several long argv tokens. */
    enum { NTOK = 8, TOKLEN = 600 };
    static char tokstore[NTOK][TOKLEN];
    char *argv[NTOK + 1];
    argv[0] = self_exe;
    for (int t = 1; t <= NTOK; t++) {
      memset(tokstore[t - 1], 'a', TOKLEN - 1);
      tokstore[t - 1][TOKLEN - 1] = '\0';
      argv[t] = tokstore[t - 1];
    }

    /* Build into an EXACT-size buffer (as the runtime does, char[4096]) so
     * any out-of-bounds write would be caught by ASan.  With the fix the
     * builder clamps final_length, so this stays in bounds. */
    char command[COMMAND_SIZE];
    size_t fl = build_launch_command(command, cwd, self_exe, 1, NTOK + 1, argv);

    printf("INFO: accumulated final_length=%zu, COMMAND_SIZE=%d\n", fl,
           COMMAND_SIZE);

    /* Property 1: final_length is clamped within the buffer. */
    if (fl >= (size_t)COMMAND_SIZE) {
      REPORT("final_length=%zu not clamped below COMMAND_SIZE=%d — the "
             "bounded-append discipline failed, `command[final_length]='\\0'` "
             "would write OUT OF BOUNDS.",
             fl, COMMAND_SIZE);
    }

    /* Property 2: the terminating write is in bounds (the runtime line). */
    command[fl] = '\0'; /* must be in [0, COMMAND_SIZE) */

    /* Property 3: the result is a valid null-terminated string no longer than
     * the buffer. */
    if (strlen(command) >= (size_t)COMMAND_SIZE) {
      REPORT("produced string is not bounded by COMMAND_SIZE");
    }

#ifdef ARTS_T192_TRIGGER_OOB
    /* Reproduce the runtime exactly with a heap buffer sized to COMMAND_SIZE.
     * With the fix, real_fl is clamped to COMMAND_SIZE-1, so the terminating
     * write is in bounds and ASan stays silent.  (Pre-fix this is the line
     * that produced heap-buffer-overflow.) */
    {
      char *heap_cmd = (char *)malloc(COMMAND_SIZE);
      size_t real_fl =
          build_launch_command(heap_cmd, cwd, self_exe, 1, NTOK + 1, argv);
      if (real_fl >= (size_t)COMMAND_SIZE) {
        REPORT("heap-buffer terminating index real_fl=%zu out of bounds",
               real_fl);
      } else {
        heap_cmd[real_fl] = '\0'; /* in bounds with the fix */
      }
      free(heap_cmd);
    }
#endif
  }

  /* ============================================================ *
   * Quoting boundary.  The builder interpolates cwd raw           *
   * into `cd %s && `; the OUTER layer (arts_shell_quote) single-  *
   * quotes the whole command for `sh -c`.  Verify the builder     *
   * produces the expected pre-quote prefix exactly and remains    *
   * overflow-safe with a space-bearing cwd.                       *
   * ============================================================ */
  {
    char command[COMMAND_SIZE];
    const char *cwd = "/home/user/my project"; /* contains a space */
    const char *self_exe = "/home/user/my project/bin/app";
    char *argv[1] = {(char *)self_exe};
    size_t fl = build_launch_command(command, cwd, self_exe, 2, 1, argv);
    if (fl >= (size_t)COMMAND_SIZE) {
      REPORT("builder overflowed on space-bearing cwd");
    }
    command[fl] = '\0';

    /* The inner (pre-quote) command begins with the literal `cd <cwd> && `.
     * Shell safety of the space is provided by the outer arts_shell_quote, not
     * by the builder; here we only verify the builder emitted the exact
     * pre-quote bytes the outer quoter then wraps. */
    const char *prefix = "cd /home/user/my project && ";
    if (strncmp(command, prefix, strlen(prefix)) != 0) {
      REPORT("builder did not emit the expected pre-quote prefix `%s`; "
             "got `%.*s`",
             prefix, (int)strlen(prefix), command);
    }
  }

  /* ============================================================ *
   * kill_mode pkill command is overflow-safe + correctly          *
   * formed for a long basename.  The basename branch emits        *
   * `pkill <basename>`; verify it is bounded and well-formed.     *
   * ============================================================ */
  {
    char command[COMMAND_SIZE];
    const char *long_name = "arts_really_long_binary_name_exe"; /* 32 chars */
    size_t fl = build_kill_command(command, long_name);
    if (fl >= (size_t)COMMAND_SIZE) {
      REPORT("kill command overflowed");
    }
    command[fl] = '\0';

    const char *expect = "pkill arts_really_long_binary_name_exe";
    if (strcmp(command, expect) != 0) {
      REPORT("kill command malformed; got `%s` expected `%s`", command,
             expect);
    }
  }

  if (g_fail) {
    fprintf(stderr,
            "FAIL launcher_ssh_command_build (bounded-append discipline did "
            "not hold — see messages above)\n");
    return 1;
  }
  printf("PASS launcher_ssh_command_build (overflow-safe; builder "
         "produces bounded, well-formed pre-quote commands)\n");
  return 0;
}
