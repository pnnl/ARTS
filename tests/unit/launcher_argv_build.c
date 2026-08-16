/* SPDX-License-Identifier: Apache-2.0
 *
 * pure_unit test for the child arg-vector construction inside
 *   void arts_launcher_local_startup_processes(struct arts_launcher_s *)
 *
 * The builder is NOT a standalone function — it is inlined in a routine that
 * also readlink()s /proc/self/exe, fork()s and execv()s, so it cannot be
 * invoked in isolation without spawning real processes.  This test therefore
 * verifies the *exact* arg-vector construction algorithm, with the launcher's
 * source lines reproduced verbatim in build_child_argv() below so any future
 * divergence in the algorithm is caught by re-reading both side by side.
 *
 * The launcher builds the child's argv as  [self_exe, argv[1..argc-1], NULL].
 * Contract / invariants checked:
 *
 *   - new_argc = argc > 0 ? argc : 1.  The allocation is (new_argc+1) slots.
 *   - new_argv[0] is ALWAYS self_exe (the resolved executable path), even when
 *     argc==0 (the degenerate case: a caller that handed in no argv at all).
 *   - new_argv[new_argc] == NULL terminator.
 *   - The argc==0 degenerate case yields new_argv = {self_exe, NULL} — i.e.
 *     new_argv[1] == NULL, exactly one usable element, no read of the
 *     (absent) caller argv.  This is the row's named focus.
 *   - For argc>=1, argv[0] is REPLACED by self_exe and argv[1..] are copied
 *     by pointer (aliasing, not duplicating) — the child execv()s self_exe but
 *     keeps the caller's later arguments.
 *   - No NULL pointer appears before slot new_argc (a NULL mid-vector would
 *     truncate execv's argument list).
 *
 * No runtime is started.  Prints "PASS launcher_argv_build ..." on success.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Verbatim mirror of arts_launcher_local_startup_processes's arg-vector build, parameterized so
 * we can drive it with arbitrary (argc, argv, self_exe).  Uses libc malloc in
 * place of arts_malloc (identical semantics for this test; arts_malloc aborts
 * on OOM, irrelevant here).  Returns the built vector AND the new_argc via
 * out-param so the caller can assert the terminator slot index. */
static char **build_child_argv(unsigned int argc, char **argv, char *self_exe,
                               unsigned int *out_new_argc) {
  unsigned int new_argc = argc > 0 ? argc : 1;
  char **new_argv = (char **)malloc((new_argc + 1) * sizeof(char *));
  new_argv[0] = self_exe;
  for (unsigned int j = 1; j < argc; j++) {
    new_argv[j] = argv[j];
  }
  new_argv[new_argc] = NULL;
  *out_new_argc = new_argc;
  return new_argv;
}

static int g_fail = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL launcher_argv_build: " __VA_ARGS__);               \
      fprintf(stderr, "\n  (at %s:%d)\n", __FILE__, __LINE__);                 \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

int main(void) {
  char self_exe[] = "/abs/path/to/exe";

  /* --- degenerate argc==0: must yield {self_exe, NULL} --- */
  {
    unsigned int na = 999;
    char **v = build_child_argv(0, NULL, self_exe, &na);
    CHECK(na == 1, "argc==0: new_argc must be 1, got %u", na);
    CHECK(v[0] == self_exe, "argc==0: new_argv[0] must be self_exe");
    CHECK(v[1] == NULL, "argc==0: new_argv[1] must be NULL terminator");
    /* terminator sits exactly at new_argc */
    CHECK(v[na] == NULL, "argc==0: new_argv[new_argc] must be NULL");
    free(v);
  }

  /* --- argc==1: only argv[0]; child gets {self_exe, NULL} --- */
  {
    char *a0 = "ignored_caller_argv0";
    char *argv[] = {a0};
    unsigned int na = 0;
    char **v = build_child_argv(1, argv, self_exe, &na);
    CHECK(na == 1, "argc==1: new_argc must be 1, got %u", na);
    CHECK(v[0] == self_exe, "argc==1: slot0 must be self_exe (argv0 replaced)");
    CHECK(v[0] != a0, "argc==1: caller argv[0] must NOT survive");
    CHECK(v[1] == NULL, "argc==1: terminator");
    free(v);
  }

  /* --- argc>=2: argv[0] replaced, rest aliased through --- */
  {
    char *a0 = "self_argv0";
    char *a1 = "--config";
    char *a2 = "foo.cfg";
    char *a3 = "extra";
    char *argv[] = {a0, a1, a2, a3};
    unsigned int na = 0;
    char **v = build_child_argv(4, argv, self_exe, &na);
    CHECK(na == 4, "argc==4: new_argc must be 4, got %u", na);
    CHECK(v[0] == self_exe, "argc==4: slot0 must be self_exe");
    CHECK(v[1] == a1, "argc==4: argv[1] aliased through (pointer identity)");
    CHECK(v[2] == a2, "argc==4: argv[2] aliased through");
    CHECK(v[3] == a3, "argc==4: argv[3] aliased through");
    CHECK(v[4] == NULL, "argc==4: terminator at slot new_argc");
    /* no NULL appears before the terminator */
    for (unsigned int j = 0; j < na; j++) {
      CHECK(v[j] != NULL, "argc==4: unexpected NULL mid-vector at slot %u", j);
    }
    free(v);
  }

  /* --- sweep argc 1..16: terminator always at new_argc, no premature NULL,
   * slot0 always self_exe, content preserved --- */
  {
    enum { MAX = 16 };
    char *argv[MAX];
    static char names[MAX][8];
    for (int i = 0; i < MAX; i++) {
      snprintf(names[i], sizeof(names[i]), "a%d", i);
      argv[i] = names[i];
    }
    for (unsigned int argc = 1; argc <= MAX; argc++) {
      unsigned int na = 0;
      char **v = build_child_argv(argc, argv, self_exe, &na);
      CHECK(na == argc, "sweep argc=%u: new_argc mismatch %u", argc, na);
      CHECK(v[0] == self_exe, "sweep argc=%u: slot0 not self_exe", argc);
      CHECK(v[na] == NULL, "sweep argc=%u: terminator slot not NULL", argc);
      for (unsigned int j = 1; j < argc; j++) {
        CHECK(v[j] == argv[j], "sweep argc=%u: slot %u not aliased", argc, j);
        CHECK(v[j] != NULL, "sweep argc=%u: premature NULL at %u", argc, j);
      }
      free(v);
    }
  }

  if (g_fail) {
    fprintf(stderr, "FAIL launcher_argv_build\n");
    return 1;
  }
  printf("PASS launcher_argv_build (argc==0 degenerate -> {self_exe,NULL}, "
         "argv0 replace, alias-through, terminator at new_argc)\n");
  return 0;
}
