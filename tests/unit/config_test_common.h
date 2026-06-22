/* SPDX-License-Identifier: Apache-2.0
 *
 * Common stubs for the config.c pure-unit tests (C21 cluster).
 *
 * The config parser's static helpers (parse_port_spec, handle_launcher,
 * handle_cxl_db_allocation_strategy, config_compute_derived, ...) have no
 * header prototype, so each test TU pulls the whole translation unit in with
 *     #include "../../libs/src/core/system/config.c"
 * (precedent: tests build edt_gpu.cu by #include'ing edt.c).  That drags in the
 * runtime's allocator / identity / abort / launcher-backend symbols, none of
 * which we want the real runtime for.  This header supplies libc-backed shims
 * for them so the test links standalone.
 *
 * IMPORTANT: include this header AFTER the config.c include, because config.c
 * brings in print.h (which declares arts_global_rank_id / arts_thread_info used
 * by the log macros) and runtime_state.h (which defines
 * struct arts_runtime_private_s).
 *
 * arts_abort() is stubbed with longjmp-or-_exit so a test can probe the
 * ARTS_ERROR (== arts_abort) death paths without killing the harness.
 */
#ifndef CONFIG_TEST_COMMON_H
#define CONFIG_TEST_COMMON_H

#include <setjmp.h>
#include <stdlib.h>
#include <unistd.h>

/* ── libc-backed allocator shims ─────────────────────────────────────────── */
void *arts_malloc(size_t s) { return malloc(s); }
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void *arts_realloc(void *p, size_t s) { return realloc(p, s); }
void arts_free(void *p) { free(p); }

/* ── identity + thread-local referenced by print.h log macros ────────────── */
unsigned int arts_global_rank_id = 0;
unsigned int arts_global_rank_count = 1;
unsigned int arts_global_master_rank_id = 0;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

/* ── abort stub: jump back to a test-installed landing pad if armed,
 *    otherwise _exit so death-tests have a deterministic code. ───────────── */
static jmp_buf g_abort_jmp;
static volatile int g_abort_armed = 0;
static volatile int g_abort_fired = 0;

void arts_abort(uint8_t code) {
  g_abort_fired = 1;
  if (g_abort_armed) {
    g_abort_armed = 0;
    longjmp(g_abort_jmp, (int)code + 1);
  }
  _exit(70);
}

/* ── launcher backend symbols (address-taken only by config_setup_*) ─────── */
void arts_launcher_ssh_startup_processes(struct arts_launcher_s *l) { (void)l; }
void arts_launcher_ssh_cleanup_processes(struct arts_launcher_s *l) { (void)l; }
void arts_launcher_local_startup_processes(struct arts_launcher_s *l) {
  (void)l;
}
void arts_launcher_local_cleanup_processes(struct arts_launcher_s *l) {
  (void)l;
}

#endif /* CONFIG_TEST_COMMON_H */
