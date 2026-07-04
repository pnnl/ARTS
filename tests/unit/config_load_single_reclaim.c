/* SPDX-License-Identifier: Apache-2.0
 *
 * T211 — single-node thread reclaim (config_compute_derived).
 *
 * Property: when a `launcher=local` config resolves to ONE node, the derived
 * thread counts must fold progress_threads back into the worker pool (a
 * single node has no fabric to progress).  Per config_compute_derived's
 * `table_length <= 1` branch:
 *
 *     worker += progress;  progress = 0;
 *     thread_count = worker + progress;
 *
 * With a cfg of worker=2 / progress=4 on 1 node we therefore expect
 *     worker_thread_count   == 6
 *     progress_thread_count == 0
 *     thread_count          == 6
 *
 * This is a config-parser test: it calls arts_config_load() directly against a
 * crafted temp cfg (via ARTS_CONFIG) and inspects the resolved struct.  It does
 * NOT start the runtime (no arts_rt), so it binds no ports and spawns no
 * threads.  It overrides ARTS_CONFIG (and clears the harness `default_ports`
 * override) so it is self-contained regardless of the registered config.
 *
 * Orthogonal to the coherence protocol axis (config.c is protocol-agnostic).
 */

#include "arts.h"
#include "arts/system/config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int write_cfg(char *path_out, size_t path_cap) {
  /* Unique temp path in the test working dir to avoid /tmp tmpfs pressure. */
  snprintf(path_out, path_cap, "config_single_reclaim_%ld.cfg", (long)getpid());
  FILE *f = fopen(path_out, "w");
  if (!f) {
    return -1;
  }
  fputs("[ARTS]\n"
        "launcher=local\n"
        "node_count=1\n"
        "worker_threads=2\n"
        "progress_threads=4\n"
        "route_table_size=14\n",
        f);
  (void)fclose(f);
  return 0;
}

int main(void) {
  char cfg_path[256];
  if (write_cfg(cfg_path, sizeof(cfg_path)) != 0) {
    printf("FAIL config_load_single_reclaim: cannot write temp cfg\n");
    return 1;
  }

  /* Self-contained: point the loader at our cfg and drop any inherited
     per-test `default_ports` override (it is irrelevant single-node). */
  setenv("ARTS_CONFIG", cfg_path, 1);
  unsetenv("default_ports");
  unsetenv("port_count");

  struct arts_config_s config;
  arts_config_load(&config);

  int ok = 1;
  if (config.table_length > 1) {
    printf("FAIL config_load_single_reclaim: table_length=%u expected 1\n",
           config.table_length);
    ok = 0;
  }
  if (config.worker_thread_count != 6) {
    printf("FAIL config_load_single_reclaim: worker=%u expected 6\n",
           config.worker_thread_count);
    ok = 0;
  }
  if (config.progress_thread_count != 0) {
    printf("FAIL config_load_single_reclaim: progress=%u expected 0\n",
           config.progress_thread_count);
    ok = 0;
  }
  if (config.thread_count != 6) {
    printf("FAIL config_load_single_reclaim: thread_count=%u expected 6\n",
           config.thread_count);
    ok = 0;
  }

  arts_config_destroy(&config);
  (void)remove(cfg_path);

  if (ok) {
    printf("PASS config_load_single_reclaim worker=6 progress=0 "
           "thread_count=6\n");
    return 0;
  }
  return 1;
}
