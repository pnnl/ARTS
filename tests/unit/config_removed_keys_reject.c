/* SPDX-License-Identifier: Apache-2.0
 *
 * T298 — removed cfg-key hard errors (config_reject_removed_keys, config.c).
 *
 * Property under test
 * -------------------
 * The config-surface cleanup deleted the `sender_threads` key outright (the
 * transport injects directly from workers now; there is no sender role left
 * to configure) and renamed `receiver_threads` to `progress_threads`.  Both
 * old keys must now be a HARD ERROR — arts_config_load() calls ARTS_ERROR
 * (== arts_abort(1) == exit) as soon as either key is present in the parsed
 * variable list, before any of their now-nonexistent defaults could silently
 * apply.  This is deliberately NOT a warning-and-fold like the sender fold
 * used to be: a stale cfg must fail loudly, not run with an unintended
 * thread count.
 *
 * ARTS_ERROR aborts the process, so each rejection case is driven in a forked
 * child (same discipline as config_port_count_agreement.c) and the parent
 * asserts the child died with a nonzero status.  A third case loads a cfg
 * using ONLY the new `progress_threads` key and asserts it loads normally (in
 * the parent process, since nothing should abort) — proving the hard error is
 * specific to the old key names, not an overzealous rejection of any
 * networking key.
 *
 * This is a config-parser test: it calls arts_config_load() directly against
 * crafted temp cfgs (no runtime started, no ports bound).  Self-contained: it
 * sets its own ARTS_CONFIG and clears the inherited `ports`/
 * `port_count` env overrides.  Orthogonal to the coherence protocol axis.
 */

#include "arts.h"
#include "arts/system/config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int write_cfg(char *path_out, size_t path_cap, const char *tag,
                     const char *extra_line) {
  snprintf(path_out, path_cap, "config_removed_keys_reject_%s_%ld.cfg", tag,
           (long)getpid());
  FILE *f = fopen(path_out, "w");
  if (!f) {
    return -1;
  }
  fputs("[ARTS]\n"
        "launcher=local\n"
        "node_count=2\n"
        "worker_threads=2\n",
        f);
  fputs(extra_line, f);
  fputs("route_table_size=14\n", f);
  (void)fclose(f);
  return 0;
}

/* Fork, load the given cfg in the child, and report whether it was rejected
 * (nonzero exit / signal) or accepted (clean exit 0). */
static int child_load_rejected(const char *cfg_path) {
  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return -1;
  }
  if (pid == 0) {
    setenv("ARTS_CONFIG", cfg_path, 1);
    unsetenv("ports");
    unsetenv("port_count");
    struct arts_config_s config;
    arts_config_load(&config);
    /* Reached only if the removed-key check did NOT fire. */
    _exit(0);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    perror("waitpid");
    return -1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status) != 0;
  }
  return WIFSIGNALED(status) ? 1 : 0;
}

int main(void) {
  int fails = 0;
  char cfg_path[256];

  /* (A) sender_threads present -> hard error. */
  if (write_cfg(cfg_path, sizeof(cfg_path), "sender", "sender_threads=1\n") !=
      0) {
    printf("FAIL config_removed_keys_reject: cannot write sender_threads "
           "cfg\n");
    return 1;
  }
  int sender_rejected = child_load_rejected(cfg_path);
  (void)remove(cfg_path);
  if (sender_rejected != 1) {
    printf("FAIL config_removed_keys_reject: sender_threads was ACCEPTED "
           "(expected hard error)\n");
    fails++;
  }

  /* (B) receiver_threads present -> hard error. */
  if (write_cfg(cfg_path, sizeof(cfg_path), "receiver",
               "receiver_threads=1\n") != 0) {
    printf("FAIL config_removed_keys_reject: cannot write receiver_threads "
           "cfg\n");
    return 1;
  }
  int receiver_rejected = child_load_rejected(cfg_path);
  (void)remove(cfg_path);
  if (receiver_rejected != 1) {
    printf("FAIL config_removed_keys_reject: receiver_threads was ACCEPTED "
           "(expected hard error)\n");
    fails++;
  }

  /* (B2) default_ports present -> hard error (renamed to ports). */
  if (write_cfg(cfg_path, sizeof(cfg_path), "defaultports",
                "default_ports=25000\n") != 0) {
    printf("FAIL config_removed_keys_reject: cannot write default_ports cfg\n");
    return 1;
  }
  int default_ports_rejected = child_load_rejected(cfg_path);
  (void)remove(cfg_path);
  if (default_ports_rejected != 1) {
    printf("FAIL config_removed_keys_reject: default_ports was ACCEPTED "
           "(expected hard error)\n");
    fails++;
  }

  /* (C) sanity: progress_threads (the new key) loads normally -- the hard
   * error is specific to the removed key NAMES, not networking keys in
   * general.  Run directly (not forked): this must NOT abort. */
  if (write_cfg(cfg_path, sizeof(cfg_path), "progress",
               "progress_threads=1\n") != 0) {
    printf("FAIL config_removed_keys_reject: cannot write progress_threads "
           "cfg\n");
    return 1;
  }
  setenv("ARTS_CONFIG", cfg_path, 1);
  unsetenv("ports");
  unsetenv("port_count");
  struct arts_config_s config;
  arts_config_load(&config);
  if (config.progress_thread_count != 1) {
    printf("FAIL config_removed_keys_reject: progress_threads=1 not honored "
           "(progress=%u)\n",
           config.progress_thread_count);
    fails++;
  }
  arts_config_destroy(&config);
  (void)remove(cfg_path);

  if (fails == 0) {
    printf("PASS config_removed_keys_reject: sender_threads + "
           "receiver_threads both hard-error; progress_threads loads "
           "normally\n");
    return 0;
  }
  return 1;
}
