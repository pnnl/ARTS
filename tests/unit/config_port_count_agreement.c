/* SPDX-License-Identifier: Apache-2.0
 *
 * T212 — port_count / ports_count agreement check
 *        (config_compute_derived).
 *
 * Property: for a multi-node config, the explicit `port_count` and the number
 * of ports parsed from `ports` must agree.  CLAUDE.md states the rule:
 * "The override must carry the same port COUNT as the cfg".  config_compute_
 * derived enforces it:
 *
 *     if (config->ports_count != config->port_count)
 *         ARTS_ERROR("ports specifies %u ports but port_count=%u",
 * ...);
 *
 * ARTS_ERROR calls arts_abort(1) -> exit(1).  We therefore drive a deliberately
 * inconsistent cfg (port_count=2 but a single-port ports) IN A CHILD
 * PROCESS and assert the child dies with a nonzero status; doing it in-process
 * would terminate the test binary.  The parent reports PASS only when the
 * mismatch was actually rejected.
 *
 * This is a config-parser test: the child calls arts_config_load() directly
 * against a crafted temp cfg (no runtime started, no ports bound).  It is
 * self-contained: it sets its own ARTS_CONFIG and clears the inherited
 * `ports`/`port_count` env overrides so the harness cfg is irrelevant.
 * Orthogonal to the coherence protocol axis.
 */

#include "arts.h"
#include "arts/system/config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int write_cfg(char *path_out, size_t path_cap) {
  snprintf(path_out, path_cap, "config_port_count_mismatch_%ld.cfg",
           (long)getpid());
  FILE *f = fopen(path_out, "w");
  if (!f) {
    return -1;
  }
  /* node_count=2 forces the multi-node port-derivation branch; port_count=2 but
     ports has a single port -> ports_count(1) != port_count(2).
   */
  fputs("[ARTS]\n"
        "launcher=ssh\n"
        "nodes=n01,n02\n"
        "node_count=2\n"
        "worker_threads=2\n"
        "progress_threads=1\n"
        "port_count=2\n"
        "ports=25000\n"
        "route_table_size=14\n",
        f);
  (void)fclose(f);
  return 0;
}

int main(void) {
  char cfg_path[256];
  if (write_cfg(cfg_path, sizeof(cfg_path)) != 0) {
    printf("FAIL config_port_count_agreement: cannot write temp cfg\n");
    return 1;
  }

  pid_t pid = fork();
  if (pid < 0) {
    printf("FAIL config_port_count_agreement: fork failed\n");
    (void)remove(cfg_path);
    return 1;
  }

  if (pid == 0) {
    /* Child: load the inconsistent cfg; this MUST abort (exit(1)) inside
       config_compute_derived.  If it returns, the check did not fire -> exit 0
       so the parent flags the missing validation. */
    setenv("ARTS_CONFIG", cfg_path, 1);
    /* Drop any inherited per-test overrides that would mask the mismatch. */
    unsetenv("ports");
    unsetenv("port_count");
    struct arts_config_s config;
    arts_config_load(&config);
    /* Reached only if the agreement check did NOT abort. */
    _exit(0);
  }

  /* Parent: wait and assert the child died abnormally / nonzero. */
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    printf("FAIL config_port_count_agreement: waitpid failed\n");
    (void)remove(cfg_path);
    return 1;
  }
  (void)remove(cfg_path);

  int rejected = 0;
  if (WIFEXITED(status)) {
    rejected = (WEXITSTATUS(status) != 0);
  } else if (WIFSIGNALED(status)) {
    rejected = 1; /* aborted via signal also counts as rejection */
  }

  if (rejected) {
    printf("PASS config_port_count_agreement: port_count!=ports_count "
           "rejected (status=%d)\n",
           status);
    return 0;
  }

  printf("FAIL config_port_count_agreement: inconsistent port_count/"
         "ports was ACCEPTED (child exited 0)\n");
  return 1;
}
