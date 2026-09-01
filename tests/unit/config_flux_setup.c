/* SPDX-License-Identifier: Apache-2.0 */

/// @file config_flux_setup.c
/// @brief The flux launcher arm of config loading, against a stub flux CLI.
///
/// launcher=flux is opt-in: the cfg names it, and the process must be a task
/// of a flux job (rank from the environment, roster from the flux CLI —
/// flux publishes no hostlist variable).  The contract this pins:
///
///   - a two-node task env + a two-host expanded hostlist -> a two-row
///     routing table in list order, master from row 0, no thread-count
///     override (the cfg's worker/progress counts govern)
///   - hostlist delimiters are normalized (spaces, tabs, commas alike)
///   - task count != node count            -> rejected (one task per node)
///   - hostlist host count != node count   -> rejected
///   - empty / failing / absent flux CLI   -> rejected
///   - a non-expanded (bracketed) hostlist -> rejected
///   - launcher=flux with no flux task env -> rejected
///
/// Config-parser test in the config_ports_contract idiom: arts_config_load()
/// against crafted temp cfgs, a stub `flux` executable the test writes into a
/// temp dir and prepends to PATH, and every ARTS_ERROR death path probed in a
/// forked child.  No runtime is started and nothing binds.

#include "arts.h"
#include "arts/system/config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static char g_stub_dir[256];
static char g_orig_path[4096];

/* The stub prints $FLUX_STUB_HOSTLIST and exits $FLUX_STUB_RC, whatever the
 * subcommand — the runtime only ever asks it for `hostlist -e local`. */
static int write_stub_flux(void) {
  snprintf(g_stub_dir, sizeof(g_stub_dir), "/tmp/config_flux_setup_%ld",
           (long)getpid());
  if (mkdir(g_stub_dir, 0700) != 0) {
    return -1;
  }
  char path[300];
  snprintf(path, sizeof(path), "%s/flux", g_stub_dir);
  FILE *f = fopen(path, "w");
  if (!f) {
    return -1;
  }
  fputs("#!/bin/sh\n"
        "printf '%s\\n' \"${FLUX_STUB_HOSTLIST-}\"\n"
        "exit \"${FLUX_STUB_RC-0}\"\n",
        f);
  (void)fclose(f);
  if (chmod(path, 0700) != 0) {
    return -1;
  }
  const char *orig = getenv("PATH");
  snprintf(g_orig_path, sizeof(g_orig_path), "%s", orig ? orig : "");
  char with_stub[4400];
  snprintf(with_stub, sizeof(with_stub), "%s:%s", g_stub_dir,
           orig ? orig : "");
  setenv("PATH", with_stub, 1);
  return 0;
}

static void remove_stub_flux(void) {
  char path[300];
  snprintf(path, sizeof(path), "%s/flux", g_stub_dir);
  (void)remove(path);
  (void)rmdir(g_stub_dir);
  setenv("PATH", g_orig_path, 1);
}

static void set_flux_env(const char *rank, const char *nnodes,
                         const char *size, const char *hostlist,
                         const char *stub_rc) {
  if (rank) {
    setenv("FLUX_TASK_RANK", rank, 1);
  } else {
    unsetenv("FLUX_TASK_RANK");
  }
  if (nnodes) {
    setenv("FLUX_JOB_NNODES", nnodes, 1);
  } else {
    unsetenv("FLUX_JOB_NNODES");
  }
  if (size) {
    setenv("FLUX_JOB_SIZE", size, 1);
  } else {
    unsetenv("FLUX_JOB_SIZE");
  }
  setenv("FLUX_STUB_HOSTLIST", hostlist ? hostlist : "", 1);
  setenv("FLUX_STUB_RC", stub_rc ? stub_rc : "0", 1);
}

static int write_cfg(char *path_out, size_t path_cap, const char *tag) {
  snprintf(path_out, path_cap, "config_flux_setup_%s_%ld.cfg", tag,
           (long)getpid());
  FILE *f = fopen(path_out, "w");
  if (!f) {
    return -1;
  }
  fputs("[ARTS]\n"
        "launcher=flux\n"
        "worker_threads=3\n"
        "progress_threads=1\n"
        "route_table_size=14\n"
        "ports=25000\n",
        f);
  (void)fclose(f);
  return 0;
}

/* Load the cfg in a forked child; 1 = rejected (nonzero exit / signal),
 * 0 = accepted, -1 = harness failure. */
static int load_rejected(const char *cfg_path) {
  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return -1;
  }
  if (pid == 0) {
    setenv("ARTS_CONFIG", cfg_path, 1);
    struct arts_config_s config;
    arts_config_load(&config);
    _exit(0);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    perror("waitpid");
    return -1;
  }
  if (WIFSIGNALED(status)) {
    return 1;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

static void expect_rejected(const char *tag, const char *cfg_path,
                            int *fails) {
  int rejected = load_rejected(cfg_path);
  if (rejected != 1) {
    printf("FAIL config_flux_setup: %s was ACCEPTED (expected hard error)\n",
           tag);
    (*fails)++;
  }
}

int main(void) {
  int fails = 0;
  char cfg_path[256];

  if (write_stub_flux() != 0) {
    printf("FAIL config_flux_setup: cannot stage the stub flux\n");
    return 1;
  }
  if (write_cfg(cfg_path, sizeof(cfg_path), "main") != 0) {
    printf("FAIL config_flux_setup: cannot write cfg\n");
    remove_stub_flux();
    return 1;
  }
  setenv("ARTS_CONFIG", cfg_path, 1);

  /* (A) two tasks on two nodes, mixed-whitespace expanded hostlist. */
  set_flux_env("1", "2", "2", "hostA  \thostB", "0");
  struct arts_config_s config;
  arts_config_load(&config);
  if (config.launcher == NULL || strcmp(config.launcher, "flux") != 0) {
    printf("FAIL config_flux_setup: launcher is '%s', want flux\n",
           config.launcher ? config.launcher : "(null)");
    fails++;
  }
  if (config.master_boot) {
    printf("FAIL config_flux_setup: master_boot set — flux spawns nobody\n");
    fails++;
  }
  if (config.table_length != 2 || config.table == NULL) {
    printf("FAIL config_flux_setup: table_length %u, want 2\n",
           config.table_length);
    fails++;
  } else if (config.table[0].ip_address == NULL ||
             config.table[1].ip_address == NULL ||
             strcmp(config.table[0].ip_address, "hostA") != 0 ||
             strcmp(config.table[1].ip_address, "hostB") != 0) {
    printf("FAIL config_flux_setup: table is [%s, %s], want [hostA, hostB]\n",
           config.table[0].ip_address ? config.table[0].ip_address : "(null)",
           config.table[1].ip_address ? config.table[1].ip_address : "(null)");
    fails++;
  }
  if (config.master_node == NULL ||
      strcmp(config.master_node, "hostA") != 0 || config.master_rank != 0) {
    printf("FAIL config_flux_setup: master is %s/rank %u, want hostA/0\n",
           config.master_node ? config.master_node : "(null)",
           config.master_rank);
    fails++;
  }
  /* The cfg's thread counts govern: no scheduler-env override exists. */
  if (config.worker_thread_count != 3 || config.progress_thread_count != 1) {
    printf("FAIL config_flux_setup: threads %u+%u, want 3+1 from the cfg\n",
           config.worker_thread_count, config.progress_thread_count);
    fails++;
  }
  arts_config_destroy(&config);

  /* (B) comma-delimited output is accepted the same way. */
  set_flux_env("0", "2", "2", "hostA,hostB", "0");
  struct arts_config_s comma;
  arts_config_load(&comma);
  if (comma.table_length != 2 || comma.table[1].ip_address == NULL ||
      strcmp(comma.table[1].ip_address, "hostB") != 0) {
    printf("FAIL config_flux_setup: comma-delimited hostlist mis-parsed\n");
    fails++;
  }
  arts_config_destroy(&comma);

  /* (C) more tasks than nodes: one task per node is the launch contract. */
  set_flux_env("0", "2", "4", "hostA hostB", "0");
  expect_rejected("tasks-exceed-nodes", cfg_path, &fails);

  /* (D) hostlist shorter than the node count. */
  set_flux_env("0", "2", "2", "hostA", "0");
  expect_rejected("short-hostlist", cfg_path, &fails);

  /* (E) empty hostlist. */
  set_flux_env("0", "2", "2", "", "0");
  expect_rejected("empty-hostlist", cfg_path, &fails);

  /* (F) flux CLI failing. */
  set_flux_env("0", "2", "2", "hostA hostB", "1");
  expect_rejected("flux-cli-fails", cfg_path, &fails);

  /* (G) a bracketed (non-expanded) hostlist must not reach the bracket
     parser. */
  set_flux_env("0", "2", "2", "host[1-2]", "0");
  expect_rejected("bracketed-hostlist", cfg_path, &fails);

  /* (H) launcher=flux with no flux task env. */
  set_flux_env(NULL, NULL, NULL, "hostA hostB", "0");
  expect_rejected("no-task-env", cfg_path, &fails);

  /* (I) flux CLI absent from PATH entirely (the shell exits 127).  PATH
     names only a nonexistent directory so a real flux on the machine can
     never answer for the stub. */
  setenv("PATH", "/nonexistent-config-flux-setup", 1);
  set_flux_env("0", "2", "2", "hostA hostB", "0");
  expect_rejected("flux-not-on-path", cfg_path, &fails);
  setenv("PATH", g_orig_path, 1);

  (void)remove(cfg_path);
  remove_stub_flux();
  unsetenv("FLUX_TASK_RANK");
  unsetenv("FLUX_JOB_NNODES");
  unsetenv("FLUX_JOB_SIZE");
  unsetenv("FLUX_STUB_HOSTLIST");
  unsetenv("FLUX_STUB_RC");

  if (fails) {
    printf("FAIL config_flux_setup: %d check(s) failed\n", fails);
    return 1;
  }
  printf("PASS config_flux_setup: opt-in flux arm builds the roster from the "
         "CLI and rejects every broken contract\n");
  return 0;
}
