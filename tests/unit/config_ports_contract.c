/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/

/// @file config_ports_contract.c
/// @brief Who fixes the listen ports, and who may not.
///
/// A local run puts every rank on this machine, so the runtime always claims a
/// free block itself; naming ports by hand there would only invite the
/// collisions that machinery exists to avoid.  Every other launcher is the
/// mirror image: its ranks resolve each other's ports from configs they read
/// independently, and a probe on the launching machine says nothing about a
/// remote host, so the config must name them.
///
///   - launcher=local, ports absent  -> runtime chooses (port_count of them)
///   - launcher=local, ports named   -> rejected, it is not the operator's call
///   - launcher=ssh,   ports absent  -> rejected, nothing else can supply them
///   - launcher=ssh,   ports named   -> used verbatim by every rank
///   - a list shorter than port_count -> rejected (it would be read past its
///     end, since the count doubles as the parallel-connection count)
///
/// Config-parser test: arts_config_load() against crafted temp cfgs, no runtime
/// started and no ports bound.  The rejections are ARTS_ERROR death paths, so
/// they are probed in a forked child.

#include "arts.h"
#include "arts/system/config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int write_cfg(char *path_out, size_t path_cap, const char *tag,
                     const char *body) {
  snprintf(path_out, path_cap, "config_ports_contract_%s_%ld.cfg", tag,
           (long)getpid());
  FILE *f = fopen(path_out, "w");
  if (!f) {
    return -1;
  }
  fputs("[ARTS]\n"
        "worker_threads=2\n"
        "progress_threads=1\n"
        "route_table_size=14\n",
        f);
  fputs(body, f);
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
    unsetenv("ports");
    unsetenv("port_count");
    unsetenv(ARTS_RESOLVED_PORTS_ENV);
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

static int expect_rejected(const char *tag, const char *body, int *fails) {
  char cfg_path[256];
  if (write_cfg(cfg_path, sizeof(cfg_path), tag, body) != 0) {
    printf("FAIL config_ports_contract: cannot write %s cfg\n", tag);
    return -1;
  }
  int rejected = load_rejected(cfg_path);
  (void)remove(cfg_path);
  if (rejected != 1) {
    printf("FAIL config_ports_contract: %s was ACCEPTED (expected hard "
           "error)\n",
           tag);
    (*fails)++;
  }
  return 0;
}

int main(void) {
  int fails = 0;
  char cfg_path[256];

  /* (A) local + no ports -> the runtime picks them, one per port_count. */
  if (write_cfg(cfg_path, sizeof(cfg_path), "localauto",
                "launcher=local\nnode_count=2\nport_count=2\n") != 0) {
    printf("FAIL config_ports_contract: cannot write localauto cfg\n");
    return 1;
  }
  setenv("ARTS_CONFIG", cfg_path, 1);
  unsetenv("ports");
  unsetenv("port_count");
  unsetenv(ARTS_RESOLVED_PORTS_ENV);
  struct arts_config_s config;
  arts_config_load(&config);
  if (config.ports == NULL || config.ports_count != config.port_count) {
    printf("FAIL config_ports_contract: unnamed ports not resolved "
           "(ports=%p count=%u port_count=%u)\n",
           (void *)config.ports, config.ports_count, config.port_count);
    fails++;
  }
  /* The seed must land in the window the search is bounded to, and the ranks
     must take disjoint blocks off it. */
  if (config.ports != NULL &&
      (config.ports[0] < ARTS_PORT_WINDOW_LO ||
       config.ports[0] >= ARTS_PORT_WINDOW_HI)) {
    printf("FAIL config_ports_contract: seed %u outside the window %u-%u\n",
           config.ports[0], ARTS_PORT_WINDOW_LO, ARTS_PORT_WINDOW_HI);
    fails++;
  }
  for (unsigned int i = 0; config.table && i < config.table_length; i++) {
    for (unsigned int j = 0; j < config.port_count; j++) {
      unsigned int want = config.ports[j] + (i * config.port_count);
      if (config.table[i].ports == NULL || config.table[i].ports[j] != want) {
        printf("FAIL config_ports_contract: node %u port %u is %u, want %u\n",
               i, j, config.table[i].ports ? config.table[i].ports[j] : 0,
               want);
        fails++;
        i = config.table_length;
        break;
      }
    }
  }
  arts_config_destroy(&config);
  (void)remove(cfg_path);

  /* (B) a local run may not be told which ports to take. */
  if (expect_rejected("localnamed",
                      "launcher=local\nnode_count=2\nports=25000\n",
                      &fails) != 0) {
    return 1;
  }

  /* (C) a remote launcher naming no ports at all. */
  if (expect_rejected("sshnoports",
                      "launcher=ssh\nnodes=n01,n02\nnode_count=2\n",
                      &fails) != 0) {
    return 1;
  }

  /* (D) a list shorter than the connection count it doubles as. */
  if (expect_rejected("shortlist", "launcher=ssh\nnodes=n01,n02\n"
                                   "node_count=2\nport_count=2\nports=25000\n",
                      &fails) != 0) {
    return 1;
  }

  /* (E) a remote launcher's ports are taken verbatim, the same on every node:
     its ranks sit on different hosts, so they never contend. */
  if (write_cfg(cfg_path, sizeof(cfg_path), "sshnamed",
                "launcher=ssh\nnodes=n01,n02\nnode_count=2\n"
                "ports=[25000-25001]\n") != 0) {
    printf("FAIL config_ports_contract: cannot write sshnamed cfg\n");
    return 1;
  }
  setenv("ARTS_CONFIG", cfg_path, 1);
  unsetenv("ports");
  unsetenv("port_count");
  unsetenv(ARTS_RESOLVED_PORTS_ENV);
  struct arts_config_s named;
  arts_config_load(&named);
  if (named.port_count != 2) {
    printf("FAIL config_ports_contract: port_count %u not derived from the "
           "ports list\n",
           named.port_count);
    fails++;
  }
  for (unsigned int i = 0; named.table && i < named.table_length; i++) {
    if (named.table[i].ports == NULL || named.table[i].ports[0] != 25000 ||
        named.table[i].ports[1] != 25001) {
      printf("FAIL config_ports_contract: node %u did not take the named "
             "ports\n",
             i);
      fails++;
      break;
    }
  }
  arts_config_destroy(&named);
  (void)remove(cfg_path);

  /* (F) a spawned local rank takes the block the spawning rank claimed. */
  if (write_cfg(cfg_path, sizeof(cfg_path), "localresolved",
                "launcher=local\nnode_count=2\n") != 0) {
    printf("FAIL config_ports_contract: cannot write localresolved cfg\n");
    return 1;
  }
  setenv("ARTS_CONFIG", cfg_path, 1);
  unsetenv("ports");
  unsetenv("port_count");
  setenv(ARTS_RESOLVED_PORTS_ENV, "21234", 1);
  struct arts_config_s spawned;
  arts_config_load(&spawned);
  if (spawned.ports == NULL || spawned.ports[0] != 21234) {
    printf("FAIL config_ports_contract: spawned rank ignored the resolved "
           "block\n");
    fails++;
  }
  if (spawned.table && spawned.table[1].ports != NULL &&
      spawned.table[1].ports[0] != 21235) {
    printf("FAIL config_ports_contract: spawned rank derived peer port %u, "
           "want 21235\n",
           spawned.table[1].ports[0]);
    fails++;
  }
  arts_config_destroy(&spawned);
  unsetenv(ARTS_RESOLVED_PORTS_ENV);
  (void)remove(cfg_path);

  if (fails) {
    printf("FAIL config_ports_contract: %d check(s) failed\n", fails);
    return 1;
  }
  printf("PASS config_ports_contract: a local run claims its own ports, every "
         "other launcher must name them\n");
  return 0;
}
