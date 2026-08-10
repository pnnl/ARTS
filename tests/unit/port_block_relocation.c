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

/// @file port_block_relocation.c
/// @brief A local run must move off a port block someone else already holds.
///
/// The spawning rank picks the block every rank will bind, so an occupied port
/// anywhere in it has to be found BEFORE the peers exist — once they are told a
/// base, nothing can move it.  This holds a listener on the block the config
/// seeded and checks the selector slides the whole run somewhere free, keeping
/// the ranks' blocks disjoint and the base list in step with the table.
///
/// Pure unit: binds sockets and resolves a config, but starts no runtime and
/// spawns no ranks.

#include "arts.h"
#include "arts/system/config.h"
#include "arts/transport/socket.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define NODES 3

static int fails = 0;

/* Hold a port the way a foreign process would: a live listener, which no
 * SO_REUSEADDR on the prober's side can bind past. */
static int hold_port(unsigned int port) {
  int fd = socket(PF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (char *)&one, sizeof(one));
  struct sockaddr_in a;
  memset(&a, 0, sizeof(a));
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_ANY);
  a.sin_port = htons((uint16_t)port);
  if (bind(fd, (struct sockaddr *)&a, sizeof(a)) < 0 || listen(fd, 1) < 0) {
    close(fd);
    return -1;
  }
  return fd;
}

int main(void) {
  char cfg_path[256];
  snprintf(cfg_path, sizeof(cfg_path), "port_block_relocation_%ld.cfg",
           (long)getpid());
  FILE *f = fopen(cfg_path, "w");
  if (!f) {
    printf("FAIL port_block_relocation: cannot write cfg\n");
    return 1;
  }
  fputs("[ARTS]\n"
        "launcher=local\n"
        "node_count=3\n"
        "worker_threads=2\n"
        "progress_threads=1\n"
        "route_table_size=14\n",
        f);
  (void)fclose(f);

  setenv("ARTS_CONFIG", cfg_path, 1);
  unsetenv("ports");
  unsetenv("port_count");
  unsetenv(ARTS_RESOLVED_PORTS_ENV);

  struct arts_config_s cfg;
  arts_config_load(&cfg);
  (void)remove(cfg_path);

  if (cfg.table_length != NODES || cfg.ports == NULL || cfg.port_count != 1) {
    printf("FAIL port_block_relocation: config did not resolve (len=%u "
           "ports=%p port_count=%u)\n",
           cfg.table_length, (void *)cfg.ports, cfg.port_count);
    return 1;
  }

  const unsigned int seed = cfg.ports[0];

  /* Occupy the seeded block.  One held port is enough to disqualify it, but
     holding all of them also proves the selector checks every rank's share,
     not just the first. */
  int held[NODES];
  unsigned int n_held = 0;
  for (unsigned int i = 0; i < NODES; i++) {
    held[i] = hold_port(seed + i);
    if (held[i] >= 0) {
      n_held++;
    }
  }
  if (n_held == 0) {
    printf("SKIP port_block_relocation: could not hold the seeded block at "
           "%u\n",
           seed);
    arts_config_destroy(&cfg);
    return 0;
  }

  if (!arts_transport_select_local_ports(&cfg)) {
    printf("FAIL port_block_relocation: selector found no free block at all\n");
    fails++;
  }

  /* It must have moved: the seeded block is held. */
  if (cfg.ports[0] == seed) {
    printf("FAIL port_block_relocation: stayed on the occupied block %u\n",
           seed);
    fails++;
  }
  /* And it must not overlap what is held. */
  for (unsigned int i = 0; i < NODES; i++) {
    if (held[i] >= 0 && cfg.ports[0] <= seed + i &&
        seed + i < cfg.ports[0] + NODES) {
      printf("FAIL port_block_relocation: new block %u still covers held port "
             "%u\n",
             cfg.ports[0], seed + i);
      fails++;
      break;
    }
  }

  /* The table must follow the base list, one disjoint block per rank. */
  for (unsigned int i = 0; i < NODES; i++) {
    unsigned int want = cfg.ports[0] + (i * cfg.port_count);
    if (cfg.table[i].ports == NULL || cfg.table[i].ports[0] != want) {
      printf("FAIL port_block_relocation: rank %u has port %u, want %u\n", i,
             cfg.table[i].ports ? cfg.table[i].ports[0] : 0, want);
      fails++;
      break;
    }
  }

  /* The chosen block has to be inside the window the search is bounded to. */
  if (cfg.ports[0] < ARTS_PORT_WINDOW_LO ||
      cfg.ports[0] + NODES > ARTS_PORT_WINDOW_HI) {
    printf("FAIL port_block_relocation: block %u..%u outside the window "
           "%u-%u\n",
           cfg.ports[0], cfg.ports[0] + NODES - 1, ARTS_PORT_WINDOW_LO,
           ARTS_PORT_WINDOW_HI);
    fails++;
  }

  for (unsigned int i = 0; i < NODES; i++) {
    if (held[i] >= 0) {
      close(held[i]);
    }
  }
  arts_config_destroy(&cfg);

  if (fails) {
    printf("FAIL port_block_relocation: %d check(s) failed\n", fails);
    return 1;
  }
  printf("PASS port_block_relocation: slid off the occupied block\n");
  return 0;
}
