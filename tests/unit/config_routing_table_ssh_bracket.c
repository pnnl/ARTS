/* SPDX-License-Identifier: Apache-2.0
 *
 * T209 — arts_config_create_routing_table SSH bracket-range path.
 *
 * Property: with master_boot=true (SSH parser), node_list "n[01-03]" must
 * expand to three zero-padded hostnames, one per rank:
 *   table[0] = {rank 0, "n01"}
 *   table[1] = {rank 1, "n02"}
 *   table[2] = {rank 2, "n03"}
 * and all allocations free cleanly (each node owns its own ip_address malloc).
 * The list carries hostnames only — ports are never named per node, they come
 * from the one shared base every rank derives from.
 *
 * Pure unit + ASan: exercises the per-node malloc + "%0*u" zero-pad snprintf
 * that is otherwise entirely uncovered.  The node count (3) matches
 * config->nodes (3) so this is the in-bounds happy-path companion to T203's
 * overflow case.
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

int main(void) {
  struct arts_config_s cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.master_boot = true;
  cfg.nodes = 3;

  char node_list[] = "n[01-03]";
  struct arts_config_s *p = &cfg;
  arts_config_create_routing_table(&p, node_list);

  if (cfg.table == NULL || cfg.table_length != 3) {
    fprintf(stderr, "FAIL ssh_bracket: table_length=%u (want 3)\n",
            cfg.table_length);
    return 1;
  }

  const char *want_ip[3] = {"n01", "n02", "n03"};
  for (unsigned int i = 0; i < 3; i++) {
    if (cfg.table[i].rank != i) {
      fprintf(stderr, "FAIL ssh_bracket: table[%u].rank=%u\n", i,
              cfg.table[i].rank);
      fails++;
    }
    if (cfg.table[i].ip_address == NULL ||
        strcmp(cfg.table[i].ip_address, want_ip[i]) != 0) {
      fprintf(stderr, "FAIL ssh_bracket: table[%u].ip='%s' want '%s'\n", i,
              cfg.table[i].ip_address ? cfg.table[i].ip_address : "(null)",
              want_ip[i]);
      fails++;
    }
    /* The nodes list never carries ports; the shared base fills these in
       later, in config_compute_derived. */
    if (cfg.table[i].ports != NULL) {
      fprintf(stderr,
              "FAIL ssh_bracket: table[%u] got ports from the nodes list\n", i);
      fails++;
    }
  }

  /* Each hostname must be its own allocation. */
  if (cfg.table[0].ip_address == cfg.table[1].ip_address) {
    fprintf(stderr, "FAIL ssh_bracket: hostnames aliased across nodes\n");
    fails++;
  }

  arts_config_destroy(&cfg);

  if (fails) {
    return 1;
  }
  printf("PASS config_routing_table_ssh_bracket: n01..n03, hostnames only\n");
  return 0;
}
