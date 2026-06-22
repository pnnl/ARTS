/* SPDX-License-Identifier: Apache-2.0
 *
 * T203 — arts_config_create_routing_table heap overflow on count mismatch.
 * Targets B097 (config.c:321, HIGH): the routing table is calloc'd with exactly
 * (*config)->nodes entries, and current_node is incremented per emitted node
 * with NO bound check.  If the node_list string parses to MORE nodes than
 * config->nodes, table[current_node] writes past the calloc'd array — a heap
 * buffer overflow.
 *
 * Reproduction: master_boot=true (SSH path), config->nodes = 2, but
 * node_list = "a,b,c" parses to 3 comma-separated hosts.  The third host write
 * (table[2]) is out of bounds for a 2-entry array.
 *
 * Run under ASan: a clean run means the overflow was fixed (bounds clamp); a
 * heap-buffer-overflow report means the bug is live.  This test is authored
 * correct-and-failing: it documents the live defect (exposes_runtime_bug=true).
 * It returns 0 only if NO overflow occurred (i.e. the parser clamped to nodes),
 * which is the desired post-fix behavior.
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <string.h>

int main(void) {
  struct arts_config_s cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.master_boot = true; /* SSH parser path */
  cfg.nodes = 2;          /* calloc 2 entries ... */

  /* ... but the list parses to 3 hosts. A node_count key disagreeing with the
     nodes string is a plausible user error. */
  char node_list[] = "a,b,c";

  struct arts_config_s *p = &cfg;
  arts_config_create_routing_table(&p, node_list);

  /* If we get here without ASan aborting, the parser must have written exactly
     table_length == nodes entries (a clamp).  table_length is forced to nodes
     by the function, so verify the parser did not run past it by checking the
     in-bounds entries are populated and reporting the count it would have
     emitted.  We cannot read table[2] (OOB) safely, so success == ASan-clean.
   */
  if (cfg.table == NULL) {
    fprintf(stderr, "FAIL routing_table_overflow: table not allocated\n");
    return 1;
  }

  /* Free only the in-bounds (allocated) entries. */
  for (unsigned int i = 0; i < cfg.nodes; i++) {
    if (cfg.table[i].ip_address) {
      arts_free(cfg.table[i].ip_address);
    }
    if (cfg.table[i].ports) {
      arts_free(cfg.table[i].ports);
    }
  }
  arts_free(cfg.table);

  printf("PASS config_routing_table_overflow: no OOB write past nodes (%u)\n",
         cfg.nodes);
  return 0;
}
