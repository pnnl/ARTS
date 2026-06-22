/* SPDX-License-Identifier: Apache-2.0
 *
 * T210 — local multi-node port auto-offset discipline (config_compute_derived
 *        + config_setup_local).
 *
 * This is the property the ENTIRE multinode-CTest port-isolation discipline
 * rests on (see CLAUDE.md "Port isolation is structural").  For a
 * `launcher=local` config with node_count > 1, the loader must:
 *   - mark config->shared_pu_pool == true (all nodes share PUs on 127.0.0.1),
 *   - force every table[i].ip_address to "127.0.0.1",
 *   - and, because all nodes share the loopback interface, give each node a
 *     DISJOINT port block:  table[i].ports[j] == default_ports[j] +
 * i*port_count. A regression here makes every local multinode run collide on
 * the same TCP ports and flake.
 *
 * Harness: runtime_multinode.  Registered for 2n/3n/4n/2n_io; each variant's
 * ARTS_CONFIG points at configs/local/<variant>.cfg and the per-test
 * `default_ports` env override (set by register_multinode_test) replaces the
 * cfg's default_ports base.  The test does NOT start the runtime — it calls
 * arts_config_load() directly and inspects the resolved table, so it binds no
 * ports and never oversubscribes.  config.c is protocol-agnostic, so this runs
 * meaningfully in every coherence build.
 *
 * Note: arts_config_load() reads the `default_ports` env override itself (via
 * arts_config_find_variable), so after load config->default_ports already holds
 * the harness-assigned base block; we verify the table offset against THAT, so
 * the assertion is independent of which base the harness happened to hand us.
 */

#include "arts.h"
#include "arts/system/config.h"

#include <stdio.h>
#include <string.h>

int main(void) {
  struct arts_config_s config;
  arts_config_load(&config);

  /* Single-node cfg (or a build that resolved to 1 node) cannot exercise the
     offset path — skip cleanly rather than false-fail. */
  if (config.table_length <= 1) {
    printf("SKIP config_load_multinode_port_offset: single node "
           "(table_length=%u)\n",
           config.table_length);
    arts_config_destroy(&config);
    return 0;
  }

  int ok = 1;

  if (!config.shared_pu_pool) {
    printf("FAIL config_load_multinode_port_offset: shared_pu_pool not set for "
           "local multi-node\n");
    ok = 0;
  }
  if (config.default_ports == NULL || config.default_ports_count == 0) {
    printf(
        "FAIL config_load_multinode_port_offset: no default_ports resolved\n");
    arts_config_destroy(&config);
    return 1;
  }
  if (config.port_count != config.default_ports_count) {
    printf("FAIL config_load_multinode_port_offset: port_count=%u != "
           "default_ports_count=%u\n",
           config.port_count, config.default_ports_count);
    ok = 0;
  }
  if (config.table == NULL) {
    printf("FAIL config_load_multinode_port_offset: routing table is NULL\n");
    arts_config_destroy(&config);
    return 1;
  }

  for (unsigned int i = 0; i < config.table_length; i++) {
    if (config.table[i].ip_address == NULL ||
        strcmp(config.table[i].ip_address, "127.0.0.1") != 0) {
      printf("FAIL config_load_multinode_port_offset: node %u ip=%s expected "
             "127.0.0.1\n",
             i,
             config.table[i].ip_address ? config.table[i].ip_address
                                        : "(null)");
      ok = 0;
      continue;
    }
    if (config.table[i].ports == NULL) {
      printf("FAIL config_load_multinode_port_offset: node %u has no ports\n",
             i);
      ok = 0;
      continue;
    }
    for (unsigned int j = 0; j < config.port_count; j++) {
      unsigned int expected = config.default_ports[j] + (i * config.port_count);
      if (config.table[i].ports[j] != expected) {
        printf("FAIL config_load_multinode_port_offset: node %u port[%u]=%u "
               "expected %u (base=%u + %u*%u)\n",
               i, j, config.table[i].ports[j], expected,
               config.default_ports[j], i, config.port_count);
        ok = 0;
      }
    }
  }

  /* Cross-node disjointness: no two nodes may share a port (the whole point of
     the offset).  Verified implicitly by the formula above (offset =
     i*port_count with port_count contiguous ports per node), but assert the
     first ports are strictly increasing across nodes as a belt-and-suspenders
     check. */
  for (unsigned int i = 1; i < config.table_length; i++) {
    if (config.table[i].ports[0] <= config.table[i - 1].ports[0]) {
      printf("FAIL config_load_multinode_port_offset: node %u base port %u not "
             "above node %u base port %u\n",
             i, config.table[i].ports[0], i - 1, config.table[i - 1].ports[0]);
      ok = 0;
    }
  }

  unsigned int n = config.table_length;
  arts_config_destroy(&config);

  if (ok) {
    printf("PASS config_load_multinode_port_offset %u nodes 127.0.0.1 "
           "offset-disjoint ports\n",
           n);
    return 0;
  }
  return 1;
}
