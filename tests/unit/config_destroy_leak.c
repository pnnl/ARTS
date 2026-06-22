/* SPDX-License-Identifier: Apache-2.0
 *
 * T216 — arts_config_destroy leak audit (LSan) + double-free documentation.
 * Documents B107 (config_destroy launcher_data raw-free + no NULL-after-free).
 *
 * Drives a full arts_config_load() end-to-end over an EMBEDDED cfg (we set the
 * file-static arts_config_override_data directly — legal because this test TU
 * #includes config.c, so the static is in-scope; config_open_file fmemopen's
 * it).  The cfg populates: launcher (ssh), routing table (3 hosts with per-node
 * ports), default_ports, master_node, net_interface, counter_folder, and a
 * launcher_data via arts_launcher_create.  Then arts_config_destroy frees them.
 *
 * Property (LSan): a single load+destroy leaks ZERO bytes.  Note the documented
 * caveats kept as comments (NOT triggered, to keep LSan green):
 *   - B107a: launcher_data is freed with a raw arts_free, no launcher-specific
 *     destructor — fine HERE because our stubbed launcher allocs nothing
 *     internal, but a real launcher with internal allocations would leak them.
 *   - B107b: arts_config_destroy does NOT NULL-out freed pointers, so a SECOND
 *     destroy double-frees.  We do NOT call destroy twice (that would be the
 *     bug, not a leak); we document it.
 *
 * Built with -fsanitize=address (LSan is part of ASan) to assert no leaks.
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <string.h>

int main(void) {
  /* Embedded cfg: ssh launcher, 3 hosts via bracket range with per-node ports,
     plus the string/array fields config_destroy must free. */
  static char cfg_text[] = "launcher=ssh\n"
                           "nodes=n[01-03]:[50000-50001]\n"
                           "node_count=3\n"
                           "net_interface=eth0\n"
                           "counter_folder=/tmp/artscnt\n"
                           "master_node=n01\n"
                           "worker_threads=4\n";
  arts_config_override_data = cfg_text;
  arts_config_override_path = NULL;

  struct arts_config_s cfg;
  arts_config_load(&cfg);

  /* Sanity: the fields config_destroy will free are actually populated. */
  if (cfg.launcher == NULL || cfg.table == NULL || cfg.table_length != 3) {
    fprintf(stderr,
            "FAIL destroy_leak: load did not populate (launcher=%p "
            "table=%p len=%u)\n",
            (void *)cfg.launcher, (void *)cfg.table, cfg.table_length);
    return 1;
  }
  if (cfg.master_node == NULL || cfg.net_interface == NULL ||
      cfg.counter_folder == NULL) {
    fprintf(stderr, "FAIL destroy_leak: string fields not populated\n");
    return 1;
  }

  /* Single destroy must free EVERYTHING (LSan asserts zero leaks at exit). */
  arts_config_destroy(&cfg);

  /* DELIBERATELY NOT calling arts_config_destroy(&cfg) a second time:
     destroy leaves the freed pointers dangling (no NULL reset), so a second
     call would double-free (B107b).  Documented, not exercised. */

  printf("PASS config_destroy_leak: full load+destroy leak-free "
         "(double-free on 2nd destroy documented)\n");
  return 0;
}
