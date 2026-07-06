/* SPDX-License-Identifier: Apache-2.0
 *
 * arts_config_create_routing_table — Slurm hostlist path (master_boot=false).
 *
 * Property: every Slurm-style compressed nodelist must expand to exactly
 * config->nodes ranked hostnames, in nodelist order, with each name owning
 * its own allocation:
 *   - zero-padded ranges keep their padding width ("j[001-048]" → j001..j048)
 *   - a bracket group may mix ranges and singles ("j[001-003,005,007-008]")
 *   - multiple comma-separated partitions with distinct prefixes
 *     ("j[031-032],k007") expand in order
 *   - unpadded ranges that cross a digit-width boundary ("j[8-12]") must
 *     widen per element (j8, j9, j10, j11, j12), not truncate to the start
 *     element's width
 *   - descending ranges expand high→low
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

static void expect_table(const char *label, char *node_list,
                         const char **want, unsigned int n) {
  struct arts_config_s cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.master_boot = false;
  cfg.nodes = n;

  struct arts_config_s *p = &cfg;
  arts_config_create_routing_table(&p, node_list);

  if (cfg.table == NULL || cfg.table_length != n) {
    fprintf(stderr, "FAIL %s: table_length=%u (want %u)\n", label,
            cfg.table ? cfg.table_length : 0, n);
    fails++;
    arts_config_destroy(&cfg);
    return;
  }
  for (unsigned int i = 0; i < n; i++) {
    if (cfg.table[i].rank != i) {
      fprintf(stderr, "FAIL %s: table[%u].rank=%u\n", label, i,
              cfg.table[i].rank);
      fails++;
    }
    if (cfg.table[i].ip_address == NULL ||
        strcmp(cfg.table[i].ip_address, want[i]) != 0) {
      fprintf(stderr, "FAIL %s: table[%u].ip='%s' want '%s'\n", label, i,
              cfg.table[i].ip_address ? cfg.table[i].ip_address : "(null)",
              want[i]);
      fails++;
    }
  }
  /* Each entry owns a distinct allocation: mutating one must not alias. */
  for (unsigned int i = 1; i < n; i++) {
    if (cfg.table[i].ip_address == cfg.table[0].ip_address) {
      fprintf(stderr, "FAIL %s: ip_address aliased (%u vs 0)\n", label, i);
      fails++;
    }
  }
  arts_config_destroy(&cfg);
}

int main(void) {
  {
    /* Padded contiguous range (the canonical sbatch/srun output form). */
    char list[] = "j[001-006]";
    const char *want[] = {"j001", "j002", "j003", "j004", "j005", "j006"};
    expect_table("padded_range", list, want, 6);
  }
  {
    /* One bracket group mixing ranges and singles. */
    char list[] = "j[001-003,005,007-008]";
    const char *want[] = {"j001", "j002", "j003", "j005", "j007", "j008"};
    expect_table("mixed_group", list, want, 6);
  }
  {
    /* Multiple partitions, distinct prefixes, bracket + bare hostname. */
    char list[] = "j[031-032],k007";
    const char *want[] = {"j031", "j032", "k007"};
    expect_table("multi_partition", list, want, 3);
  }
  {
    /* Single bare hostname (1-node allocation). */
    char list[] = "j017";
    const char *want[] = {"j017"};
    expect_table("single_host", list, want, 1);
  }
  {
    /* Unpadded range crossing a digit-width boundary: each element renders
       at its own natural width. */
    char list[] = "j[8-12]";
    const char *want[] = {"j8", "j9", "j10", "j11", "j12"};
    expect_table("width_crossing", list, want, 5);
  }
  {
    /* Descending range expands high→low. */
    char list[] = "j[003-001]";
    const char *want[] = {"j003", "j002", "j001"};
    expect_table("descending", list, want, 3);
  }

  if (fails) {
    return 1;
  }
  printf("PASS config_routing_table_slurm_hostlist: 6 forms expanded\n");
  return 0;
}
