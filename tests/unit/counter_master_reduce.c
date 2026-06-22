/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

/// @file counter_master_reduce.c
/// @brief EXPOSES B136: cluster-level MASTER reduction assumes the master
///        rank == node index 0.
///
/// arts_apply_reduction(REDUCE_MASTER) returns the source value only when
/// source_index == 0.  arts_counter_write_cluster (counter.c) iterates node
/// index n and passes n as source_index, so MASTER always selects node 0's
/// value — not the value of arts_global_master_rank_id.  Compounding this,
/// arts_counter_write_node emits a MASTER CLUSTER counter ONLY on the master
/// rank; so when the master rank != 0, node 0's n0.json contains no
/// TIME_INIT/TIME_TOTAL object, the cluster reducer reads the calloc-zeroed
/// slot for node 0, and the cluster value collapses to 0 (wrong).
///
/// TIME_INIT and TIME_TOTAL are ONCE,CLUSTER,MASTER in both counters.cfg and
/// full_counters.cfg.  The master writes cluster.json at shutdown after polling
/// every n{n}.json.  This test (run on the master rank) reads cluster.json and
/// asserts TIME_TOTAL's cluster value is non-zero — the master's real
/// end-to-end time.  This holds iff master rank == 0; if a config ever sets a
/// non-zero master rank the value is 0 and the assertion fails, surfacing B136.
///
/// Multinode: requires node_count > 1 (single node has no cluster aggregation).
/// SKIPs cleanly on a single node.  If cluster.json or TIME_TOTAL is absent
/// (CLUSTER counters disabled in this build's counter config) it SKIPs.
/// Config-agnostic across coherence protocols.  A stranded run is reaped by the
/// ctest TIMEOUT.

#include "arts.h"
#include "arts/system/identity.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void busy_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* A little work on each rank so per-node counter files are produced. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  unsigned int ranks = arts_get_total_ranks();
  for (unsigned int r = 0; r < ranks; r++) {
    arts_edt_create(busy_edt, 0, NULL, 0,
                    &(arts_edt_hint_t){.rank = r, .finish_event = fe});
  }
  arts_event_wait(fe);

  arts_shutdown();
}

/* Read counter `name`'s "value" from a cluster/node JSON. */
static bool read_counter_value(const char *json, const char *name,
                               uint64_t *out) {
  char key[128];
  (void)snprintf(key, sizeof(key), "\"%s\"", name);
  const char *obj = strstr(json, key);
  if (!obj) {
    return false;
  }
  const char *v = strstr(obj, "\"value\"");
  if (!v) {
    return false;
  }
  v = strchr(v + strlen("\"value\""), ':');
  if (!v) {
    return false;
  }
  v++;
  while (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r') {
    v++;
  }
  char *end = NULL;
  uint64_t val = strtoull(v, &end, 10);
  if (end == v) {
    return false;
  }
  *out = val;
  return true;
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);

  /* Only the master rank writes (and should validate) cluster.json. */
  if (arts_global_rank_id != arts_global_master_rank_id) {
    printf("counter_master_reduce: rank %u not master, done\n",
           arts_global_rank_id);
    return 0;
  }

  if (arts_global_rank_count < 2) {
    printf(
        "SKIP counter_master_reduce: single node (no cluster aggregation)\n");
    return 0;
  }

  FILE *fp = fopen("./counters/cluster.json", "r");
  if (!fp) {
    printf("SKIP counter_master_reduce: no ./counters/cluster.json (CLUSTER "
           "counters disabled?)\n");
    return 0;
  }
  (void)fseek(fp, 0, SEEK_END);
  long sz = ftell(fp);
  (void)fseek(fp, 0, SEEK_SET);
  if (sz <= 0) {
    (void)fclose(fp);
    printf("SKIP counter_master_reduce: empty cluster.json\n");
    return 0;
  }
  char *buf = (char *)malloc((size_t)sz + 1);
  size_t got = fread(buf, 1, (size_t)sz, fp);
  (void)fclose(fp);
  buf[got] = '\0';

  uint64_t total = 0;
  bool found = read_counter_value(buf, "TIME_TOTAL", &total);
  free(buf);

  if (!found) {
    printf("SKIP counter_master_reduce: TIME_TOTAL absent in cluster.json\n");
    return 0;
  }

  /* B136: when master rank != 0 the MASTER reduce selects node 0 (which never
     emitted the counter) and collapses to 0.  A correct master-aware reduce
     yields the master's real (non-zero) TIME_TOTAL. */
  if (total == 0) {
    printf("FAIL counter_master_reduce: cluster TIME_TOTAL == 0 (master rank "
           "%u != node-index 0 -> B136 MASTER-reduce picked wrong node)\n",
           arts_global_master_rank_id);
    return 1;
  }

  printf("PASS counter_master_reduce: cluster TIME_TOTAL=%llu from master rank "
         "%u\n",
         (unsigned long long)total, arts_global_master_rank_id);
  return 0;
}
