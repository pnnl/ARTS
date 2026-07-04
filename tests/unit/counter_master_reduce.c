/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

/// @file counter_master_reduce.c
/// @brief Cluster-level counter aggregation: the master rank must collect
///        EVERY node's per-node counter file (n{n}.json) and reduce each
///        CLUSTER counter across ALL nodes into cluster.json.
///
/// The workload finishes exactly (ranks + 1) EDTs — the main EDT on rank 0
/// plus one affinity-pinned EDT per rank — so NUM_EDT_FINISH (PERIODIC,
/// CLUSTER, SUM in the stock counter configs) must aggregate to at least
/// ranks + 1 in cluster.json.  A smaller sum means some node's counter file
/// was dropped from (or mis-selected by) the reduction: that node's pinned
/// EDT goes uncounted.  This guards the reducer's node-coverage contract —
/// the historical failure mode where the cluster reduce reads the wrong
/// source node's slot (a calloc-zeroed value) instead of a real one.
///
/// Multinode: requires node_count > 1 (single node has no cluster
/// aggregation).  SKIPs cleanly on a single node, when cluster.json is
/// absent, or when NUM_EDT_FINISH is not part of this build's counter
/// config (ARTS_COUNTER_CONFIG is a build-time selection; the assertion is
/// only meaningful for counters the build actually captures).
/// Config-agnostic across coherence protocols.  A stranded run is reaped by
/// the ctest TIMEOUT.

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

  /* One EDT pinned to each rank so every node's counter file must
   * contribute to the cluster sum. */
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

  uint64_t finished = 0;
  bool found = read_counter_value(buf, "NUM_EDT_FINISH", &finished);
  free(buf);

  if (!found) {
    printf(
        "SKIP counter_master_reduce: NUM_EDT_FINISH absent in cluster.json "
        "(not enabled in this build's counter config)\n");
    return 0;
  }

  /* main EDT + one pinned EDT per rank all completed before shutdown, so a
   * node-complete SUM reduce must see at least ranks + 1.  Anything less
   * means a node's file was dropped from the cluster reduction. */
  uint64_t expected_min = (uint64_t)arts_global_rank_count + 1;
  if (finished < expected_min) {
    printf("FAIL counter_master_reduce: cluster NUM_EDT_FINISH=%llu < %llu "
           "(ranks+1) — a node's counter file was dropped from the cluster "
           "reduce\n",
           (unsigned long long)finished, (unsigned long long)expected_min);
    return 1;
  }

  printf("PASS counter_master_reduce: cluster NUM_EDT_FINISH=%llu >= %llu "
         "(ranks+1) from master rank %u\n",
         (unsigned long long)finished, (unsigned long long)expected_min,
         arts_global_master_rank_id);
  return 0;
}
