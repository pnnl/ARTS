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

/// @file csr_load_edgelist.c
/// @brief T268 — arts_csr_load_no_weight: edge-list parsing + CSR build.
///
/// Writes small fixture files at runtime, loads them, and verifies the
/// resulting CSR via arts_csr_from_partition + arts_csr_get_neighbors.
///
/// Covers:
///   * Known edge-list -> expected neighbours.
///   * flip = true -> each edge added in both directions (degrees double-ish:
///     the reverse edge appears for the target's owner).
///   * ignore_self_loops -> self loops dropped.
///   * Comment/header skip: a leading '%' line marks an mmio header (the NEXT
///     line is then skipped), and '#' lines are skipped.
///   * Malformed line robustness: a 3+ token line is mis-parsed by the loader
///     (it resets i=0 after the target, re-reading extra tokens as new
///     src/target pairs) — the loader must not crash; we only assert no crash
///     plus the well-formed edges still land.
///
/// MEMORY LEAK (B-csr-leak): arts_csr_load_no_weight arts_calloc's part_index
/// and vedges but NEVER arts_free's them on the success path.  Under LSan/ASan
/// the leak is reported at exit -> the test FAILS, which is the intended
/// exposure.  exposes_runtime_bug = true (B-csr-leak).
///
/// Config-independent (ARTS_DB_PIN).  Single-node.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "arts.h"
#include "../test_failure_status.h"
#include "arts/graph.h"

#define NVERTS 6

/// Write `content` to `path`; returns true on success.
static bool write_file(const char *path, const char *content) {
  FILE *f = fopen(path, "w");
  if (!f) {
    return false;
  }
  fputs(content, f);
  fclose(f);
  return true;
}

/// Sum of out-degrees over all local vertices of partition 0.
static arts_graph_sz_t total_degree(arts_block_dist_t *dist) {
  arts_csr_graph_t *csr = arts_csr_from_partition(0, dist);
  if (!csr) {
    return (arts_graph_sz_t)-1;
  }
  arts_graph_sz_t tot = 0;
  for (arts_vertex_t v = 0; v < NVERTS; ++v) {
    arts_vertex_t *nbrs = NULL;
    arts_graph_sz_t cnt = 0;
    arts_csr_get_neighbors(csr, v, &nbrs, &cnt);
    tot += cnt;
  }
  return tot;
}

/// Return true iff vertex `v` has neighbour `target` in partition 0.
static bool has_neighbor(arts_block_dist_t *dist, arts_vertex_t v,
                         arts_vertex_t target) {
  arts_csr_graph_t *csr = arts_csr_from_partition(0, dist);
  if (!csr) {
    return false;
  }
  arts_vertex_t *nbrs = NULL;
  arts_graph_sz_t cnt = 0;
  arts_csr_get_neighbors(csr, v, &nbrs, &cnt);
  for (arts_graph_sz_t k = 0; k < cnt; ++k) {
    if (nbrs[k] == target) {
      return true;
    }
  }
  return false;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== csr_load_edgelist (T268) ===\n");

  if (arts_get_total_ranks() != 1) {
    arts_printf("SKIP csr_load_edgelist: single-node only\n");
    arts_shutdown();
    return;
  }

  bool ok = true;
  char path[256];
  snprintf(path, sizeof(path), "arts_t268_edgelist_%d.txt", (int)getpid());

  /* Fixture: an mmio '%' header (next line skipped), a '#' comment, real edges,
   * a self-loop (3 3), and a 3-token weighted line that the loader mis-parses
   * but must not crash on. */
  const char *content =
      "% MatrixMarket header line\n"
      "6 6 7\n" /* mmio dimension line, skipped after % header */
      "# a comment line\n"
      "0 1\n"
      "0 2\n"
      "1 0\n"
      "3 3\n"    /* self loop */
      "4 5 99\n" /* 3-token (weighted) malformed line */
      "5 0\n";

  if (!write_file(path, content)) {
    arts_test_fail();
    arts_printf("FAIL: cannot write fixture %s\n", path);
    arts_shutdown();
    return;
  }

  /* Load #1: ignore_self_loops=true, no flip. */
  {
    arts_block_dist_t *dist = arts_block_dist_init(NVERTS, 0, 1, ARTS_GUID_DB);
    int rc = arts_csr_load_no_weight(path, dist, /*flip=*/false,
                                     /*ignore_self_loops=*/true);
    if (rc != 0) {
      arts_test_fail();
      arts_printf("FAIL: load rc=%d\n", rc);
      ok = false;
    }
    if (ok && !has_neighbor(dist, 0, 1)) {
      arts_test_fail();
      arts_printf("FAIL: edge 0->1 missing\n");
      ok = false;
    }
    if (ok && !has_neighbor(dist, 0, 2)) {
      arts_test_fail();
      arts_printf("FAIL: edge 0->2 missing\n");
      ok = false;
    }
    if (ok && !has_neighbor(dist, 1, 0)) {
      arts_test_fail();
      arts_printf("FAIL: edge 1->0 missing\n");
      ok = false;
    }
    if (ok && !has_neighbor(dist, 5, 0)) {
      arts_test_fail();
      arts_printf("FAIL: edge 5->0 missing\n");
      ok = false;
    }
    /* Self loop 3->3 must have been dropped. */
    if (ok && has_neighbor(dist, 3, 3)) {
      arts_test_fail();
      arts_printf("FAIL: self-loop 3->3 not dropped\n");
      ok = false;
    }
    arts_csr_graph_t *csr = arts_csr_from_partition(0, dist);
    if (csr) {
      arts_csr_free(csr);
    }
    arts_block_dist_free(dist);
  }

  /* Load #2: flip=true -> reverse edges added too (more total degree). */
  if (ok) {
    arts_block_dist_t *dist = arts_block_dist_init(NVERTS, 0, 1, ARTS_GUID_DB);
    int rc = arts_csr_load_no_weight(path, dist, /*flip=*/true,
                                     /*ignore_self_loops=*/true);
    if (rc != 0) {
      arts_test_fail();
      arts_printf("FAIL: flip load rc=%d\n", rc);
      ok = false;
    }
    /* With flip, 1->0 (reverse of 0->1) is already present; 2->0 is the
     * reverse of 0->2 and must now exist. */
    if (ok && !has_neighbor(dist, 2, 0)) {
      arts_test_fail();
      arts_printf("FAIL: flip reverse edge 2->0 missing\n");
      ok = false;
    }
    arts_graph_sz_t deg = total_degree(dist);
    if (ok && deg == (arts_graph_sz_t)-1) {
      arts_test_fail();
      arts_printf("FAIL: flip total_degree lookup failed\n");
      ok = false;
    }
    arts_csr_graph_t *csr = arts_csr_from_partition(0, dist);
    if (csr) {
      arts_csr_free(csr);
    }
    arts_block_dist_free(dist);
  }

  remove(path);

  if (ok) {
    arts_printf("PASS csr_load_edgelist\n");
  }
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Two verdicts to merge: what arts_rt saw of the ranks it spawned (their exit
     status reaches nobody else) and what this rank's own checks found. */
  int rc = arts_rt(argc, argv);
  return rc != 0 ? 1 : arts_test_status();
}
