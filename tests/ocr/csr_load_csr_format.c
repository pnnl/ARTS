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

/// @file csr_load_csr_format.c
/// @brief T269 — arts_csr_load_no_weight_csr: CSR-adjacency-format parsing.
///
/// Format: first line "num_verts num_edges"; each subsequent line is the
/// (1-based) adjacency list of the next source vertex (src increments per
/// line).  Targets are decremented by 1 (1-based -> 0-based).
///
/// Covers:
///   * Valid round-trip: build a graph from a well-formed CSR-format file and
///     verify neighbours (with the 1-based -> 0-based decrement).
///   * Mismatched header (num_verts != actual line count): the loader builds
///     NOTHING but still returns 0 — assert the CSR was not materialised.
///   * Token "0" -> target = (uint64_t)-1 (B-csr-token-zero): the decrement of
///     a "0" token underflows to a huge value routed to a bogus owner.  We
///     feed a file containing a "0" adjacency token; under ASan the resulting
///     huge target either routes to no local partition (silently wrong) or, in
///     the build, drives an OOB.  The test asserts the SANE behaviour (no huge
///     bogus neighbour appears) so it FAILS while the bug stands.
///     exposes_runtime_bug = true (B-csr-token-zero).
///
/// MEMORY LEAK (B-csr-leak): arts_csr_load_no_weight_csr frees part_index /
/// vedges only on the early header-read-fail path, never on the success path —
/// reported by LSan at exit.
///
/// Config-independent (ARTS_DB_PIN).  Single-node.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "arts.h"
#include "arts/graph.h"

#define NVERTS 4

static bool write_file(const char *path, const char *content) {
  FILE *f = fopen(path, "w");
  if (!f) {
    return false;
  }
  fputs(content, f);
  fclose(f);
  return true;
}

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

  arts_printf("=== csr_load_csr_format (T269) ===\n");

  if (arts_get_total_ranks() != 1) {
    arts_printf("SKIP csr_load_csr_format: single-node only\n");
    arts_shutdown();
    return;
  }

  bool ok = true;
  char path[256];

  /* Case 1: valid CSR-adjacency format, 4 verts.  1-based targets.
   *   line src=0: "2 3"   -> neighbours {1, 2}
   *   line src=1: "1"     -> neighbours {0}
   *   line src=2: "4"     -> neighbours {3}
   *   line src=3: "1 3"   -> neighbours {0, 2}
   * total 6 edges. */
  snprintf(path, sizeof(path), "/tmp/arts_t269_valid_%d.txt", (int)getpid());
  if (!write_file(path, "4 6\n2 3\n1\n4\n1 3\n")) {
    arts_printf("FAIL: cannot write valid fixture\n");
    arts_shutdown();
    return;
  }
  {
    arts_block_dist_t *dist = arts_block_dist_init(NVERTS, 0, 1, ARTS_GUID_DB);
    int rc = arts_csr_load_no_weight_csr(path, dist, /*flip=*/false,
                                         /*ignore_self_loops=*/true);
    if (rc != 0) {
      arts_printf("FAIL: valid load rc=%d\n", rc);
      ok = false;
    }
    /* 1-based decrement: token "2" -> neighbour 1, "3" -> 2, etc. */
    if (ok && !(has_neighbor(dist, 0, 1) && has_neighbor(dist, 0, 2))) {
      arts_printf("FAIL: vertex 0 neighbours wrong\n");
      ok = false;
    }
    if (ok && !has_neighbor(dist, 1, 0)) {
      arts_printf("FAIL: vertex 1 neighbour wrong\n");
      ok = false;
    }
    if (ok && !has_neighbor(dist, 2, 3)) {
      arts_printf("FAIL: vertex 2 neighbour wrong\n");
      ok = false;
    }
    if (ok && !(has_neighbor(dist, 3, 0) && has_neighbor(dist, 3, 2))) {
      arts_printf("FAIL: vertex 3 neighbours wrong\n");
      ok = false;
    }
    arts_csr_graph_t *csr = arts_csr_from_partition(0, dist);
    if (csr) {
      arts_csr_free(csr);
    }
    arts_block_dist_free(dist);
  }
  remove(path);

  /* Case 2: mismatched header — claims 4 verts but only 2 adjacency lines.
   * The loader detects src != num_verts and builds nothing, returning 0.
   * The CSR DB must therefore NOT exist. */
  if (ok) {
    snprintf(path, sizeof(path), "/tmp/arts_t269_mismatch_%d.txt",
             (int)getpid());
    if (!write_file(path, "4 3\n2 3\n1\n")) {
      arts_printf("FAIL: cannot write mismatch fixture\n");
      ok = false;
    } else {
      arts_block_dist_t *dist =
          arts_block_dist_init(NVERTS, 0, 1, ARTS_GUID_DB);
      int rc = arts_csr_load_no_weight_csr(path, dist, false, true);
      if (rc != 0) {
        arts_printf("FAIL: mismatch load rc=%d (expected 0)\n", rc);
        ok = false;
      }
      if (ok && arts_csr_from_partition(0, dist) != NULL) {
        arts_printf("FAIL: mismatched header still built a CSR\n");
        ok = false;
      }
      arts_block_dist_free(dist);
      remove(path);
    }
  }

  /* Case 3: B-csr-token-zero — a "0" adjacency token decrements to
   * (uint64_t)-1.  A correct loader would reject / not produce a neighbour;
   * the bug routes a huge bogus target.  We feed src=0 a "0 2" line: the "0"
   * underflows.  Assert that vertex 0 does NOT end up with the huge bogus
   * neighbour (uint64_t)-1 (i.e. the loader behaved sanely).  While the bug
   * stands, either an OOB build occurs (ASan) or the huge value slips in. */
  if (ok) {
    snprintf(path, sizeof(path), "/tmp/arts_t269_zero_%d.txt", (int)getpid());
    /* 2 verts: line src=0 "0 2", line src=1 "1". */
    if (!write_file(path, "2 3\n0 2\n1\n")) {
      arts_printf("FAIL: cannot write zero-token fixture\n");
      ok = false;
    } else {
      arts_block_dist_t *dist = arts_block_dist_init(2, 0, 1, ARTS_GUID_DB);
      (void)arts_csr_load_no_weight_csr(path, dist, false, true);
      if (has_neighbor(dist, 0, (arts_vertex_t)-1)) {
        arts_printf("FAIL: B-csr-token-zero produced bogus neighbour -1\n");
        ok = false;
      }
      arts_csr_graph_t *csr = arts_csr_from_partition(0, dist);
      if (csr) {
        arts_csr_free(csr);
      }
      arts_block_dist_free(dist);
      remove(path);
    }
  }

  if (ok) {
    arts_printf("PASS csr_load_csr_format\n");
  }
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
