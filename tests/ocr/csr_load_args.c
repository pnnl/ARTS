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

/// @file csr_load_args.c
/// @brief T270 — arts_csr_load_from_args argument dispatch.
///
/// Covers:
///   * --csr-format selects the CSR-adjacency loader; absence selects the
///     edge-list loader.  We verify by feeding a file in each format and
///     checking the neighbours produced.
///   * --flip / --keep-self-loops plumbing: with --keep-self-loops a self loop
///     survives; without it (default) a self loop is dropped.
///   * Missing --file: dispatch reaches a loader with file==NULL -> returns -1.
///   * --file as the LAST argv token (B-args-oob): argv[i+1] read past the end.
///     Under ASan this aborts -> the intended failure.  exposes_runtime_bug =
///     true (B-args-oob).
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

static void destroy_csr(arts_block_dist_t *dist) {
  arts_csr_graph_t *csr = arts_csr_from_partition(0, dist);
  if (csr) {
    arts_csr_free(csr);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== csr_load_args (T270) ===\n");

  if (arts_get_total_ranks() != 1) {
    arts_printf("SKIP csr_load_args: single-node only\n");
    arts_shutdown();
    return;
  }

  bool ok = true;
  char el_path[256];
  char csr_path[256];
  snprintf(el_path, sizeof(el_path), "/tmp/arts_t270_el_%d.txt", (int)getpid());
  snprintf(csr_path, sizeof(csr_path), "/tmp/arts_t270_csr_%d.txt",
           (int)getpid());

  /* Edge-list fixture with a self-loop 2->2. */
  if (!write_file(el_path, "0 1\n1 2\n2 2\n3 0\n")) {
    arts_printf("FAIL: cannot write edge-list fixture\n");
    arts_shutdown();
    return;
  }
  /* CSR-format fixture, 4 verts, 1-based. */
  if (!write_file(csr_path, "4 4\n2\n3\n3\n1\n")) {
    arts_printf("FAIL: cannot write csr fixture\n");
    remove(el_path);
    arts_shutdown();
    return;
  }

  /* (1) edge-list dispatch, default (drop self loops). */
  {
    arts_block_dist_t *dist = arts_block_dist_init(NVERTS, 0, 1, ARTS_GUID_DB);
    char *argv[] = {(char *)"prog", (char *)"--file", el_path};
    int rc = arts_csr_load_from_args(dist, 3, argv);
    if (rc != 0) {
      arts_printf("FAIL: edge-list dispatch rc=%d\n", rc);
      ok = false;
    }
    if (ok && !has_neighbor(dist, 0, 1)) {
      arts_printf("FAIL: edge-list 0->1 missing\n");
      ok = false;
    }
    if (ok && has_neighbor(dist, 2, 2)) {
      arts_printf("FAIL: self loop 2->2 not dropped by default\n");
      ok = false;
    }
    destroy_csr(dist);
    arts_block_dist_free(dist);
  }

  /* (2) edge-list dispatch with --keep-self-loops: 2->2 survives. */
  if (ok) {
    arts_block_dist_t *dist = arts_block_dist_init(NVERTS, 0, 1, ARTS_GUID_DB);
    char *argv[] = {(char *)"prog", (char *)"--file", el_path,
                    (char *)"--keep-self-loops"};
    int rc = arts_csr_load_from_args(dist, 4, argv);
    if (rc != 0) {
      arts_printf("FAIL: keep-self-loops dispatch rc=%d\n", rc);
      ok = false;
    }
    if (ok && !has_neighbor(dist, 2, 2)) {
      arts_printf("FAIL: --keep-self-loops did not retain 2->2\n");
      ok = false;
    }
    destroy_csr(dist);
    arts_block_dist_free(dist);
  }

  /* (3) --csr-format dispatch selects the CSR-adjacency loader. */
  if (ok) {
    arts_block_dist_t *dist = arts_block_dist_init(NVERTS, 0, 1, ARTS_GUID_DB);
    char *argv[] = {(char *)"prog", (char *)"--file", csr_path,
                    (char *)"--csr-format"};
    int rc = arts_csr_load_from_args(dist, 4, argv);
    if (rc != 0) {
      arts_printf("FAIL: csr-format dispatch rc=%d\n", rc);
      ok = false;
    }
    /* line src=0 "2" -> neighbour 1 (1-based decrement). */
    if (ok && !has_neighbor(dist, 0, 1)) {
      arts_printf("FAIL: csr-format vertex 0 neighbour wrong\n");
      ok = false;
    }
    destroy_csr(dist);
    arts_block_dist_free(dist);
  }

  /* (4) Missing --file: loader gets NULL path, returns -1. */
  if (ok) {
    arts_block_dist_t *dist = arts_block_dist_init(NVERTS, 0, 1, ARTS_GUID_DB);
    char *argv[] = {(char *)"prog"};
    int rc = arts_csr_load_from_args(dist, 1, argv);
    if (rc != -1) {
      arts_printf("FAIL: missing --file rc=%d (expected -1)\n", rc);
      ok = false;
    }
    arts_block_dist_free(dist);
  }

  remove(el_path);
  remove(csr_path);

  if (ok) {
    arts_printf("  arg-dispatch paths OK; now triggering B-args-oob\n");
  }

  /* (5) BUG EXPOSURE: --file as the last token -> argv[i+1] OOB read.  The
   * array is sized exactly 2, so argv[2] is past the end.  ASan aborts. */
  {
    arts_block_dist_t *dist = arts_block_dist_init(NVERTS, 0, 1, ARTS_GUID_DB);
    char *argv[] = {(char *)"prog", (char *)"--file"};
    int rc = arts_csr_load_from_args(dist, 2, argv);
    arts_printf("FAIL: B-args-oob not caught -- --file last token returned "
                "rc=%d without abort\n",
                rc);
    arts_block_dist_free(dist);
  }

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
