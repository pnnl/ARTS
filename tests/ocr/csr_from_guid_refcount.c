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

/// @file csr_from_guid_refcount.c
/// @brief T266 — arts_csr_from_guid / arts_csr_from_partition lookup +
///        refcount discipline.
///
/// Checks:
///   * Local lookup returns the SAME raw pointer as arts_csr_init produced
///     (the CSR lives in the DB payload, just past the descriptor).
///   * Repeated lookups (N times) return the identical, stable pointer and
///     never crash.  arts_csr_from_guid does arts_shared_get then
///     arts_shared_release on every call; if it leaked a ref each time the DB
///     descriptor would be pinned, and if it dropped one too many the repeated
///     dereference below would be a use-after-free (caught by ASan).  A stable,
///     dereferenceable pointer across many calls is the observable proof that
///     the per-call get/release is balanced.
///   * A never-created local GUID returns NULL.
///   * A remote GUID (a DB GUID reserved on another rank when multi-node)
///     returns NULL even though the route table may know of it.
///   * arts_csr_from_partition(0) matches arts_csr_from_guid of partition 0's
///     GUID.
///
/// DOCUMENTED ASSUMPTION (suspected ref-drop-then-use UAF): arts_csr_from_guid
/// intentionally returns the payload pointer AFTER releasing the cb ref,
/// relying on "graph DBs are never destroyed during computation."  This test
/// honours that contract (it never destroys the DB before the lookups) — it
/// does NOT race a destroy against a lookup, which would be the genuine UAF.
///
/// Config-independent (ARTS_DB_PIN).  Single-node body; remote-GUID branch only
/// meaningfully exercised when run with >1 rank, but is harmless single-node.

#include "arts.h"
#include "arts/graph.h"

#define LOOKUPS 2048

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== csr_from_guid_refcount (T266) ===\n");

  /* One block per rank so partition 0 is local on rank 0, and (multinode)
   * higher partitions are remote. */
  unsigned int ranks = arts_get_total_ranks();
  arts_block_dist_t *dist = arts_block_dist_init(64, 3, ranks, ARTS_GUID_DB);

  /* Build a tiny CSR on the local partition (index 0 is rank 0). */
  arts_edge_vector_t vec;
  arts_edge_vector_init(&vec, 16);
  /* vertices 0,1,2 belong to partition 0 (block_sz >= 64/ranks >= 1). */
  arts_edge_vector_push_back(&vec, 0, 1, 0);
  arts_edge_vector_push_back(&vec, 0, 2, 0);
  arts_edge_vector_push_back(&vec, 1, 0, 0);
  arts_edge_vector_sort_by_source(&vec);

  arts_graph_sz_t localv = arts_block_dist_block_size(0, dist);
  arts_guid_t g0 = arts_block_dist_guid_for_partition(dist, 0);
  arts_csr_graph_t *built =
      arts_csr_init(0, localv, vec.used, dist, &vec, true, g0);
  arts_edge_vector_free(&vec);

  if (built == NULL) {
    arts_printf("FAIL: local init returned NULL\n");
    arts_block_dist_free(dist);
    arts_shutdown();
    return;
  }

  bool ok = true;

  /* Local lookup must equal the init result, repeatedly and stably. */
  for (int i = 0; i < LOOKUPS && ok; ++i) {
    arts_csr_graph_t *got = arts_csr_from_guid(g0);
    if (got != built) {
      arts_printf("FAIL: lookup %d returned %p, expected %p\n", i, (void *)got,
                  (void *)built);
      ok = false;
      break;
    }
    /* Dereference the descriptor field to prove it is still alive (would fault
     * under ASan if a missing release had freed it). */
    if (got->partGuid != g0) {
      arts_printf("FAIL: lookup %d partGuid mismatch\n", i);
      ok = false;
      break;
    }
  }

  /* from_partition(0) == from_guid(g0). */
  if (ok && arts_csr_from_partition(0, dist) != built) {
    arts_printf("FAIL: from_partition(0) != built\n");
    ok = false;
  }

  /* Never-created local GUID -> NULL.  Reserve a fresh local DB GUID that was
   * never installed. */
  if (ok) {
    arts_guid_t never =
        arts_guid_reserve(ARTS_GUID_DB, arts_get_current_rank());
    if (arts_csr_from_guid(never) != NULL) {
      arts_printf("FAIL: never-created GUID lookup not NULL\n");
      ok = false;
    }
  }

  /* Remote partition (multinode) -> NULL. */
  if (ok && ranks > 1) {
    arts_csr_graph_t *rem = arts_csr_from_partition(1, dist);
    if (rem != NULL) {
      arts_printf("FAIL: remote partition lookup not NULL (got %p)\n",
                  (void *)rem);
      ok = false;
    }
  }

  if (ok) {
    arts_printf("PASS csr_from_guid_refcount\n");
  }

  arts_csr_free(built);
  arts_block_dist_free(dist);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
