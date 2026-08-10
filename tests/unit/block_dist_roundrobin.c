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

/// @file block_dist_roundrobin.c
/// @brief T263 — multinode: verify arts_block_dist_init round-robins block
///        GUIDs across ranks exactly as the constructor's nested loop does.
///
/// The round-robin assignment is: rank i gets (num_blocks/ranks) blocks, plus
/// one extra for the first (num_blocks%ranks) ranks, filled in contiguous
/// partition-index order (rank 0 gets the lowest block indices, etc.).  The
/// rank packed into each block's GUID must match this assignment.
///
/// Cases: num_blocks == ranks (1:1), num_blocks not divisible by ranks
/// (remainder ranks get +1), num_blocks < ranks (some ranks get 0 blocks).
///
/// Config-independent (block GUIDs are reserved, not coherence-managed).  Must
/// run with >= 2 ranks; SKIPs cleanly otherwise.

#include "arts.h"
#include "arts/gas/guid.h" /* ARTS_GUID_GET_RANK */
#include "arts/graph.h"

/// Recompute the expected rank for partition index `blk` under the same
/// contiguous round-robin the constructor uses, then compare against the rank
/// encoded in the reserved GUID.  Returns true on full match.
static bool check_roundrobin(unsigned int num_blocks, unsigned int ranks) {
  arts_block_dist_t *dist =
      arts_block_dist_init(1024, 0, num_blocks, ARTS_GUID_DB);
  if (dist == NULL) {
    arts_printf("FAIL: init returned NULL for nb=%u\n", num_blocks);
    return false;
  }

  unsigned int blocks_per_node = num_blocks / ranks;
  unsigned int mod = num_blocks % ranks;
  unsigned int blk = 0;
  bool ok = true;
  for (unsigned int r = 0; r < ranks && ok; ++r) {
    unsigned int this_rank_blocks = blocks_per_node + (r < mod ? 1u : 0u);
    for (unsigned int j = 0; j < this_rank_blocks; ++j) {
      arts_guid_t g = arts_block_dist_guid_for_partition(dist, blk);
      unsigned int grank = (unsigned int)ARTS_GUID_GET_RANK(g);
      if (grank != r) {
        arts_printf("FAIL: nb=%u block %u expected rank %u got %u\n",
                    num_blocks, blk, r, grank);
        ok = false;
        break;
      }
      ++blk;
    }
  }
  if (ok && blk != num_blocks) {
    arts_printf("FAIL: nb=%u filled %u blocks, expected %u\n", num_blocks, blk,
                num_blocks);
    ok = false;
  }
  arts_block_dist_free(dist);
  return ok;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int ranks = arts_get_total_ranks();
  arts_printf("=== block_dist_roundrobin (T263) ranks=%u ===\n", ranks);
  if (ranks < 2) {
    arts_printf("SKIP block_dist_roundrobin: needs >= 2 ranks\n");
    arts_shutdown();
    return;
  }

  bool ok = true;
  /* 1:1 — one block per rank. */
  ok &= check_roundrobin(ranks, ranks);
  /* not divisible — remainder ranks get +1 (e.g. 2*ranks+1 blocks). */
  ok &= check_roundrobin(2u * ranks + 1u, ranks);
  /* fewer blocks than ranks — only the first num_blocks ranks own a block. */
  ok &= check_roundrobin(ranks - 1u, ranks);
  /* many blocks per rank. */
  ok &= check_roundrobin(3u * ranks, ranks);

  if (ok) {
    arts_printf("PASS block_dist_roundrobin\n");
  }
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
