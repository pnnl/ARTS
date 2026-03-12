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
#include <assert.h>

#include "arts/BlockDistribution.h"

int main(int argc, char **argv) {

#ifdef NDEBUG
  PRINTF("[WARN] asserts are disabled. Verification will not run.\n");
#endif

  arts_block_dist_t *dist = initBlockDistributionBlock(64, 0, 2, ARTS_DB_READ);
  assert(dist->num_vertices == 64);
  assert(dist->num_blocks == 2);
  assert(dist->block_sz == 32);
  assert(getOwnerDistr(5, dist) == 0);
  assert(getOwnerDistr(31, dist) == 0);
  assert(getOwnerDistr(45, dist) == 1);
  assert(partitionStartDistr(0, dist) == 0);
  assert(partitionEndDistr(0, dist) == 31);
  assert(partitionStartDistr(1, dist) == 32);
  assert(partitionEndDistr(1, dist) == 63);
  freeDistribution(dist);

  dist = initBlockDistributionBlock(8, 0, 3, ARTS_DB_WRITE);
  assert(dist->num_vertices == 8);
  assert(dist->num_blocks == 3);
  assert(dist->block_sz == 3);
  assert(getOwnerDistr(5, dist) == 1);
  assert(getOwnerDistr(6, dist) == 2);
  assert(getOwnerDistr(2, dist) == 0);
  assert(partitionStartDistr(0, dist) == 0);
  assert(partitionEndDistr(0, dist) == 2);
  assert(partitionStartDistr(2, dist) == 6);
  assert(partitionEndDistr(2, dist) == 7);
  freeDistribution(dist);
}
