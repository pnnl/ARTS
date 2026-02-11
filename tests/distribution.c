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

#include "arts/block_distribution.h"

int main(int argc, char **argv) {

(void)argc;

(void)argv;

#ifdef NDEBUG
  arts_printf("[WARN] asserts are disabled. Verification will not run.\n");
#endif

  arts_block_dist_t *dist = init_block_distribution_block(64, 0, 2, ARTS_DB);
  assert(dist->num_vertices == 64);
  assert(dist->num_blocks == 2);
  assert(dist->block_sz == 32);
  assert(get_owner_distr(5, dist) == 0);
  assert(get_owner_distr(31, dist) == 0);
  assert(get_owner_distr(45, dist) == 1);
  assert(partition_start_distr(0, dist) == 0);
  assert(partition_end_distr(0, dist) == 31);
  assert(partition_start_distr(1, dist) == 32);
  assert(partition_end_distr(1, dist) == 63);
  free_distribution(dist);

  dist = init_block_distribution_block(8, 0, 3, ARTS_DB);
  assert(dist->num_vertices == 8);
  assert(dist->num_blocks == 3);
  assert(dist->block_sz == 3);
  assert(get_owner_distr(5, dist) == 1);
  assert(get_owner_distr(6, dist) == 2);
  assert(get_owner_distr(2, dist) == 0);
  assert(partition_start_distr(0, dist) == 0);
  assert(partition_end_distr(0, dist) == 2);
  assert(partition_start_distr(2, dist) == 6);
  assert(partition_end_distr(2, dist) == 7);
  free_distribution(dist);
}
