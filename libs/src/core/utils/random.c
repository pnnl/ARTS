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
#include <stdint.h>

#include "arts/runtime_state.h"
#include "arts/utils/random.h"

/* Scrambles an integer into a value with no visible relation to its
 * neighbours: xor-shift-multiply rounds, each bit of the result depending on
 * every input bit.  Used both to draw and to derive a thread's key, so that
 * adjacent thread ids do not yield adjacent streams. */
static inline uint64_t arts_random_mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

void arts_thread_random_init(unsigned int rank, unsigned int thread_id) {
  /* The identity is (rank, thread) so a stream is reproducible run to run and
     unique across the whole job -- a wall-clock seed would be neither: its
     resolution is coarser than thread startup, so threads would collide. */
  arts_thread_info.rng_key =
      arts_random_mix(((uint64_t)rank << 32) | (uint64_t)thread_id);
  arts_thread_info.rng_counter = 0;
}

uint64_t arts_thread_safe_random(void) {
  /* Counter mode: the key selects the stream, the counter walks it.  Two
   * threads therefore run genuinely different sequences -- unlike a shared
   * recurrence seeded at different points, where one thread's stream is
   * another's shifted, and neighbouring seeds stay visibly related. */
  return arts_random_mix(arts_thread_info.rng_counter++ ^
                         arts_thread_info.rng_key);
}
