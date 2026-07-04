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

/// @file net_addr_frame_bound.c
/// @brief Whitebox check for the bootstrap address-frame length bound.
///
/// arts_net_exchange_addresses() reads one length-prefixed address frame per
/// peer and accepts it only via arts_net_addr_frame_ok(): the embedded rank
/// must be in range AND the blob length must EXACTLY equal this run's fixed
/// provider address length.  The strict equality (not a <= bound) is what keeps
/// a corrupt/misframed peer from being silently truncated.  This restores the
/// coverage lost when socket_size_dos_bound.c was deleted with the old socket
/// receive path — the predicate is extracted into net.h precisely so this bound
/// is testable in isolation from the socket I/O around it.

#include <stdio.h>

#include "arts/transport/net.h"

int main(void) {
#ifdef ARTS_TRANSPORT_OFI
  const unsigned NRANKS = 4;
  const unsigned OWN = 32; /* fixed provider address length for this run */

  /* Accept: rank in range and length exactly own_len. */
  if (!arts_net_addr_frame_ok(0, OWN, NRANKS, OWN) ||
      !arts_net_addr_frame_ok(NRANKS - 1, OWN, NRANKS, OWN)) {
    printf("FAIL: rejected a valid frame\n");
    return 1;
  }

  /* Reject: rank out of range (>= nranks). */
  if (arts_net_addr_frame_ok(NRANKS, OWN, NRANKS, OWN) ||
      arts_net_addr_frame_ok(NRANKS + 7, OWN, NRANKS, OWN)) {
    printf("FAIL: accepted an out-of-range rank\n");
    return 1;
  }

  /* Reject: length mismatch in EITHER direction — the equality is strict. */
  if (arts_net_addr_frame_ok(0, OWN - 1, NRANKS, OWN) ||
      arts_net_addr_frame_ok(0, OWN + 1, NRANKS, OWN) ||
      arts_net_addr_frame_ok(0, 0, NRANKS, OWN)) {
    printf("FAIL: accepted a mismatched blob length\n");
    return 1;
  }

  printf("PASS: address-frame bound rejects out-of-range rank and any length "
         "deviation\n");
  return 0;
#else
  printf("SKIP: OFI transport disabled — no bootstrap address exchange\n");
  return 0;
#endif
}
