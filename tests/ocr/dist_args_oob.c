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

/// @file dist_args_oob.c
/// @brief T262 — arg-parsing of arts_block_dist_init_from_args /
///        arts_csr_load_from_args.
///
/// Two behaviours are exercised:
///   (1) SAFE / CORRECT: when --num-vertices / --num-edges are missing, the
///       constructor must return NULL (and arts_csr_load_from_args with no
///       --file must dispatch into a loader that returns -1 on a NULL path).
///   (2) A VALUE-BEARING FLAG IN THE LAST POSITION.  Its value token does not
///       exist, so the parser must leave the flag unset rather than reach past
///       the argv array for it: the constructor then sees no vertex count and
///       returns NULL, exactly as it does when the flag is absent entirely.
///
/// Runtime test: _from_args calls arts_get_total_ranks(), so it must run
/// inside main_edt.  Config-independent (no coherence protocol dependence).

#include <stdio.h>
#include <stdlib.h>

#include "arts.h"
#include "../test_failure_status.h"
#include "arts/graph.h"

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== dist_args_oob (T262) ===\n");

  /* (1a) Both flags missing -> NULL. */
  {
    char *argv0[] = {(char *)"prog"};
    arts_block_dist_t *d = arts_block_dist_init_from_args(1, argv0);
    if (d != NULL) {
      arts_test_fail();
      arts_printf("FAIL: missing both args should yield NULL\n");
      arts_block_dist_free(d);
      arts_shutdown();
      return;
    }
  }

  /* (1b) Only --num-vertices present (with a value) -> still NULL (m==0). */
  {
    char *argv1[] = {(char *)"prog", (char *)"--num-vertices", (char *)"16"};
    arts_block_dist_t *d = arts_block_dist_init_from_args(3, argv1);
    if (d != NULL) {
      arts_test_fail();
      arts_printf("FAIL: only --num-vertices should yield NULL\n");
      arts_block_dist_free(d);
      arts_shutdown();
      return;
    }
  }

  /* (1c) arts_csr_load_from_args with no --file -> loader sees NULL path and
   *      returns -1 (a real dist is required for the local-partition scan). */
  {
    arts_block_dist_t *dist =
        arts_block_dist_init(8, 4, arts_get_total_ranks(), ARTS_GUID_DB);
    char *argv2[] = {(char *)"prog"};
    int rc = arts_csr_load_from_args(dist, 1, argv2);
    if (rc != -1) {
      arts_test_fail();
      arts_printf("FAIL: load_from_args with no --file should return -1, "
                  "got %d\n",
                  rc);
      arts_block_dist_free(dist);
      arts_shutdown();
      return;
    }
    arts_block_dist_free(dist);
  }

  arts_printf("  arg-NULL paths OK\n");

  /* (2) A value-bearing flag with no value token.  argv here is tightly sized
   *     at 2, so argv[2] is past the end; the parser must not go looking there.
   *     Reading it is an out-of-bounds access a sanitizer build aborts on, and
   *     on any build it would feed the constructor a garbage vertex count. */
  {
    char *argv3[] = {(char *)"prog", (char *)"--num-vertices"};
    arts_block_dist_t *d = arts_block_dist_init_from_args(2, argv3);
    if (d != NULL) {
      arts_test_fail();
      arts_printf("FAIL: --num-vertices with no value produced a distribution "
                  "(%p) -- the missing value token was read anyway\n",
                  (void *)d);
      arts_block_dist_free(d);
    } else {
      arts_printf("  PASS: trailing --num-vertices left the count unset\n");
    }
  }

  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Two verdicts to merge: what arts_rt saw of the ranks it spawned (their exit
     status reaches nobody else) and what this rank's own checks found. */
  int rc = arts_rt(argc, argv);
  return rc != 0 ? 1 : arts_test_status();
}
