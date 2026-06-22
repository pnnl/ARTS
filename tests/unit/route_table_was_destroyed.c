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

/// @file route_table_was_destroyed.c
/// @brief T031 — discriminate destroyed (value NULL + gen>0) vs never-created
///        (no slot / gen==0) via arts_route_table_was_destroyed.
///
/// arts_route_table_was_destroyed is consulted ONLY by the MRSW ownership
/// release guard (it lets a pending acquire-request handler distinguish a
/// post-destroy absent slot — fail the acquire — from a pre-create absent slot
/// — keep deferring).  The gen field is bumped by set_destroyed in every
/// protocol, but the helper is dead outside MRSW; so this test is meaningful
/// only under MRSW builds and self-skips elsewhere.
///
/// This is a white-box test: it drives the route table directly (install a
/// sentinel with a NULL deleter, destroy it, query was_destroyed) rather than a
/// full DB lifecycle, so the three cases are isolated and deterministic.

#include "arts.h"

#if !defined(ARTS_PROTOCOL_MRSW)
#include <stdio.h>
int main(void) {
  printf("SKIP route_table_was_destroyed: MRSW-only (only consumer of "
         "arts_route_table_was_destroyed)\n");
  return 0;
}
#else

#include "arts/gas/route_table.h"

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== route_table_was_destroyed ===\n");
  bool all_pass = true;

  /* A stable backing object the sentinel cb points at.  NULL deleter => the
   * route table never frees it (we own it), so detach/destroy is leak-free. */
  static int sentinel_obj = 7;

  /* Case "never created": a fresh key with no slot (or a slot whose gen has
   * never been bumped) must report was_destroyed == false. */
  arts_guid_t fresh = arts_guid_reserve(ARTS_GUID_DB, arts_get_current_rank());
  if (arts_route_table_was_destroyed(fresh)) {
    arts_printf("  FAIL: never-created key reported destroyed\n");
    all_pass = false;
  } else {
    arts_printf("  PASS: never-created key not reported destroyed\n");
  }

  /* Case "live": install the sentinel; value != NULL => not destroyed. */
  arts_guid_t key = arts_guid_reserve(ARTS_GUID_DB, arts_get_current_rank());
  arts_route_table_install_with_deleter(&sentinel_obj, key, NULL);
  if (arts_route_table_was_destroyed(key)) {
    arts_printf("  FAIL: live (just-installed) key reported destroyed\n");
    all_pass = false;
  } else {
    arts_printf("  PASS: live key not reported destroyed\n");
  }

  /* Case "destroyed": set_destroyed NULLs the slot AND bumps gen>0; now the
   * helper must report true — value==NULL AND gen>0 distinguishes this from
   * the never-created case above. */
  bool detached = arts_route_table_set_destroyed(key);
  if (!detached) {
    arts_printf("  FAIL: set_destroyed did not detach the live cb\n");
    all_pass = false;
  } else {
    arts_printf("  PASS: set_destroyed detached the live cb (single-flight)\n");
  }
  if (!arts_route_table_was_destroyed(key)) {
    arts_printf("  FAIL: destroyed key not reported destroyed "
                "(value NULL + gen>0 expected)\n");
    all_pass = false;
  } else {
    arts_printf("  PASS: destroyed key reported destroyed\n");
  }

  /* set_destroyed is idempotent / single-flight: a second call detaches
   * nothing (already NULL) but was_destroyed stays true. */
  bool detached2 = arts_route_table_set_destroyed(key);
  if (detached2) {
    arts_printf("  FAIL: second set_destroyed claimed to detach a NULL slot\n");
    all_pass = false;
  } else {
    arts_printf("  PASS: second set_destroyed is a no-op (single-flight)\n");
  }
  if (!arts_route_table_was_destroyed(key)) {
    arts_printf("  FAIL: key stopped reporting destroyed after idempotent "
                "re-destroy\n");
    all_pass = false;
  } else {
    arts_printf(
        "  PASS: destroyed state stable across idempotent re-destroy\n");
  }

  arts_printf("=== route_table_was_destroyed: %s ===\n",
              all_pass ? "ALL PASSED" : "FAILED");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRSW */
