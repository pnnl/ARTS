/* SPDX-License-Identifier: Apache-2.0
 *
 * T217 — handle_cxl_db_allocation_strategy (config.c, ARTS_USE_CXL only).
 *
 * Property: the CXL DB allocation-strategy handler resolves:
 *   value="round_robin"                 -> strategy ROUND_ROBIN
 *   value="static" + cxl_db_allocation_device=3 -> strategy STATIC, device 3
 *   value=NULL                          -> strategy STATIC, device 0 (default)
 *   any other value                     -> strategy STATIC (+ device lookup)
 * The device index for the STATIC path is read from a second config key via
 * config_lookup, so the test supplies a vars list with
 * cxl_db_allocation_device.
 *
 * BUILD NOTE (needs_full_build=true): handle_cxl_db_allocation_strategy is
 * compiled only under -DARTS_USE_CXL, and config.c's include of
 * runtime_state.h -> cxl/deque.h then requires the rapid API (GLOBAL_MALLOC_DEV
 * etc.), which is unavailable to a standalone gcc.  This test therefore builds
 * ONLY inside an ARTS_USE_CXL CMake configuration (rapid includes on the path).
 * Outside a CXL build it compiles to a self-skipping no-op so the file is still
 * registerable everywhere.
 */

#ifndef ARTS_USE_CXL

#include <stdio.h>
int main(void) {
  printf("SKIP config_cxl_alloc_strategy: requires ARTS_USE_CXL build\n");
  return 0;
}

#else /* ARTS_USE_CXL */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

/* Build a one-node vars list "cxl_db_allocation_device=<dev>". */
static struct arts_config_variable_s *dev_vars(const char *dev_str) {
  unsigned int size = (unsigned int)strlen(dev_str);
  struct arts_config_variable_s *v =
      (struct arts_config_variable_s *)arts_malloc(
          sizeof(struct arts_config_variable_s) + size + 1);
  v->size = size;
  v->next = NULL;
  strncpy(v->variable, "cxl_db_allocation_device", 254);
  v->variable[254] = '\0';
  memcpy(v->value, dev_str, size + 1);
  return v;
}

int main(void) {
  /* round_robin. */
  {
    struct arts_config_s c;
    memset(&c, 0, sizeof(c));
    struct arts_config_variable_s *vars = NULL;
    handle_cxl_db_allocation_strategy(&c, "round_robin", &vars);
    if (c.cxl_db_allocation_strategy != ARTS_CXL_DB_ALLOC_ROUND_ROBIN) {
      fprintf(stderr, "FAIL cxl: round_robin -> strategy %d\n",
              (int)c.cxl_db_allocation_strategy);
      fails++;
    }
    config_free_variables(vars);
  }

  /* static + device=3. */
  {
    struct arts_config_s c;
    memset(&c, 0, sizeof(c));
    struct arts_config_variable_s *vars = dev_vars("3");
    handle_cxl_db_allocation_strategy(&c, "static", &vars);
    if (c.cxl_db_allocation_strategy != ARTS_CXL_DB_ALLOC_STATIC ||
        c.cxl_db_allocation_device != 3) {
      fprintf(stderr, "FAIL cxl: static dev3 -> strategy %d dev %u\n",
              (int)c.cxl_db_allocation_strategy, c.cxl_db_allocation_device);
      fails++;
    }
    config_free_variables(vars);
  }

  /* NULL value -> STATIC, device 0. */
  {
    struct arts_config_s c;
    memset(&c, 0, sizeof(c));
    struct arts_config_variable_s *vars = NULL;
    handle_cxl_db_allocation_strategy(&c, NULL, &vars);
    if (c.cxl_db_allocation_strategy != ARTS_CXL_DB_ALLOC_STATIC ||
        c.cxl_db_allocation_device != 0) {
      fprintf(stderr, "FAIL cxl: NULL -> strategy %d dev %u (want STATIC,0)\n",
              (int)c.cxl_db_allocation_strategy, c.cxl_db_allocation_device);
      fails++;
    }
    config_free_variables(vars);
  }

  if (fails) {
    return 1;
  }
  printf("PASS config_cxl_alloc_strategy: round_robin / static+dev3 / "
         "NULL->static,0\n");
  return 0;
}

#endif /* ARTS_USE_CXL */
