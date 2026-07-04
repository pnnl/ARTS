/* SPDX-License-Identifier: Apache-2.0
 *
 * T299 — `provider` and `regpool_slab_mb` cfg-key parsing (config.c).
 *
 * Property under test
 * -------------------
 * Two new optional networking/memory keys added by the config-surface
 * cleanup:
 *   provider         - CONFIG_STRING, passed through to fi_getinfo's
 *                       prov_name hint by arts_net_init(); unset/empty means
 *                       auto-select (NULL field).
 *   regpool_slab_mb   - CONFIG_UINT, feeds the registered-memory slab pool
 *                       size; default 64 (matching the pre-cfg-key
 *                       ARTS_REGPOOL_SLAB_BYTES_DEFAULT constant it replaced,
 *                       so an unconfigured cfg keeps the historical
 *                       behaviour).
 *
 * This is a config-parser test: it calls arts_config_load() directly against
 * crafted temp cfgs (no runtime started, no fabric brought up, no ports
 * bound).  Self-contained: it sets its own ARTS_CONFIG and clears the
 * inherited `default_ports`/`port_count` env overrides.  Orthogonal to the
 * coherence protocol axis (config.c is protocol-agnostic).
 */

#include "arts.h"
#include "arts/system/config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int write_cfg(char *path_out, size_t path_cap, const char *tag,
                     const char *extra_lines) {
  snprintf(path_out, path_cap, "config_provider_regpool_parse_%s_%ld.cfg",
           tag, (long)getpid());
  FILE *f = fopen(path_out, "w");
  if (!f) {
    return -1;
  }
  fputs("[ARTS]\n"
        "launcher=local\n"
        "node_count=1\n"
        "worker_threads=2\n"
        "route_table_size=14\n",
        f);
  fputs(extra_lines, f);
  (void)fclose(f);
  return 0;
}

static int load_cfg(const char *tag, const char *extra_lines,
                    struct arts_config_s *out) {
  char cfg_path[256];
  if (write_cfg(cfg_path, sizeof(cfg_path), tag, extra_lines) != 0) {
    return -1;
  }
  setenv("ARTS_CONFIG", cfg_path, 1);
  unsetenv("default_ports");
  unsetenv("port_count");
  arts_config_load(out);
  (void)remove(cfg_path);
  return 0;
}

int main(void) {
  int fails = 0;

  /* (A) Explicit provider + regpool_slab_mb parse into the struct verbatim. */
  {
    struct arts_config_s config;
    if (load_cfg("explicit", "provider=verbs\nregpool_slab_mb=128\n",
                 &config) != 0) {
      printf("FAIL config_provider_regpool_parse: cannot write explicit "
             "cfg\n");
      return 1;
    }
    if (!config.provider || strcmp(config.provider, "verbs") != 0) {
      printf("FAIL config_provider_regpool_parse: provider=%s expected "
             "\"verbs\"\n",
             config.provider ? config.provider : "(null)");
      fails++;
    }
    if (config.regpool_slab_mb != 128) {
      printf("FAIL config_provider_regpool_parse: regpool_slab_mb=%u "
             "expected 128\n",
             config.regpool_slab_mb);
      fails++;
    }
    arts_config_destroy(&config);
  }

  /* (B) Unconfigured: provider stays NULL (auto), regpool_slab_mb defaults
   * to 64 (the historical ARTS_REGPOOL_SLAB_BYTES_DEFAULT this key replaced,
   * so an unconfigured cfg's behavior is unchanged from before the key
   * existed). */
  {
    struct arts_config_s config;
    if (load_cfg("default", "", &config) != 0) {
      printf(
          "FAIL config_provider_regpool_parse: cannot write default cfg\n");
      return 1;
    }
    if (config.provider != NULL) {
      printf("FAIL config_provider_regpool_parse: provider=%s expected "
             "unset (auto)\n",
             config.provider);
      fails++;
    }
    if (config.regpool_slab_mb != 64) {
      printf("FAIL config_provider_regpool_parse: regpool_slab_mb=%u "
             "expected default 64\n",
             config.regpool_slab_mb);
      fails++;
    }
    arts_config_destroy(&config);
  }

  if (fails == 0) {
    printf("PASS config_provider_regpool_parse: provider + regpool_slab_mb "
           "parse explicit values and the unconfigured default (NULL, 64)\n");
    return 0;
  }
  return 1;
}
