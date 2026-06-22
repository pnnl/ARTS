/* SPDX-License-Identifier: Apache-2.0
 *
 * T297 — arts_edt_hint_t.edt_id profiling -> object-counter table.
 *
 * Target: the profiling identifier arts_edt_hint_t.edt_id, which the runtime
 * stores into edt->arts_id and, on EDT execution, records via
 * arts_object_record_edt(edt->arts_id, ...) into the per-thread object-counter
 * EDT table (libs/src/core/scheduler.c).  At shutdown the per-node tables are
 * merged and written to <counter_folder>/object_n<rank>.json as the
 * "edt_objects" array (libs/src/core/counter/object_counter.c).  edt_id is
 * never set by any existing test, so this propagation path is untested.
 *
 * Correct behavior pinned:
 *   - EDTs created with distinct non-zero edt_id values that actually run land
 *     as entries in the object-counter EDT table keyed by that id, so the
 *     post-shutdown object_n0.json contains each edt_id in its "edt_objects"
 *     array (an edt_id == 0 is the "disabled" sentinel and is never recorded).
 *
 * The test sets edt_id on N worker EDTs, runs them under a finish scope, and
 * after arts_rt returns it parses <counter_folder>/object_n0.json and asserts
 * every chosen edt_id appears.  This mirrors counter_smoke.c's post-shutdown
 * JSON-file assertion.
 *
 * Config-specific (full_counters): the object-counter EDT table is compiled in
 * only when the build's ARTS_COUNTER_CONFIG enables an OBJ_* EDT counter (the
 * full_counters profile).  The header-derived ARTS_OBJECT_EDT_TABLE_ENABLED
 * macro gates the real body; in any other counter build the file is a clean
 * SKIP that still links against libarts.  Mirrors the protocol/GPU self-skip
 * idiom and arts_id.c's use of the same macro.
 *
 * exposes_runtime_bug = false (pins edt_id -> object-counter propagation).
 */
#include "arts.h"
#include "arts/counter/object_counter.h" /* ARTS_OBJECT_EDT_TABLE_ENABLED */

#if !ARTS_OBJECT_EDT_TABLE_ENABLED

#include <stdio.h>
int main(void) {
  printf("SKIP api_edt_id_profiling: requires an OBJ EDT counter build "
         "(full_counters)\n");
  return 0;
}

#else /* ARTS_OBJECT_EDT_TABLE_ENABLED — real body */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_EDTS 8u
#define EDT_ID_BASE 0xE17A0000ull /* distinct, non-zero, easy to spot */

/* Worker EDT: pure profiling carrier; its edt_id (set via the hint) is recorded
 * into the object-counter table when it runs. */
void worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== api_edt_id_profiling ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int i = 0; i < NUM_EDTS; i++) {
    arts_edt_hint_t h = ARTS_EDT_HINT_DEFAULTS;
    h.rank = 0;
    h.finish_event = fe;
    h.edt_id = EDT_ID_BASE + i;
    arts_edt_create(worker_edt, 0, NULL, 0, &h);
  }
  arts_event_wait(fe);
  arts_shutdown();
}

/* Parse object_n0.json and confirm every chosen edt_id appears as an arts_id.
 * Returns 0 if all found, nonzero otherwise. */
static int verify_object_json(void) {
  /* Default counter_folder is "./counters" (configs set it); object JSON is
   * <folder>/object_n0.json. */
  const char *candidates[] = {"./counters/object_n0.json",
                              "counters/object_n0.json"};
  FILE *fp = NULL;
  for (unsigned int c = 0; c < sizeof(candidates) / sizeof(candidates[0]);
       c++) {
    fp = fopen(candidates[c], "r");
    if (fp) {
      break;
    }
  }
  if (!fp) {
    fprintf(stderr, "FAIL api_edt_id_profiling: object_n0.json not found "
                    "(object counters not written)\n");
    return 1;
  }

  /* Slurp the file. */
  fseek(fp, 0, SEEK_END);
  long sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (sz <= 0) {
    fclose(fp);
    fprintf(stderr, "FAIL api_edt_id_profiling: object_n0.json empty\n");
    return 1;
  }
  char *buf = (char *)malloc((size_t)sz + 1);
  if (!buf) {
    fclose(fp);
    return 1;
  }
  size_t rd = fread(buf, 1, (size_t)sz, fp);
  buf[rd] = '\0';
  fclose(fp);

  int missing = 0;
  for (unsigned int i = 0; i < NUM_EDTS; i++) {
    char needle[64];
    (void)snprintf(needle, sizeof(needle), "%llu",
                   (unsigned long long)(EDT_ID_BASE + i));
    if (strstr(buf, needle) == NULL) {
      fprintf(stderr,
              "FAIL api_edt_id_profiling: edt_id %s not present in "
              "object_n0.json\n",
              needle);
      missing = 1;
    }
  }
  free(buf);
  if (!missing) {
    printf("PASS api_edt_id_profiling: all %u edt_id values recorded in "
           "object_n0.json\n",
           NUM_EDTS);
  }
  return missing;
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return verify_object_json();
}

#endif /* ARTS_OBJECT_EDT_TABLE_ENABLED */
