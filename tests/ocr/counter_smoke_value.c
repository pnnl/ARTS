/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

/// @file counter_smoke_value.c
/// @brief Strengthens the existence-only counter smoke test (counter_smoke.c)
///        by asserting an actual counter VALUE in n0.json, exercising the
///        increment path (arts_counter_increment_by) end-to-end through node
///        reduction + JSON write.
///
/// Under the default counters.cfg, NUM_EDT_CREATE is PERIODIC,CLUSTER, so the
/// per-node JSON (n0.json) carries a "NUM_EDT_CREATE" object with a numeric
/// "value" reduced (SUM) across this node's worker threads.  We create a known
/// number N of EDTs; the recorded NUM_EDT_CREATE value must be >= N (the
/// runtime also creates internal EDTs, so it is a lower bound, not equality).
///
/// This is config-specific on the *counter* axis: it requires a build whose
/// counter config enables NUM_EDT_CREATE at CLUSTER/NODE level (the default
/// counters.cfg and full_counters.cfg both do).  If the counter is absent from
/// n0.json (disabled in this build's counter config) the test SKIPs cleanly
/// instead of asserting a value that cannot exist.  Config-agnostic across
/// coherence protocols.

#include "arts.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_EDTS 64

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

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (int i = 0; i < N_EDTS; i++) {
    arts_edt_create(worker_edt, 0, NULL, 0,
                    &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  }
  arts_event_wait(fe);
  arts_shutdown();
}

/* Locate counter object `name` inside the JSON, then read its "value" field.
   Returns true and sets *out on success; false if the counter or value is
   missing. */
static bool read_counter_value(const char *json, const char *name,
                               uint64_t *out) {
  char key[128];
  (void)snprintf(key, sizeof(key), "\"%s\"", name);
  const char *obj = strstr(json, key);
  if (!obj) {
    return false;
  }
  /* The "value" key inside this counter's object is the first "value" after
     the counter name.  Counter objects are small and self-contained, so the
     nearest "value" belongs to this counter. */
  const char *v = strstr(obj, "\"value\"");
  if (!v) {
    return false;
  }
  v = strchr(v + strlen("\"value\""), ':');
  if (!v) {
    return false;
  }
  v++;
  while (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r') {
    v++;
  }
  char *end = NULL;
  uint64_t val = strtoull(v, &end, 10);
  if (end == v) {
    return false;
  }
  *out = val;
  return true;
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);

  FILE *fp = fopen("./counters/n0.json", "r");
  if (!fp) {
    printf("FAIL counter_smoke_value: ./counters/n0.json not found\n");
    return 1;
  }
  (void)fseek(fp, 0, SEEK_END);
  long sz = ftell(fp);
  (void)fseek(fp, 0, SEEK_SET);
  if (sz <= 0) {
    (void)fclose(fp);
    printf("FAIL counter_smoke_value: n0.json empty\n");
    return 1;
  }
  char *buf = (char *)malloc((size_t)sz + 1);
  size_t got = fread(buf, 1, (size_t)sz, fp);
  (void)fclose(fp);
  buf[got] = '\0';

  uint64_t value = 0;
  bool found = read_counter_value(buf, "NUM_EDT_CREATE", &value);
  free(buf);

  if (!found) {
    /* NUM_EDT_CREATE disabled in this build's counter config. */
    printf("SKIP counter_smoke_value: NUM_EDT_CREATE not present in n0.json\n");
    return 0;
  }

  if (value < (uint64_t)N_EDTS) {
    printf("FAIL counter_smoke_value: NUM_EDT_CREATE value %llu < expected "
           ">= %d\n",
           (unsigned long long)value, N_EDTS);
    return 1;
  }

  printf("PASS counter_smoke_value: NUM_EDT_CREATE value %llu >= %d\n",
         (unsigned long long)value, N_EDTS);
  return 0;
}
