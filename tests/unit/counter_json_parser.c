/* SPDX-License-Identifier: Apache-2.0
 *
 * T251 — counter.c cluster-aggregation JSON parser: malformed / truncated
 *        n{n}.json must not read or write out of bounds.
 *
 * Target subsystem: the *file-local* JSON parser used by the master node when
 * it aggregates per-node counter files (`arts_counter_write_cluster` →
 * `arts_read_node_counter_file` → `arts_json_find_key` /
 * `arts_json_find_object_end` / `arts_json_parse_u_int64` /
 * `arts_json_parse_capture_history`).  Those are all `static` in
 * libs/src/core/counter/counter.c, so this TU compiles the real source in-place
 * via  #include "counter.c"  (the same idiom used by tests/unit/*_gpu.cu which
 * #include edt.c).  We supply libc-backed shims for the runtime symbols
 * counter.c links against (arts_malloc/free/calloc, the atomics, array-list,
 * transport) and the two runtime globals/objects (arts_node_info,
 * arts_global_rank_id); the JSON *writer* is the real json.c, linked alongside.
 * None of the stubbed symbols are reached on the parse path under test.
 *
 * Property under test (well-formed inputs):
 *   - A correct n{n}.json with a CLUSTER-level counter object parses its
 *     "value" and "captureHistory" exactly.
 *
 * Defect targeted (B134 / suspected-bug HIGH, census 33-counter.md §1):
 *   In arts_read_node_counter_file, after locating a CLUSTER counter's key the
 *   code does
 *       counter_end = arts_json_find_object_end(counter_obj);
 *       counter_len = counter_end - counter_obj;          // NO NULL CHECK
 *       counter_json = arts_malloc(counter_len + 1);
 *       memcpy(counter_json, counter_obj, counter_len);
 *   arts_json_find_object_end returns NULL whenever *counter_obj != '{'.  That
 *   happens for a perfectly plausible malformed/partial file in which the
 *   counter key's value is a scalar rather than an object, e.g.
 *       "counters": { "<CLUSTER_NAME>": 12345 }
 *   or where the counter NAME is matched inside a string value rather than as a
 *   key.  Then counter_len = (NULL - counter_obj) is an enormous size_t, and
 * the subsequent arts_malloc(huge+1) + memcpy(...,huge) is a wild-pointer
 *   read/overflow.  This file is master-polled while a peer is still writing
 * it, so a truncated object is a realistic input.
 *
 * Expectation: a robust parser rejects/skips the malformed counter and returns
 * without OOB.  Under ASan the current code is expected to ABORT (heap overflow
 * / allocation-size-too-big) on the malformed case — that is the bug, reported
 * with exposes_runtime_bug=true.  We do NOT weaken the test to hide it.
 */

#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Pull the real counter.c in (gives us the static parser + the static
 *      const arts_counter_{names,mode_array,level_array} from Preamble.h, and
 *      the file-local arts_cluster_counter_data_t typedef).
 * -------------------*/
#include "counter.c"

/* ---- Runtime symbols counter.c references but that the parse path never
 *      actually invokes; defined here only so the TU links.
 * ------------------*/
void *arts_malloc(size_t n) { return malloc(n); }
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void arts_free(void *p) { free(p); }
void arts_abort(uint8_t code) {
  (void)code;
  abort();
}

uint64_t arts_get_time_stamp(void) { return 0; }
uint64_t arts_atomic_cswap_u64(volatile uint64_t *d, uint64_t e, uint64_t s) {
  (void)d;
  (void)e;
  (void)s;
  return 0;
}
uint64_t arts_atomic_swap_u64(volatile uint64_t *d, uint64_t s) {
  (void)d;
  (void)s;
  return 0;
}
uint64_t arts_atomic_fetch_add_u64(volatile uint64_t *d, uint64_t s) {
  (void)d;
  (void)s;
  return 0;
}
uint64_t arts_atomic_fetch_sub_u64(volatile uint64_t *d, uint64_t s) {
  (void)d;
  (void)s;
  return 0;
}
uint64_t arts_atomic_read_u64(const volatile uint64_t *d) {
  (void)d;
  return 0;
}
uint64_t arts_push_to_array_list(arts_array_list_t *a, void *e) {
  (void)a;
  (void)e;
  return 0;
}
void arts_array_list_iter_init(arts_array_list_iterator_t *it,
                               arts_array_list_t *a) {
  (void)it;
  (void)a;
}
void *arts_array_list_next(arts_array_list_iterator_t *it) {
  (void)it;
  return NULL;
}
bool arts_array_list_has_next(arts_array_list_iterator_t *it) {
  (void)it;
  return false;
}
void arts_transport_send_async(int rank, char *buf, unsigned int n) {
  (void)rank;
  (void)buf;
  (void)n;
}

/* Runtime globals/objects referenced by counter.c (parse path leaves them
 * untouched, but they must exist). */
unsigned int arts_global_rank_id = 0;
unsigned int arts_global_master_rank_id = 0;
struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

/* ------------------------------------------------------------------------- */

/* Pick the first counter that the compiled-in counters.cfg marks CLUSTER-level
 * and non-OFF.  arts_read_node_counter_file only parses such counters, so the
 * crafted JSON must name one of them for the parse path to be entered. */
static int first_cluster_counter(void) {
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (arts_counter_mode_array[i] != ARTS_COUNTER_MODE_OFF &&
        arts_counter_level_array[i] == ARTS_COUNTER_LEVEL_CLUSTER) {
      return (int)i;
    }
  }
  return -1;
}

static void write_file(const char *path, const char *contents) {
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "FAIL counter_json_parser: cannot open %s for write\n",
            path);
    exit(1);
  }
  (void)fputs(contents, f);
  (void)fclose(f);
}

/* The fixtures live in the test's working directory, so they have to be taken
 * away again: a leaked tree would accumulate one directory per run. */
static void cleanup_tmpdir(const char *dir) {
  static const char *names[] = {"good.json", "bad_scalar.json", "trunc.json"};
  char path[1280];
  for (unsigned int i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
    (void)snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
    (void)remove(path);
  }
  (void)rmdir(dir);
}

int main(void) {
  int ci = first_cluster_counter();
  if (ci < 0) {
    /* No CLUSTER counter enabled in this counters.cfg → the malformed-object
     * parse branch is unreachable in this build.  Report PASS-as-skip so the
     * suite stays green where the path simply does not exist. */
    printf(
        "PASS counter_json_parser (skip: no CLUSTER-level counter in cfg)\n");
    return 0;
  }
  const char *cname = arts_counter_names[ci];

  char tmpl[] = "arts_counter_json_XXXXXX";
  if (!mkdtemp(tmpl)) {
    fprintf(stderr, "FAIL counter_json_parser: mkdtemp failed\n");
    return 1;
  }
  char path[1280];

  /* node_data array indexed by counter type, as the real caller allocates. */
  arts_cluster_counter_data_t *data = (arts_cluster_counter_data_t *)calloc(
      NUM_COUNTER_TYPES, sizeof(arts_cluster_counter_data_t));

  /* ---- (A) well-formed file: value + captureHistory parse exactly -------- */
  {
    char good[2048];
    (void)snprintf(good, sizeof(good),
                   "{\n"
                   "  \"counters\": {\n"
                   "    \"%s\": {\n"
                   "      \"captureMode\": \"PERIODIC\",\n"
                   "      \"captureLevel\": \"CLUSTER\",\n"
                   "      \"value\": 4242,\n"
                   "      \"captureHistory\": [[1,10],[2,20],[3,30]]\n"
                   "    }\n"
                   "  }\n"
                   "}\n",
                   cname);
    (void)snprintf(path, sizeof(path), "%s/good.json", tmpl);
    write_file(path, good);

    memset(data, 0, NUM_COUNTER_TYPES * sizeof(*data));
    bool ok = arts_read_node_counter_file(path, data);
    if (!ok) {
      fprintf(stderr, "FAIL counter_json_parser: well-formed file rejected\n");
      return 1;
    }
    if (data[ci].value != 4242) {
      fprintf(stderr, "FAIL counter_json_parser: value=%llu expected 4242\n",
              (unsigned long long)data[ci].value);
      return 1;
    }
    if (data[ci].captureCount != 3 || data[ci].captureEpochs[0] != 1 ||
        data[ci].captureValues[0] != 10 || data[ci].captureEpochs[2] != 3 ||
        data[ci].captureValues[2] != 30) {
      fprintf(
          stderr,
          "FAIL counter_json_parser: captureHistory mis-parsed (count=%llu)\n",
          (unsigned long long)data[ci].captureCount);
      return 1;
    }
    if (data[ci].captureEpochs)
      free(data[ci].captureEpochs);
    if (data[ci].captureValues)
      free(data[ci].captureValues);
  }

  /* ---- (B) malformed file: CLUSTER counter's value is a scalar, not an
   *         object.  arts_json_find_object_end(counter_obj) sees *p != '{' and
   *         returns NULL → counter_end - counter_obj is a wild pointer
   *         subtraction → arts_malloc(huge)+memcpy OOB.  A correct parser
   *         returns without OOB; the current code is expected to trip ASan. --
   */
  {
    char bad[1024];
    (void)snprintf(bad, sizeof(bad),
                   "{\n"
                   "  \"counters\": {\n"
                   "    \"%s\": 12345\n"
                   "  }\n"
                   "}\n",
                   cname);
    (void)snprintf(path, sizeof(path), "%s/bad_scalar.json", tmpl);
    write_file(path, bad);

    memset(data, 0, NUM_COUNTER_TYPES * sizeof(*data));
    /* If this returns at all (no crash/OOB), a robust parser must NOT have
     * fabricated a bogus value from the wild copy.  Reaching here is itself the
     * pass for the "no OOB" property. */
    (void)arts_read_node_counter_file(path, data);
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (data[i].captureEpochs)
        free(data[i].captureEpochs);
      if (data[i].captureValues)
        free(data[i].captureValues);
    }
  }

  /* ---- (C) truncated file: object opened but never closed (peer still
   *         writing).  find_object_end walks to the trailing NUL and returns a
   *         bounded pointer (counter_end <= end-of-buffer), so this must parse
   *         without OOB. -----------------------------------------------------
   */
  {
    char trunc[1024];
    (void)snprintf(trunc, sizeof(trunc),
                   "{\n"
                   "  \"counters\": {\n"
                   "    \"%s\": {\n"
                   "      \"captureMode\": \"PERIODIC\",\n"
                   "      \"value\": 77", /* abrupt EOF mid-object */
                   cname);
    (void)snprintf(path, sizeof(path), "%s/trunc.json", tmpl);
    write_file(path, trunc);

    memset(data, 0, NUM_COUNTER_TYPES * sizeof(*data));
    (void)arts_read_node_counter_file(path, data);
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (data[i].captureEpochs)
        free(data[i].captureEpochs);
      if (data[i].captureValues)
        free(data[i].captureValues);
    }
  }

  free(data);
  cleanup_tmpdir(tmpl);
  printf("PASS counter_json_parser (well-formed parse + malformed/truncated "
         "no-OOB, counter=%s)\n",
         cname);
  return 0;
}
