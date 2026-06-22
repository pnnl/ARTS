/* SPDX-License-Identifier: Apache-2.0
 *
 * T250 — arts_json_writer_* well-formedness and string escaping.
 *
 * The JSON writer (libs/src/core/counter/json.c) is a tiny stateful streaming
 * emitter over a FILE* with a depth stack and per-depth comma bookkeeping.  It
 * is pure and single-threaded, and has ZERO existing test coverage.  This test
 * exercises it in isolation: it is header-only against json.c (the only
 * dependency is <string.h>), so it needs no ARTS runtime and no Preamble.h.
 *
 * Outputs are captured into an in-memory FILE* (open_memstream) and checked
 * both for exact bytes (where the format is fully determined) and for parse
 * well-formedness via a small recursive-descent JSON validator embedded below
 * (so the "well-formed" claim is mechanically verified, not eyeballed).
 *
 * Properties covered (census 33-counter.md §json.c "Boundary properties"):
 *   1. Nested object/array are well-formed and correctly comma-separated.
 *   2. Empty object {} and empty array [] format with no spurious
 * newline/comma.
 *   3. String escaping: backslash, double-quote, \n, \r, \t, and control chars
 *      < 0x20 as \u00XX; printable bytes pass through; '/' is NOT escaped.
 *   4. NULL key (value-only emit) and NULL string value ("").
 *   5. write_double uses %.6f.
 *   6. Top-level object with multiple entries: commas between siblings, none
 *      before the first.
 *   7. Depth overflow (> ARTS_JSON_MAX_DEPTH-1 nesting): the writer silently
 *      stops pushing; finish() must still close every brace it opened so the
 *      stream the writer believes it produced stays brace-balanced.  (This
 *      documents the silent-drop behavior, suspected-bug LOW B138.)
 *
 * On any mismatch: prints FAIL ... to stderr and returns 1.  Deterministic.
 */

#include "arts/counter/json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- minimal JSON well-formedness validator ----------------- *
 * Accepts the RFC-8259 grammar subset the writer can emit (objects, arrays,
 * strings, numbers incl. fixed-point, true/false/null) and rejects malformed
 * structure (unbalanced braces, trailing/leading commas, bad escapes).  Used
 * only to certify "well-formed"; value-level checks are done by byte compare.
 */
static const char *jv_value(const char *p); /* fwd */

static const char *jv_ws(const char *p) {
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
    p++;
  }
  return p;
}

static const char *jv_string(const char *p) {
  if (*p != '"') {
    return NULL;
  }
  p++;
  while (*p && *p != '"') {
    if (*p == '\\') {
      p++;
      switch (*p) {
      case '"':
      case '\\':
      case '/':
      case 'b':
      case 'f':
      case 'n':
      case 'r':
      case 't':
        p++;
        break;
      case 'u':
        p++;
        for (int i = 0; i < 4; i++) {
          char c = *p++;
          int hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                    (c >= 'A' && c <= 'F');
          if (!hex) {
            return NULL;
          }
        }
        break;
      default:
        return NULL; /* invalid escape */
      }
    } else if ((unsigned char)*p < 0x20) {
      return NULL; /* raw control char must have been escaped */
    } else {
      p++;
    }
  }
  if (*p != '"') {
    return NULL;
  }
  return p + 1;
}

static const char *jv_number(const char *p) {
  const char *start = p;
  if (*p == '-') {
    p++;
  }
  if (*p < '0' || *p > '9') {
    return NULL;
  }
  while (*p >= '0' && *p <= '9') {
    p++;
  }
  if (*p == '.') {
    p++;
    if (*p < '0' || *p > '9') {
      return NULL;
    }
    while (*p >= '0' && *p <= '9') {
      p++;
    }
  }
  return (p > start) ? p : NULL;
}

static const char *jv_object(const char *p) {
  p++; /* { */
  p = jv_ws(p);
  if (*p == '}') {
    return p + 1;
  }
  for (;;) {
    p = jv_ws(p);
    p = jv_string(p);
    if (!p) {
      return NULL;
    }
    p = jv_ws(p);
    if (*p != ':') {
      return NULL;
    }
    p = jv_ws(p + 1);
    p = jv_value(p);
    if (!p) {
      return NULL;
    }
    p = jv_ws(p);
    if (*p == ',') {
      p++;
      continue;
    }
    if (*p == '}') {
      return p + 1;
    }
    return NULL;
  }
}

static const char *jv_array(const char *p) {
  p++; /* [ */
  p = jv_ws(p);
  if (*p == ']') {
    return p + 1;
  }
  for (;;) {
    p = jv_ws(p);
    p = jv_value(p);
    if (!p) {
      return NULL;
    }
    p = jv_ws(p);
    if (*p == ',') {
      p++;
      continue;
    }
    if (*p == ']') {
      return p + 1;
    }
    return NULL;
  }
}

static const char *jv_value(const char *p) {
  p = jv_ws(p);
  switch (*p) {
  case '{':
    return jv_object(p);
  case '[':
    return jv_array(p);
  case '"':
    return jv_string(p);
  case 't':
    return (strncmp(p, "true", 4) == 0) ? p + 4 : NULL;
  case 'f':
    return (strncmp(p, "false", 5) == 0) ? p + 5 : NULL;
  case 'n':
    return (strncmp(p, "null", 4) == 0) ? p + 4 : NULL;
  default:
    return jv_number(p);
  }
}

static int json_is_well_formed(const char *s) {
  const char *p = jv_value(s);
  if (!p) {
    return 0;
  }
  p = jv_ws(p);
  return *p == '\0';
}

/* ------------------------- capture helper -------------------------------- */
/* Run `build` against a memstream, NUL-terminate, return the captured text in
 * a static buffer (single-threaded test, one capture live at a time). */
static char g_cap[8192];

static void capture(void (*build)(arts_json_writer_t *), unsigned indent) {
  char *buf = NULL;
  size_t len = 0;
  FILE *fp = open_memstream(&buf, &len);
  if (!fp) {
    fprintf(stderr, "FAIL json_writer: open_memstream failed\n");
    exit(1);
  }
  arts_json_writer_t w;
  arts_json_writer_init(&w, fp, indent);
  build(&w);
  arts_json_writer_finish(&w);
  fflush(fp);
  fclose(fp);
  if (len + 1 > sizeof(g_cap)) {
    fprintf(stderr, "FAIL json_writer: capture too large (%zu)\n", len);
    free(buf);
    exit(1);
  }
  memcpy(g_cap, buf, len);
  g_cap[len] = '\0';
  free(buf);
}

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL json_writer: %s\n  captured:\n%s\n", msg, g_cap);  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

/* ----------------------------- builders ---------------------------------- */

static void b_nested(arts_json_writer_t *w) {
  arts_json_writer_begin_object(w, NULL);
  arts_json_writer_write_u_int64(w, "a", 1);
  arts_json_writer_begin_object(w, "obj");
  arts_json_writer_write_u_int64(w, "x", 2);
  arts_json_writer_write_u_int64(w, "y", 3);
  arts_json_writer_end_object(w);
  arts_json_writer_begin_array(w, "arr");
  arts_json_writer_end_array(w); /* empty array */
  arts_json_writer_end_object(w);
}

static void b_empty_obj(arts_json_writer_t *w) {
  arts_json_writer_begin_object(w, NULL);
  arts_json_writer_end_object(w);
}

static void b_escape(arts_json_writer_t *w) {
  arts_json_writer_begin_object(w, NULL);
  /* value containing: backslash, quote, newline, tab, CR, a control char
   * (0x01), a forward slash (must stay raw), and printable text. */
  arts_json_writer_write_string(w, "s", "a\\b\"c\n\td\re/\x01z");
  arts_json_writer_write_string(w, "nullval", NULL); /* NULL → "" */
  arts_json_writer_end_object(w);
}

static void b_double(arts_json_writer_t *w) {
  arts_json_writer_begin_object(w, NULL);
  arts_json_writer_write_double(w, "d", 1.5);
  arts_json_writer_end_object(w);
}

static void b_multi_top(arts_json_writer_t *w) {
  arts_json_writer_begin_object(w, NULL);
  arts_json_writer_write_u_int64(w, "first", 10);
  arts_json_writer_write_u_int64(w, "second", 20);
  arts_json_writer_write_u_int64(w, "third", 30);
  arts_json_writer_end_object(w);
}

/* Open more than ARTS_JSON_MAX_DEPTH objects to drive the silent-drop path. */
static void b_overflow(arts_json_writer_t *w) {
  for (int i = 0; i < ARTS_JSON_MAX_DEPTH + 8; i++) {
    arts_json_writer_begin_object(w, (i == 0) ? NULL : "n");
  }
  arts_json_writer_write_u_int64(w, "leaf", 99);
  /* Do NOT manually end; rely on finish() to close everything the writer
   * actually pushed.  Matches the counter.c usage (close + finish). */
}

int main(void) {
  /* 1. nested obj/array well-formed + exact bytes (indent 2). */
  capture(b_nested, 2);
  CHECK(json_is_well_formed(g_cap), "nested not well-formed");
  {
    const char *expect = "{\n"
                         "  \"a\": 1,\n"
                         "  \"obj\": {\n"
                         "    \"x\": 2,\n"
                         "    \"y\": 3\n"
                         "  },\n"
                         "  \"arr\": []\n"
                         "}";
    CHECK(strcmp(g_cap, expect) == 0, "nested exact bytes mismatch");
  }

  /* 2a. empty object. */
  capture(b_empty_obj, 2);
  CHECK(json_is_well_formed(g_cap), "empty object not well-formed");
  CHECK(strcmp(g_cap, "{}") == 0, "empty object not '{}'");

  /* 3. escaping. */
  capture(b_escape, 2);
  CHECK(json_is_well_formed(g_cap), "escaped output not well-formed");
  {
    /* expected escaped form of "a\\b\"c\n\td\re/\x01z":
     *   \\ \" \n \t \r  then '/' raw  then  */
    const char *needle = "\"s\": \"a\\\\b\\\"c\\n\\td\\re/\\u0001z\"";
    CHECK(strstr(g_cap, needle) != NULL, "string escaping mismatch");
    CHECK(strstr(g_cap, "\"nullval\": \"\"") != NULL,
          "NULL value not emitted as empty string");
    /* '/' must NOT be escaped to '\/' */
    CHECK(strstr(g_cap, "\\/") == NULL, "forward slash was escaped");
  }

  /* 5. double %.6f. */
  capture(b_double, 2);
  CHECK(json_is_well_formed(g_cap), "double output not well-formed");
  CHECK(strstr(g_cap, "\"d\": 1.500000") != NULL, "double not %.6f");

  /* 6. top-level multiple entries: exactly two ',' separators, none leading. */
  capture(b_multi_top, 2);
  CHECK(json_is_well_formed(g_cap), "multi-top not well-formed");
  {
    /* first child must not be preceded by a comma. */
    CHECK(strstr(g_cap, "{\n  \"first\"") != NULL,
          "leading comma before first entry");
    int commas = 0;
    for (const char *p = g_cap; *p; p++) {
      if (*p == ',') {
        commas++;
      }
    }
    CHECK(commas == 2, "expected exactly 2 sibling commas at top level");
  }

  /* 7. depth overflow (exposes B138, census §json.c L99/§suspected-bug 6):
   *    When more than ARTS_JSON_MAX_DEPTH-1 objects are opened,
   * json_writer_push refuses to increment depth but begin_object STILL emits
   * `"key": {`, and prepare_entry STILL inserts a comma between those
   * same-depth siblings. The result is structurally malformed JSON — sequences
   * like `{,` (a brace immediately followed by a comma) and an unbalanced brace
   * count, because finish() only pops the capped depth and leaves the surplus
   * `{` unclosed. A correct writer would either error, drop the over-deep entry
   * entirely, or keep the document brace-balanced.  This assertion documents
   * the defect and is expected to FAIL on the current code; we do NOT relax it.
   */
  capture(b_overflow, 2);
  {
    int opens = 0, closes = 0;
    for (const char *p = g_cap; *p; p++) {
      if (*p == '{') {
        opens++;
      } else if (*p == '}') {
        closes++;
      }
    }
    /* Surface the concrete malformation in the diagnostic. */
    if (opens != closes || strstr(g_cap, "{,") != NULL ||
        !json_is_well_formed(g_cap)) {
      fprintf(stderr,
              "FAIL json_writer: depth-overflow emits MALFORMED JSON "
              "(opens=%d closes=%d, has '{,'=%d, parses=%d) — exposes B138\n"
              "  captured:\n%s\n",
              opens, closes, strstr(g_cap, "{,") != NULL,
              json_is_well_formed(g_cap), g_cap);
      return 1;
    }
  }

  printf("PASS json_writer (nested/empty/escape/double/multi-top/overflow)\n");
  return 0;
}
