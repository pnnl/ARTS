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

#include "arts/introspection/json_writer.h"

#include <string.h>

static void json_writer_write_indent(const arts_json_writer_t *writer) {
  for (unsigned i = 0; i < writer->depth * writer->indent_size; i++) {
    (void)fputc(' ', writer->fp);
}
}

static void json_writer_prepare_entry(arts_json_writer_t *writer) {
  if (writer->depth && writer->needComma[writer->depth]) {
    (void)fputs(",\n", writer->fp);
  } else if (writer->depth) {
    (void)fputc('\n', writer->fp);
}

  if (writer->depth) {
    json_writer_write_indent(writer);
}

  if (writer->depth < ARTS_JSON_MAX_DEPTH) {
    writer->needComma[writer->depth] = 1;
}
}

static void json_writer_push(arts_json_writer_t *writer) {
  if (writer->depth + 1 >= ARTS_JSON_MAX_DEPTH) {
    return;
}
  writer->depth++;
  writer->needComma[writer->depth] = 0;
}

static void json_writer_pop(arts_json_writer_t *writer, char closing) {
  if (!writer->depth) {
    return;
}

  int had_entries = writer->needComma[writer->depth];
  writer->needComma[writer->depth] = 0;
  writer->depth--;

  if (had_entries) {
    (void)fputc('\n', writer->fp);
    if (writer->depth) {
      json_writer_write_indent(writer);
}
  }
  (void)fputc(closing, writer->fp);
}

void arts_json_writer_init(arts_json_writer_t *writer, FILE *fp, unsigned indent_size) {
  writer->fp = fp;
  writer->indent_size = indent_size;
  writer->depth = 0;
  memset(writer->needComma, 0, sizeof(writer->needComma));
}

void arts_json_writer_begin_object(arts_json_writer_t *writer, const char *key) {
  if (writer->depth) {
    json_writer_prepare_entry(writer);
}

  if (key) {
    (void)fprintf(writer->fp, "\"%s\": {", key);
  } else {
    (void)fputc('{', writer->fp);
}
  json_writer_push(writer);
}

void arts_json_writer_end_object(arts_json_writer_t *writer) {
  json_writer_pop(writer, '}');
}

void arts_json_writer_begin_array(arts_json_writer_t *writer, const char *key) {
  if (writer->depth) {
    json_writer_prepare_entry(writer);
}

  if (key) {
    (void)fprintf(writer->fp, "\"%s\": [", key);
  } else {
    (void)fputc('[', writer->fp);
}
  json_writer_push(writer);
}

void arts_json_writer_end_array(arts_json_writer_t *writer) {
  json_writer_pop(writer, ']');
}

static void json_writer_write_escaped(const char *value, FILE *fp) {
  (void)fputc('"', fp);
  if (!value) {
    (void)fputc('"', fp);
    return;
  }

  for (const unsigned char *cursor = (const unsigned char *)value; *cursor;
       cursor++) {
    switch (*cursor) {
    case '\\':
      (void)fputs("\\\\", fp);
      break;
    case '"':
      (void)fputs("\\\"", fp);
      break;
    case '\n':
      (void)fputs("\\n", fp);
      break;
    case '\r':
      (void)fputs("\\r", fp);
      break;
    case '\t':
      (void)fputs("\\t", fp);
      break;
    default:
      if (*cursor < 0x20) {
        (void)fprintf(fp, "\\u%04x", *cursor);
      } else {
        (void)fputc(*cursor, fp);
}
    }
  }
  (void)fputc('"', fp);
}

static void json_writer_write_key(arts_json_writer_t *writer, const char *key) {
  json_writer_prepare_entry(writer);
  if (key) {
    json_writer_write_escaped(key, writer->fp);
    (void)fputs(": ", writer->fp);
  }
}

void arts_json_writer_write_u_int64(arts_json_writer_t *writer, const char *key,
                               uint64_t value) {
  json_writer_write_key(writer, key);
  (void)fprintf(writer->fp, "%llu", (unsigned long long)value);
}

void arts_json_writer_write_double(arts_json_writer_t *writer, const char *key,
                               double value) {
  json_writer_write_key(writer, key);
  (void)fprintf(writer->fp, "%.6f", value);
}

void arts_json_writer_write_string(arts_json_writer_t *writer, const char *key,
                               const char *value) {
  json_writer_write_key(writer, key);
  json_writer_write_escaped(value ? value : "", writer->fp);
}

void arts_json_writer_write_null(arts_json_writer_t *writer, const char *key) {
  json_writer_write_key(writer, key);
  (void)fputs("null", writer->fp);
}

void arts_json_writer_write_raw_array(arts_json_writer_t *writer, const char *key,
                                 const char *raw_json) {
  json_writer_write_key(writer, key);
  (void)fputs(raw_json, writer->fp);
}

void arts_json_writer_finish(arts_json_writer_t *writer) {
  while (writer->depth) {
    json_writer_pop(writer, '}');
}
}
