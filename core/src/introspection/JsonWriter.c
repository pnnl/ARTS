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

#include "arts/introspection/JsonWriter.h"

#include <string.h>

static void jsonWriterWriteIndent(const artsJsonWriter *writer) {
  for (unsigned i = 0; i < writer->depth * writer->indentSize; i++)
    fputc(' ', writer->fp);
}

static void jsonWriterPrepareEntry(artsJsonWriter *writer) {
  if (writer->depth && writer->needComma[writer->depth])
    fputs(",\n", writer->fp);
  else if (writer->depth)
    fputc('\n', writer->fp);

  if (writer->depth)
    jsonWriterWriteIndent(writer);

  if (writer->depth < ARTS_JSON_MAX_DEPTH)
    writer->needComma[writer->depth] = 1;
}

static void jsonWriterPush(artsJsonWriter *writer) {
  if (writer->depth + 1 >= ARTS_JSON_MAX_DEPTH)
    return;
  writer->depth++;
  writer->needComma[writer->depth] = 0;
}

static void jsonWriterPop(artsJsonWriter *writer, char closing) {
  if (!writer->depth)
    return;

  int hadEntries = writer->needComma[writer->depth];
  writer->needComma[writer->depth] = 0;
  writer->depth--;

  if (hadEntries) {
    fputc('\n', writer->fp);
    if (writer->depth)
      jsonWriterWriteIndent(writer);
  }
  fputc(closing, writer->fp);
}

void artsJsonWriterInit(artsJsonWriter *writer, FILE *fp, unsigned indentSize) {
  writer->fp = fp;
  writer->indentSize = indentSize;
  writer->depth = 0;
  memset(writer->needComma, 0, sizeof(writer->needComma));
}

void artsJsonWriterBeginObject(artsJsonWriter *writer, const char *key) {
  if (writer->depth)
    jsonWriterPrepareEntry(writer);

  if (key)
    fprintf(writer->fp, "\"%s\": {", key);
  else
    fputc('{', writer->fp);
  jsonWriterPush(writer);
}

void artsJsonWriterEndObject(artsJsonWriter *writer) {
  jsonWriterPop(writer, '}');
}

void artsJsonWriterBeginArray(artsJsonWriter *writer, const char *key) {
  if (writer->depth)
    jsonWriterPrepareEntry(writer);

  if (key)
    fprintf(writer->fp, "\"%s\": [", key);
  else
    fputc('[', writer->fp);
  jsonWriterPush(writer);
}

void artsJsonWriterEndArray(artsJsonWriter *writer) {
  jsonWriterPop(writer, ']');
}

static void jsonWriterWriteEscaped(const char *value, FILE *fp) {
  fputc('"', fp);
  if (!value) {
    fputc('"', fp);
    return;
  }

  for (const unsigned char *cursor = (const unsigned char *)value; *cursor;
       cursor++) {
    switch (*cursor) {
    case '\\':
      fputs("\\\\", fp);
      break;
    case '"':
      fputs("\\\"", fp);
      break;
    case '\n':
      fputs("\\n", fp);
      break;
    case '\r':
      fputs("\\r", fp);
      break;
    case '\t':
      fputs("\\t", fp);
      break;
    default:
      if (*cursor < 0x20)
        fprintf(fp, "\\u%04x", *cursor);
      else
        fputc(*cursor, fp);
    }
  }
  fputc('"', fp);
}

static void jsonWriterWriteKey(artsJsonWriter *writer, const char *key) {
  jsonWriterPrepareEntry(writer);
  if (key) {
    jsonWriterWriteEscaped(key, writer->fp);
    fputs(": ", writer->fp);
  }
}

void artsJsonWriterWriteUInt64(artsJsonWriter *writer, const char *key,
                               uint64_t value) {
  jsonWriterWriteKey(writer, key);
  fprintf(writer->fp, "%llu", (unsigned long long)value);
}

void artsJsonWriterWriteDouble(artsJsonWriter *writer, const char *key,
                               double value) {
  jsonWriterWriteKey(writer, key);
  fprintf(writer->fp, "%.6f", value);
}

void artsJsonWriterWriteString(artsJsonWriter *writer, const char *key,
                               const char *value) {
  jsonWriterWriteKey(writer, key);
  jsonWriterWriteEscaped(value ? value : "", writer->fp);
}

void artsJsonWriterWriteNull(artsJsonWriter *writer, const char *key) {
  jsonWriterWriteKey(writer, key);
  fputs("null", writer->fp);
}

void artsJsonWriterFinish(artsJsonWriter *writer) {
  while (writer->depth)
    jsonWriterPop(writer, '}');
}
