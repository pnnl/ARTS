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
#ifndef ARTS_JSON_WRITER_H
#define ARTS_JSON_WRITER_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARTS_JSON_MAX_DEPTH 32

typedef struct {
  FILE *fp;
  unsigned indentSize;
  unsigned depth;
  uint8_t needComma[ARTS_JSON_MAX_DEPTH];
} artsJsonWriter;

void artsJsonWriterInit(artsJsonWriter *writer, FILE *fp, unsigned indentSize);
void artsJsonWriterBeginObject(artsJsonWriter *writer, const char *key);
void artsJsonWriterEndObject(artsJsonWriter *writer);
void artsJsonWriterBeginArray(artsJsonWriter *writer, const char *key);
void artsJsonWriterEndArray(artsJsonWriter *writer);
void artsJsonWriterWriteUInt64(artsJsonWriter *writer, const char *key,
                               uint64_t value);
void artsJsonWriterWriteDouble(artsJsonWriter *writer, const char *key,
                               double value);
void artsJsonWriterWriteString(artsJsonWriter *writer, const char *key,
                               const char *value);
void artsJsonWriterWriteNull(artsJsonWriter *writer, const char *key);
void artsJsonWriterWriteRawArray(artsJsonWriter *writer, const char *key,
                                 const char *rawJson);
void artsJsonWriterFinish(artsJsonWriter *writer);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_JSON_WRITER_H */
