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
#ifndef ARTS_SYSTEM_PRINT_H
#define ARTS_SYSTEM_PRINT_H

#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>

#include "arts/runtime/Globals.h"

#ifdef __cplusplus
extern "C" {
#endif

/// ARTS atomic print helper
static inline void artsAtomicPrint(const char *format, ...) {
  char buffer[4096];
  va_list args;
  int len;

  // Format the entire string
  va_start(args, format);
  len = vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  if (len > 0 && len < (int)sizeof(buffer))
    write(STDERR_FILENO, buffer, len);
  else if (len >= (int)sizeof(buffer))
    write(STDERR_FILENO, buffer, sizeof(buffer) - 1);
}

/// ARTS debug stream accessor
#define artsStream(format, ...)                                                \
  artsAtomicPrint("[%u:%u] " format, artsGlobalRankId, artsThreadInfo.groupId, \
                  ##__VA_ARGS__)

/// ARTS print stream accessor
#define artsPrintStream(format, ...)                                           \
  artsAtomicPrint("[%u:%u] " format, artsGlobalRankId, artsThreadInfo.groupId, \
                  ##__VA_ARGS__)

/// ARTS formatted print stream with rank
#define ARTS_PRINT_FORMATTED(format, ...)                                      \
  artsAtomicPrint("[%u:%u] " format "\n", artsGlobalRankId,                    \
                  artsThreadInfo.groupId, ##__VA_ARGS__)

#define ARTS_PRINT_TYPE(format, ...)                                           \
  artsAtomicPrint("[%u:%u] " format "\n", artsGlobalRankId,                    \
                  artsThreadInfo.groupId, ##__VA_ARGS__)

#define ARTS_PRINT_ONCE(format, ...)                                           \
  if (artsGlobalIWillPrint) {                                                  \
    artsAtomicPrint("[%u:%u] " format "\n", artsGlobalRankId,                  \
                    artsThreadInfo.groupId, ##__VA_ARGS__);                    \
  }

#define ARTS_PRINT_MASTER(format, ...)                                         \
  if (artsGlobalRankId == artsGlobalMasterRankId) {                            \
    artsAtomicPrint("[%u:%u] " format "\n", artsGlobalRankId,                  \
                    artsThreadInfo.groupId, ##__VA_ARGS__);                    \
  }

/// ARTS print
#define ARTS_PRINT(format, ...)                                                \
  artsAtomicPrint("[%u:%u] " format "\n", artsGlobalRankId,                    \
                  artsThreadInfo.groupId, ##__VA_ARGS__)

/// ARTS print that outputs only once per rank
#define ARTS_PRINT_ONCE(format, ...)                                           \
  if (artsGlobalIWillPrint) {                                                  \
    artsAtomicPrint("[%u:%u] " format "\n", artsGlobalRankId,                  \
                    artsThreadInfo.groupId, ##__VA_ARGS__);                    \
  }

/// ARTS print that forces output from master thread only
#define ARTS_PRINT_MASTER(format, ...)                                         \
  if (artsGlobalRankId == artsGlobalMasterRankId) {                            \
    artsAtomicPrint("[%u:%u] " format "\n", artsGlobalRankId,                  \
                    artsThreadInfo.groupId, ##__VA_ARGS__);                    \
  }

/// ARTS debug
#ifdef ARTS_DEBUG_ENABLED
#define ARTS_DEBUG(format, ...)                                                \
  artsAtomicPrint("[%u:%u] [DEBUG] " format "\n", artsGlobalRankId,            \
                  artsThreadInfo.groupId, ##__VA_ARGS__)
#define ARTS_DEBUG_ONCE(format, ...)                                           \
  if (artsGlobalIWillPrint) {                                                  \
    artsAtomicPrint("[%u:%u] [DEBUG] " format "\n", artsGlobalRankId,          \
                    artsThreadInfo.groupId, ##__VA_ARGS__);                    \
  }

#define ARTS_DEBUG_MASTER(format, ...)                                         \
  if (artsGlobalRankId == artsGlobalMasterRankId) {                            \
    artsAtomicPrint("[%u:%u] [DEBUG] " format "\n", artsGlobalRankId,          \
                    artsThreadInfo.groupId, ##__VA_ARGS__);                    \
  }
#else
#define ARTS_DEBUG(...)
#define ARTS_DEBUG_ONCE(...)
#define ARTS_DEBUG_MASTER(...)
#endif

/// ARTS info
#ifdef ARTS_INFO_ENABLED
#define ARTS_INFO(format, ...)                                                 \
  artsAtomicPrint("[%u:%u] [INFO] " format "\n", artsGlobalRankId,             \
                  artsThreadInfo.groupId, ##__VA_ARGS__)
#define ARTS_INFO_ONCE(format, ...)                                            \
  if (artsGlobalIWillPrint) {                                                  \
    artsAtomicPrint("[%u:%u] [INFO] " format "\n", artsGlobalRankId,           \
                    artsThreadInfo.groupId, ##__VA_ARGS__);                    \
  }

#define ARTS_INFO_MASTER(format, ...)                                          \
  if (artsGlobalRankId == artsGlobalMasterRankId) {                            \
    artsAtomicPrint("[%u:%u] [INFO] " format "\n", artsGlobalRankId,           \
                    artsThreadInfo.groupId, ##__VA_ARGS__);                    \
  }
#define ARTS_WARN(format, ...)                                                 \
  artsAtomicPrint("[%u:%u] [WARN] " format "\n", artsGlobalRankId,             \
                  artsThreadInfo.groupId, ##__VA_ARGS__)
#define ARTS_ERROR(format, ...)                                                \
  artsAtomicPrint("[%u:%u] [ERROR] " format "\n", artsGlobalRankId,            \
                  artsThreadInfo.groupId, ##__VA_ARGS__)

#else
#define ARTS_INFO(format, ...)
#define ARTS_INFO_ONCE(format, ...)
#define ARTS_INFO_MASTER(format, ...)
#define ARTS_WARN(format, ...)
#define ARTS_ERROR(format, ...)
#endif // ARTS_INFO_ENABLED

#ifdef __cplusplus
}
#endif

#endif // ARTS_SYSTEM_PRINT_H
