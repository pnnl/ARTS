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

#include "arts/runtime/globals.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--- Atomic stderr output (used by all logging macros) ------------------------
 *
 * Formats into a stack buffer and writes with a single write() syscall so that
 * output from concurrent threads does not interleave.
 *---------------------------------------------------------------------------*/

static inline void arts_atomic_print(const char *format, ...) {
  char buffer[4096];
  va_list args;
  va_start(args, format);
  int len = vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  if (len > 0 && len < (int)sizeof(buffer)) {
    write(STDERR_FILENO, buffer, (size_t)len);
  } else if (len >= (int)sizeof(buffer)) {
    write(STDERR_FILENO, buffer, sizeof(buffer) - 1);
  }
}

/*--- Log Level Macros --------------------------------------------------------
 *
 * ARTS_LOG_LEVEL (set via CMake, default 1):
 *   0 = ERROR   — print + abort (abort fires even when print is compiled out)
 *   1 = WARN    — print only
 *   2 = INFO    — print only
 *   3 = DEBUG   — print only
 *
 * All levels <= ARTS_LOG_LEVEL are compiled in; higher levels are no-ops.
 *---------------------------------------------------------------------------*/

#ifndef ARTS_LOG_LEVEL
#define ARTS_LOG_LEVEL 1
#endif

#define ARTS_CLR_RED    "\033[1;31m"
#define ARTS_CLR_YELLOW "\033[1;33m"
#define ARTS_CLR_CYAN   "\033[36m"
#define ARTS_CLR_DIM    "\033[2m"
#define ARTS_CLR_RESET  "\033[0m"

/* Level 0: ERROR — abort always fires */
#if ARTS_LOG_LEVEL >= 0
#define ARTS_ERROR(format, ...)                                          \
  do {                                                                   \
    arts_atomic_print(                                                   \
        ARTS_CLR_RED "[%u:%u] [ERROR] " format ARTS_CLR_RESET "\n",      \
        arts_global_rank_id, arts_thread_info.group_pos, ##__VA_ARGS__); \
    arts_abort(1);                                                       \
  } while (0)
#else
#define ARTS_ERROR(format, ...) \
  do {                          \
    arts_abort(1);              \
  } while (0)
#endif

/* Level 1: WARN */
#if ARTS_LOG_LEVEL >= 1
#define ARTS_WARN(format, ...)                                      \
  arts_atomic_print(                                                \
      ARTS_CLR_YELLOW "[%u:%u] [WARN] " format ARTS_CLR_RESET "\n", \
      arts_global_rank_id, arts_thread_info.group_pos, ##__VA_ARGS__)
#else
#define ARTS_WARN(...)
#endif

/* Level 2: INFO */
#if ARTS_LOG_LEVEL >= 2
#define ARTS_INFO(format, ...)                                    \
  arts_atomic_print(                                              \
      ARTS_CLR_CYAN "[%u:%u] [INFO] " format ARTS_CLR_RESET "\n", \
      arts_global_rank_id, arts_thread_info.group_pos, ##__VA_ARGS__)
#else
#define ARTS_INFO(...)
#endif

/* Level 3: DEBUG */
#if ARTS_LOG_LEVEL >= 3
#define ARTS_DEBUG(format, ...)                                   \
  arts_atomic_print(                                              \
      ARTS_CLR_DIM "[%u:%u] [DEBUG] " format ARTS_CLR_RESET "\n", \
      arts_global_rank_id, arts_thread_info.group_pos, ##__VA_ARGS__)
#else
#define ARTS_DEBUG(...)
#endif

#ifdef __cplusplus
}
#endif

#endif /* ARTS_SYSTEM_PRINT_H */
