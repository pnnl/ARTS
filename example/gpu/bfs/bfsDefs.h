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
#ifndef BFSDEFS_H
#define BFSDEFS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime/Globals.h"

#define PRINTF(...)
//  #define PRINTF(...) PRINTF(__VA_ARGS__)
#define TURNON(...)
// #define TURNON(...) __VA_ARGS__
#define ROOT 7
#define PARTS 8
#define GPULISTLEN 1024UL * 1024UL * 256UL
#define MAXLEVEL (unsigned int)-1
#define SMTILE 32
#define GPU_THRESHOLD (unsigned int)1024

#define USE_LC 2
#ifdef USE_LC
#define DO_SYNC(level) (level % USE_LC == 0)
#define DB_WRITE_TYPE ARTS_DB_LC
#define checkConsistency(workerId)                                             \
  if (!workerId && artsLookUpConfig(gpuLCSync) != 4) {                         \
    PRINTF("The gpuLCSync must be set to 4 (artsGetMinDbUnsignedInt) in arts " \
           "config file.\n");                                                  \
    artsShutdown();                                                            \
    return;                                                                    \
  }
#else
#define DO_SYNC(level) 0
#define DB_WRITE_TYPE ARTS_DB_GPU_WRITE
#define checkConsistency(workerId)                                             \
  if (!workerId && artsLookUpConfig(gpuLocality) != 3) {                       \
    PRINTF("The gpuLocality must be set to 3 (hashOnDBZero) in arts config "   \
           "file.\n");                                                         \
    artsShutdown();                                                            \
    return;                                                                    \
  }
#endif

#define DASHDASHFILE(argc, argv)                                               \
  if (argc == 3) {                                                             \
    argv[1] = argv[2];                                                         \
    argc--;                                                                    \
  }

#ifdef __cplusplus
}
#endif

#endif