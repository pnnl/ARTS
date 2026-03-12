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
#ifndef ARTS_CXL_WRAPPER_H
#define ARTS_CXL_WRAPPER_H
#ifdef __cpluscplus
extern "C" {
#endif

// #include "arts/cxl/SharedAlloc.h"

// JS: SOME DEFS
#define CACHELINE_SIZE 64
 
#ifdef USE_CXL
#include <SharedAlloc.h>
#include <MemOps.h>
#endif

// AA: Round up the alignment of x to a
#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

// #if CXL_FUNCTIONS
#ifdef USE_CXL
#define SHARED_MALLOC(...) SHARED_CXL_MALLOC(__VA_ARGS__)
#define GLOBAL_MALLOC(...) GLOBAL_CXL_MALLOC(__VA_ARGS__)
#define GLOBAL_FREE(...) GLOBAL_CXL_FREE(__VA_ARGS__)
#define SHARED_FREE(...) SHARED_CXL_FREE(__VA_ARGS__)
#define SHARED_MALLOC_INITIALIZED(...)                                         \
  SHARED_CXL_MALLOC_INITIALIZED(__VA_ARGS__)
#define LAST_SHARED_MALLOC(...) LAST_SHARED_CXL_MALLOC(__VA_ARGS__)
#define IS_CXL_PTR(...) IS_FAM_PTR(__VA_ARGS__)
#else
#define SHARED_MALLOC(...) malloc(__VA_ARGS__)
#define GLOBAL_MALLOC(...) malloc(__VA_ARGS__)
#define GLOBAL_FREE(...) free(__VA_ARGS__)
#define SHARED_FREE(...) free(__VA_ARGS__)
#define FLUSH_FENCE_PRODUCER(...)
#define FLUSH_FENCE_CONSUMER(...)
#define SHARED_MALLOC_INITIALIZED(...)
#define LAST_SHARED_MALLOC(...) malloc(__VA_ARGS__)
#define IS_CXL_PTR(...) ((void)0, 0)
#endif

#ifdef __cplusplus
}
#endif
#endif /* ARTS_CXL_WRAPPER_H */
