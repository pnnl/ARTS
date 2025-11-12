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
#ifndef ARTS_RUNTIME_MEMORY_TWINDIFF_H
#define ARTS_RUNTIME_MEMORY_TWINDIFF_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Twin-Diff flags for struct artsDb
#define ARTS_DB_HAS_TWIN 0x00000001

// Merge threshold for run-length encoding (bytes)
#define ARTS_DIFF_MERGE_THRESHOLD 128

// Dirty ratio threshold for fallback to full DB send
#define ARTS_DIFF_DIRTY_THRESHOLD 0.75

// Diff region descriptor
struct artsDiffRegion {
  uint32_t offset;
  uint32_t length;
  struct artsDiffRegion *next;
};

// Diff list metadata
struct artsDiffList {
  uint32_t regionCount;
  uint32_t totalBytes;
  float dirtyRatio;
  struct artsDiffRegion *head;
  struct artsDiffRegion *tail;
};

// Forward declaration
struct artsDb;

// Twin allocation/deallocation
void *artsDbAllocateTwin(struct artsDb *db);
void artsDbFreeTwin(struct artsDb *db);

// Diff computation
struct artsDiffList *artsComputeDiffs(void *working, void *twin, size_t size);
void artsFreeDiffList(struct artsDiffList *diffs);

// Dirty ratio estimation (fast scan before full diff)
float artsEstimateDirtyRatio(void *working, void *twin, size_t size);

#ifdef __cplusplus
}
#endif
#endif /* ARTS_RUNTIME_MEMORY_TWINDIFF_H */
