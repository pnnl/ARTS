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

#include "arts/runtime/memory/TwinDiff.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arts/introspection/Counter.h"
#include "arts/introspection/Preamble.h"
#include "arts/runtime/RT.h"
#include "arts/system/ArtsPrint.h"

// Twin-diff uses simple word-by-word comparison
// Allocate twin copy for diff computation
void *artsDbAllocateTwin(struct artsDb *db) {
  if (!db)
    return NULL;

  size_t dbDataSize = db->header.size - sizeof(struct artsDb);
  void *twin = malloc(dbDataSize);
  if (!twin) {
    ARTS_ERROR("Failed to allocate twin of size %zu", dbDataSize);
    return NULL;
  }

  // Copy current DB data to twin
  void *dbData = (void *)(db + 1);
  memcpy(twin, dbData, dbDataSize);

  // Mark DB as having twin
  db->twin = twin;
  db->twinFlags |= ARTS_DB_HAS_TWIN;
  db->twinSize = dbDataSize;

  ARTS_DEBUG("Allocated twin for DB [Guid: %lu, size: %zu]", db->guid,
             dbDataSize);
  return twin;
}

// Free twin copy
void artsDbFreeTwin(struct artsDb *db) {
  if (!db || !db->twin) {
    return;
  }

  free(db->twin);
  db->twin = NULL;
  db->twinFlags &= ~ARTS_DB_HAS_TWIN;
  db->twinSize = 0;

  ARTS_DEBUG("Freed twin for DB [Guid: %lu]", db->guid);
}

// Helper: Get current time in microseconds (for metrics)
static inline uint64_t getCurrentTimeUs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000 + (uint64_t)ts.tv_nsec / 1000;
}

// Estimate dirty ratio with fast sampling (every 64 words)
float artsEstimateDirtyRatio(void *working, void *twin, size_t size) {
  if (!working || !twin || size == 0) {
    return 1.0;
  }

  const size_t sampleStride = 64 * sizeof(uint64_t); // Sample every 64 words
  uint64_t *work64 = (uint64_t *)working;
  uint64_t *twin64 = (uint64_t *)twin;
  size_t numWords = size / sizeof(uint64_t);

  size_t dirtyCount = 0;
  size_t sampleCount = 0;

  for (size_t i = 0; i < numWords; i += 64) {
    if (work64[i] != twin64[i])
      dirtyCount++;
    sampleCount++;
  }

  if (sampleCount == 0) {
    return 0.0;
  }

  return (float)dirtyCount / (float)sampleCount;
}

// Compute diffs with word-by-word comparison and RLE
struct artsDiffList *artsComputeDiffs(void *working, void *twin, size_t size) {
  if (!working || !twin || size == 0) {
    return NULL;
  }

  uint64_t startTime = getCurrentTimeUs();

  struct artsDiffList *diffs =
      (struct artsDiffList *)malloc(sizeof(struct artsDiffList));
  if (!diffs) {
    return NULL;
  }

  diffs->regionCount = 0;
  diffs->totalBytes = 0;
  diffs->dirtyRatio = 0.0;
  diffs->head = NULL;
  diffs->tail = NULL;

  // Word-by-word comparison
  uint64_t *work64 = (uint64_t *)working;
  uint64_t *twin64 = (uint64_t *)twin;
  size_t numWords = size / sizeof(uint64_t);
  size_t remainder = size % sizeof(uint64_t);

  uint32_t regionStart = 0;
  uint32_t regionEnd = 0;
  uint32_t currentGap = 0;
  bool inRegion = false;
  size_t dirtyWords = 0;

  // Adaptive merge threshold based on DB size
  uint32_t mergeThreshold = ARTS_DIFF_MERGE_THRESHOLD;
  // For DBs > 4MB
  if (size > (4 * 1024 * 1024))
    mergeThreshold = 256; // More aggressive merging

  // Scan word-by-word to find dirty regions
  for (size_t i = 0; i < numWords; i++) {
    if (work64[i] != twin64[i]) {
      dirtyWords++;
      if (!inRegion) {
        regionStart = i * sizeof(uint64_t);
        inRegion = true;
      }
      // Include any clean gap we were tracking since the last dirty word
      if (currentGap && inRegion) {
        regionEnd += currentGap;
        currentGap = 0;
      }
      regionEnd = (i + 1) * sizeof(uint64_t);
    } else {
      // Check if we should close the current region
      if (inRegion) {
        currentGap += sizeof(uint64_t);
        if (currentGap > mergeThreshold) {
          // Close current region
          struct artsDiffRegion *region =
              (struct artsDiffRegion *)malloc(sizeof(struct artsDiffRegion));
          if (region) {
            region->offset = regionStart;
            region->length = regionEnd - regionStart;
            region->next = NULL;

            diffs->totalBytes += region->length;
            diffs->regionCount++;

            if (!diffs->head) {
              diffs->head = region;
              diffs->tail = region;
            } else {
              diffs->tail->next = region;
              diffs->tail = region;
            }
          }
          inRegion = false;
          currentGap = 0;
        } else {
          // Still within merge threshold; keep gap pending
        }
      }
    }
  }

  // Close final region if needed
  if (inRegion) {
    struct artsDiffRegion *region =
        (struct artsDiffRegion *)malloc(sizeof(struct artsDiffRegion));
    if (region) {
      region->offset = regionStart;
      region->length = regionEnd - regionStart;
      region->next = NULL;

      diffs->totalBytes += region->length;
      diffs->regionCount++;

      if (!diffs->head) {
        diffs->head = region;
        diffs->tail = region;
      } else {
        diffs->tail->next = region;
        diffs->tail = region;
      }
    }
  }

  // Handle remainder bytes
  if (remainder > 0) {
    uint8_t *work8 = (uint8_t *)working;
    uint8_t *twin8 = (uint8_t *)twin;
    size_t baseOffset = numWords * sizeof(uint64_t);

    bool hasDiff = false;
    for (size_t i = 0; i < remainder; i++) {
      if (work8[baseOffset + i] != twin8[baseOffset + i]) {
        hasDiff = true;
        break;
      }
    }

    if (hasDiff) {
      struct artsDiffRegion *region =
          (struct artsDiffRegion *)malloc(sizeof(struct artsDiffRegion));
      if (region) {
        region->offset = baseOffset;
        region->length = remainder;
        region->next = NULL;

        diffs->totalBytes += region->length;
        diffs->regionCount++;

        if (!diffs->head) {
          diffs->head = region;
          diffs->tail = region;
        } else {
          diffs->tail->next = region;
          diffs->tail = region;
        }
      }
    }
  }

  // Calculate dirty ratio
  diffs->dirtyRatio = (float)diffs->totalBytes / (float)size;

  // Update metrics
  uint64_t endTime = getCurrentTimeUs();
  uint64_t diffTimeUs = endTime - startTime;
  uint64_t bytesSaved = size - diffs->totalBytes;

  // Update twin-diff counters (only when enabled in counter.cfg)
  INCREMENT_TWIN_DIFF_USED_BY(1);
  INCREMENT_TWIN_DIFF_BYTES_SAVED_BY(bytesSaved);
  INCREMENT_TWIN_DIFF_COMPUTE_TIME_BY(diffTimeUs);

  ARTS_DEBUG("Computed diffs: %u regions, %u bytes, %.2f%% dirty (took %lu us, "
             "saved %lu bytes)",
             diffs->regionCount, diffs->totalBytes, diffs->dirtyRatio * 100.0,
             diffTimeUs, bytesSaved);

  return diffs;
}

// Free diff list
void artsFreeDiffList(struct artsDiffList *diffs) {
  if (!diffs) {
    return;
  }

  struct artsDiffRegion *region = diffs->head;
  while (region) {
    struct artsDiffRegion *next = region->next;
    free(region);
    region = next;
  }

  free(diffs);
}
