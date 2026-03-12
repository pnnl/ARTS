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

#ifndef ARTS_SMART_DB_H
#define ARTS_SMART_DB_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "arts/runtime/RT.h"

// Memory placement hints for the memory sensor
typedef enum {
  ARTS_MEM_PLACE_DEFAULT = 0,         // Default placement (system decides)
  ARTS_MEM_PLACE_DRAM = 1 << 0,       // Place in DRAM
  ARTS_MEM_PLACE_HBM = 1 << 1,        // Place in High-Bandwidth Memory
  ARTS_MEM_PLACE_GPU = 1 << 2,        // Place on GPU
  ARTS_MEM_PLACE_REMOTE = 1 << 3,     // Place in remote memory pool
  ARTS_MEM_PLACE_NUMA_LOCAL = 1 << 4, // Place in NUMA-local memory
} artsMemPlacement_t;

// Memory access patterns for optimization
typedef enum {
  ARTS_ACCESS_PATTERN_UNKNOWN = 0,
  ARTS_ACCESS_PATTERN_SEQUENTIAL = 1 << 0,
  ARTS_ACCESS_PATTERN_RANDOM = 1 << 1,
  ARTS_ACCESS_PATTERN_STREAMING = 1 << 2,
  ARTS_ACCESS_PATTERN_REUSE = 1 << 3,
} artsAccessPattern_t;

// Memory sensor metrics structure
typedef struct {
  uint64_t accessCount;        // Number of times data has been accessed
  uint64_t lastAccessTime;     // Timestamp of last access
  uint64_t totalAccessLatency; // Cumulative access latency
  uint64_t totalAccessBytes;   // Total bytes accessed
  artsAccessPattern_t pattern; // Observed access pattern
  float contentionScore;       // Score indicating memory contention (0-1)
  bool isHot;                  // Whether data is frequently accessed
} artsMemMetrics_t;

// SmartDB metadata flags
typedef enum {
  ARTS_SMART_DB_NONE = 0,
  ARTS_SMART_DB_PERSISTENT = 1 << 0,   // Data should persist across iterations
  ARTS_SMART_DB_PINNED = 1 << 1,       // Data should be pinned in memory
  ARTS_SMART_DB_GPU = 1 << 2,          // Data should be on GPU
  ARTS_SMART_DB_NUMA_LOCAL = 1 << 3,   // Data should be NUMA local
  ARTS_SMART_DB_READ_ONLY = 1 << 4,    // Data is read-only
  ARTS_SMART_DB_AUTO_MIGRATE = 1 << 5, // Enable automatic data migration
  ARTS_SMART_DB_REPLICATE = 1 << 6,    // Enable data replication
} artsSmartDbFlags_t;

// SmartDB structure that combines a DataBlock with persistent events and memory
// sensors
typedef struct artsSmartDb {
  // Core DataBlock components
  artsGuid_t dbGuid;    // GUID of the underlying DataBlock
  artsGuid_t eventGuid; // GUID of the persistent event
  // artsGuid_t dataGuid;  // GUID for the data being tracked
  uint64_t size;   // Size of the data
  artsType_t type; // Type of the DataBlock

  // Readiness sensor components
  unsigned int numProducers; // Number of producers
  unsigned int numConsumers; // Number of consumers
  unsigned int version;      // Version number for tracking updates
  unsigned int latchCount;   // Current latch count for readiness
  bool isReady;              // Current readiness state

  // Memory sensor components
  artsSmartDbFlags_t flags;     // Metadata flags
  artsMemPlacement_t placement; // Current memory placement
  artsMemMetrics_t metrics;     // Memory access metrics
  unsigned int numaNode;        // Current NUMA node
  unsigned int gpuDevice;       // Current GPU device (if applicable)
  float accessCost;             // Estimated access cost (latency + bandwidth)

  // Memory management
  void *memRef;          // Reference to actual memory location
  size_t memRefSize;     // Size of memory reference
  bool isMigrating;      // Whether data is currently being migrated
  unsigned int homeNode; // Node where the SmartDB currently resides

// Access pattern detection (circular buffer)
#define ARTS_SMART_DB_ACCESS_HISTORY 8
  uint64_t accessOffsets[ARTS_SMART_DB_ACCESS_HISTORY];
  unsigned int accessHistoryIdx;
  unsigned int accessHistoryCount;
} artsSmartDb_t;

// Migration message for distributed migration
typedef struct {
  uint64_t size;
  artsType_t type;
  artsSmartDbFlags_t flags;
  unsigned int version;
  unsigned int numProducers;
  unsigned int numConsumers;
  unsigned int latchCount;
  bool isReady;
  artsMemPlacement_t placement;
  artsMemMetrics_t metrics;
  unsigned int numaNode;
  unsigned int gpuDevice;
  float accessCost;
  size_t memRefSize;
  // For simplicity, we send the data inline after the struct
  // char data[];
} artsSmartDbMigrationMsg_t;

// Migration handler prototype
void artsSmartDbMigrationHandler(void *args);

// Create a new SmartDB with the given size and type
artsSmartDb_t *artsSmartDbCreate(uint64_t size, artsType_t type,
                                 artsSmartDbFlags_t flags);

// Create a SmartDB with a specific GUID
artsSmartDb_t *artsSmartDbCreateWithGuid(artsGuid_t guid, uint64_t size,
                                         artsSmartDbFlags_t flags);

// Destroy a SmartDB and its associated resources
void artsSmartDbDestroy(artsSmartDb_t *smartDb);

// Readiness sensor operations
void artsSmartDbAddProducer(artsSmartDb_t *smartDb);
void artsSmartDbAddConsumer(artsSmartDb_t *smartDb);
void artsSmartDbProducerComplete(artsSmartDb_t *smartDb);
void artsSmartDbConsumerComplete(artsSmartDb_t *smartDb);
bool artsSmartDbIsReady(artsSmartDb_t *smartDb);
unsigned int artsSmartDbGetVersion(artsSmartDb_t *smartDb);
void artsSmartDbIncrementVersion(artsSmartDb_t *smartDb);

// Memory sensor operations
void artsSmartDbUpdateMetrics(artsSmartDb_t *smartDb, uint64_t accessSize,
                              uint64_t latency);
void artsSmartDbSetPlacement(artsSmartDb_t *smartDb,
                             artsMemPlacement_t placement);
artsMemPlacement_t artsSmartDbGetPlacement(artsSmartDb_t *smartDb);
void artsSmartDbSetAccessPattern(artsSmartDb_t *smartDb,
                                 artsAccessPattern_t pattern);
artsAccessPattern_t artsSmartDbGetAccessPattern(artsSmartDb_t *smartDb);
float artsSmartDbGetAccessCost(artsSmartDb_t *smartDb);
bool artsSmartDbShouldMigrate(artsSmartDb_t *smartDb);
bool artsSmartDbShouldReplicate(artsSmartDb_t *smartDb);

// Data operations with memory awareness
void *artsSmartDbGetData(artsSmartDb_t *smartDb);
void artsSmartDbSetData(artsSmartDb_t *smartDb, void *data, uint64_t size);
void artsSmartDbMigrate(artsSmartDb_t *smartDb,
                        artsMemPlacement_t newPlacement);
void artsSmartDbReplicate(artsSmartDb_t *smartDb, unsigned int numCopies);

// Dependence management
void artsSmartDbAddDependence(artsSmartDb_t *smartDb, artsGuid_t edtGuid,
                              uint32_t slot);

// Metadata operations
artsSmartDbFlags_t artsSmartDbGetFlags(artsSmartDb_t *smartDb);
void artsSmartDbSetFlags(artsSmartDb_t *smartDb, artsSmartDbFlags_t flags);
unsigned int artsSmartDbGetNumProducers(artsSmartDb_t *smartDb);
unsigned int artsSmartDbGetNumConsumers(artsSmartDb_t *smartDb);

// Migration API
void artsSmartDbMigrateToNode(artsSmartDb_t *smartDb, unsigned int newNode);

// Sophisticated access pattern detection
void artsSmartDbRecordAccess(artsSmartDb_t *smartDb, uint64_t offset);
void artsSmartDbAnalyzeAccessPattern(artsSmartDb_t *smartDb);

#ifdef __cplusplus
}
#endif

#endif // ARTS_SMART_DB_H