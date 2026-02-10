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

#include "arts/runtime/rt.h"

// Memory placement hints for the memory sensor
typedef enum {
  ARTS_MEM_PLACE_DEFAULT = 0,         // Default placement (system decides)
  ARTS_MEM_PLACE_DRAM = 1 << 0,       // Place in DRAM
  ARTS_MEM_PLACE_HBM = 1 << 1,        // Place in High-Bandwidth Memory
  ARTS_MEM_PLACE_GPU = 1 << 2,        // Place on GPU
  ARTS_MEM_PLACE_REMOTE = 1 << 3,     // Place in remote memory pool
  ARTS_MEM_PLACE_NUMA_LOCAL = 1 << 4, // Place in NUMA-local memory
} arts_mem_placement_t;

// Memory access patterns for optimization
typedef enum {
  ARTS_ACCESS_PATTERN_UNKNOWN = 0,
  ARTS_ACCESS_PATTERN_SEQUENTIAL = 1 << 0,
  ARTS_ACCESS_PATTERN_RANDOM = 1 << 1,
  ARTS_ACCESS_PATTERN_STREAMING = 1 << 2,
  ARTS_ACCESS_PATTERN_REUSE = 1 << 3,
} arts_access_pattern_t;

// Memory sensor metrics structure
typedef struct {
  uint64_t accessCount;        // Number of times data has been accessed
  uint64_t lastAccessTime;     // Timestamp of last access
  uint64_t totalAccessLatency; // Cumulative access latency
  uint64_t totalAccessBytes;   // Total bytes accessed
  arts_access_pattern_t pattern; // Observed access pattern
  float contentionScore;       // Score indicating memory contention (0-1)
  bool isHot;                  // Whether data is frequently accessed
} arts_mem_metrics_t;

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
} arts_smart_db_flags_t;

// SmartDB structure that combines a DataBlock with persistent events and memory
// sensors
typedef struct arts_smart_db {
  // Core DataBlock components
  arts_guid_t db_guid;    // GUID of the underlying DataBlock
  arts_guid_t event_guid; // GUID of the persistent event
  // arts_guid_t data_guid;  // GUID for the data being tracked
  uint64_t size;   // Size of the data
  arts_type_t type; // Type of the DataBlock

  // Readiness sensor components
  unsigned int numProducers; // Number of producers
  unsigned int numConsumers; // Number of consumers
  unsigned int version;      // Version number for tracking updates
  unsigned int latch_count;   // Current latch count for readiness
  bool isReady;              // Current readiness state

  // Memory sensor components
  arts_smart_db_flags_t flags;     // Metadata flags
  arts_mem_placement_t placement; // Current memory placement
  arts_mem_metrics_t metrics;     // Memory access metrics
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
} arts_smart_db_t;

// Migration message for distributed migration
typedef struct {
  uint64_t size;
  arts_type_t type;
  arts_smart_db_flags_t flags;
  unsigned int version;
  unsigned int numProducers;
  unsigned int numConsumers;
  unsigned int latch_count;
  bool isReady;
  arts_mem_placement_t placement;
  arts_mem_metrics_t metrics;
  unsigned int numaNode;
  unsigned int gpuDevice;
  float accessCost;
  size_t memRefSize;
  // For simplicity, we send the data inline after the struct
  // char data[];
} arts_smart_db_migration_msg_t;

// Migration handler prototype
void arts_smart_db_migration_handler(void *args);

// Create a new SmartDB with the given size and type
arts_smart_db_t *arts_smart_db_create(uint64_t size, arts_type_t type,
                                 arts_smart_db_flags_t flags);

// Create a SmartDB with a specific GUID
arts_smart_db_t *arts_smart_db_create_with_guid(arts_guid_t guid, uint64_t size,
                                         arts_smart_db_flags_t flags);

// Destroy a SmartDB and its associated resources
void arts_smart_db_destroy(arts_smart_db_t *smart_db);

// Readiness sensor operations
void arts_smart_db_add_producer(arts_smart_db_t *smart_db);
void arts_smart_db_add_consumer(arts_smart_db_t *smart_db);
void arts_smart_db_producer_complete(arts_smart_db_t *smart_db);
void arts_smart_db_consumer_complete(arts_smart_db_t *smart_db);
bool arts_smart_db_is_ready(arts_smart_db_t *smart_db);
unsigned int arts_smart_db_get_version(arts_smart_db_t *smart_db);
void arts_smart_db_increment_version(arts_smart_db_t *smart_db);

// Memory sensor operations
void arts_smart_db_update_metrics(arts_smart_db_t *smart_db, uint64_t access_size,
                              uint64_t latency);
void arts_smart_db_set_placement(arts_smart_db_t *smart_db,
                             arts_mem_placement_t placement);
arts_mem_placement_t arts_smart_db_get_placement(arts_smart_db_t *smart_db);
void arts_smart_db_set_access_pattern(arts_smart_db_t *smart_db,
                                 arts_access_pattern_t pattern);
arts_access_pattern_t arts_smart_db_get_access_pattern(arts_smart_db_t *smart_db);
float arts_smart_db_get_access_cost(arts_smart_db_t *smart_db);
bool arts_smart_db_should_migrate(arts_smart_db_t *smart_db);
bool arts_smart_db_should_replicate(arts_smart_db_t *smart_db);

// Data operations with memory awareness
void *arts_smart_db_get_data(arts_smart_db_t *smart_db);
void arts_smart_db_set_data(arts_smart_db_t *smart_db, void *data, uint64_t size);
void arts_smart_db_migrate(arts_smart_db_t *smart_db,
                        arts_mem_placement_t new_placement);
void arts_smart_db_replicate(arts_smart_db_t *smart_db, unsigned int num_copies);

// Dependence management
void arts_smart_db_add_dependence(arts_smart_db_t *smart_db, arts_guid_t edt_guid,
                              uint32_t slot);

// Metadata operations
arts_smart_db_flags_t arts_smart_db_get_flags(arts_smart_db_t *smart_db);
void arts_smart_db_set_flags(arts_smart_db_t *smart_db, arts_smart_db_flags_t flags);
unsigned int arts_smart_db_get_num_producers(arts_smart_db_t *smart_db);
unsigned int arts_smart_db_get_num_consumers(arts_smart_db_t *smart_db);

// Migration API
void arts_smart_db_migrate_to_node(arts_smart_db_t *smart_db, unsigned int new_node);

// Sophisticated access pattern detection
void arts_smart_db_record_access(arts_smart_db_t *smart_db, uint64_t offset);
void arts_smart_db_analyze_access_pattern(arts_smart_db_t *smart_db);

#ifdef __cplusplus
}
#endif

#endif // ARTS_SMART_DB_H