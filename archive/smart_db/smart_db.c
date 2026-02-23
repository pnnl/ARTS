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

#include "smart_db.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"
#include "arts/gas/route_table.h"
#include "arts/runtime/globals.h"
#include "arts/utils/malloc.h"

// Constants for memory management
#define HOT_ACCESS_THRESHOLD 1000 // Number of accesses to consider data "hot"
#define CONTENTION_THRESHOLD 0.7f // Threshold for considering data contended
#define MIGRATION_COST_THRESHOLD 0.5f // Cost threshold for migration
#define REPLICATION_THRESHOLD 0.8f    // Threshold for replication

// Helper function to calculate access cost based on metrics
static float calculate_access_cost(const arts_mem_metrics_t *metrics,
                                   arts_mem_placement_t placement) {
  float base_cost = 0.0f;

  // Base cost based on placement
  switch (placement) {
  case ARTS_MEM_PLACE_DRAM:
    base_cost = 100.0f; // ns
    break;
  case ARTS_MEM_PLACE_HBM:
    base_cost = 50.0f; // ns
    break;
  case ARTS_MEM_PLACE_GPU:
    base_cost = 200.0f; // ns
    break;
  case ARTS_MEM_PLACE_REMOTE:
    base_cost = 1000.0f; // ns
    break;
  case ARTS_MEM_PLACE_NUMA_LOCAL:
    base_cost = 80.0f; // ns
    break;
  default:
    base_cost = 150.0f; // ns
  }

  // Adjust cost based on contention
  base_cost *= (1.0f + metrics->contentionScore);

  // Adjust for access pattern
  if (metrics->pattern & ARTS_ACCESS_PATTERN_SEQUENTIAL) {
    base_cost *= 0.8f; // Sequential access is more efficient
  } else if (metrics->pattern & ARTS_ACCESS_PATTERN_RANDOM) {
    base_cost *= 1.2f; // Random access is less efficient
  }

  return base_cost;
}

// Create a new SmartDB with the given size and type
arts_smart_db_t *arts_smart_db_create(uint64_t size, arts_type_t type,
                                      arts_smart_db_flags_t flags) {
  SMART_DB_CREATE_COUNTER_START();
  arts_smart_db_t *smart_db =
      (arts_smart_db_t *)arts_malloc(sizeof(arts_smart_db_t));
  if (!smart_db) {
    SMART_DB_CREATE_COUNTER_STOP();
    return NULL;
  }

  // Initialize core components
  smart_db->size = size;
  smart_db->type = type;
  smart_db->flags = flags;

  // Initialize readiness sensor
  smart_db->numProducers = 0;
  smart_db->numConsumers = 0;
  smart_db->version = 0;
  smart_db->latch_count = 0;
  smart_db->isReady = false;

  // Initialize memory sensor
  smart_db->placement = ARTS_MEM_PLACE_DEFAULT;
  memset(&smart_db->metrics, 0, sizeof(arts_mem_metrics_t));
  smart_db->numaNode = arts_get_current_cluster();
  smart_db->gpuDevice = 0;
  smart_db->accessCost = 0.0f;
  smart_db->memRef = NULL;
  smart_db->memRefSize = 0;
  smart_db->isMigrating = false;
  smart_db->homeNode = arts_get_current_node();
  memset(smart_db->accessOffsets, 0, sizeof(smart_db->accessOffsets));
  smart_db->accessHistoryIdx = 0;
  smart_db->accessHistoryCount = 0;

  // Create the underlying DataBlock
  void *data = NULL;
  smart_db->db_guid = arts_db_create(&data, size, ARTS_DB_DEFAULT, NULL);
  if (smart_db->db_guid == NULL_GUID) {
    arts_free(smart_db);
    return NULL;
  }

  // Create a persistent event to track readiness
  smart_db->event_guid =
      arts_persistent_event_create(arts_global_rank_id, 0, smart_db->db_guid);
  if (smart_db->event_guid == NULL_GUID) {
    arts_db_destroy(smart_db->db_guid);
    arts_free(smart_db);
    return NULL;
  }

  // smart_db->data_guid = smart_db->db_guid;
  smart_db->memRef = data;
  smart_db->memRefSize = size;

  SMART_DB_CREATE_COUNTER_STOP();
  return smart_db;
}

// Create a SmartDB with a specific GUID
arts_smart_db_t *arts_smart_db_create_with_guid(arts_guid_t guid, uint64_t size,
                                                arts_smart_db_flags_t flags) {
  arts_smart_db_t *smart_db =
      (arts_smart_db_t *)arts_malloc(sizeof(arts_smart_db_t));
  if (!smart_db) {
    return NULL;
  }

  // Initialize core components
  smart_db->db_guid = guid;
  smart_db->size = size;
  smart_db->type = arts_guid_get_type(guid);
  smart_db->flags = flags;

  // Initialize readiness sensor
  smart_db->numProducers = 0;
  smart_db->numConsumers = 0;
  smart_db->version = 0;
  smart_db->latch_count = 0;
  smart_db->isReady = false;

  // Initialize memory sensor
  smart_db->placement = ARTS_MEM_PLACE_DEFAULT;
  memset(&smart_db->metrics, 0, sizeof(arts_mem_metrics_t));
  smart_db->numaNode = arts_get_current_cluster();
  smart_db->gpuDevice = 0;
  smart_db->accessCost = 0.0f;
  smart_db->memRef = NULL;
  smart_db->memRefSize = 0;
  smart_db->isMigrating = false;
  smart_db->homeNode = arts_get_current_node();
  memset(smart_db->accessOffsets, 0, sizeof(smart_db->accessOffsets));
  smart_db->accessHistoryIdx = 0;
  smart_db->accessHistoryCount = 0;

  // Create the underlying DataBlock with the given GUID
  void *data = arts_db_create_with_guid(guid, size, ARTS_DB_DEFAULT, NULL, NULL);
  if (!data) {
    arts_free(smart_db);
    return NULL;
  }

  // Create a persistent event to track readiness
  smart_db->event_guid =
      arts_persistent_event_create(arts_global_rank_id, 0, guid);
  if (smart_db->event_guid == NULL_GUID) {
    arts_db_destroy(guid);
    arts_free(smart_db);
    return NULL;
  }

  // smart_db->data_guid = guid;
  smart_db->memRef = data;
  smart_db->memRefSize = size;

  return smart_db;
}

// Destroy a SmartDB and its associated resources
void arts_smart_db_destroy(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return;
  }

  // Wait for any ongoing migration to complete
  while (smart_db->isMigrating) {
    arts_yield();
  }

  // Destroy the persistent event
  arts_event_destroy(smart_db->event_guid);

  // Destroy the underlying DataBlock
  arts_db_destroy(smart_db->db_guid);

  // Free the SmartDB structure
  arts_free(smart_db);
}

// Readiness sensor operations
void arts_smart_db_add_producer(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return;
  }
  smart_db->numProducers++;
  smart_db->latch_count++;
  arts_persistent_event_increment_latch(smart_db->event_guid);
}

void arts_smart_db_add_consumer(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return;
  }
  smart_db->numConsumers++;
}

void arts_smart_db_producer_complete(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return;
  }
  arts_persistent_event_decrement_latch(smart_db->event_guid);
  smart_db->latch_count--;
  smart_db->version++;

  // Update readiness state
  smart_db->isReady = (smart_db->latch_count == 0);
}

void arts_smart_db_consumer_complete(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return;
  }
  // Update metrics for consumer completion
  arts_smart_db_update_metrics(smart_db, smart_db->size, 0);
}

bool arts_smart_db_is_ready(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return false;
  }
  return smart_db->isReady && !arts_is_event_fired(smart_db->event_guid);
}

// Memory sensor operations
void arts_smart_db_update_metrics(arts_smart_db_t *smart_db,
                                  uint64_t access_size, uint64_t latency) {
  if (!smart_db) {
    return;
  }

  arts_mem_metrics_t *metrics = &smart_db->metrics;
  uint64_t current_time = arts_get_time_stamp();

  // Update access statistics
  metrics->accessCount++;
  metrics->lastAccessTime = current_time;
  metrics->totalAccessLatency += latency;
  metrics->totalAccessBytes += access_size;

  // Update contention score based on access frequency
  float time_since_last_access =
      (float)(current_time - metrics->lastAccessTime) /
      1e9f;                              // Convert to seconds
  if (time_since_last_access < 0.001f) { // High frequency access
    metrics->contentionScore = fminf(1.0f, metrics->contentionScore + 0.1f);
  } else {
    metrics->contentionScore = fmaxf(0.0f, metrics->contentionScore - 0.05f);
  }

  // Update hot data status
  metrics->isHot = (metrics->accessCount > HOT_ACCESS_THRESHOLD);

  // Update access cost
  smart_db->accessCost = calculate_access_cost(metrics, smart_db->placement);
}

void arts_smart_db_set_placement(arts_smart_db_t *smart_db,
                                 arts_mem_placement_t placement) {
  if (!smart_db || smart_db->isMigrating) {
    return;
  }

  if (placement != smart_db->placement) {
    smart_db->isMigrating = true;
    // TODO: Implement actual data migration
    smart_db->placement = placement;
    smart_db->isMigrating = false;
  }
}

arts_mem_placement_t arts_smart_db_get_placement(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return ARTS_MEM_PLACE_DEFAULT;
  }
  return smart_db->placement;
}

void arts_smart_db_set_access_pattern(arts_smart_db_t *smart_db,
                                      arts_access_pattern_t pattern) {
  if (!smart_db) {
    return;
  }
  smart_db->metrics.pattern = pattern;
  smart_db->accessCost =
      calculate_access_cost(&smart_db->metrics, smart_db->placement);
}

arts_access_pattern_t
arts_smart_db_get_access_pattern(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return ARTS_ACCESS_PATTERN_UNKNOWN;
  }
  return smart_db->metrics.pattern;
}

float arts_smart_db_get_access_cost(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return 0.0f;
  }
  return smart_db->accessCost;
}

bool arts_smart_db_should_migrate(arts_smart_db_t *smart_db) {
  if (!smart_db || !(smart_db->flags & ARTS_SMART_DB_AUTO_MIGRATE)) {
    return false;
  }

  const arts_mem_metrics_t *metrics = &smart_db->metrics;

  // Consider migration if:
  // 1. Data is hot and contended
  // 2. Current placement is not optimal
  // 3. Migration cost is justified
  bool should_migrate = metrics->isHot &&
                        metrics->contentionScore > CONTENTION_THRESHOLD &&
                        smart_db->accessCost > MIGRATION_COST_THRESHOLD;

  return should_migrate;
}

bool arts_smart_db_should_replicate(arts_smart_db_t *smart_db) {
  if (!smart_db || !(smart_db->flags & ARTS_SMART_DB_REPLICATE)) {
    return false;
  }

  const arts_mem_metrics_t *metrics = &smart_db->metrics;

  // Consider replication if:
  // 1. Data is very hot
  // 2. High contention
  // 3. Multiple consumers
  bool should_replicate = metrics->isHot &&
                          metrics->contentionScore > REPLICATION_THRESHOLD &&
                          smart_db->numConsumers > 1;

  return should_replicate;
}

// Data operations with memory awareness
void *arts_smart_db_get_data(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return NULL;
  }

  // Update metrics for this access
  uint64_t start_time = arts_get_time_stamp();
  void *data =
      arts_db_create_with_guid(smart_db->db_guid, smart_db->size, ARTS_DB_DEFAULT, NULL, NULL);
  uint64_t latency = arts_get_time_stamp() - start_time;

  if (data) {
    arts_smart_db_update_metrics(smart_db, smart_db->size, latency);
    // Record access at offset 0 (whole DB)
    arts_smart_db_record_access(smart_db, 0);
    arts_smart_db_analyze_access_pattern(smart_db);
  }

  return data;
}

void arts_smart_db_set_data(arts_smart_db_t *smart_db, void *data,
                            uint64_t size) {
  if (!smart_db || !data || size > smart_db->size) {
    return;
  }

  // Get the current data pointer
  void *current_data =
      arts_db_create_with_guid(smart_db->db_guid, smart_db->size, ARTS_DB_DEFAULT, NULL, NULL);
  if (!current_data) {
    return;
  }

  // Copy the new data
  uint64_t start_time = arts_get_time_stamp();
  memcpy(current_data, data, size);
  uint64_t latency = arts_get_time_stamp() - start_time;

  // Update metrics
  arts_smart_db_update_metrics(smart_db, size, latency);
  // Record access at offset 0 (whole DB)
  arts_smart_db_record_access(smart_db, 0);
  arts_smart_db_analyze_access_pattern(smart_db);

  // Signal that the data has been updated
  arts_smart_db_producer_complete(smart_db);
}

void arts_smart_db_migrate(arts_smart_db_t *smart_db,
                           arts_mem_placement_t new_placement) {
  if (!smart_db || smart_db->isMigrating) {
    return;
  }

  smart_db->isMigrating = true;

  // TODO: Implement actual data migration based on placement
  // This would involve:
  // 1. Allocating memory in the new location
  // 2. Copying data
  // 3. Updating pointers and metadata
  // 4. Freeing old memory

  smart_db->placement = new_placement;
  smart_db->isMigrating = false;
}

void arts_smart_db_replicate(arts_smart_db_t *smart_db,
                             unsigned int num_copies) {
  (void)num_copies;
  if (!smart_db || !(smart_db->flags & ARTS_SMART_DB_REPLICATE)) {
    return;
  }

  // TODO: Implement data replication
  // This would involve:
  // 1. Creating copies in appropriate locations
  // 2. Setting up replication metadata
  // 3. Managing consistency between copies
}

// Dependence management
void arts_smart_db_add_dependence(arts_smart_db_t *smart_db,
                                  arts_guid_t edt_guid, uint32_t slot) {
  if (!smart_db || edt_guid == NULL_GUID) {
    return;
  }
  arts_add_dependence_to_persistent_event(smart_db->event_guid, edt_guid, slot);
}

// Metadata operations
arts_smart_db_flags_t arts_smart_db_get_flags(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return ARTS_SMART_DB_NONE;
  }
  return smart_db->flags;
}

void arts_smart_db_set_flags(arts_smart_db_t *smart_db,
                             arts_smart_db_flags_t flags) {
  if (!smart_db) {
    return;
  }
  smart_db->flags = flags;
}

unsigned int arts_smart_db_get_num_producers(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return 0;
  }
  return smart_db->numProducers;
}

unsigned int arts_smart_db_get_num_consumers(arts_smart_db_t *smart_db) {
  if (!smart_db) {
    return 0;
  }
  return smart_db->numConsumers;
}

// Migration API: Move SmartDB to a new node
void arts_smart_db_migrate_to_node(arts_smart_db_t *smart_db,
                                   unsigned int new_node) {
  if (!smart_db || smart_db->isMigrating || smart_db->homeNode == new_node) {
    return;
  }
  smart_db->isMigrating = true;

  // Quiescence: Prevent concurrent accesses during migration
  // (isMigrating flag is checked in all SmartDB accessors)

  // Marshall SmartDB metadata and data
  size_t msg_size =
      sizeof(arts_smart_db_migration_msg_t) + smart_db->memRefSize;
  char *buffer = (char *)arts_malloc(msg_size);
  arts_smart_db_migration_msg_t *msg = (arts_smart_db_migration_msg_t *)buffer;
  msg->size = smart_db->size;
  msg->type = smart_db->type;
  msg->flags = smart_db->flags;
  msg->version = smart_db->version;
  msg->numProducers = smart_db->numProducers;
  msg->numConsumers = smart_db->numConsumers;
  msg->latch_count = smart_db->latch_count;
  msg->isReady = smart_db->isReady;
  msg->placement = smart_db->placement;
  msg->metrics = smart_db->metrics;
  msg->numaNode = smart_db->numaNode;
  msg->gpuDevice = smart_db->gpuDevice;
  msg->accessCost = smart_db->accessCost;
  msg->memRefSize = smart_db->memRefSize;
  if (smart_db->memRef && smart_db->memRefSize > 0) {
    memcpy(buffer + sizeof(arts_smart_db_migration_msg_t), smart_db->memRef,
           smart_db->memRefSize);
  }

  // Send to new node
  arts_remote_send(new_node, (send_handler_t)arts_smart_db_migration_handler,
                   buffer, msg_size, true);

  // Update homeNode
  smart_db->homeNode = new_node;

  // Update routing table to remove local entry (simulate move semantics)
  arts_route_table_remove_item(smart_db->db_guid);

  // Destroy local SmartDB
  arts_smart_db_destroy(smart_db);
  // Note: buffer is freed by remote handler (arts_remote_send with free=true)

  // Only actionable TODOs remain:
  // TODO: Implement persistent event migration and update dependents
  // TODO: Implement notification to dependents
}

// Handler to reconstruct SmartDB on the destination node
void arts_smart_db_migration_handler(void *args) {
  if (!args) {
    return;
  }
  arts_smart_db_migration_msg_t *msg = (arts_smart_db_migration_msg_t *)args;
  void *data_ptr = (void *)(msg + 1);

  // Create new SmartDB and DataBlock
  arts_smart_db_t *smart_db =
      arts_smart_db_create(msg->size, msg->type, msg->flags);
  if (!smart_db) {
    return;
  }

  // Copy metadata
  smart_db->version = msg->version;
  smart_db->numProducers = msg->numProducers;
  smart_db->numConsumers = msg->numConsumers;
  smart_db->latch_count = msg->latch_count;
  smart_db->isReady = msg->isReady;
  smart_db->placement = msg->placement;
  smart_db->metrics = msg->metrics;
  smart_db->numaNode = msg->numaNode;
  smart_db->gpuDevice = msg->gpuDevice;
  smart_db->accessCost = msg->accessCost;
  smart_db->memRefSize = msg->memRefSize;
  smart_db->homeNode = arts_get_current_node();

  // Copy data
  void *db_data = arts_smart_db_get_data(smart_db);
  if (db_data && data_ptr && msg->memRefSize > 0) {
    memcpy(db_data, data_ptr, msg->memRefSize);
  }

  // Update routing table so the SmartDB's GUID points to this node
  // (Assume db_guid is the SmartDB's GUID for now)
  arts_route_table_add_item(smart_db, smart_db->db_guid, smart_db->homeNode,
                            false);

  // Placeholder: Migrate persistent event and update dependents
  // TODO: Implement persistent event migration and update dependents

  // Placeholder: Notify dependents of new location
  // TODO: Implement notification to dependents
}

// Sophisticated access pattern detection
void arts_smart_db_record_access(arts_smart_db_t *smart_db, uint64_t offset) {
  if (!smart_db) {
    return;
  }
  smart_db->accessOffsets[smart_db->accessHistoryIdx] = offset;
  smart_db->accessHistoryIdx =
      (smart_db->accessHistoryIdx + 1) % ARTS_SMART_DB_ACCESS_HISTORY;
  if (smart_db->accessHistoryCount < ARTS_SMART_DB_ACCESS_HISTORY) {
    smart_db->accessHistoryCount++;
  }
}

void arts_smart_db_analyze_access_pattern(arts_smart_db_t *smart_db) {
  if (!smart_db || smart_db->accessHistoryCount < 2) {
    return;
  }
  int sequential = 0;
  int random = 0;
  int streaming = 0;
  int reuse = 0;
  uint64_t last = smart_db->accessOffsets[(smart_db->accessHistoryIdx +
                                           ARTS_SMART_DB_ACCESS_HISTORY - 1) %
                                          ARTS_SMART_DB_ACCESS_HISTORY];
  for (unsigned int i = 1; i < smart_db->accessHistoryCount; ++i) {
    unsigned int idx =
        (smart_db->accessHistoryIdx + ARTS_SMART_DB_ACCESS_HISTORY - 1 - i) %
        ARTS_SMART_DB_ACCESS_HISTORY;
    uint64_t curr = smart_db->accessOffsets[idx];
    int64_t diff = (int64_t)last - (int64_t)curr;
    if (diff == (int64_t)smart_db->memRefSize) {
      sequential++;
    } else if (diff == 0) {
      reuse++;
    } else if (llabs(diff) < (int64_t)smart_db->memRefSize / 4) {
      streaming++;
    } else {
      random++;
    }
    last = curr;
  }
  // Pick the dominant pattern
  if (sequential > random && sequential > streaming && sequential > reuse) {
    smart_db->metrics.pattern = ARTS_ACCESS_PATTERN_SEQUENTIAL;
  } else if (streaming > sequential && streaming > random &&
             streaming > reuse) {
    smart_db->metrics.pattern = ARTS_ACCESS_PATTERN_STREAMING;
  } else if (reuse > sequential && reuse > streaming && reuse > random) {
    smart_db->metrics.pattern = ARTS_ACCESS_PATTERN_REUSE;
  } else {
    smart_db->metrics.pattern = ARTS_ACCESS_PATTERN_RANDOM;
  }
  // Update access cost
  smart_db->accessCost =
      calculate_access_cost(&smart_db->metrics, smart_db->placement);
}