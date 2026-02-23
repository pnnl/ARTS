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
#include "arts/counter/object_counter.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "arts.h"
#include "arts/counter/json.h"
#include "arts/runtime/globals.h"
#include "arts/utils/malloc.h"

// ============================================================================
// Thread-local storage definitions
// ============================================================================

#if ARTS_OBJECT_EDT_TABLE_ENABLED || ARTS_OBJECT_DB_TABLE_ENABLED
ARTS_THREAD_LOCAL arts_object_table_t arts_object_tls_table;
#endif
#if ARTS_OBJECT_EDT_TRACE_ENABLED
ARTS_THREAD_LOCAL arts_array_list_t *arts_object_tls_edt_traces = NULL;
#endif
#if ARTS_OBJECT_DB_TRACE_ENABLED
ARTS_THREAD_LOCAL arts_array_list_t *arts_object_tls_db_traces = NULL;
#endif

// ============================================================================
// Hash table internals
// ============================================================================

// FNV-inspired hash for distributing arts_id values
static inline uint32_t arts_object_hash(uint64_t arts_id) {
  uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
  hash ^= arts_id;
  hash *= 1099511628211ULL;  // FNV prime
  return (uint32_t)(hash & (ARTS_OBJECT_TABLE_SIZE - 1));
}

// Get current time in nanoseconds
static inline uint64_t arts_object_get_time_ns(void) {
  struct timespec ts;
  (void)clock_gettime(CLOCK_MONOTONIC, &ts);
  return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}

// Find or create EDT slot using linear probing
#if ARTS_OBJECT_EDT_TABLE_ENABLED
static inline arts_object_edt_entry_t *arts_object_find_edt_slot(
    arts_object_edt_entry_t *table, uint64_t arts_id, uint64_t *collisions) {
  uint32_t idx = arts_object_hash(arts_id);
  uint32_t start_idx = idx;

  while (table[idx].valid && table[idx].arts_id != arts_id) {
    idx = (idx + 1) & (ARTS_OBJECT_TABLE_SIZE - 1);
    if (idx == start_idx) {
      return NULL;  // Table full
    }
    (*collisions)++;
  }

  if (!table[idx].valid) {
    memset(&table[idx], 0, sizeof(arts_object_edt_entry_t));
    table[idx].arts_id = arts_id;
    table[idx].valid = true;
  }

  return &table[idx];
}
#endif

// Find or create DB slot using linear probing
#if ARTS_OBJECT_DB_TABLE_ENABLED
static inline arts_object_db_entry_t *arts_object_find_db_slot(
    arts_object_db_entry_t *table, uint64_t arts_id, uint64_t *collisions) {
  uint32_t idx = arts_object_hash(arts_id);
  uint32_t start_idx = idx;

  while (table[idx].valid && table[idx].arts_id != arts_id) {
    idx = (idx + 1) & (ARTS_OBJECT_TABLE_SIZE - 1);
    if (idx == start_idx) {
      return NULL;  // Table full
    }
    (*collisions)++;
  }

  if (!table[idx].valid) {
    memset(&table[idx], 0, sizeof(arts_object_db_entry_t));
    table[idx].arts_id = arts_id;
    table[idx].valid = true;
  }

  return &table[idx];
}
#endif

// ============================================================================
// Recording functions (runtime hot path)
// ============================================================================

void arts_object_record_edt(uint64_t arts_id, uint64_t exec_ns,
                            uint64_t stall_ns) {
#if ARTS_OBJECT_EDT_TABLE_ENABLED
  if (arts_id == 0) {
    return;
  }

  arts_object_edt_entry_t *slot =
      arts_object_find_edt_slot(arts_object_tls_table.edt_table, arts_id,
                                &arts_object_tls_table.edt_collisions);

  if (slot) {
    // TLS — no atomics needed, only the owning thread accesses this
    slot->count++;
    slot->exec_ns += exec_ns;
    slot->stall_ns += stall_ns;
  }
#else
  (void)arts_id;
  (void)exec_ns;
  (void)stall_ns;
#endif
}

void arts_object_record_db(uint64_t arts_id, uint64_t bytes_local,
                           uint64_t bytes_remote, uint64_t cache_misses) {
#if ARTS_OBJECT_DB_TABLE_ENABLED
  if (arts_id == 0) {
    return;
  }

  arts_object_db_entry_t *slot =
      arts_object_find_db_slot(arts_object_tls_table.db_table, arts_id,
                               &arts_object_tls_table.db_collisions);

  if (slot) {
    slot->count++;
    slot->bytes_local += bytes_local;
    slot->bytes_remote += bytes_remote;
    slot->cache_misses += cache_misses;
  }
#else
  (void)arts_id;
  (void)bytes_local;
  (void)bytes_remote;
  (void)cache_misses;
#endif
}

void arts_object_trace_edt(uint64_t arts_id, uint64_t exec_ns,
                           uint64_t stall_ns) {
#if ARTS_OBJECT_EDT_TRACE_ENABLED
  if (arts_id == 0 || !arts_object_tls_edt_traces) {
    return;
  }

  arts_object_edt_trace_t trace = {.arts_id = arts_id,
                                   .timestamp_ns = arts_object_get_time_ns(),
                                   .exec_ns = exec_ns,
                                   .stall_ns = stall_ns,
                                   .node = arts_get_current_node(),
                                   .thread = arts_thread_info.thread_id};

  arts_push_to_array_list(arts_object_tls_edt_traces, &trace);
#else
  (void)arts_id;
  (void)exec_ns;
  (void)stall_ns;
#endif
}

void arts_object_trace_db(uint64_t arts_id, uint64_t bytes_accessed,
                          uint8_t access_type) {
#if ARTS_OBJECT_DB_TRACE_ENABLED
  if (arts_id == 0 || !arts_object_tls_db_traces) {
    return;
  }

  arts_object_db_trace_t trace = {.arts_id = arts_id,
                                  .timestamp_ns = arts_object_get_time_ns(),
                                  .bytes_accessed = bytes_accessed,
                                  .node = arts_get_current_node(),
                                  .access_type = access_type};

  arts_push_to_array_list(arts_object_tls_db_traces, &trace);
#else
  (void)arts_id;
  (void)bytes_accessed;
  (void)access_type;
#endif
}

// ============================================================================
// Table initialization and reduction
// ============================================================================

void arts_object_init_table(arts_object_table_t *table) {
  if (table) {
    memset(table, 0, sizeof(arts_object_table_t));
  }
}

void arts_object_reduce_tables(arts_object_table_t *dest,
                               const arts_object_table_t *src) {
  if (!dest || !src) {
    return;
  }

#if ARTS_OBJECT_EDT_TABLE_ENABLED
  for (uint32_t i = 0; i < ARTS_OBJECT_TABLE_SIZE; i++) {
    if (!src->edt_table[i].valid) {
      continue;
    }

    uint64_t id = src->edt_table[i].arts_id;
    arts_object_edt_entry_t *slot =
        arts_object_find_edt_slot(dest->edt_table, id, &dest->edt_collisions);

    if (slot) {
      slot->count += src->edt_table[i].count;
      slot->exec_ns += src->edt_table[i].exec_ns;
      slot->stall_ns += src->edt_table[i].stall_ns;
    }
  }
#endif

#if ARTS_OBJECT_DB_TABLE_ENABLED
  for (uint32_t i = 0; i < ARTS_OBJECT_TABLE_SIZE; i++) {
    if (!src->db_table[i].valid) {
      continue;
    }

    uint64_t id = src->db_table[i].arts_id;
    arts_object_db_entry_t *slot =
        arts_object_find_db_slot(dest->db_table, id, &dest->db_collisions);

    if (slot) {
      slot->count += src->db_table[i].count;
      slot->bytes_local += src->db_table[i].bytes_local;
      slot->bytes_remote += src->db_table[i].bytes_remote;
      slot->cache_misses += src->db_table[i].cache_misses;
    }
  }
#endif
}

// ============================================================================
// Node storage lifecycle
// ============================================================================

void arts_object_alloc_node_storage(unsigned int thread_count) {
#if ARTS_OBJECT_EDT_TABLE_ENABLED || ARTS_OBJECT_DB_TABLE_ENABLED
  arts_node_info.object_tables = (arts_object_table_t **)arts_calloc(
      thread_count, sizeof(arts_object_table_t *));
#endif
#if ARTS_OBJECT_EDT_TRACE_ENABLED
  arts_node_info.object_edt_traces = (arts_array_list_t **)arts_calloc(
      thread_count, sizeof(arts_array_list_t *));
#endif
#if ARTS_OBJECT_DB_TRACE_ENABLED
  arts_node_info.object_db_traces = (arts_array_list_t **)arts_calloc(
      thread_count, sizeof(arts_array_list_t *));
#endif
  (void)thread_count;
}

void arts_object_save_thread_data(unsigned int thread_id) {
#if ARTS_OBJECT_EDT_TABLE_ENABLED || ARTS_OBJECT_DB_TABLE_ENABLED
  arts_node_info.object_tables[thread_id] =
      (arts_object_table_t *)arts_malloc(sizeof(arts_object_table_t));
  memcpy(arts_node_info.object_tables[thread_id], &arts_object_tls_table,
         sizeof(arts_object_table_t));
#endif
#if ARTS_OBJECT_EDT_TRACE_ENABLED
  arts_node_info.object_edt_traces[thread_id] = arts_object_tls_edt_traces;
  arts_object_tls_edt_traces = NULL;  // Transfer ownership
#endif
#if ARTS_OBJECT_DB_TRACE_ENABLED
  arts_node_info.object_db_traces[thread_id] = arts_object_tls_db_traces;
  arts_object_tls_db_traces = NULL;
#endif
  (void)thread_id;
}

void arts_object_cleanup_node_storage(unsigned int thread_count) {
#if ARTS_OBJECT_EDT_TABLE_ENABLED || ARTS_OBJECT_DB_TABLE_ENABLED
  if (arts_node_info.object_tables) {
    for (unsigned int t = 0; t < thread_count; t++) {
      if (arts_node_info.object_tables[t]) {
        arts_free(arts_node_info.object_tables[t]);
      }
    }
    arts_free(arts_node_info.object_tables);
    arts_node_info.object_tables = NULL;
  }
#endif
#if ARTS_OBJECT_EDT_TRACE_ENABLED
  if (arts_node_info.object_edt_traces) {
    for (unsigned int t = 0; t < thread_count; t++) {
      if (arts_node_info.object_edt_traces[t]) {
        arts_delete_array_list(arts_node_info.object_edt_traces[t]);
      }
    }
    arts_free(arts_node_info.object_edt_traces);
    arts_node_info.object_edt_traces = NULL;
  }
#endif
#if ARTS_OBJECT_DB_TRACE_ENABLED
  if (arts_node_info.object_db_traces) {
    for (unsigned int t = 0; t < thread_count; t++) {
      if (arts_node_info.object_db_traces[t]) {
        arts_delete_array_list(arts_node_info.object_db_traces[t]);
      }
    }
    arts_free(arts_node_info.object_db_traces);
    arts_node_info.object_db_traces = NULL;
  }
#endif
  (void)thread_count;
}

// ============================================================================
// JSON output — object_n{id}.json
// ============================================================================

void arts_object_write_node(const char *output_folder, unsigned int node_id,
                            unsigned int thread_count) {
#if !ARTS_OBJECT_ANY_ENABLED
  (void)output_folder;
  (void)node_id;
  (void)thread_count;
#else
  if (!output_folder) {
    return;
  }

  // Reduce tables across all threads into a single merged table
  arts_object_table_t merged;
  arts_object_init_table(&merged);

#if ARTS_OBJECT_EDT_TABLE_ENABLED || ARTS_OBJECT_DB_TABLE_ENABLED
  for (unsigned int t = 0; t < thread_count; t++) {
    if (arts_node_info.object_tables && arts_node_info.object_tables[t]) {
      arts_object_reduce_tables(&merged, arts_node_info.object_tables[t]);
    }
  }
#endif

  // Open output file
  struct stat st = {0};
  if (stat(output_folder, &st) == -1) {
    mkdir(output_folder, 0755);
  }
  char filepath[1024];
  (void)snprintf(filepath, sizeof(filepath), "%s/object_n%u.json",
                 output_folder, node_id);
  FILE *fp = fopen(filepath, "w");
  if (!fp) {
    return;
  }

  arts_json_writer_t writer;
  arts_json_writer_init(&writer, fp, 2);
  arts_json_writer_begin_object(&writer, NULL);

  // Metadata
  arts_json_writer_begin_object(&writer, "metadata");
  arts_json_writer_write_u_int64(&writer, "node_id", node_id);
  arts_json_writer_write_u_int64(&writer, "timestamp", (uint64_t)time(NULL));
  arts_json_writer_write_string(&writer, "version", "1.7.0");
  arts_json_writer_write_u_int64(&writer, "total_threads", thread_count);
  arts_json_writer_end_object(&writer);

  // EDT objects (aggregate)
#if ARTS_OBJECT_EDT_TABLE_ENABLED
  arts_json_writer_begin_array(&writer, "edt_objects");
  for (uint32_t i = 0; i < ARTS_OBJECT_TABLE_SIZE; i++) {
    if (!merged.edt_table[i].valid) {
      continue;
    }
    arts_json_writer_begin_object(&writer, NULL);
    arts_json_writer_write_u_int64(&writer, "arts_id",
                                   merged.edt_table[i].arts_id);
    arts_json_writer_write_u_int64(&writer, "count", merged.edt_table[i].count);
    arts_json_writer_write_u_int64(&writer, "exec_ns",
                                   merged.edt_table[i].exec_ns);
    arts_json_writer_write_u_int64(&writer, "stall_ns",
                                   merged.edt_table[i].stall_ns);
    arts_json_writer_end_object(&writer);
  }
  arts_json_writer_end_array(&writer);
  arts_json_writer_write_u_int64(&writer, "edt_collisions",
                                 merged.edt_collisions);
#endif

  // DB objects (aggregate)
#if ARTS_OBJECT_DB_TABLE_ENABLED
  arts_json_writer_begin_array(&writer, "db_objects");
  for (uint32_t i = 0; i < ARTS_OBJECT_TABLE_SIZE; i++) {
    if (!merged.db_table[i].valid) {
      continue;
    }
    arts_json_writer_begin_object(&writer, NULL);
    arts_json_writer_write_u_int64(&writer, "arts_id",
                                   merged.db_table[i].arts_id);
    arts_json_writer_write_u_int64(&writer, "count", merged.db_table[i].count);
    arts_json_writer_write_u_int64(&writer, "bytes_local",
                                   merged.db_table[i].bytes_local);
    arts_json_writer_write_u_int64(&writer, "bytes_remote",
                                   merged.db_table[i].bytes_remote);
    arts_json_writer_write_u_int64(&writer, "cache_misses",
                                   merged.db_table[i].cache_misses);
    arts_json_writer_end_object(&writer);
  }
  arts_json_writer_end_array(&writer);
  arts_json_writer_write_u_int64(&writer, "db_collisions",
                                 merged.db_collisions);
#endif

  // EDT traces (detailed per-invocation)
#if ARTS_OBJECT_EDT_TRACE_ENABLED
  arts_json_writer_begin_array(&writer, "edt_traces");
  for (unsigned int t = 0; t < thread_count; t++) {
    arts_array_list_t *traces = arts_node_info.object_edt_traces
                                    ? arts_node_info.object_edt_traces[t]
                                    : NULL;
    if (!traces || traces->index == 0) {
      continue;
    }
    arts_array_list_iterator_t iter;
    arts_array_list_iter_init(&iter, traces);
    while (arts_array_list_has_next(&iter)) {
      arts_object_edt_trace_t *tr =
          (arts_object_edt_trace_t *)arts_array_list_next(&iter);
      if (!tr) {
        break;
      }
      arts_json_writer_begin_object(&writer, NULL);
      arts_json_writer_write_u_int64(&writer, "arts_id", tr->arts_id);
      arts_json_writer_write_u_int64(&writer, "timestamp_ns", tr->timestamp_ns);
      arts_json_writer_write_u_int64(&writer, "exec_ns", tr->exec_ns);
      arts_json_writer_write_u_int64(&writer, "stall_ns", tr->stall_ns);
      arts_json_writer_write_u_int64(&writer, "node", tr->node);
      arts_json_writer_write_u_int64(&writer, "thread", tr->thread);
      arts_json_writer_end_object(&writer);
    }
  }
  arts_json_writer_end_array(&writer);
#endif

  // DB traces (detailed per-invocation)
#if ARTS_OBJECT_DB_TRACE_ENABLED
  arts_json_writer_begin_array(&writer, "db_traces");
  for (unsigned int t = 0; t < thread_count; t++) {
    arts_array_list_t *traces = arts_node_info.object_db_traces
                                    ? arts_node_info.object_db_traces[t]
                                    : NULL;
    if (!traces || traces->index == 0) {
      continue;
    }
    arts_array_list_iterator_t iter;
    arts_array_list_iter_init(&iter, traces);
    while (arts_array_list_has_next(&iter)) {
      arts_object_db_trace_t *tr =
          (arts_object_db_trace_t *)arts_array_list_next(&iter);
      if (!tr) {
        break;
      }
      arts_json_writer_begin_object(&writer, NULL);
      arts_json_writer_write_u_int64(&writer, "arts_id", tr->arts_id);
      arts_json_writer_write_u_int64(&writer, "timestamp_ns", tr->timestamp_ns);
      arts_json_writer_write_u_int64(&writer, "bytes_accessed",
                                     tr->bytes_accessed);
      arts_json_writer_write_u_int64(&writer, "node", tr->node);
      arts_json_writer_write_u_int64(&writer, "access_type", tr->access_type);
      arts_json_writer_end_object(&writer);
    }
  }
  arts_json_writer_end_array(&writer);
#endif

  arts_json_writer_end_object(&writer);
  arts_json_writer_finish(&writer);
  (void)fputc('\n', fp);
  (void)fclose(fp);
#endif  // ARTS_OBJECT_ANY_ENABLED
}

// ============================================================================
// JSON output — object.json (cluster aggregation)
// ============================================================================

void arts_object_write_cluster(const char *output_folder,
                               unsigned int node_count) {
  // Cluster aggregation for object counters requires reading and parsing
  // object_n{id}.json files from all nodes, similar to cluster.json.
  // For now, individual node files are self-contained. Cluster-level
  // aggregation can be added as a post-processing step if needed.
  (void)output_folder;
  (void)node_count;
}
