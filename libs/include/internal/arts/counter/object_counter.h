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
#ifndef ARTS_COUNTER_OBJECT_COUNTER_H
#define ARTS_COUNTER_OBJECT_COUNTER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#include "arts/defs.h"
#include "arts/counter/Preamble.h"
#include "arts/utils/array_list.h"

// Hash table size (must be power of 2 for fast modulo)
#define ARTS_OBJECT_TABLE_SIZE 1024

// Composite enable flags (auto-derived from individual Preamble ENABLE_* flags)
#define ARTS_OBJECT_EDT_TABLE_ENABLED \
  (ENABLE_OBJ_NUM_EDT || ENABLE_OBJ_TIME_EDT_EXEC || ENABLE_OBJ_TIME_EDT_STALL)
#define ARTS_OBJECT_DB_TABLE_ENABLED                 \
  (ENABLE_OBJ_NUM_DB || ENABLE_OBJ_BYTES_DB_LOCAL || \
   ENABLE_OBJ_BYTES_DB_REMOTE || ENABLE_OBJ_NUM_DB_CACHE_MISS)
#define ARTS_OBJECT_EDT_TRACE_ENABLED ENABLE_OBJ_TRACE_EDT
#define ARTS_OBJECT_DB_TRACE_ENABLED  ENABLE_OBJ_TRACE_DB
#define ARTS_OBJECT_ANY_ENABLED                                     \
  (ARTS_OBJECT_EDT_TABLE_ENABLED || ARTS_OBJECT_DB_TABLE_ENABLED || \
   ARTS_OBJECT_EDT_TRACE_ENABLED || ARTS_OBJECT_DB_TRACE_ENABLED)

// Per-object EDT hash entry
typedef struct {
  uint64_t arts_id;
  uint64_t count;
  uint64_t exec_ns;
  uint64_t stall_ns;
  bool valid;
} arts_object_edt_entry_t;

// Per-object DB hash entry
typedef struct {
  uint64_t arts_id;
  uint64_t count;
  uint64_t bytes_local;
  uint64_t bytes_remote;
  uint64_t cache_misses;
  bool valid;
} arts_object_db_entry_t;

// Per-thread object counter table (contains both EDT and DB hash tables)
typedef struct {
  arts_object_edt_entry_t edt_table[ARTS_OBJECT_TABLE_SIZE];
  arts_object_db_entry_t db_table[ARTS_OBJECT_TABLE_SIZE];
  uint64_t edt_collisions;
  uint64_t db_collisions;
} arts_object_table_t;

// Per-invocation EDT trace record
typedef struct {
  uint64_t arts_id;
  uint64_t timestamp_ns;
  uint64_t exec_ns;
  uint64_t stall_ns;
  uint32_t node;
  uint32_t thread;
} arts_object_edt_trace_t;

// Per-invocation DB trace record
typedef struct {
  uint64_t arts_id;
  uint64_t timestamp_ns;
  uint64_t bytes_accessed;
  uint32_t node;
  uint8_t access_type;  // 0=READ, 1=WRITE
} arts_object_db_trace_t;

// Thread-local storage (declared here, defined in object_counter.c)
#if ARTS_OBJECT_EDT_TABLE_ENABLED || ARTS_OBJECT_DB_TABLE_ENABLED
extern ARTS_THREAD_LOCAL arts_object_table_t arts_object_tls_table;
#endif
#if ARTS_OBJECT_EDT_TRACE_ENABLED
extern ARTS_THREAD_LOCAL arts_array_list_t *arts_object_tls_edt_traces;
#endif
#if ARTS_OBJECT_DB_TRACE_ENABLED
extern ARTS_THREAD_LOCAL arts_array_list_t *arts_object_tls_db_traces;
#endif

// Recording functions (runtime hot path)
void arts_object_record_edt(uint64_t arts_id, uint64_t exec_ns,
                            uint64_t stall_ns);
void arts_object_record_db(uint64_t arts_id, uint64_t bytes_local,
                           uint64_t bytes_remote, uint64_t cache_misses);
void arts_object_trace_edt(uint64_t arts_id, uint64_t exec_ns,
                           uint64_t stall_ns);
void arts_object_trace_db(uint64_t arts_id, uint64_t bytes_accessed,
                          uint8_t access_type);

// Lifecycle functions (called from runtime init/shutdown)
void arts_object_alloc_node_storage(unsigned int thread_count);
void arts_object_save_thread_data(unsigned int thread_id);
void arts_object_write_node(const char *output_folder, unsigned int node_id,
                            unsigned int thread_count);
void arts_object_write_cluster(const char *output_folder,
                               unsigned int node_count);
void arts_object_cleanup_node_storage(unsigned int thread_count);

// Reduction and initialization
void arts_object_reduce_tables(arts_object_table_t *dest,
                               const arts_object_table_t *src);
void arts_object_init_table(arts_object_table_t *table);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_COUNTER_OBJECT_COUNTER_H */
