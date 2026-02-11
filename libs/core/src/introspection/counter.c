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
#include "arts/introspection/counter.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include "arts.h"
#include "arts/utils/malloc.h"
#include "arts/introspection/arts_id_counter.h"
#include "arts/introspection/json_writer.h"
#include "arts/network/remote.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/network/remote_functions.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/utils/atomics.h"

// Access to network ports for setting up inbound queues during counter collection
extern unsigned int ports;

// Arrays are defined as static const in Preamble.h (included via arts.h)

// Thread-local counter storage - simple array of counters.
// Each thread updates these directly during execution.
// No captures here - capture thread handles periodic snapshots separately.
__thread arts_counter_t arts_thread_local_counters[NUM_COUNTER_TYPES];

// arts_id tracking stored separately per-thread (compile-time conditional)
#if ENABLE_ARTS_ID_EDT_METRICS || ENABLE_ARTS_ID_DB_METRICS
__thread arts_id_hash_table_t arts_thread_local_arts_id_metrics;
#endif
#if ENABLE_ARTS_ID_EDT_CAPTURES
__thread arts_array_list_t *arts_thread_local_edt_capture_list = NULL;
#endif
#if ENABLE_ARTS_ID_DB_CAPTURES
__thread arts_array_list_t *arts_thread_local_db_capture_list = NULL;
#endif

// Capture thread state - only used for periodic counter capture
static pthread_t capture_thread;
static volatile bool capture_thread_running = false;

// Time synchronization variables (exported for RemoteFunctions.c)
// timeOffset = workerTime - masterTime (positive if worker is ahead)
// To get synchronized time: local_time - timeOffset = masterTime
volatile int64_t arts_counter_time_offset = 0;
volatile bool arts_counter_time_sync_received = false;

// Get synchronized timestamp (adjusted to master node's clock)
// Call this function only after time synchronization is complete
// arts_counter_time_sync_received == true
static inline uint64_t arts_get_synced_time_stamp(void) {
  return (uint64_t)((int64_t)arts_get_time_stamp() - arts_counter_time_offset);
}

static uint64_t arts_counter_capture_counter(arts_counter_t *counter) {
  uint64_t expected = counter->start;
  while (expected) {
    uint64_t start = arts_get_time_stamp();
    if (arts_atomic_cswap_u64(&counter->start, expected, start) != expected) {
      expected = counter->start;
    } else {
      arts_atomic_fetch_add_u64(&counter->count, start - expected);
      expected = 0;
    }
  }
  return counter->count;
}

static void *arts_counter_capture_thread(void *args) {
  (void)args; // Unused - capture thread doesn't need its own counters

  // Validate counter_capture_interval to prevent division by zero
  if (arts_node_info.counter_capture_interval == 0) {
    ARTS_INFO("counter_capture_interval is 0, capture thread exiting");
    return NULL;
  }

  // Validate interval to prevent overflow (max ~18 billion ms before overflow)
  if (arts_node_info.counter_capture_interval > UINT64_MAX / 1000000) {
    ARTS_INFO("counter_capture_interval too large, capture thread exiting");
    return NULL;
  }

  // milli to nano
  uint64_t interval_ns = arts_node_info.counter_capture_interval * 1000000;

  // Time synchronization: align captures to synchronized time.
  // All nodes use master node's time (via arts_get_synced_time_stamp) so they
  // all capture at the same logical time, preventing drift between nodes.
  //
  // Epochs are RELATIVE to capture thread start, not absolute timestamps.
  // This makes epoch numbers small (0, 1, 2...) and meaningful for comparing
  // captures across different runs. All nodes use the same baseline alignment
  // so their epoch numbers match.
  uint64_t synced_time = arts_get_synced_time_stamp();

  // Compute baseline: align to the current interval boundary
  // This ensures all nodes that start within the same interval get the same
  // baseline, producing identical epoch numbers across the cluster.
  uint64_t baseline_time = (synced_time / interval_ns) * interval_ns;

  // Calculate the capture epoch relative to baseline (starts at 0)
  uint64_t capture_epoch = 0;
  // Next capture will be at the first interval boundary after baseline
  // (i.e., baseline + interval_ns for epoch 0)
  uint64_t next_capture_time = baseline_time + interval_ns;

  while (capture_thread_running) {
    synced_time = arts_get_synced_time_stamp();
    // Calculate sleep time until next aligned capture (in synced time)
    int64_t sleep_ns = (int64_t)(next_capture_time - synced_time);

    // Track current epoch for this capture
    uint64_t current_epoch = capture_epoch + 1;

    if (sleep_ns > 0) {
      nanosleep((const struct timespec[]){{sleep_ns / 1000000000,
                                           sleep_ns % 1000000000}},
                NULL);
      // Advance to the next expected capture time
      capture_epoch++;
      next_capture_time += interval_ns;
    } else {
      // We are late: next_capture_time is already in the past.
      // Compute how many intervals we are behind and jump forward
      // to the next future aligned capture time to avoid drift.
      uint64_t intervals_behind =
          (uint64_t)(((-sleep_ns) / (int64_t)interval_ns) + 1);
      ARTS_INFO(
          "Counter capture lagging: synced_time=%lf ms, "
          "next_capture_time=%lf ms, lateBy=%lf ms, skipping %lu interval(s)",
          (double)synced_time / 1000000.0, (double)next_capture_time / 1000000.0,
          (double)(-sleep_ns) / 1000000.0, intervals_behind);
      capture_epoch += intervals_behind;
      current_epoch = capture_epoch;
      next_capture_time += intervals_behind * interval_ns;
    }

    // Capture counters from all threads - store in nodeInfo.capture_arrays
    // Skip threads with NULL live_counters (not registered or already closed)
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_PERIODIC) {
        for (unsigned int t = 0; t < arts_node_info.total_thread_count; t++) {
          // Read counter value from thread's __thread storage via live_counters
          // pointer NULL means thread hasn't registered yet or has already
          // closed
          arts_counter_t *thread_counters = arts_node_info.live_counters[t];
          if (!thread_counters) {
            continue;
          }
          arts_counter_capture_t capture;
          capture.epoch = current_epoch;
          capture.value = arts_counter_capture_counter(&thread_counters[i]);
          if (arts_node_info.capture_arrays[t][i]) {
            arts_push_to_array_list(arts_node_info.capture_arrays[t][i], &capture);
          }
        }
      }
    }
  }
  return NULL;
}

void arts_counter_capture_start() {
  if (capture_thread_running) {
    ARTS_DEBUG("Trying to start capture thread which is already running");
    arts_debug_generate_seg_fault();
  }

  bool need_capture_thread = false;

  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    // Need capture thread for PERIODIC mode counters
    if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_PERIODIC) {
      need_capture_thread = true;
      break;
    }
  }

  if (need_capture_thread) {
    // RTT-based time synchronization: workers send request to master,
    // master responds with its timestamp, workers calculate offset using RTT.
    // This provides better accuracy than one-way broadcast.

    if (arts_global_rank_id == arts_global_master_rank_id) {
      // Master node: no offset needed, just set ready
      arts_counter_time_offset = 0;
      arts_counter_time_sync_received = true;
      ARTS_INFO("Time sync: Master node (rank %u), offset=0",
                arts_global_master_rank_id);
    } else {
      // Worker nodes: send sync request and wait for response
      arts_remote_time_sync_request();
      uint64_t timeout = arts_get_time_stamp() + 5000000000ULL; // 5 seconds
      while (!arts_counter_time_sync_received && arts_get_time_stamp() < timeout) {
        usleep(1000); // Wait 1ms
      }
      if (!arts_counter_time_sync_received) {
        ARTS_INFO("Time sync: Timeout waiting for master response, "
                  "using local time (offset=0)");
        arts_counter_time_offset = 0;
        arts_counter_time_sync_received = true;
      }
    }

    capture_thread_running = true;

    int ret =
        pthread_create(&capture_thread, NULL, arts_counter_capture_thread, NULL);
    if (ret) {
      ARTS_DEBUG("Failed to create capture thread: %d", ret);
      capture_thread_running = false;
    } else {
      ARTS_INFO("Counter capture thread started");
    }
  }
}

void arts_counter_capture_stop() {
  if (!capture_thread_running) {
    // No capture thread to stop - this is fine, counters still work
    return;
  }
  capture_thread_running = false;
  int ret = pthread_join(capture_thread, NULL);
  if (ret) {
    ARTS_DEBUG("Failed to join capture thread: %d", ret);
  }
}

void arts_counter_increment_by(arts_counter_t *counter, uint64_t num) {
  arts_atomic_fetch_add_u64(&counter->count, num);
}

void arts_counter_decrement_by(arts_counter_t *counter, uint64_t num) {
  arts_atomic_fetch_sub_u64(&counter->count, num);
}

void arts_counter_timer_start(arts_counter_t *counter) {
  if (arts_atomic_cswap_u64(&counter->start, 0, arts_get_time_stamp())) {
    ARTS_DEBUG("Trying to start a timer that is already started");
    arts_debug_generate_seg_fault();
  }
}

void arts_counter_timer_end(arts_counter_t *counter) {
  uint64_t end = arts_get_time_stamp();
  uint64_t start = arts_atomic_swap_u64(&counter->start, 0);
  if (!start) {
    ARTS_DEBUG("Trying to end a timer that is not started");
    arts_debug_generate_seg_fault();
  }
  arts_atomic_fetch_add_u64(&counter->count, end - start);
}

// Helper: apply one reduction step
static inline uint64_t arts_apply_reduction(uint64_t accumulator, uint64_t value,
                                          arts_counter_reduce_method_t reduce_method,
                                          unsigned int source_index) {
  switch (reduce_method) {
  case ARTS_COUNTER_REDUCE_SUM:
    return accumulator + value;
  case ARTS_COUNTER_REDUCE_MAX:
    return (accumulator < value) ? value : accumulator;
  case ARTS_COUNTER_REDUCE_MIN:
    return (accumulator > value) ? value : accumulator;
  case ARTS_COUNTER_REDUCE_MASTER:
    return (source_index == 0) ? value : accumulator;
  }
  return accumulator;
}

// Helper: convert reduce method to string for JSON output
static inline const char *
arts_reduce_method_to_string(arts_counter_reduce_method_t reduce_method) {
  switch (reduce_method) {
  case ARTS_COUNTER_REDUCE_SUM:
    return "SUM";
  case ARTS_COUNTER_REDUCE_MAX:
    return "MAX";
  case ARTS_COUNTER_REDUCE_MIN:
    return "MIN";
  case ARTS_COUNTER_REDUCE_MASTER:
    return "MASTER";
  }
  return "SUM";
}

// Helper: convert capture mode to string for JSON output
static inline const char *arts_counter_mode_to_string(unsigned int mode) {
  switch (mode) {
  case ARTS_COUNTER_MODE_ONCE:
    return "ONCE";
  case ARTS_COUNTER_MODE_PERIODIC:
    return "PERIODIC";
  default:
    return "OFF";
  }
}

// Helper: check if counter name contains "Time" (case-insensitive)
static bool arts_counter_is_time_counter(const char *name) {
  if (!name) {
    return false;
}
  const char *p = name;
  while (*p) {
    if ((*p == 'T' || *p == 't') && (*(p + 1) == 'I' || *(p + 1) == 'i') &&
        (*(p + 2) == 'M' || *(p + 2) == 'm') &&
        (*(p + 3) == 'E' || *(p + 3) == 'e')) {
      return true;
    }
    p++;
  }
  return false;
}

// Helper: write capture history array to JSON as compact single line
// Format: [[epoch, value], [epoch, value], ...]
// Epochs are absolute (synced across nodes) for proper offline merging
static void arts_write_capture_history(arts_json_writer_t *writer, uint64_t *epochs,
                                    uint64_t *values, uint64_t count) {
  if (count == 0) {
    return;
}

  // Build compact JSON string: [[e,v],[e,v],...]
  // Estimate size: each entry is at most ~40 chars, plus brackets
  size_t buf_size = (count * 45) + 10;
  char *buf = (char *)arts_malloc(buf_size);
  char *p = buf;
  *p++ = '[';

  for (uint64_t c = 0; c < count; c++) {
    if (c > 0) {
      *p++ = ',';
}
    p += sprintf(p, "[%llu,%llu]", (unsigned long long)epochs[c],
                 (unsigned long long)values[c]);
  }
  *p++ = ']';
  *p = '\0';

  arts_json_writer_write_raw_array(writer, "captureHistory", buf);
  arts_free(buf);
}

// Helper: write common metadata fields to JSON
static void arts_write_common_metadata(arts_json_writer_t *writer) {
  arts_json_writer_write_u_int64(writer, "timestamp", (uint64_t)time(NULL));
  arts_json_writer_write_string(writer, "version", "1.7.0");
  if (arts_node_info.counter_folder) {
    arts_json_writer_write_string(writer, "counter_folder",
                              arts_node_info.counter_folder);
  }
}

// Helper: check if any counters exist at given level, return count
static unsigned int arts_counters_at_level(unsigned int level) {
  unsigned int count = 0;
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (arts_counter_mode_array[i] != ARTS_COUNTER_MODE_OFF &&
        arts_counter_level_array[i] == level) {
      count++;
    }
  }
  return count;
}

// Helper: open output file, creating directory if needed
static FILE *arts_open_counter_file(const char *output_folder,
                                 const char *filename) {
  struct stat st = {0};
  if (stat(output_folder, &st) == -1) {
    mkdir(output_folder, 0755);
  }
  char filepath[1024];
  (void)snprintf(filepath, sizeof(filepath), "%s/%s", output_folder, filename);
  return fopen(filepath, "w");
}

// Helper: finalize and close JSON file
static void arts_close_counter_file(arts_json_writer_t *writer, FILE *fp) {
  arts_json_writer_end_object(writer);
  arts_json_writer_finish(writer);
  (void)fputc('\n', fp);
  (void)fclose(fp);
}

// Helper function to compute node-level reduced value across all threads
// Safe to call after threads have closed - uses saved_counters data
static uint64_t arts_compute_node_reduced_value(unsigned int index) {
  arts_counter_reduce_method_t reduce_method = arts_counter_reduce_method_array[index];
  uint64_t node_value = (reduce_method == ARTS_COUNTER_REDUCE_MIN) ? UINT64_MAX : 0;
  for (unsigned int t = 0; t < arts_node_info.total_thread_count; t++) {
    arts_counter_t *saved = arts_node_info.saved_counters[t];
    if (!saved) {
      continue; // Thread never registered or data not available
    }
    node_value =
        arts_apply_reduction(node_value, saved[index].count, reduce_method, t);
  }
  return node_value;
}

// Helper function to compute node-level reduced captures for PERIODIC mode
// Returns arrays of epochs and values, sets count. Caller must free both.
static void arts_compute_node_reduced_captures(unsigned int counter_index,
                                           uint64_t **out_epochs,
                                           uint64_t **out_values,
                                           uint64_t *out_count) {
  *out_epochs = NULL;
  *out_values = NULL;
  *out_count = 0;

  // Find the max number of captures across all threads
  uint64_t max_captures = 0;
  for (unsigned int t = 0; t < arts_node_info.total_thread_count; t++) {
    arts_array_list_t *thread_list = arts_node_info.capture_arrays[t][counter_index];
    if (thread_list && thread_list->index > max_captures) {
      max_captures = thread_list->index;
    }
  }

  if (max_captures == 0) {
    return;
  }

  // Allocate output arrays (upper bound size)
  *out_epochs = (uint64_t *)arts_malloc(max_captures * sizeof(uint64_t));
  *out_values = (uint64_t *)arts_malloc(max_captures * sizeof(uint64_t));

  // Create iterators for all threads
  arts_array_list_iterator_t **iters = (arts_array_list_iterator_t **)arts_calloc(
      arts_node_info.total_thread_count, sizeof(arts_array_list_iterator_t *));
  arts_counter_capture_t **current_captures = (arts_counter_capture_t **)arts_calloc(
      arts_node_info.total_thread_count, sizeof(arts_counter_capture_t *));

  for (unsigned int t = 0; t < arts_node_info.total_thread_count; t++) {
    arts_array_list_t *thread_list = arts_node_info.capture_arrays[t][counter_index];
    if (thread_list && thread_list->index > 0) {
      iters[t] = arts_new_array_list_iterator(thread_list);
      if (arts_array_list_has_next(iters[t])) {
        current_captures[t] = (arts_counter_capture_t *)arts_array_list_next(iters[t]);
      }
    }
  }

  // Reduce by epoch - process captures in epoch order
  uint64_t captures_written = 0;
  while (captures_written < max_captures) {
    // Find minimum epoch among all current captures
    uint64_t min_epoch = UINT64_MAX;
    for (unsigned int t = 0; t < arts_node_info.total_thread_count; t++) {
      if (current_captures[t] && current_captures[t]->epoch < min_epoch) {
        min_epoch = current_captures[t]->epoch;
      }
    }
    if (min_epoch == UINT64_MAX) {
      break;
}

    // Reduce all captures at this epoch
    arts_counter_reduce_method_t reduce_method =
        arts_counter_reduce_method_array[counter_index];
    uint64_t reduced_value =
        (reduce_method == ARTS_COUNTER_REDUCE_MIN) ? UINT64_MAX : 0;
    bool has_value = false;
    for (unsigned int t = 0; t < arts_node_info.total_thread_count; t++) {
      if (current_captures[t] && current_captures[t]->epoch == min_epoch) {
        has_value = true;
        reduced_value = arts_apply_reduction(
            reduced_value, current_captures[t]->value, reduce_method, t);
        // Advance this thread's iterator
        if (arts_array_list_has_next(iters[t])) {
          current_captures[t] =
              (arts_counter_capture_t *)arts_array_list_next(iters[t]);
        } else {
          current_captures[t] = NULL;
        }
      }
    }

    if (has_value) {
      (*out_epochs)[captures_written] = min_epoch;
      (*out_values)[captures_written] = reduced_value;
      captures_written++;
    }
  }

  *out_count = captures_written;

  // Cleanup iterators
  for (unsigned int t = 0; t < arts_node_info.total_thread_count; t++) {
    if (iters[t]) {
      arts_delete_array_list_iterator(iters[t]);
    }
  }
  arts_free(iters);
  arts_free(current_captures);
}

// Helper: write arts_id metrics to JSON
#if ENABLE_ARTS_ID_EDT_METRICS || ENABLE_ARTS_ID_DB_METRICS
static void artsWriteArtsIdMetrics(arts_json_writer_t *writer,
                                   arts_id_hash_table_t *table, bool writeEdt,
                                   bool writeDb) {
  arts_json_writer_begin_object(writer, "arts_id_metrics_t");

  if (writeEdt) {
    arts_json_writer_begin_array(writer, "edts");
    for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
      if (table->edt_metrics[i].valid) {
        arts_json_writer_begin_object(writer, NULL);
        arts_json_writer_write_u_int64(writer, "arts_id",
                                  table->edt_metrics[i].arts_id);
        arts_json_writer_write_u_int64(writer, "invocations",
                                  table->edt_metrics[i].invocations);
        arts_json_writer_write_u_int64(writer, "total_exec_ns",
                                  table->edt_metrics[i].total_exec_ns);
        arts_json_writer_write_u_int64(writer, "total_stall_ns",
                                  table->edt_metrics[i].total_stall_ns);
        arts_json_writer_end_object(writer);
      }
    }
    arts_json_writer_end_array(writer);
    arts_json_writer_write_u_int64(writer, "edt_collisions", table->edt_collisions);
  }

  if (writeDb) {
    arts_json_writer_begin_array(writer, "dbs");
    for (uint32_t i = 0; i < ARTS_ID_HASH_SIZE; i++) {
      if (table->db_metrics[i].valid) {
        arts_json_writer_begin_object(writer, NULL);
        arts_json_writer_write_u_int64(writer, "arts_id",
                                  table->db_metrics[i].arts_id);
        arts_json_writer_write_u_int64(writer, "invocations",
                                  table->db_metrics[i].invocations);
        arts_json_writer_write_u_int64(writer, "bytes_local",
                                  table->db_metrics[i].bytes_local);
        arts_json_writer_write_u_int64(writer, "bytes_remote",
                                  table->db_metrics[i].bytes_remote);
        arts_json_writer_write_u_int64(writer, "cache_misses",
                                  table->db_metrics[i].cache_misses);
        arts_json_writer_end_object(writer);
      }
    }
    arts_json_writer_end_array(writer);
    arts_json_writer_write_u_int64(writer, "db_collisions", table->db_collisions);
  }

  arts_json_writer_end_object(writer);
}
#endif

static void arts_counter_write_thread(const char *output_folder,
                                   unsigned int node_id, unsigned int thread_id) {
  if (!arts_counters_at_level(ARTS_COUNTER_LEVEL_THREAD)) {
    return;
}

  char filename[64];
  (void)snprintf(filename, sizeof(filename), "n%u_t%u.json", node_id, thread_id);
  FILE *fp = arts_open_counter_file(output_folder, filename);
  if (!fp) {
    return;
}

  arts_json_writer_t writer;
  arts_json_writer_init(&writer, fp, 2);
  arts_json_writer_begin_object(&writer, NULL);

  // Metadata
  arts_json_writer_begin_object(&writer, "metadata");
  arts_json_writer_write_u_int64(&writer, "node_id", node_id);
  arts_json_writer_write_u_int64(&writer, "thread_id", thread_id);
  arts_write_common_metadata(&writer);
  arts_json_writer_write_u_int64(&writer, "counter_capture_interval",
                            arts_node_info.counter_capture_interval);
  arts_json_writer_end_object(&writer);

  // Counters
  arts_json_writer_begin_object(&writer, "counters");
  arts_counter_t *saved = arts_node_info.saved_counters[thread_id];
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_OFF ||
        arts_counter_level_array[i] != ARTS_COUNTER_LEVEL_THREAD) {
      continue;
}

    arts_json_writer_begin_object(&writer, arts_counter_names[i]);
    arts_json_writer_write_string(&writer, "captureMode",
                              arts_counter_mode_to_string(arts_counter_mode_array[i]));
    arts_json_writer_write_string(&writer, "captureLevel", "THREAD");

    // Always write final value
    uint64_t final_value = saved[i].count;
    arts_json_writer_write_u_int64(&writer, "value", final_value);
    if (arts_counter_is_time_counter(arts_counter_names[i])) {
      arts_json_writer_write_double(&writer, "value_ms",
                                (double)final_value / 1000000.0);
    }

    if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_PERIODIC) {
      arts_array_list_t *capture_list = arts_node_info.capture_arrays[thread_id][i];
      if (capture_list && capture_list->index > 0) {
        uint64_t count = capture_list->index;
        uint64_t *epochs = (uint64_t *)arts_malloc(count * sizeof(uint64_t));
        uint64_t *values = (uint64_t *)arts_malloc(count * sizeof(uint64_t));
        arts_array_list_iterator_t *iter = arts_new_array_list_iterator(capture_list);
        for (uint64_t idx = 0; arts_array_list_has_next(iter) && idx < count;
             idx++) {
          arts_counter_capture_t *cap =
              (arts_counter_capture_t *)arts_array_list_next(iter);
          if (cap) {
            epochs[idx] = cap->epoch;
            values[idx] = cap->value;
          }
        }
        arts_delete_array_list_iterator(iter);
        arts_write_capture_history(&writer, epochs, values, count);
        arts_free(epochs);
        arts_free(values);
      }
    }
    arts_json_writer_end_object(&writer);
  }
  arts_json_writer_end_object(&writer);

#if ENABLE_ARTS_ID_EDT_METRICS || ENABLE_ARTS_ID_DB_METRICS
  bool edtThread =
      arts_counter_mode_array[ARTS_ID_EDT_METRICS] != ARTS_COUNTER_MODE_OFF &&
      arts_counter_level_array[ARTS_ID_EDT_METRICS] == ARTS_COUNTER_LEVEL_THREAD;
  bool dbThread =
      arts_counter_mode_array[ARTS_ID_DB_METRICS] != ARTS_COUNTER_MODE_OFF &&
      arts_counter_level_array[ARTS_ID_DB_METRICS] == ARTS_COUNTER_LEVEL_THREAD;
  if (edtThread || dbThread) {
    artsWriteArtsIdMetrics(
        &writer, &arts_node_info.saved_counters[thread_id]->artsIdMetricsTable,
        edtThread, dbThread);
  }
#endif

  arts_close_counter_file(&writer, fp);
}

static void arts_counter_write_node(const char *output_folder,
                                 unsigned int node_id) {
  // Write if we have NODE-level counters OR CLUSTER-level counters
  if (!arts_counters_at_level(ARTS_COUNTER_LEVEL_NODE) &&
      !arts_counters_at_level(ARTS_COUNTER_LEVEL_CLUSTER)) {
    return;
}

  char filename[64];
  (void)snprintf(filename, sizeof(filename), "n%u.json", node_id);
  FILE *fp = arts_open_counter_file(output_folder, filename);
  if (!fp) {
    return;
}

  arts_json_writer_t writer;
  arts_json_writer_init(&writer, fp, 2);
  arts_json_writer_begin_object(&writer, NULL);

  // Metadata
  arts_json_writer_begin_object(&writer, "metadata");
  arts_json_writer_write_u_int64(&writer, "node_id", node_id);
  arts_write_common_metadata(&writer);
  arts_json_writer_write_u_int64(&writer, "captureInterval",
                            arts_node_info.counter_capture_interval);
  arts_json_writer_write_u_int64(&writer, "total_threads",
                            arts_node_info.total_thread_count);
  arts_json_writer_end_object(&writer);

  // Counters
  arts_json_writer_begin_object(&writer, "counters");
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_OFF ||
        arts_counter_level_array[i] != ARTS_COUNTER_LEVEL_NODE) {
      continue;
}

    arts_json_writer_begin_object(&writer, arts_counter_names[i]);
    arts_json_writer_write_string(&writer, "captureMode",
                              arts_counter_mode_to_string(arts_counter_mode_array[i]));
    arts_json_writer_write_string(&writer, "captureLevel", "NODE");
    arts_json_writer_write_string(
        &writer, "reduce_method",
        arts_reduce_method_to_string(arts_counter_reduce_method_array[i]));

    uint64_t final_value = arts_compute_node_reduced_value(i);
    arts_json_writer_write_u_int64(&writer, "value", final_value);
    if (arts_counter_is_time_counter(arts_counter_names[i])) {
      arts_json_writer_write_double(&writer, "value_ms",
                                (double)final_value / 1000000.0);
    }

    if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_PERIODIC) {
      uint64_t *epochs;
      uint64_t *values;
      uint64_t count;
      arts_compute_node_reduced_captures(i, &epochs, &values, &count);
      arts_write_capture_history(&writer, epochs, values, count);
      if (epochs) {
        arts_free(epochs);
}
      if (values) {
        arts_free(values);
}
    }
    arts_json_writer_end_object(&writer);
  }

  // Write CLUSTER-level counters (node-reduced values for later cluster reduction)
  // These will be read by master node for file-based cluster aggregation
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_OFF ||
        arts_counter_level_array[i] != ARTS_COUNTER_LEVEL_CLUSTER) {
      continue;
}

    // MASTER reduce method counters: only emit on master node
    // Workers don't need to emit these since cluster reduction uses master's value only
    if (arts_counter_reduce_method_array[i] == ARTS_COUNTER_REDUCE_MASTER &&
        arts_global_rank_id != arts_global_master_rank_id) {
      continue;
}

    arts_json_writer_begin_object(&writer, arts_counter_names[i]);
    arts_json_writer_write_string(&writer, "captureMode",
                              arts_counter_mode_to_string(arts_counter_mode_array[i]));
    arts_json_writer_write_string(&writer, "captureLevel", "CLUSTER");
    arts_json_writer_write_string(
        &writer, "reduce_method",
        arts_reduce_method_to_string(arts_counter_reduce_method_array[i]));

    uint64_t final_value = arts_compute_node_reduced_value(i);
    arts_json_writer_write_u_int64(&writer, "value", final_value);
    if (arts_counter_is_time_counter(arts_counter_names[i])) {
      arts_json_writer_write_double(&writer, "value_ms",
                                (double)final_value / 1000000.0);
    }

    if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_PERIODIC) {
      uint64_t *epochs;
      uint64_t *values;
      uint64_t count;
      arts_compute_node_reduced_captures(i, &epochs, &values, &count);
      arts_write_capture_history(&writer, epochs, values, count);
      if (epochs) {
        arts_free(epochs);
}
      if (values) {
        arts_free(values);
}
    }
    arts_json_writer_end_object(&writer);
  }

  arts_json_writer_end_object(&writer);

#if ENABLE_ARTS_ID_EDT_METRICS || ENABLE_ARTS_ID_DB_METRICS
  bool edtNode =
      arts_counter_mode_array[ARTS_ID_EDT_METRICS] != ARTS_COUNTER_MODE_OFF &&
      arts_counter_level_array[ARTS_ID_EDT_METRICS] == ARTS_COUNTER_LEVEL_NODE;
  bool dbNode = arts_counter_mode_array[ARTS_ID_DB_METRICS] != ARTS_COUNTER_MODE_OFF &&
                arts_counter_level_array[ARTS_ID_DB_METRICS] == ARTS_COUNTER_LEVEL_NODE;
  if (edtNode || dbNode) {
    // Use thread 0's metrics as representative for node level
    // TODO: implement proper node-level reduction of arts_id metrics
    if (arts_node_info.saved_counters[0]) {
      artsWriteArtsIdMetrics(&writer,
                             &arts_node_info.saved_counters[0]->artsIdMetricsTable,
                             edtNode, dbNode);
    }
  }
#endif

  arts_close_counter_file(&writer, fp);
}

// Unified counter write function - single entry point
void arts_counter_write(const char *output_folder, unsigned int node_id,
                      unsigned int thread_id) {
  if (!output_folder) {
    return;
}

  // Write thread-level counters for this thread
  arts_counter_write_thread(output_folder, node_id, thread_id);

  // Thread 0 handles node level output
  if (thread_id == 0) {
    arts_counter_write_node(output_folder, node_id);
  }
}

// ============================================================================
// arts_id tracking wrapper functions (integrated with counter infrastructure)
// ============================================================================

void arts_counter_record_arts_id_edt(uint64_t arts_id, uint64_t exec_ns,
                                uint64_t stall_ns) {
#if ENABLE_ARTS_ID_EDT_METRICS
  if (arts_counter_mode_t[ARTS_ID_EDT_METRICS] != ARTS_COUNTER_MODE_OFF) {
    arts_id_record_edt_metrics(arts_id, exec_ns, stall_ns,
                           &arts_thread_local_arts_id_metrics);
  }
#else
  (void)arts_id;
  (void)exec_ns;
  (void)stall_ns;
#endif
}

void arts_counter_record_arts_id_db(uint64_t arts_id, uint64_t bytes_local,
                               uint64_t bytes_remote, uint64_t cache_misses) {
#if ENABLE_ARTS_ID_DB_METRICS
  if (arts_counter_mode_t[ARTS_ID_DB_METRICS] != ARTS_COUNTER_MODE_OFF) {
    arts_id_record_db_metrics(arts_id, bytes_local, bytes_remote, cache_misses,
                          &arts_thread_local_arts_id_metrics);
  }
#else
  (void)arts_id;
  (void)bytes_local;
  (void)bytes_remote;
  (void)cache_misses;
#endif
}

void arts_counter_capture_arts_id_edt(uint64_t arts_id, uint64_t exec_ns,
                                 uint64_t stall_ns) {
#if ENABLE_ARTS_ID_EDT_CAPTURES
  if (arts_counter_mode_t[ARTS_ID_EDT_CAPTURES] != ARTS_COUNTER_MODE_OFF) {
    arts_id_capture_edt_execution(arts_id, exec_ns, stall_ns,
                              arts_thread_local_edt_capture_list);
  }
#else
  (void)arts_id;
  (void)exec_ns;
  (void)stall_ns;
#endif
}

void arts_counter_capture_arts_id_db(uint64_t arts_id, uint64_t bytes_accessed,
                                uint8_t access_type) {
#if ENABLE_ARTS_ID_DB_CAPTURES
  if (arts_counter_mode_t[ARTS_ID_DB_CAPTURES] != ARTS_COUNTER_MODE_OFF) {
    arts_id_capture_db_access(arts_id, bytes_accessed, access_type,
                          arts_thread_local_db_capture_list);
  }
#else
  (void)arts_id;
  (void)bytes_accessed;
  (void)access_type;
#endif
}

// ============================================================================
// Cluster-level counter aggregation (master node reads all n{id}.json files)
// ============================================================================

// Maximum capture history entries per counter
#define MAX_CAPTURE_HISTORY 10000

// Parsed counter data from a single node's JSON file
typedef struct {
  uint64_t value;
  uint64_t captureCount;
  uint64_t *captureEpochs;
  uint64_t *captureValues;
} arts_cluster_counter_data_t;

// Skip whitespace in JSON
static const char *arts_json_skip_whitespace(const char *p) {
  while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
    p++;
}
  return p;
}

// Find a key in JSON object, return pointer to value start
static const char *arts_json_find_key(const char *json, const char *key) {
  char search_key[256];
  (void)snprintf(search_key, sizeof(search_key), "\"%s\"", key);
  const char *found = strstr(json, search_key);
  if (!found) {
    return NULL;
}
  found += strlen(search_key);
  found = arts_json_skip_whitespace(found);
  if (*found != ':') {
    return NULL;
}
  return arts_json_skip_whitespace(found + 1);
}

// Parse uint64 from JSON
static uint64_t arts_json_parse_u_int64(const char *p) {
  return strtoull(p, NULL, 10);
}

// Parse capture history array: [[epoch,value],[epoch,value],...]
// Returns number of entries parsed
static uint64_t arts_json_parse_capture_history(const char *p, uint64_t *epochs,
                                            uint64_t *values,
                                            uint64_t max_entries) {
  if (!p || *p != '[') {
    return 0;
}
  p++; // Skip opening [

  uint64_t count = 0;
  while (*p && *p != ']' && count < max_entries) {
    p = arts_json_skip_whitespace(p);
    if (*p == ',') {
      p++;
}
    p = arts_json_skip_whitespace(p);
    if (*p != '[') {
      break;
}
    p++; // Skip inner [

    // Parse epoch
    epochs[count] = strtoull(p, (char **)&p, 10);
    p = arts_json_skip_whitespace(p);
    if (*p == ',') {
      p++;
}
    p = arts_json_skip_whitespace(p);

    // Parse value
    values[count] = strtoull(p, (char **)&p, 10);
    p = arts_json_skip_whitespace(p);
    if (*p == ']') {
      p++; // Skip inner ]
}
    count++;
  }
  return count;
}

// Find the end of a JSON object (matching braces)
static const char *arts_json_find_object_end(const char *p) {
  if (*p != '{') {
    return NULL;
}
  int depth = 1;
  p++;
  while (*p && depth > 0) {
    if (*p == '{') {
      depth++;
    } else if (*p == '}') {
      depth--;
    } else if (*p == '"') {
      p++;
      while (*p && *p != '"') {
        if (*p == '\\') {
          p++;
}
        p++;
      }
    }
    if (*p) {
      p++;
}
  }
  return p;
}

// Read and parse a node's counter JSON file
static bool arts_read_node_counter_file(const char *filepath,
                                    arts_cluster_counter_data_t *counter_data) {
  FILE *fp = fopen(filepath, "r");
  if (!fp) {
    return false;
}

  // Get file size
  (void)fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  (void)fseek(fp, 0, SEEK_SET);

  if (file_size <= 0 || file_size > 10L * 1024 * 1024) { // Max 10MB
    (void)fclose(fp);
    return false;
  }

  char *json = (char *)arts_malloc(file_size + 1);
  size_t bytes_read = fread(json, 1, file_size, fp);
  (void)fclose(fp);
  json[bytes_read] = '\0';

  // Find "counters" object
  const char *counters = arts_json_find_key(json, "counters");
  if (!counters) {
    arts_free(json);
    return false;
  }

  // Parse each CLUSTER-level counter
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_OFF ||
        arts_counter_level_array[i] != ARTS_COUNTER_LEVEL_CLUSTER) {
      continue;
    }

    // Find this counter's object
    const char *counter_obj = arts_json_find_key(counters, arts_counter_names[i]);
    if (!counter_obj) {
      continue;
}

    // Find the end of this counter object for scoped searching
    const char *counter_end = arts_json_find_object_end(counter_obj);
    size_t counter_len = counter_end - counter_obj;
    char *counter_json = (char *)arts_malloc(counter_len + 1);
    memcpy(counter_json, counter_obj, counter_len);
    counter_json[counter_len] = '\0';

    // Parse "value"
    const char *value_ptr = arts_json_find_key(counter_json, "value");
    if (value_ptr) {
      counter_data[i].value = arts_json_parse_u_int64(value_ptr);
    }

    // Parse "captureHistory" if present
    const char *history_ptr = arts_json_find_key(counter_json, "captureHistory");
    if (history_ptr) {
      counter_data[i].captureEpochs =
          (uint64_t *)arts_malloc(MAX_CAPTURE_HISTORY * sizeof(uint64_t));
      counter_data[i].captureValues =
          (uint64_t *)arts_malloc(MAX_CAPTURE_HISTORY * sizeof(uint64_t));
      counter_data[i].captureCount = arts_json_parse_capture_history(
          history_ptr, counter_data[i].captureEpochs,
          counter_data[i].captureValues, MAX_CAPTURE_HISTORY);
    }

    arts_free(counter_json);
  }

  arts_free(json);
  return true;
}

// Merge capture histories from all nodes for a single counter
static void arts_merge_capture_histories(arts_cluster_counter_data_t *node_data,
                                      unsigned int node_count,
                                      unsigned int counter_index,
                                      uint64_t **out_epochs, uint64_t **out_values,
                                      uint64_t *out_count) {
  *out_epochs = NULL;
  *out_values = NULL;
  *out_count = 0;

  // Find max capture count across all nodes
  uint64_t max_captures = 0;
  for (unsigned int n = 0; n < node_count; n++) {
    if (node_data[(n * NUM_COUNTER_TYPES) + counter_index].captureCount >
        max_captures) {
      max_captures = node_data[(n * NUM_COUNTER_TYPES) + counter_index].captureCount;
    }
  }
  if (max_captures == 0) {
    return;
}

  // Allocate output arrays
  *out_epochs = (uint64_t *)arts_malloc(max_captures * sizeof(uint64_t));
  *out_values = (uint64_t *)arts_malloc(max_captures * sizeof(uint64_t));

  // Track current index for each node
  uint64_t *node_indices =
      (uint64_t *)arts_calloc(node_count, sizeof(uint64_t));

  uint64_t count = 0;
  arts_counter_reduce_method_t reduce_method =
      arts_counter_reduce_method_array[counter_index];

  while (count < max_captures) {
    // Find minimum epoch among current positions
    uint64_t min_epoch = UINT64_MAX;
    for (unsigned int n = 0; n < node_count; n++) {
      arts_cluster_counter_data_t *data =
          &node_data[(n * NUM_COUNTER_TYPES) + counter_index];
      if (node_indices[n] < data->captureCount &&
          data->captureEpochs[node_indices[n]] < min_epoch) {
        min_epoch = data->captureEpochs[node_indices[n]];
      }
    }
    if (min_epoch == UINT64_MAX) {
      break;
}

    // Reduce all values at this epoch
    uint64_t reduced_value =
        (reduce_method == ARTS_COUNTER_REDUCE_MIN) ? UINT64_MAX : 0;
    bool has_value = false;

    for (unsigned int n = 0; n < node_count; n++) {
      arts_cluster_counter_data_t *data =
          &node_data[(n * NUM_COUNTER_TYPES) + counter_index];
      if (node_indices[n] < data->captureCount &&
          data->captureEpochs[node_indices[n]] == min_epoch) {
        has_value = true;
        reduced_value = arts_apply_reduction(
            reduced_value, data->captureValues[node_indices[n]], reduce_method, n);
        node_indices[n]++;
      }
    }

    if (has_value) {
      (*out_epochs)[count] = min_epoch;
      (*out_values)[count] = reduced_value;
      count++;
    }
  }

  *out_count = count;
  arts_free(node_indices);
}

// Write cluster-aggregated counter file
void arts_counter_write_cluster(const char *output_folder, unsigned int node_count) {
  if (!output_folder || node_count == 0) {
    return;
}

  // Check if we have any CLUSTER-level counters
  if (!arts_counters_at_level(ARTS_COUNTER_LEVEL_CLUSTER)) {
    return;
}

  ARTS_INFO("Aggregating cluster counters from %u nodes", node_count);

  // Allocate storage for all nodes' counter data
  // Layout: node_data[node_id * NUM_COUNTER_TYPES + counter_index]
  arts_cluster_counter_data_t *node_data = (arts_cluster_counter_data_t *)arts_calloc(
      (size_t)node_count * NUM_COUNTER_TYPES, sizeof(arts_cluster_counter_data_t));

  // Read each node's JSON file, polling until all are available
  unsigned int nodes_read = 0;
  bool *node_read = (bool *)arts_calloc(node_count, sizeof(bool));
  int max_retries = 100; // 100 * 100ms = 10 seconds

  for (int attempt = 0; attempt < max_retries && nodes_read < node_count;
       attempt++) {
    for (unsigned int n = 0; n < node_count; n++) {
      if (node_read[n]) {
        continue;
}
      char filepath[1024];
      (void)snprintf(filepath, sizeof(filepath), "%s/n%u.json", output_folder, n);
      if (arts_read_node_counter_file(filepath,
                                  &node_data[(size_t)n * NUM_COUNTER_TYPES])) {
        node_read[n] = true;
        nodes_read++;
      }
    }
    if (nodes_read < node_count) {
      usleep(100000); // 100ms
}
  }
  for (unsigned int n = 0; n < node_count; n++) {
    if (!node_read[n]) {
      ARTS_INFO("Warning: Could not read counter file for node %u after "
                "timeout",
                n);
}
  }
  arts_free(node_read);

  if (nodes_read == 0) {
    ARTS_INFO("No node counter files found, skipping cluster aggregation");
    arts_free(node_data);
    return;
  }

  // Open output file
  FILE *fp = arts_open_counter_file(output_folder, "cluster.json");
  if (!fp) {
    arts_free(node_data);
    return;
  }

  arts_json_writer_t writer;
  arts_json_writer_init(&writer, fp, 2);
  arts_json_writer_begin_object(&writer, NULL);

  // Metadata
  arts_json_writer_begin_object(&writer, "metadata");
  arts_json_writer_write_string(&writer, "type", "cluster");
  arts_json_writer_write_u_int64(&writer, "node_count", node_count);
  arts_json_writer_write_u_int64(&writer, "nodes_read", nodes_read);
  arts_write_common_metadata(&writer);
  arts_json_writer_write_u_int64(&writer, "captureInterval",
                            arts_node_info.counter_capture_interval);
  arts_json_writer_end_object(&writer);

  // Counters
  arts_json_writer_begin_object(&writer, "counters");
  for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
    if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_OFF ||
        arts_counter_level_array[i] != ARTS_COUNTER_LEVEL_CLUSTER) {
      continue;
    }

    arts_counter_reduce_method_t reduce_method = arts_counter_reduce_method_array[i];

    arts_json_writer_begin_object(&writer, arts_counter_names[i]);
    arts_json_writer_write_string(&writer, "captureMode",
                              arts_counter_mode_to_string(arts_counter_mode_array[i]));
    arts_json_writer_write_string(&writer, "captureLevel", "CLUSTER");
    arts_json_writer_write_string(&writer, "reduce_method",
                              arts_reduce_method_to_string(reduce_method));

    // Reduce final values across all nodes
    uint64_t cluster_value =
        (reduce_method == ARTS_COUNTER_REDUCE_MIN) ? UINT64_MAX : 0;
    for (unsigned int n = 0; n < node_count; n++) {
      cluster_value = arts_apply_reduction(
          cluster_value, node_data[(n * NUM_COUNTER_TYPES) + i].value,
          reduce_method, n);
    }
    arts_json_writer_write_u_int64(&writer, "value", cluster_value);
    if (arts_counter_is_time_counter(arts_counter_names[i])) {
      arts_json_writer_write_double(&writer, "value_ms",
                                (double)cluster_value / 1000000.0);
    }

    // Merge and write capture histories if PERIODIC
    if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_PERIODIC) {
      uint64_t *epochs;
      uint64_t *values;
      uint64_t count;
      arts_merge_capture_histories(node_data, node_count, i, &epochs, &values,
                                &count);
      if (count > 0) {
        arts_write_capture_history(&writer, epochs, values, count);
        arts_free(epochs);
        arts_free(values);
      }
    }

    arts_json_writer_end_object(&writer);
  }
  arts_json_writer_end_object(&writer);

  arts_close_counter_file(&writer, fp);

  // Cleanup allocated capture history arrays
  for (unsigned int n = 0; n < node_count; n++) {
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      arts_cluster_counter_data_t *data = &node_data[(n * NUM_COUNTER_TYPES) + i];
      if (data->captureEpochs) {
        arts_free(data->captureEpochs);
}
      if (data->captureValues) {
        arts_free(data->captureValues);
}
    }
  }
  arts_free(node_data);

  ARTS_INFO("Cluster counter aggregation complete: %s/cluster.json",
            output_folder);
}
