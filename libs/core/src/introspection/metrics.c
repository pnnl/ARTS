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

#include "arts/introspection/metrics.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// #include "arts.h"
#include "arts/utils/malloc.h"
#include "arts/runtime/globals.h"
// #include "arts/runtime/network/remote_functions.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/utils/atomics.h"

#define NANOSECS 1000000000
#define LOCAL_TIME_STAMP arts_get_time_stamp
#define GLOBAL_TIME_STAMP arts_get_time_stamp

const char *const arts_metric_name[] = {"ARTS_METRIC_EDT_THROUGHPUT",
                                      "ARTS_METRIC_EDT_QUEUE",
                                      "ARTS_METRIC_EDT_STEAL_ATTEMPT",
                                      "ARTS_METRIC_EDT_STEAL",
                                      "ARTS_METRIC_EDT_LAST_LOCAL_HIT",
                                      "ARTS_METRIC_EDT_SIGNAL_THROUGHPUT",
                                      "ARTS_METRIC_EVENT_SIGNAL_THROUGHPUT",
                                      "ARTS_METRIC_PERSISTENT_EVENT_SIGNAL_THROUGHPUT",
                                      "ARTS_METRIC_GET_BW",
                                      "ARTS_METRIC_PUT_BW",
                                      "ARTS_METRIC_NETWORK_SEND_BW",
                                      "ARTS_METRIC_NETWORK_RECIEVE_BW",
                                      "ARTS_METRIC_NETWORK_QUEUE_PUSH",
                                      "ARTS_METRIC_NETWORK_QUEUE_POP",
                                      "ARTS_METRIC_YIELD_BW",
                                      "ARTS_METRIC_GPU_EDT",
                                      "ARTS_METRIC_GPU_GC",
                                      "ARTS_METRIC_GPU_GCBW",
                                      "ARTS_METRIC_GPU_BW_PUSH",
                                      "ARTS_METRIC_GPU_BW_PULL",
                                      "ARTS_METRIC_GPU_BUFFER_FLUSH",
                                      "ARTS_METRIC_GPU_SYNC",
                                      "ARTS_METRIC_GPU_SYNC_DELETE",
                                      "ARTS_METRIC_MALLOC_BW",
                                      "ARTS_METRIC_FREE_BW",
                                      "ARTS_METRIC_REMOTE_SHUTDOWN_MSG",
                                      "ARTS_METRIC_REMOTE_EDT_SIGNAL_MSG",
                                      "ARTS_METRIC_REMOTE_SIGNAL_EDT_WITH_PTR_MSG",
                                      "ARTS_METRIC_REMOTE_EVENT_SATISFY_SLOT_MSG",
                                      "ARTS_METRIC_REMOTE_ADD_DEPENDENCE_MSG",
                                      "ARTS_METRIC_REMOTE_DB_REQUEST_MSG",
                                      "ARTS_METRIC_REMOTE_DB_SEND_MSG",
                                      "ARTS_METRIC_REMOTE_INVALIDATE_DB_MSG",
                                      "ARTS_METRIC_REMOTE_DB_UPDATE_GUID_MSG",
                                      "ARTS_METRIC_REMOTE_DB_UPDATE_MSG",
                                      "ARTS_METRIC_REMOTE_DB_DESTROY_MSG",
                                      "ARTS_METRIC_REMOTE_DB_DESTROY_FORWARD_MSG",
                                      "ARTS_METRIC_REMOTE_DB_CLEAN_FORWARD_MSG",
                                      "ARTS_METRIC_REMOTE_DB_MOVE_REQ_MSG",
                                      "ARTS_METRIC_REMOTE_EDT_MOVE_MSG",
                                      "ARTS_METRIC_REMOTE_EVENT_MOVE_MSG",
                                      "ARTS_METRIC_REMOTE_DB_MOVE_MSG",
                                      "ARTS_METRIC_REMOTE_PINGPONG_TEST_MSG",
                                      "ARTS_METRIC_REMOTE_METRIC_UPDATE_MSG",
                                      "ARTS_METRIC_REMOTE_DB_FULL_REQUEST_MSG",
                                      "ARTS_METRIC_REMOTE_DB_FULL_SEND_MSG",
                                      "ARTS_METRIC_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG",
                                      "ARTS_METRIC_REMOTE_GET_FROM_DB_MSG",
                                      "ARTS_METRIC_REMOTE_PUT_IN_DB_MSG",
                                      "ARTS_METRIC_REMOTE_SEND_MSG",
                                      "ARTS_METRIC_EPOCH_INIT_MSG",
                                      "ARTS_METRIC_EPOCH_INIT_POOL_MSG",
                                      "ARTS_METRIC_EPOCH_REQ_MSG",
                                      "ARTS_METRIC_EPOCH_SEND_MSG",
                                      "ARTS_METRIC_EPOCH_DELETE_MSG",
                                      "ARTS_METRIC_ATOMIC_ADD_ARRAYDB_MSG",
                                      "ARTS_METRIC_ATOMIC_CAS_ARRAYDB_MSG",
                                      "ARTS_METRIC_REMOTE_BUFFER_SEND_MSG",
                                      "ARTS_METRIC_REMOTE_CONTEXT_SIG_MSG",
                                      "ARTS_METRIC_REMOTE_DB_RENAME_MSG",
                                      "ARTS_METRIC_DEFAULT_MEMORY_SIZE",
                                      "ARTS_METRIC_EDT_MEMORY_SIZE",
                                      "ARTS_METRIC_EVENT_MEMORY_SIZE",
                                      "ARTS_METRIC_PERSISTENT_EVENT_MEMORY_SIZE",
                                      "ARTS_METRIC_DB_MEMORY_SIZE",
                                      "ARTS_METRIC_BUFFER_MEMORY_SIZE",
                                      "ARTS_METRIC_DB_COUNT"};

uint64_t **count_window;
uint64_t **time_window;
uint64_t **max_total;

char *print_totals_to_file = NULL;
volatile unsigned int inspector_on = 0;
arts_inspector_t *inspector = NULL;
arts_inspector_stats_t *stats = NULL;
arts_inspector_shots_t *inspector_shots = NULL;
arts_packet_inspector_t *packet_inspector = NULL;

__thread bool inspector_ignore = 0;

static int metric_default_enabled = 1;
static int metric_enabled_override[ARTS_METRIC_LAST_TYPE];
static bool metric_override_initialized = false;

static void ensure_metric_overrides() {
  if (!metric_override_initialized) {
    for (int i = 0; i < ARTS_METRIC_LAST_TYPE; i++) {
      metric_enabled_override[i] = -1;
}
    metric_override_initialized = true;
  }
}

static int metric_index_from_name(const char *name) {
  if (!name) {
    return -1;
}
  for (int i = 0; i < ARTS_METRIC_LAST_TYPE; i++) {
    const char *candidate = arts_metric_name[i];
    if (candidate && !strcasecmp(candidate, name)) {
      return i;
}
  }
  return -1;
}

static inline bool metric_is_enabled(arts_metric_type_t type) {
  if (!metric_override_initialized) {
    return metric_default_enabled;
}
  int override = metric_enabled_override[type];
  if (override != -1) {
    return override;
}
  return metric_default_enabled;
}

static arts_performance_unit_t *get_metric(arts_metric_type_t type,
                                      arts_metric_level_t level);
static arts_metric_level_t update_performance_core_metric(unsigned int core,
                                                   arts_metric_type_t type,
                                                   arts_metric_level_t level,
                                                   uint64_t to_add, bool sub);

// void ARTS_METRICS_CONFIG_SET_DEFAULT_ENABLED(bool enabled) {
//   ensure_metric_overrides();
//   metric_default_enabled = enabled ? 1 : 0;
// }

// void ARTS_METRICS_CONFIG_SET_ENABLED(const char *name, bool enabled) {
//   ensure_metric_overrides();
//   int index = metric_index_from_name(name);
//   if (index >= 0)
//     metric_enabled_override[index] = enabled ? 1 : 0;
// }

// void ARTS_METRICS_TRIGGER_EVENT(arts_metric_type_t metricType, arts_metric_level_t
// level,
//                              uint64_t value) {
//   if (!inspector_on || inspector_ignore || !metric_is_enabled(metricType))
//     return;

//   update_performance_core_metric(arts_thread_info.thread_id, metricType, level,
//   value,
//                               false);
// }

// void ARTS_METRICS_TRIGGER_TIMER_EVENT(arts_metric_type_t metricType,
//                                   arts_metric_level_t level, bool start) {
//   if (!inspector_on || inspector_ignore || !metric_is_enabled(metricType))
//     return;

//   arts_performance_unit_t *metric = get_metric(metricType, level);
//   if (metric) {
//     if (start)
//       metric->firstTimeStamp = metric->timeMethod();
//   }
// }

// void ARTS_METRICS_TOGGLE_THREAD() {
//   inspector_ignore = !inspector_ignore;
//   ARTS_DEBUG("II: %u\n", inspector_ignore);
// }

// uint64_t ARTS_METRICS_GET_INSPECTOR_TIME() {
//   return inspector ? inspector->startTimeStamp : 0;
// }

// bool ARTS_METRICS_IS_ACTIVE() { return (inspector_on); }

// void ARTS_METRICS_START(unsigned int startPoint) {
//   if (inspector && inspector->startPoint == startPoint) {
//     inspector_on = 1;
//     inspector->startTimeStamp = GLOBAL_TIME_STAMP();
//   }
// }

// void ARTS_METRICS_STOP() {
//   if (inspector) {
//     inspector_on = 0;
//     inspector->endTimeStamp = GLOBAL_TIME_STAMP();
//   }
// }

static void print_metrics() {
  for (unsigned int i = 0; i < ARTS_METRIC_LAST_TYPE; i++) {
    ARTS_INFO("%35s %ld %ld %ld %ld %ld %ld %ld %ld %ld\n", arts_metric_name[i],
              count_window[i][0], count_window[i][1], count_window[i][2],
              time_window[i][0], time_window[i][1], time_window[i][2],
              max_total[i][0], max_total[i][1], max_total[i][2]);
  }
}

// void ARTS_METRICS_INIT_INTROSPECTOR(unsigned int startPoint) {
//   ARTS_DEBUG("count_window %u\n", sizeof(uint64_t *) * ARTS_METRIC_LAST_TYPE);
//   count_window = arts_malloc(sizeof(uint64_t *) * ARTS_METRIC_LAST_TYPE);
//   ARTS_DEBUG("time_window %u\n", sizeof(uint64_t *) * ARTS_METRIC_LAST_TYPE);
//   time_window = arts_malloc(sizeof(uint64_t *) * ARTS_METRIC_LAST_TYPE);
//   ARTS_DEBUG("max_total %u\n", sizeof(uint64_t *) * ARTS_METRIC_LAST_TYPE);
//   max_total = arts_malloc(sizeof(uint64_t *) * ARTS_METRIC_LAST_TYPE);

//   for (unsigned int i = 0; i < ARTS_METRIC_LAST_TYPE; i++) {
//     ARTS_DEBUG("count_window[%u] %u\n", i, sizeof(uint64_t) *
//     ARTS_METRICLEVELS); count_window[i] = arts_malloc(sizeof(uint64_t) *
//     ARTS_METRICLEVELS); ARTS_DEBUG("time_window[%u] %u\n", i, sizeof(uint64_t)
//     * ARTS_METRICLEVELS); time_window[i] = arts_malloc(sizeof(uint64_t) *
//     ARTS_METRICLEVELS); ARTS_DEBUG("max_total[%u] %u\n", i, sizeof(uint64_t) *
//     ARTS_METRICLEVELS); max_total[i] = arts_malloc(sizeof(uint64_t) *
//     ARTS_METRICLEVELS); for (unsigned int j = 0; j < ARTS_METRICLEVELS; j++) {
//       count_window[i][j] = -1;
//       time_window[i][j] = -1;
//       max_total[i][j] = -1;
//     }
//   }

//   if (!arts_global_rank_id)
//     print_metrics();
//   ARTS_DEBUG("inspector %u\n", sizeof(arts_inspector_t));
//   inspector = arts_calloc(1, sizeof(arts_inspector_t));
//   inspector->startPoint = startPoint;
//   ARTS_DEBUG("inspector->coreMetric %u\n", sizeof(arts_performance_unit_t) *
//                                                ARTS_METRIC_LAST_TYPE *
//                                                arts_node_info.total_thread_count);
//   inspector->coreMetric =
//       arts_calloc(ARTS_METRIC_LAST_TYPE * arts_node_info.total_thread_count,
//                  sizeof(arts_performance_unit_t));
//   for (unsigned int i = 0; i < arts_node_info.total_thread_count; i++) {
//     for (unsigned int j = 0; j < ARTS_METRIC_LAST_TYPE; j++) {
//       inspector->coreMetric[i * ARTS_METRIC_LAST_TYPE + j].max_total =
//           max_total[j][0];
//       inspector->coreMetric[i * ARTS_METRIC_LAST_TYPE + j].timeMethod =
//           LOCAL_TIME_STAMP;
//     }
//   }

//   inspector->nodeMetric =
//       arts_calloc(ARTS_METRIC_LAST_TYPE, sizeof(arts_performance_unit_t));
//   for (unsigned int j = 0; j < ARTS_METRIC_LAST_TYPE; j++) {
//     inspector->nodeMetric[j].max_total = max_total[j][1];
//     inspector->nodeMetric[j].timeMethod = GLOBAL_TIME_STAMP;
//   }

//   inspector->systemMetric =
//       arts_calloc(ARTS_METRIC_LAST_TYPE, sizeof(arts_performance_unit_t));
//   for (unsigned int j = 0; j < ARTS_METRIC_LAST_TYPE; j++) {
//     inspector->systemMetric[j].max_total = max_total[j][2];
//     inspector->systemMetric[j].timeMethod = GLOBAL_TIME_STAMP;
//   }

//   ARTS_DEBUG("stats %u\n", sizeof(arts_inspector_stats_t));
//   stats = arts_calloc(1, sizeof(arts_inspector_stats_t));
//   ARTS_DEBUG("packet_inspector %u\n", sizeof(arts_packet_inspector_t));
//   packet_inspector = arts_calloc(1, sizeof(arts_packet_inspector_t));
//   packet_inspector->minPacket = (uint64_t)-1;
//   packet_inspector->maxPacket = 0;
//   packet_inspector->intervalMin = (uint64_t)-1;
//   packet_inspector->intervalMax = 0;
// }

static bool metric_try_lock(arts_metric_level_t level, arts_performance_unit_t *metric) {
  if (level == ARTS_METRIC_THREAD) {
    return true;
}

  unsigned int local;
  while (1) {
    local = arts_atomic_cswap(&metric->lock, 0U, 1U);
    if (local != 2U) {
      break;
}
  }
  return (local == 0U);
}

static void metric_lock(arts_metric_level_t level, arts_performance_unit_t *metric) {
  if (level == ARTS_METRIC_THREAD) {
    return;
}
  while (!arts_atomic_cswap(&metric->lock, 0U, 1U)) {
    ;
}
}

static void metric_unlock(arts_performance_unit_t *metric) { metric->lock = 0U; }

static arts_performance_unit_t *get_metric(arts_metric_type_t type,
                                      arts_metric_level_t level) {
  arts_performance_unit_t *metric = NULL;
  if (inspector) {
    switch (level) {
    case ARTS_METRIC_THREAD:
      metric =
          &inspector->coreMetric[(arts_thread_info.thread_id * ARTS_METRIC_LAST_TYPE) +
                                 type];
      break;
    case ARTS_METRIC_NODE:
      metric = &inspector->nodeMetric[type];
      break;
    case ARTS_METRIC_SYSTEM:
      metric = &inspector->systemMetric[type];
      break;
    default:
      metric = NULL;
      break;
    }
  }
  return metric;
}

// uint64_t ARTS_METRICS_GET_TOTAL(arts_metric_type_t type, arts_metric_level_t level) {
//   arts_performance_unit_t *metric = get_metric(type, level);
//   return (metric) ? metric->totalCount : 0;
// }

// double ARTS_METRICS_GET_RATE(arts_metric_type_t type, arts_metric_level_t level,
//                           bool last) {
//   arts_performance_unit_t *metric = get_metric(type, level);
//   if (metric) {
//     uint64_t local_window_time_stamp;
//     uint64_t local_window_count_stamp;
//     uint64_t localCurrentCountStamp;
//     uint64_t localCurrentTimeStamp;

//     metric_lock(level, metric);
//     if (last) {
//       local_window_time_stamp = metric->lastWindowTimeStamp;
//       local_window_count_stamp = metric->lastWindowCountStamp;
//       localCurrentCountStamp = metric->windowCountStamp;
//       localCurrentTimeStamp = metric->windowTimeStamp;
//       metric_unlock(metric);
//     } else {
//       local_window_time_stamp = metric->windowTimeStamp;
//       local_window_count_stamp = metric->windowCountStamp;
//       metric_unlock(metric);
//       localCurrentCountStamp = metric->totalCount;
//       localCurrentTimeStamp = metric->timeMethod();
//     }

//     if (localCurrentCountStamp && localCurrentTimeStamp) {
//       double num = (double)(localCurrentCountStamp - local_window_count_stamp);
//       double den = (double)(localCurrentTimeStamp - local_window_time_stamp);
//       ARTS_INFO("%u %s %lf / %lf\n", level, arts_metric_name[type], num, den);
//       return num / den / 1E9;
//     }
//   }
//   return 0;
// }

// double ARTS_METRICS_GET_TOTAL_RATE(arts_metric_type_t type, arts_metric_level_t level) {
//   arts_performance_unit_t *metric = get_metric(type, level);
//   if (metric) {
//     double num = (double)metric->totalCount;
//     double den = (double)metric->timeMethod() - inspector->startTimeStamp;
//     ARTS_INFO("%u %s %lf / %lf\n", level, arts_metric_name[type], num, den);
//     return num / den;
//   }
//   return 0;
// }

// double ARTS_METRICS_TEST(arts_metric_type_t type, arts_metric_level_t level,
//                        uint64_t num) {
//   arts_performance_unit_t *metric = get_metric(type, level);
//   if (metric && num) {
//     double tot = (double)metric->totalCount;
//     if (tot) {
//       double dif = (double)metric->timeMethod() - inspector->startTimeStamp;
//       double temp = ((double)num * dif) / tot;
//       return temp;
//     }
//     return 100000;
//   }
//   return 0;
// }

// uint64_t ARTS_METRICS_GET_RATE_U64(arts_metric_type_t type, arts_metric_level_t level,
//                                bool last) {
//   arts_performance_unit_t *metric = get_metric(type, level);
//   if (metric) {
//     uint64_t local_window_time_stamp;
//     uint64_t local_window_count_stamp;
//     uint64_t localCurrentCountStamp;
//     uint64_t localCurrentTimeStamp;

//     metric_lock(level, metric);
//     if (last) {
//       local_window_time_stamp = metric->lastWindowTimeStamp;
//       local_window_count_stamp = metric->lastWindowCountStamp;
//       localCurrentCountStamp = metric->windowCountStamp;
//       localCurrentTimeStamp = metric->windowTimeStamp;
//       metric_unlock(metric);
//     } else {
//       local_window_time_stamp = metric->windowTimeStamp;
//       local_window_count_stamp = metric->windowCountStamp;
//       metric_unlock(metric);
//       localCurrentCountStamp = metric->totalCount;
//       localCurrentTimeStamp = metric->timeMethod();
//     }

//     if (localCurrentCountStamp && localCurrentTimeStamp &&
//         localCurrentCountStamp > local_window_count_stamp) {
//       return (localCurrentTimeStamp - local_window_time_stamp) /
//              (localCurrentCountStamp - local_window_count_stamp);
//     }
//   }
//   return 0;
// }

// uint64_t ARTS_METRICS_GET_RATE_U64_DIFF(arts_metric_type_t type, arts_metric_level_t
// level,
//                                    uint64_t *total) {
//   arts_performance_unit_t *metric = get_metric(type, level);
//   if (metric) {
//     metric_lock(level, metric);
//     uint64_t local_window_time_stamp = metric->windowTimeStamp;
//     uint64_t local_window_count_stamp = metric->windowCountStamp;
//     uint64_t lastWindowTimeStamp = metric->lastWindowTimeStamp;
//     uint64_t lastWindowCountStamp = metric->lastWindowCountStamp;
//     metric_unlock(metric);

//     uint64_t localCurrentCountStamp = metric->totalCount;
//     uint64_t localCurrentTimeStamp = metric->timeMethod();
//     *total = localCurrentCountStamp;
//     if (localCurrentCountStamp) {
//       uint64_t diff = localCurrentCountStamp - local_window_count_stamp;
//       if (diff && local_window_time_stamp) {
//         return (localCurrentTimeStamp - local_window_time_stamp) / diff;
//       }
//       diff = local_window_count_stamp - lastWindowCountStamp;
//       if (diff && local_window_count_stamp && lastWindowTimeStamp) {
//         return (local_window_count_stamp - lastWindowTimeStamp) / diff;
//       }
//     }
//   }
//   return 0;
// }

// uint64_t ARTS_METRICS_GET_TOTAL_RATE_U64(arts_metric_type_t type, arts_metric_level_t
// level,
//                                     uint64_t *total, uint64_t *time_stamp) {
//   arts_performance_unit_t *metric = get_metric(type, level);
//   if (metric) {
//     uint64_t localCurrentCountStamp = *total = metric->totalCount;
//     uint64_t localCurrentTimeStamp = metric->timeMethod();
//     *time_stamp = localCurrentTimeStamp;
//     uint64_t start_time = metric->firstTimeStamp;
//     if (start_time && localCurrentCountStamp) {
//       return (localCurrentTimeStamp - start_time) / localCurrentCountStamp;
//     }
//   }
//   return 0;
// }

// void ARTS_METRICS_HANDLE_REMOTE_UPDATE(arts_metric_type_t type, arts_metric_level_t
// level,
//                                    uint64_t to_add, bool sub) {
//   arts_performance_unit_t *metric = get_metric(type, level);
//   if (metric) {
//     metric_lock(level, metric);
//     if (sub) {
//       metric->windowCountStamp -= to_add;
//       metric->totalCount -= to_add;
//     } else {
//       metric->windowCountStamp += to_add;
//       metric->totalCount += to_add;
//     }
//     metric_unlock(metric);
//     arts_atomic_add_u64(&stats->remoteUpdates, 1);
//   }
// }

static void internal_update_max(arts_metric_level_t level,
                              arts_performance_unit_t *metric, uint64_t total) {
  uint64_t entry = metric->max_total;
  uint64_t local_max = metric->max_total;
  if (local_max > total) {
    return;
}
  if (level == ARTS_METRIC_THREAD) {
    metric->max_total = total;
  } else {
    while (local_max < total) {
      local_max = arts_atomic_cswap_u64(&metric->max_total, local_max, total);
    }
  }
}

static uint64_t internal_observe_max(arts_metric_level_t level,
                                   arts_performance_unit_t *metric) {
  uint64_t max = -1;
  if (metric->max_total != -1) {
    if (level == ARTS_METRIC_THREAD) {
      max = metric->max_total;
      metric->max_total = metric->totalCount;
    } else {
      max = arts_atomic_swap_u64(&metric->max_total, metric->totalCount);
    }
  }
  return max;
}

static bool single_metric_update(arts_metric_type_t type, arts_metric_level_t level,
                               uint64_t *to_add, bool *sub,
                               arts_performance_unit_t *metric) {
  if (!count_window[type][level] || !time_window[type][level]) {
    return true;
}

  if (count_window[type][level] == -1 && time_window[type][level] == -1) {
    return false;
}

  uint64_t total_stamp = 0;
  if (*to_add) {
    if (*sub) {
      if (metric->totalCount < *to_add) {
        ARTS_INFO(
            "Potential Inspection Underflow Detected! Level: %s Type: %s\n",
            level, arts_metric_name[type]);
      }
      total_stamp = (level == ARTS_METRIC_THREAD)
                       ? metric->totalCount -= *to_add
                       : arts_atomic_sub_u64(&metric->totalCount, *to_add);
    } else {
      total_stamp = (level == ARTS_METRIC_THREAD)
                       ? metric->totalCount += *to_add
                       : arts_atomic_add_u64(&metric->totalCount, *to_add);
    }
    internal_update_max(level, metric, total_stamp);
  }

  uint64_t local_window_time_stamp = metric->windowTimeStamp;
  uint64_t local_window_count_stamp = metric->windowCountStamp;

  uint64_t time_stamp = metric->timeMethod();
  if (!local_window_time_stamp) {
    if (!arts_atomic_cswap_u64(&metric->windowTimeStamp, 0, time_stamp)) {
      metric->firstTimeStamp = metric->windowTimeStamp;
}
    return false;
  }

  uint64_t elapsed =
      (time_stamp > local_window_time_stamp) ? time_stamp - local_window_time_stamp : 0;
  uint64_t last = (total_stamp > local_window_count_stamp)
                      ? total_stamp - local_window_count_stamp
                      : local_window_count_stamp - total_stamp;

  if (last >= count_window[type][level] || elapsed >= time_window[type][level]) {
    if (!metric_try_lock(level, metric)) {
      return false;
}
    if (local_window_time_stamp != metric->windowTimeStamp) {
      metric_unlock(metric);
      return false;
    }
    ARTS_DEBUG("Check metric %d %d %" PRIu64 " %" PRIu64 " vs %" PRIu64
               " %" PRIu64 "\n",
               level, type, last, elapsed, count_window[type][level],
               time_window[type][level]);
    ARTS_DEBUG("Updating metric %d %d %" PRIu64 " %" PRIu64 "\n", level, type,
               metric->windowCountStamp, metric->windowTimeStamp);
    metric->lastWindowTimeStamp = metric->windowTimeStamp;
    metric->lastWindowCountStamp = metric->windowCountStamp;
    metric->lastWindowMaxTotal = metric->windowMaxTotal;
    metric->windowCountStamp = metric->totalCount;
    metric->windowMaxTotal = internal_observe_max(level, metric);
    metric->windowTimeStamp = metric->timeMethod();
    if (metric->windowCountStamp > metric->lastWindowCountStamp) {
      *to_add = metric->windowCountStamp - metric->lastWindowCountStamp;
      *sub = false;
    } else {
      *to_add = metric->lastWindowCountStamp - metric->windowCountStamp;
      *sub = true;
    }
    metric_unlock(metric);
    return true;
  }
  return false;
}

static void take_rate_shot(arts_metric_type_t type, arts_metric_level_t level,
                         bool last) {
  if (inspector_shots && level >= inspector_shots->traceLevel) {
    if (!count_window[type][level] || !time_window[type][level]) {
      return;
}
    ARTS_DEBUG("TRACING LEVEL %d\n", level);

    // int traceOn = arts_thread_info.malloc_trace;
    // arts_thread_info.malloc_trace = 0;
    arts_performance_unit_t *metric = get_metric(type, level);
    if (metric) {
      arts_array_list_t *list = NULL;
      unsigned int *lock = NULL;
      switch (level) {
      case ARTS_METRIC_THREAD:
        list = inspector_shots
                   ->coreMetric[(arts_thread_info.thread_id * ARTS_METRIC_LAST_TYPE) +
                                type];
        lock = NULL;
        break;

      case ARTS_METRIC_NODE:
        list = inspector_shots->nodeMetric[type];
        lock = &inspector_shots->nodeLock[type];
        break;

      case ARTS_METRIC_SYSTEM:
        list = inspector_shots->systemMetric[type];
        lock = &inspector_shots->systemLock[type];
        break;

      default:
        list = NULL;
        lock = NULL;
        break;
      }

      if (list) {
        if (lock) {
          unsigned int local;
          while (1) {
            local = arts_atomic_cswap(lock, 0U, 2U);
            if (local == 2U) {
              // arts_thread_info.malloc_trace = 1;
              return;
            }
            if (!local) {
              break;
}
          }
        }
        arts_metric_shot_t shot;
        if (last) {
          metric_lock(level, metric);
          shot.max_total = metric->windowMaxTotal;
          shot.windowTimeStamp = metric->lastWindowTimeStamp;
          shot.windowCountStamp = metric->lastWindowCountStamp;
          shot.currentTimeStamp = metric->windowTimeStamp;
          shot.currentCountStamp = metric->windowCountStamp;
          metric_unlock(metric);
        } else {
          metric_lock(level, metric);
          shot.windowTimeStamp = metric->windowTimeStamp;
          shot.windowCountStamp = metric->windowCountStamp;
          metric_unlock(metric);

          shot.max_total = metric->max_total;
          shot.currentCountStamp = metric->totalCount;
          shot.currentTimeStamp = metric->timeMethod();
        }

        // arts_thread_info.malloc_trace = 0;
        arts_push_to_array_list(list, &shot);
        // arts_thread_info.malloc_trace = 1;

        if (lock) {
          *lock = 0U;
}
      }
    }
    // arts_thread_info.malloc_trace = traceOn;
  }
}

static arts_metric_level_t update_performance_core_metric(unsigned int core,
                                                   arts_metric_type_t type,
                                                   arts_metric_level_t level,
                                                   uint64_t to_add, bool sub) {
  if (type <= ARTS_METRIC_FIRST_TYPE || type >= ARTS_METRIC_LAST_TYPE) {
    ARTS_ERROR("Metrics: invalid type %d", type);
  }

  arts_metric_level_t updated_level = ARTS_METRIC_NO_LEVEL;
  if (inspector_on) {
    switch (level) {
    case ARTS_METRIC_THREAD:
      ARTS_DEBUG("Thread updated up to %d %" PRIu64 " %u %s\n", updated_level,
                 to_add, sub, arts_metric_name[type]);
      if (!single_metric_update(
              type, ARTS_METRIC_THREAD, &to_add, &sub,
              &inspector->coreMetric[(core * ARTS_METRIC_LAST_TYPE) + type])) {
        break;
}
      take_rate_shot(type, ARTS_METRIC_THREAD, true);
      updated_level = ARTS_METRIC_THREAD;

    case ARTS_METRIC_NODE:
      ARTS_DEBUG("Node   updated up to %d %" PRIu64 " %u %s\n", updated_level,
                 to_add, sub, arts_metric_name[type]);
      if (!single_metric_update(type, ARTS_METRIC_NODE, &to_add, &sub,
                              &inspector->nodeMetric[type])) {
        break;
}
      arts_atomic_add_u64(&stats->nodeUpdates, 1);
      take_rate_shot(type, ARTS_METRIC_NODE, true);
      updated_level = ARTS_METRIC_NODE;

    case ARTS_METRIC_SYSTEM:
      ARTS_DEBUG("System updated up to %d %" PRIu64 " %u %s\n", updated_level,
                 to_add, sub, arts_metric_name[type]);
      if (single_metric_update(type, ARTS_METRIC_SYSTEM, &to_add, &sub,
                             &inspector->systemMetric[type])) {
        uint64_t time_to_send = inspector->systemMetric[type].timeMethod();
        // int traceOn = arts_thread_info.malloc_trace;
        // arts_thread_info.malloc_trace = 0;
        // for (unsigned int i = 0; i < arts_global_rank_count; i++)
        //   if (i != arts_global_rank_id)
        //     arts_remote_metric_update(i, type, level, time_to_send, to_add, sub);
        // arts_thread_info.malloc_trace = traceOn;
        arts_atomic_add_u64(&stats->systemUpdates, 1);
        if (arts_global_rank_count > 1) {
          arts_atomic_add_u64(&stats->systemMessages, arts_global_rank_count - 1);
}
        take_rate_shot(type, ARTS_METRIC_SYSTEM, true);
        updated_level = ARTS_METRIC_SYSTEM;
      }
    default:
      break;
    }
  }
  return updated_level;
}

static void set_thread_metric(arts_metric_type_t type, uint64_t value) {
  if (count_window[type][ARTS_METRIC_THREAD] == -1 && time_window[type][ARTS_METRIC_THREAD] == -1) {
    return;
}

  arts_performance_unit_t *metric = get_metric(type, ARTS_METRIC_THREAD);
  if (metric) {
    bool shot = true;

    metric->lastWindowCountStamp = metric->windowCountStamp;
    metric->lastWindowTimeStamp = metric->windowTimeStamp;
    metric->lastWindowMaxTotal = metric->max_total;

    uint64_t local_time = metric->timeMethod();
    if (!metric->firstTimeStamp) {
      shot = false;
      metric->firstTimeStamp = local_time;
    }

    metric->totalCount = value;
    metric->windowCountStamp = value;
    metric->windowTimeStamp = local_time;
    if (metric->max_total < value) {
      metric->max_total = value;
}
    if (shot) {
      take_rate_shot(type, ARTS_METRIC_THREAD, true);
    }
  }
}

void arts_metrics_read_config_file(char *filename) {
  char *line = NULL;
  size_t length = 0;
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    return;
}

  char temp[ARTS_MAXMETRICNAME];

  while (getline(&line, &length, fp) != -1) {
    ARTS_DEBUG("%s", line);
    if (line[0] != '#') {
      int param_read = 0;
      (void)sscanf(line, "%s", temp);
      size_t offset = strlen(temp);
      unsigned int metric_index = -1;
      for (unsigned int i = 0; i < ARTS_METRIC_LAST_TYPE; i++) {
        if (!strcmp(temp, arts_metric_name[i])) {
          metric_index = i;
          break;
        }
      }

      if (metric_index < ARTS_METRIC_LAST_TYPE) {
        for (unsigned int i = 0; i < ARTS_METRICLEVELS; i++) {
          while (line[offset] == ' ') {
            offset++;
}
          {
            char *endptr;
            count_window[metric_index][i] = strtoull(&line[offset], &endptr, 10);
            if (endptr != &line[offset]) {
              param_read++;
            }
          }
          (void)sscanf(&line[offset], "%s", temp);
          offset += strlen(temp);
          ARTS_DEBUG("temp: %s %u %u %u %" PRIu64 "\n", temp, metric_index, i,
                     offset, count_window[metric_index][i]);
        }

        for (unsigned int i = 0; i < ARTS_METRICLEVELS; i++) {
          while (line[offset] == ' ') {
            offset++;
}
          {
            char *endptr;
            time_window[metric_index][i] = strtoull(&line[offset], &endptr, 10);
            if (endptr != &line[offset]) {
              param_read++;
            }
          }
          (void)sscanf(&line[offset], "%s", temp);
          offset += strlen(temp);
          ARTS_DEBUG("temp: %s %u %u %u %" PRIu64 "\n", temp, metric_index, i,
                     offset, time_window[metric_index][i]);
        }

        for (unsigned int i = 0; i < ARTS_METRICLEVELS; i++) {
          while (line[offset] == ' ') {
            offset++;
}
          {
            char *endptr;
            max_total[metric_index][i] = strtoull(&line[offset], &endptr, 10);
            if (endptr != &line[offset]) {
              param_read++;
            }
          }
          (void)sscanf(&line[offset], "%s", temp);
          offset += strlen(temp);
          ARTS_DEBUG("temp: %s %u %u %u %" PRIu64 "\n", temp, metric_index, i,
                     offset, max_total[metric_index][i]);
        }
      }

      if (metric_index >= ARTS_METRIC_LAST_TYPE ||
          param_read < ARTS_METRICLEVELS * 2) {
        ARTS_INFO("FAILED to init metric %s\n", temp);
      }
    }
  }
  (void)fclose(fp);

  if (line) {
    free(line);
}
}

// void ARTS_METRICS_PRINT_INSPECTOR_TIME() {
//   printf("Stat 0 Node %u Start %" PRIu64 " End %" PRIu64 "\n",
//   arts_global_rank_id,
//          inspector->startTimeStamp, inspector->endTimeStamp);
// }

// void ARTS_METRICS_PRINT_INSPECTOR_STATS() {
//   printf("Stat 3 Node %u Node_Updates %" PRIu64 " System_Updates %" PRIu64
//          " Remote_Updates  %" PRIu64 " System_Messages %" PRIu64 "\n",
//          arts_global_rank_id, stats->nodeUpdates, stats->systemUpdates,
//          stats->remoteUpdates, stats->systemMessages);
// }

// void ARTS_METRICS_PRINT_MODEL_TOTAL_METRICS(arts_metric_level_t level) {
//   if (level == ARTS_METRIC_NODE)
//     printf("Stat 1 Node %u edt %" PRIu64 " edt_signal %" PRIu64
//            " event_signal %" PRIu64 " network_sent %" PRIu64
//            " network_recv %" PRIu64 " malloc %" PRIu64 " free %" PRIu64 "\n",
//            arts_global_rank_id, ARTS_METRICS_GET_TOTAL(ARTS_METRIC_EDT_THROUGHPUT, level),
//            ARTS_METRICS_GET_TOTAL(ARTS_METRIC_EDT_SIGNAL_THROUGHPUT, level),
//            ARTS_METRICS_GET_TOTAL(ARTS_METRIC_EVENT_SIGNAL_THROUGHPUT, level),
//            ARTS_METRICS_GET_TOTAL(ARTS_METRIC_NETWORK_SEND_BW, level),
//            ARTS_METRICS_GET_TOTAL(ARTS_METRIC_NETWORK_RECIEVE_BW, level),
//            ARTS_METRICS_GET_TOTAL(ARTS_METRIC_MALLOC_BW, level),
//            ARTS_METRICS_GET_TOTAL(ARTS_METRIC_FREE_BW, level));
//   else if (level == ARTS_METRIC_THREAD) {
//     ARTS_INFO("Stat 1 Thread %u edt %" PRIu64 " edt_signal %" PRIu64
//               " event_signal %" PRIu64 " network_sent %" PRIu64
//               " network_recv %" PRIu64 " malloc %" PRIu64 " free %" PRIu64
//               "\n", arts_thread_info.thread_id,
//               ARTS_METRICS_GET_TOTAL(ARTS_METRIC_EDT_THROUGHPUT, level),
//               ARTS_METRICS_GET_TOTAL(ARTS_METRIC_EDT_SIGNAL_THROUGHPUT, level),
//               ARTS_METRICS_GET_TOTAL(ARTS_METRIC_EVENT_SIGNAL_THROUGHPUT, level),
//               ARTS_METRICS_GET_TOTAL(ARTS_METRIC_NETWORK_SEND_BW, level),
//               ARTS_METRICS_GET_TOTAL(ARTS_METRIC_NETWORK_RECIEVE_BW, level),
//               ARTS_METRICS_GET_TOTAL(ARTS_METRIC_MALLOC_BW, level),
//               ARTS_METRICS_GET_TOTAL(ARTS_METRIC_FREE_BW, level));
//   }
// }

static inline void update_packet_extreme(uint64_t val, volatile uint64_t *old,
                                       bool min) {
  uint64_t local = *old;
  uint64_t res;
  if (min) {
    while (val < local) {
      res = arts_atomic_cswap_u64(old, local, val);
      if (res == local) {
        break;
}
      local = res;
    }
  } else {
    while (val > local) {
      res = arts_atomic_cswap_u64(old, local, val);
      if (res == local) {
        break;
}
      local = res;
    }
  }
}

// void ARTS_METRICS_UPDATE_PACKET_INFO(uint64_t bytes) {
//   if (packet_inspector) {
//     arts_reader_lock(&packet_inspector->reader, &packet_inspector->writer);
//     arts_atomic_add_u64(&packet_inspector->totalBytes, bytes);
//     arts_atomic_add_u64(&packet_inspector->totalPackets, 1U);
//     update_packet_extreme(bytes, &packet_inspector->maxPacket, false);
//     update_packet_extreme(bytes, &packet_inspector->minPacket, true);
//     arts_reader_unlock(&packet_inspector->reader);

//     arts_reader_lock(&packet_inspector->intervalReader,
//                    &packet_inspector->intervalWriter);
//     arts_atomic_add_u64(&packet_inspector->intervalBytes, bytes);
//     arts_atomic_add_u64(&packet_inspector->intervalPackets, 1U);
//     update_packet_extreme(bytes, &packet_inspector->intervalMax, false);
//     update_packet_extreme(bytes, &packet_inspector->intervalMin, true);
//     arts_reader_unlock(&packet_inspector->intervalReader);
//   }
// }

// void ARTS_METRICS_PACKET_STATS(uint64_t *totalBytes, uint64_t *totalPackets,
//                             uint64_t *minPacket, uint64_t *maxPacket) {
//   if (packet_inspector) {
//     arts_writer_lock(&packet_inspector->reader, &packet_inspector->writer);
//     (*totalBytes) = packet_inspector->totalBytes;
//     (*totalPackets) = packet_inspector->totalPackets;
//     (*minPacket) = packet_inspector->minPacket;
//     (*maxPacket) = packet_inspector->maxPacket;
//     arts_writer_unlock(&packet_inspector->writer);
//   }
// }

// void ARTS_METRICS_INTERVAL_PACKET_STATS(uint64_t *totalBytes,
//                                     uint64_t *totalPackets, uint64_t
//                                     *minPacket, uint64_t *maxPacket) {
//   if (packet_inspector) {
//     arts_writer_lock(&packet_inspector->intervalReader,
//                    &packet_inspector->intervalWriter);
//     (*totalBytes) = arts_atomic_swap_u64(&packet_inspector->totalBytes, 0);
//     (*totalPackets) = arts_atomic_swap_u64(&packet_inspector->totalPackets, 0);
//     (*minPacket) = arts_atomic_swap_u64(&packet_inspector->minPacket, 0);
//     (*maxPacket) = arts_atomic_swap_u64(&packet_inspector->maxPacket, 0);
//     arts_writer_unlock(&packet_inspector->intervalWriter);
//   }
// }
