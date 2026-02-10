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
#ifndef ARTSMETRICS_H
#define ARTSMETRICS_H

#include "arts/utils/array_list.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ARTS_METRICLEVELS 3
#define ARTS_MAXMETRICNAME 64

/* #define ARTS_MALLOC_WITH_TYPE(size, type) \
  ({ \
    if (ARTS_METRICS_IS_ACTIVE()) \
      arts_thread_info.malloc_type = type; \
    void *__ptr = arts_malloc(size); \
    if (ARTS_METRICS_IS_ACTIVE()) \
      arts_thread_info.malloc_type = ARTS_METRIC_DEFAULT_MEMORY_SIZE; \
    __ptr; \
  }) */

/* #define ARTS_MALLOC_ALIGN_WITH_TYPE(size, align, type) \
  ({ \
    if (ARTS_METRICS_IS_ACTIVE()) \
      arts_thread_info.malloc_type = type; \
    void *__ptr = arts_malloc_align(size, align); \
    if (ARTS_METRICS_IS_ACTIVE()) \
      arts_thread_info.malloc_type = ARTS_METRIC_DEFAULT_MEMORY_SIZE; \
    __ptr; \
  }) */

/* #define ARTS_CALLOC_WITH_TYPE(nmemb, size, type) \
  ({ \
    if (ARTS_METRICS_IS_ACTIVE()) \
      arts_thread_info.malloc_type = type; \
    void *__ptr = arts_calloc(nmemb, size); \
    if (ARTS_METRICS_IS_ACTIVE()) \
      arts_thread_info.malloc_type = ARTS_METRIC_DEFAULT_MEMORY_SIZE; \
    __ptr; \
  }) */

/* #define ARTS_CALLOC_ALIGN_WITH_TYPE(nmemb, size, align, type) \
  ({ \
    if (ARTS_METRICS_IS_ACTIVE()) \
      arts_thread_info.malloc_type = type; \
    void *__ptr = arts_calloc_align(nmemb, size, align); \
    if (ARTS_METRICS_IS_ACTIVE()) \
      arts_thread_info.malloc_type = ARTS_METRIC_DEFAULT_MEMORY_SIZE; \
    __ptr; \
  }) */

#define ARTS_MALLOC_WITH_TYPE(size, type) arts_malloc(size)
#define ARTS_MALLOC_ALIGN_WITH_TYPE(size, align, type) arts_malloc_align(size, align)
#define ARTS_CALLOC_WITH_TYPE(nmemb, size, type) arts_calloc(nmemb, size)
#define ARTS_CALLOC_ALIGN_WITH_TYPE(nmemb, size, align, type)                      \
  arts_calloc_align(nmemb, size, align)

extern const char *const arts_metric_name[];

typedef enum arts_metric_type_t {
  ARTS_METRIC_FIRST_TYPE = -1,
  ARTS_METRIC_EDT_THROUGHPUT,
  ARTS_METRIC_EDT_QUEUE,
  ARTS_METRIC_EDT_STEAL_ATTEMPT,
  ARTS_METRIC_EDT_STEAL,
  ARTS_METRIC_EDT_LAST_LOCAL_HIT,
  ARTS_METRIC_EDT_SIGNAL_THROUGHPUT,
  ARTS_METRIC_EVENT_SIGNAL_THROUGHPUT,
  ARTS_METRIC_PERSISTENT_EVENT_SIGNAL_THROUGHPUT,
  ARTS_METRIC_GET_BW,
  ARTS_METRIC_PUT_BW,
  ARTS_METRIC_NETWORK_SEND_BW,
  ARTS_METRIC_NETWORK_RECIEVE_BW,
  ARTS_METRIC_NETWORK_QUEUE_PUSH,
  ARTS_METRIC_NETWORK_QUEUE_POP,
  ARTS_METRIC_YIELD_BW,
  ARTS_METRIC_GPU_EDT,
  ARTS_METRIC_GPU_GC,
  ARTS_METRIC_GPU_GCBW,
  ARTS_METRIC_GPU_BW_PUSH,
  ARTS_METRIC_GPU_BW_PULL,
  ARTS_METRIC_GPU_BUFFER_FLUSH,
  ARTS_METRIC_GPU_SYNC,
  ARTS_METRIC_GPU_SYNC_DELETE,
  ARTS_METRIC_MALLOC_BW,
  ARTS_METRIC_FREE_BW,
  ARTS_METRIC_REMOTE_SHUTDOWN_MSG,
  ARTS_METRIC_REMOTE_EDT_SIGNAL_MSG,
  ARTS_METRIC_REMOTE_SIGNAL_EDT_WITH_PTR_MSG,
  ARTS_METRIC_REMOTE_EVENT_SATISFY_SLOT_MSG,
  ARTS_METRIC_REMOTE_ADD_DEPENDENCE_MSG,
  ARTS_METRIC_REMOTE_DB_REQUEST_MSG,
  ARTS_METRIC_REMOTE_DB_SEND_MSG,
  ARTS_METRIC_REMOTE_INVALIDATE_DB_MSG,
  ARTS_METRIC_REMOTE_DB_UPDATE_GUID_MSG,
  ARTS_METRIC_REMOTE_DB_UPDATE_MSG,
  ARTS_METRIC_REMOTE_DB_DESTROY_MSG,
  ARTS_METRIC_REMOTE_DB_DESTROY_FORWARD_MSG,
  ARTS_METRIC_REMOTE_DB_CLEAN_FORWARD_MSG,
  ARTS_METRIC_REMOTE_DB_MOVE_REQ_MSG,
  ARTS_METRIC_REMOTE_EDT_MOVE_MSG,
  ARTS_METRIC_REMOTE_EVENT_MOVE_MSG,
  ARTS_METRIC_REMOTE_DB_MOVE_MSG,
  ARTS_METRIC_REMOTE_PINGPONG_TEST_MSG,
  ARTS_METRIC_REMOTE_METRIC_UPDATE_MSG,
  ARTS_METRIC_REMOTE_DB_FULL_REQUEST_MSG,
  ARTS_METRIC_REMOTE_DB_FULL_SEND_MSG,
  ARTS_METRIC_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG,
  ARTS_METRIC_REMOTE_GET_FROM_DB_MSG,
  ARTS_METRIC_REMOTE_PUT_IN_DB_MSG,
  ARTS_METRIC_REMOTE_SEND_MSG,
  ARTS_METRIC_EPOCH_INIT_MSG,
  ARTS_METRIC_EPOCH_INIT_POOL_MSG,
  ARTS_METRIC_EPOCH_REQ_MSG,
  ARTS_METRIC_EPOCH_SEND_MSG,
  ARTS_METRIC_EPOCH_DELETE_MSG,
  ARTS_METRIC_ATOMIC_ADD_ARRAYDB_MSG,
  ARTS_METRIC_ATOMIC_CAS_ARRAYDB_MSG,
  ARTS_METRIC_REMOTE_BUFFER_SEND_MSG,
  ARTS_METRIC_REMOTE_CONTEXT_SIG_MSG,
  ARTS_METRIC_REMOTE_DB_RENAME_MSG,
  ARTS_METRIC_DEFAULT_MEMORY_SIZE,
  ARTS_METRIC_EDT_MEMORY_SIZE,
  ARTS_METRIC_EVENT_MEMORY_SIZE,
  ARTS_METRIC_PERSISTENT_EVENT_MEMORY_SIZE,
  ARTS_METRIC_DB_MEMORY_SIZE,
  ARTS_METRIC_BUFFER_MEMORY_SIZE,
  ARTS_METRIC_DB_COUNT,
  ARTS_METRIC_LAST_TYPE
} arts_metric_type_t;

typedef enum arts_metric_level_t {
  ARTS_METRIC_NO_LEVEL = -1,
  ARTS_METRIC_THREAD,
  ARTS_METRIC_NODE,
  ARTS_METRIC_SYSTEM
} arts_metric_level_t;

typedef struct {
  volatile unsigned int reader;
  char pad1[60];
  volatile unsigned int writer;
  char pad2[60];
  volatile unsigned int intervalReader;
  char pad3[60];
  volatile unsigned int intervalWriter;
  char pad4[60];
  volatile uint64_t totalBytes;
  volatile uint64_t totalPackets;
  volatile uint64_t minPacket;
  volatile uint64_t maxPacket;
  volatile uint64_t intervalBytes;
  volatile uint64_t intervalPackets;
  volatile uint64_t intervalMin;
  volatile uint64_t intervalMax;
} arts_packet_inspector_t;

struct arts_performance_unit_s {
  volatile uint64_t totalCount;
  char pad1[56];
  volatile uint64_t max_total;
  char pad2[56];
  uint64_t firstTimeStamp;
  char pad3[56];
  volatile unsigned int lock;
  char pad4[60];
  volatile uint64_t windowCountStamp;
  volatile uint64_t windowTimeStamp;
  volatile uint64_t windowMaxTotal;
  volatile uint64_t lastWindowCountStamp;
  volatile uint64_t lastWindowTimeStamp;
  volatile uint64_t lastWindowMaxTotal;
  uint64_t (*timeMethod)(void);
} __attribute__((aligned(64)));

typedef struct arts_performance_unit_s arts_performance_unit_t;

typedef struct {
  unsigned int startPoint;
  uint64_t startTimeStamp;
  uint64_t endTimeStamp;
  arts_performance_unit_t *coreMetric;
  arts_performance_unit_t *nodeMetric;
  arts_performance_unit_t *systemMetric;
} arts_inspector_t;

typedef struct {
  uint64_t nodeUpdates;
  uint64_t systemUpdates;
  uint64_t systemMessages;
  uint64_t remoteUpdates;
} arts_inspector_stats_t;

typedef struct {
  uint64_t windowCountStamp;
  uint64_t windowTimeStamp;
  uint64_t currentCountStamp;
  uint64_t currentTimeStamp;
  uint64_t max_total;
} arts_metric_shot_t;

typedef struct {
  arts_metric_level_t traceLevel;
  uint64_t initialStart;
  arts_array_list_t **coreMetric;
  arts_array_list_t **nodeMetric;
  arts_array_list_t **systemMetric;
  unsigned int *nodeLock;
  unsigned int *systemLock;
  char *prefix;
} arts_inspector_shots_t;

// void ARTS_METRICS_CONFIG_SET_DEFAULT_ENABLED(bool enabled);
// void ARTS_METRICS_CONFIG_SET_ENABLED(const char *name, bool enabled);
// void ARTS_METRICS_TRIGGER_EVENT(arts_metric_type_t metricType, arts_metric_level_t
// level,
//                              uint64_t value);
// void ARTS_METRICS_TRIGGER_TIMER_EVENT(arts_metric_type_t metricType,
//                                   arts_metric_level_t level, bool start);
// void ARTS_METRICS_TOGGLE_THREAD();
// uint64_t ARTS_METRICS_GET_INSPECTOR_TIME();
// bool ARTS_METRICS_IS_ACTIVE();
// void ARTS_METRICS_START(unsigned int startPoint);
// void ARTS_METRICS_STOP();
// void ARTS_METRICS_INIT_INTROSPECTOR(unsigned int startPoint);
// uint64_t ARTS_METRICS_GET_TOTAL(arts_metric_type_t type, arts_metric_level_t level);
// double ARTS_METRICS_GET_RATE(arts_metric_type_t type, arts_metric_level_t level,
//                           bool last);
// double ARTS_METRICS_GET_TOTAL_RATE(arts_metric_type_t type, arts_metric_level_t level);
// double ARTS_METRICS_TEST(arts_metric_type_t type, arts_metric_level_t level,
//                        uint64_t num);
// uint64_t ARTS_METRICS_GET_RATE_U64(arts_metric_type_t type, arts_metric_level_t level,
//                                bool last);
// uint64_t ARTS_METRICS_GET_RATE_U64_DIFF(arts_metric_type_t type, arts_metric_level_t
// level,
//                                    uint64_t *diff);
// uint64_t ARTS_METRICS_GET_TOTAL_RATE_U64(arts_metric_type_t type, arts_metric_level_t
// level,
//                                     uint64_t *total, uint64_t *time_stamp);
// void ARTS_METRICS_HANDLE_REMOTE_UPDATE(arts_metric_type_t type, arts_metric_level_t
// level,
//                                    uint64_t to_add, bool sub);
// void ARTS_METRICS_PRINT_INSPECTOR_TIME();
// void ARTS_METRICS_PRINT_INSPECTOR_STATS();
// void ARTS_METRICS_PRINT_MODEL_TOTAL_METRICS(arts_metric_level_t level);
// void ARTS_METRICS_UPDATE_PACKET_INFO(uint64_t bytes);
// void ARTS_METRICS_PACKET_STATS(uint64_t *totalBytes, uint64_t *totalPackets,
//                             uint64_t *minPacket, uint64_t *maxPacket);
// void ARTS_METRICS_INTERVAL_PACKET_STATS(uint64_t *totalBytes,
//                                     uint64_t *totalPackets, uint64_t
//                                     *minPacket, uint64_t *maxPacket);

#define ARTS_METRICS_CONFIG_SET_DEFAULT_ENABLED(enabled) ((void)0)
#define ARTS_METRICS_CONFIG_SET_ENABLED(name, enabled) ((void)0)
#define ARTS_METRICS_TRIGGER_EVENT(metricType, level, value) ((void)0)
#define ARTS_METRICS_TRIGGER_TIMER_EVENT(metricType, level, start) ((void)0)
#define ARTS_METRICS_TOGGLE_THREAD()
#define ARTS_METRICS_GET_INSPECTOR_TIME() 0
#define ARTS_METRICS_IS_ACTIVE() 0
#define ARTS_METRICS_START(startPoint)
#define ARTS_METRICS_STOP()
#define ARTS_METRICS_INIT_INTROSPECTOR(startPoint)
#define ARTS_METRICS_GET_TOTAL(type, level) 0
#define ARTS_METRICS_GET_RATE(type, level, last) 0
#define ARTS_METRICS_GET_TOTAL_RATE(type, level) 0
#define ARTS_METRICS_TEST(type, level, num) 0
#define ARTS_METRICS_GET_RATE_U64(type, level, last) 0
#define ARTS_METRICS_GET_RATE_U64_DIFF(type, level, diff) 0
#define ARTS_METRICS_GET_TOTAL_RATE_U64(type, level, total, time_stamp) 0
#define ARTS_METRICS_HANDLE_REMOTE_UPDATE(type, level, to_add, sub) 0
#define ARTS_METRICS_PRINT_INSPECTOR_TIME()
#define ARTS_METRICS_PRINT_INSPECTOR_STATS()
#define ARTS_METRICS_PRINT_MODEL_TOTAL_METRICS(level)
#define ARTS_METRICS_UPDATE_PACKET_INFO(bytes)
#define ARTS_METRICS_PACKET_STATS(totalBytes, totalPackets, minPacket, maxPacket)
#define ARTS_METRICS_INTERVAL_PACKET_STATS(totalBytes, totalPackets, minPacket,    \
                                       maxPacket)

#ifdef __cplusplus
}
#endif

#endif /* ARTSMETRICS_H */
