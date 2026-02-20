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
#ifndef ARTS_NETWORK_REMOTE_PROTOCOL_H
#define ARTS_NETWORK_REMOTE_PROTOCOL_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts/runtime/rt.h"
#define SEQUENCENUMBERS 1

enum artsServerMessageType {
  ARTS_REMOTE_SHUTDOWN_MSG = 0,
  ARTS_REMOTE_EDT_SIGNAL_MSG,
  ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG,
  ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG,
  ARTS_REMOTE_PERSISTENT_EVENT_SATISFY_SLOT_MSG,
  ARTS_REMOTE_ADD_DEPENDENCE_MSG,
  ARTS_REMOTE_ADD_DEPENDENCE_TO_PERSISTENT_EVENT_MSG,
  ARTS_REMOTE_ADD_DEPENDENCE_TO_PERSISTENT_EVENT_WITH_BYTE_OFFSET_MSG,
  ARTS_REMOTE_DB_ADD_DEPENDENCE_MSG,
  ARTS_REMOTE_DB_INCREMENT_LATCH_MSG,
  ARTS_REMOTE_DB_DECREMENT_LATCH_MSG,
  ARTS_REMOTE_DB_REQUEST_MSG,
  ARTS_REMOTE_DB_SEND_MSG,
  ARTS_REMOTE_INVALIDATE_DB_MSG,
  ARTS_REMOTE_DB_UPDATE_GUID_MSG,
  ARTS_REMOTE_DB_UPDATE_MSG,
  ARTS_REMOTE_DB_DESTROY_MSG,
  ARTS_REMOTE_DB_DESTROY_FORWARD_MSG,
  ARTS_REMOTE_DB_CLEAN_FORWARD_MSG,
  ARTS_REMOTE_DB_MOVE_REQ_MSG,
  ARTS_REMOTE_EDT_MOVE_MSG,
  ARTS_REMOTE_EVENT_MOVE_MSG,
  ARTS_REMOTE_PERSISTENT_EVENT_MOVE_MSG,
  ARTS_REMOTE_DB_MOVE_MSG,
  ARTS_REMOTE_PINGPONG_TEST_MSG,
  ARTS_REMOTE_METRIC_UPDATE_MSG,
  ARTS_REMOTE_DB_FULL_REQUEST_MSG,
  ARTS_REMOTE_DB_FULL_SEND_MSG,
  ARTS_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG,
  ARTS_REMOTE_GET_FROM_DB_MSG,
  ARTS_REMOTE_PUT_IN_DB_MSG,
  ARTS_REMOTE_SEND_MSG,
  ARTS_EPOCH_INIT_MSG,
  ARTS_EPOCH_INIT_POOL_MSG,
  ARTS_EPOCH_REQ_MSG,
  ARTS_EPOCH_SEND_MSG,
  ARTS_EPOCH_DELETE_MSG,
  ARTS_ATOMIC_ADD_ARRAYDB_MSG,
  ARTS_ATOMIC_CAS_ARRAYDB_MSG,
  ARTS_REMOTE_BUFFER_SEND_MSG,
  ARTS_REMOTE_CONTEXT_SIG_MSG,
  ARTS_REMOTE_DB_RENAME_MSG,
  ARTS_REMOTE_DB_PARTIAL_UPDATE_MSG,
  ARTS_REMOTE_DB_ADD_DEPENDENCE_WITH_BYTE_OFFSET_MSG,
  ARTS_REMOTE_TIME_SYNC_REQ_MSG,  // Worker -> Master: request with T1
  ARTS_REMOTE_TIME_SYNC_RESP_MSG, // Master -> Worker: response with T1, T2
};

// Header
struct __attribute__((__packed__)) arts_remote_packet_s {
  unsigned int message_type;
  uint64_t size;
  unsigned int rank;
#ifdef SEQUENCENUMBERS
  unsigned int seq_rank;
  uint64_t seq_num;
#endif
  uint64_t time_stamp;
  uint64_t proc_time_stamp;
};

struct __attribute__((__packed__)) arts_remote_guid_only_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t guid;
};

struct __attribute__((__packed__)) arts_remote_add_dependence_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t source;
  arts_guid_t destination;
  uint32_t slot;
  arts_type_t mode;
};

/// ESD: Packet for adding dependency to persistent event with byte offset
struct __attribute__((__packed__)) arts_remote_add_dependence_with_byte_offset_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t source;
  arts_guid_t destination;
  uint32_t slot;
  arts_type_t mode;
  uint64_t byte_offset;
  uint64_t size;
};

struct __attribute__((__packed__)) arts_remote_edt_signal_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t edt;
  arts_guid_t db;
  uint32_t slot;
  arts_type_t mode;
  unsigned int db_route;
};

struct __attribute__((__packed__)) arts_remote_event_satisfy_slot_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t event;
  arts_guid_t db;
  uint32_t slot;
};

struct __attribute__((__packed__)) arts_remote_persistent_event_satisfy_slot_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t event;
  uint32_t action;
  bool lock;
};

struct __attribute__((__packed__)) arts_remote_db_add_dependence_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_src;
  arts_guid_t edt_dest;
  uint32_t edt_slot;
  arts_type_t mode;
};

/// ESD: Packet for byte-offset dependencies (stencil halo exchange)
struct __attribute__((__packed__)) arts_remote_db_add_dependence_with_byte_offset_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_src;
  arts_guid_t edt_dest;
  uint32_t edt_slot;
  arts_type_t mode;
  uint64_t byte_offset; ///< Byte offset into DB for slice
  uint64_t size;       ///< Size of slice in bytes
};

struct __attribute__((__packed__)) arts_remote_db_request_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  arts_type_t mode;
};

struct __attribute__((__packed__)) arts_remote_db_send_packet_s {
  struct arts_remote_packet_s header;
};

struct __attribute__((__packed__)) arts_remote_db_full_request_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  arts_guid_t edt_guid;
  unsigned int slot;
  arts_type_t mode;
};

struct __attribute__((__packed__)) arts_remote_db_full_send_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t edt_guid;
  unsigned int slot;
  arts_type_t mode;
};

struct __attribute__((__packed__)) arts_remote_metric_update_s {
  struct arts_remote_packet_s header;
  int type;
  uint64_t time_stamp;
  uint64_t to_add;
  bool sub;
};

struct __attribute__((__packed__)) arts_remote_get_put_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t edt_guid;
  arts_guid_t db_guid;
  arts_guid_t epoch_guid;
  unsigned int slot;
  unsigned int offset;
  unsigned int size;
};

struct __attribute__((__packed__)) arts_remote_signal_edt_with_ptr_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t edt_guid;
  arts_guid_t db_guid;
  unsigned int size;
  unsigned int slot;
};

typedef void (*send_handler_t)(void *args);

struct __attribute__((__packed__)) arts_remote_send_s {
  struct arts_remote_packet_s header;
  send_handler_t fun_ptr;
};

struct __attribute__((__packed__)) arts_remote_epoch_init_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t epoch_guid;
  arts_guid_t edt_guid;
  unsigned int slot;
};

struct __attribute__((__packed__)) arts_remote_epoch_init_pool_packet_s {
  struct arts_remote_packet_s header;
  unsigned int pool_size;
  arts_guid_t start_guid;
  arts_guid_t pool_guid;
};

struct __attribute__((__packed__)) arts_remote_epoch_send_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t epoch_guid;
  unsigned int active;
  unsigned int finish;
};

struct __attribute__((__packed__)) arts_remote_atomic_add_in_array_db_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  arts_guid_t edt_guid;
  arts_guid_t epoch_guid;
  unsigned int slot;
  unsigned int index;
  unsigned int to_add;
};

struct __attribute__((__packed__))
arts_remote_atomic_compare_and_swap_in_array_db_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  arts_guid_t edt_guid;
  arts_guid_t epoch_guid;
  unsigned int slot;
  unsigned int index;
  unsigned int old_value;
  unsigned int new_value;
};

struct __attribute__((__packed__)) arts_remote_signal_context_packet_s {
  struct arts_remote_packet_s header;
  uint64_t ticket;
};

struct __attribute__((__packed__)) arts_remote_db_rename_s {
  struct arts_remote_packet_s header;
  arts_guid_t old_guid;
  arts_guid_t new_guid;
};

// Diff region descriptor
struct __attribute__((__packed__)) arts_remote_diff_region_s {
  uint32_t offset;
  uint32_t length;
};

// Partial update packet header
struct __attribute__((__packed__)) arts_remote_partial_update_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t guid;
  uint32_t region_count;
  uint32_t data_bytes;
  uint32_t flags;
  uint32_t reserved;
};

// Time synchronization packets for RTT-based clock sync
// Worker sends request with its send time T1
struct __attribute__((__packed__)) arts_remote_time_sync_req_packet_s {
  struct arts_remote_packet_s header;
  uint64_t worker_send_time; // T1: worker's local time when sending request
};

// Master responds with T1 (echoed) and T2 (master's receive time)
struct __attribute__((__packed__)) arts_remote_time_sync_resp_packet_s {
  struct arts_remote_packet_s header;
  uint64_t worker_send_time; // T1: echoed back
  uint64_t master_recv_time; // T2: master's local time when receiving request
};

#include "arts/runtime/globals.h"

static inline void arts_fill_packet_header(struct arts_remote_packet_s *header,
                                           uint64_t size,
                                           unsigned int message_type) {
  header->size = size;
  header->message_type = message_type;
  header->rank = arts_global_rank_id;
}

void out_init(unsigned int size);
void arts_remote_flush_outbound(void);
bool arts_remote_async_send();
void arts_remote_send_request_async(int rank, char *message, unsigned int length);
void arts_remote_send_request_payload_async(int rank, char *message,
                                       unsigned int length, char *payload,
                                       uint64_t size);
void arts_remote_send_request_payload_async_free(int rank, char *message,
                                           unsigned int length, char *payload,
                                           unsigned int offset,
                                           uint64_t size,
                                           void (*free_method)(void *));
void arts_remote_set_thread_outbound_queues(unsigned int start, unsigned int stop);
void arts_remote_thread_outbound_queues_cleanup();
#ifdef __cplusplus
}
#endif

#endif
