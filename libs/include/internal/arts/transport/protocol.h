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
  ARTS_REMOTE_EDT_SIGNAL_MSG = 1,
  ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG = 2,
  ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG = 3,
  ARTS_REMOTE_ADD_DEPENDENCE_MSG = 5,
  ARTS_REMOTE_CHANNEL_ADD_DEPENDENCE_MSG = 6,
  ARTS_REMOTE_CHANNEL_ADD_DEPENDENCE_WITH_BYTE_OFFSET_MSG = 7,
  ARTS_REMOTE_DB_ADD_DEPENDENCE_MSG = 8,
  ARTS_REMOTE_DB_INCREMENT_LATCH_MSG = 9,
  ARTS_REMOTE_DB_DECREMENT_LATCH_MSG = 10,
  ARTS_REMOTE_DB_REQUEST_MSG = 11,
  ARTS_REMOTE_DB_SEND_MSG = 12,
  ARTS_REMOTE_INVALIDATE_DB_MSG = 13,
  ARTS_REMOTE_DB_UPDATE_GUID_MSG = 14,
  ARTS_REMOTE_DB_UPDATE_MSG = 15,
  ARTS_REMOTE_DB_DESTROY_MSG = 16,
  ARTS_REMOTE_DB_DESTROY_FORWARD_MSG = 17,
  ARTS_REMOTE_DB_CLEAN_FORWARD_MSG = 18,
  ARTS_REMOTE_EDT_MOVE_MSG = 19,
  ARTS_REMOTE_EVENT_MOVE_MSG = 20,
  ARTS_REMOTE_DB_MOVE_MSG = 22,
  ARTS_REMOTE_PINGPONG_TEST_MSG = 23,
  ARTS_REMOTE_METRIC_UPDATE_MSG = 24,
  ARTS_REMOTE_DB_FULL_REQUEST_MSG = 25,
  ARTS_REMOTE_DB_FULL_SEND_MSG = 26,
  ARTS_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG = 27,
  ARTS_REMOTE_GET_FROM_DB_MSG = 28,
  ARTS_REMOTE_PUT_IN_DB_MSG = 29,
  ARTS_REMOTE_SEND_MSG = 30,
  ARTS_EPOCH_INIT_MSG = 31,
  ARTS_EPOCH_INIT_POOL_MSG = 32,
  ARTS_EPOCH_REQ_MSG = 33,
  ARTS_EPOCH_SEND_MSG = 34,
  ARTS_EPOCH_DELETE_MSG = 35,
  ARTS_REMOTE_BUFFER_SEND_MSG = 36,
  ARTS_REMOTE_CONTEXT_SIG_MSG = 37,
  ARTS_REMOTE_DB_RENAME_MSG = 38,
  ARTS_REMOTE_DB_PARTIAL_UPDATE_MSG = 39,
  ARTS_REMOTE_DB_ADD_DEPENDENCE_WITH_BYTE_OFFSET_MSG = 40,
  ARTS_REMOTE_TIME_SYNC_REQ_MSG = 41,
  ARTS_REMOTE_TIME_SYNC_RESP_MSG = 42,
};

// Header
struct ARTS_PACKED arts_remote_packet_s {
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

struct ARTS_PACKED arts_remote_guid_only_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t guid;
};

struct ARTS_PACKED arts_remote_add_dependence_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t source;
  arts_guid_t destination;
  uint32_t slot;
  arts_db_access_mode_t mode;
};

struct ARTS_PACKED arts_remote_add_dependence_with_byte_offset_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t source;
  arts_guid_t destination;
  uint32_t slot;
  arts_db_access_mode_t mode;
  uint64_t byte_offset;
  uint64_t size;
};

struct ARTS_PACKED arts_remote_edt_signal_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t edt;
  arts_guid_t db;
  uint32_t slot;
  arts_db_access_mode_t mode;
  unsigned int db_route;
};

struct ARTS_PACKED arts_remote_event_satisfy_slot_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t event;
  arts_guid_t db;
  uint32_t slot;
};

struct ARTS_PACKED arts_remote_db_add_dependence_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_src;
  arts_guid_t edt_dest;
  uint32_t edt_slot;
  arts_db_access_mode_t mode;
};

/// ESD: Packet for byte-offset dependencies (stencil halo exchange)
struct ARTS_PACKED arts_remote_db_add_dependence_with_byte_offset_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_src;
  arts_guid_t edt_dest;
  uint32_t edt_slot;
  arts_db_access_mode_t mode;
  uint64_t byte_offset; ///< Byte offset into DB for slice
  uint64_t size;        ///< Size of slice in bytes
};

struct ARTS_PACKED arts_remote_db_request_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  arts_db_access_mode_t mode;
};

struct ARTS_PACKED arts_remote_db_send_packet_s {
  struct arts_remote_packet_s header;
};

struct ARTS_PACKED arts_remote_db_full_request_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  arts_guid_t edt_guid;
  unsigned int slot;
  arts_db_access_mode_t mode;
};

struct ARTS_PACKED arts_remote_db_full_send_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t edt_guid;
  unsigned int slot;
  arts_db_access_mode_t mode;
};

struct ARTS_PACKED arts_remote_metric_update_s {
  struct arts_remote_packet_s header;
  int type;
  uint64_t time_stamp;
  uint64_t to_add;
  bool sub;
};

struct ARTS_PACKED arts_remote_get_put_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t edt_guid;
  arts_guid_t db_guid;
  arts_guid_t epoch_guid;
  unsigned int slot;
  unsigned int offset;
  unsigned int size;
};

struct ARTS_PACKED arts_remote_signal_edt_with_ptr_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t edt_guid;
  arts_guid_t db_guid;
  unsigned int size;
  unsigned int slot;
};

typedef void (*send_handler_t)(void *args);

struct ARTS_PACKED arts_remote_send_s {
  struct arts_remote_packet_s header;
  send_handler_t fun_ptr;
};

struct ARTS_PACKED arts_remote_epoch_init_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t epoch_guid;
  arts_guid_t edt_guid;
  unsigned int slot;
};

struct ARTS_PACKED arts_remote_epoch_init_pool_packet_s {
  struct arts_remote_packet_s header;
  unsigned int pool_size;
  arts_guid_t start_guid;
  arts_guid_t pool_guid;
};

struct ARTS_PACKED arts_remote_epoch_send_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t epoch_guid;
  unsigned int active;
  unsigned int finish;
};

struct ARTS_PACKED arts_remote_signal_context_packet_s {
  struct arts_remote_packet_s header;
  uint64_t ticket;
};

struct ARTS_PACKED arts_remote_db_rename_s {
  struct arts_remote_packet_s header;
  arts_guid_t old_guid;
  arts_guid_t new_guid;
};

// Diff region descriptor
struct ARTS_PACKED arts_remote_diff_region_s {
  uint32_t offset;
  uint32_t length;
};

// Partial update packet header
struct ARTS_PACKED arts_remote_partial_update_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t guid;
  uint32_t region_count;
  uint32_t data_bytes;
  uint32_t flags;
  uint32_t reserved;
};

// Time synchronization packets for RTT-based clock sync
// Worker sends request with its send time T1
struct ARTS_PACKED arts_remote_time_sync_req_packet_s {
  struct arts_remote_packet_s header;
  uint64_t worker_send_time; // T1: worker's local time when sending request
};

// Master responds with T1 (echoed) and T2 (master's receive time)
struct ARTS_PACKED arts_remote_time_sync_resp_packet_s {
  struct arts_remote_packet_s header;
  uint64_t worker_send_time; // T1: echoed back
  uint64_t master_recv_time; // T2: master's local time when receiving request
};

#include "arts/system/threads.h"

static inline void arts_fill_packet_header(struct arts_remote_packet_s *header,
                                           uint64_t size,
                                           unsigned int message_type) {
  header->size = size;
  header->message_type = message_type;
  header->rank = arts_global_rank_id;
}

void out_init(unsigned int size);
void out_cleanup(void);
void arts_remote_flush_outbound(void);
bool arts_remote_async_send();
void arts_remote_send_request_async(int rank, char *message,
                                    unsigned int length);
void arts_remote_send_request_payload_async(int rank, char *message,
                                            unsigned int length, char *payload,
                                            uint64_t size);
void arts_remote_send_request_payload_async_free(
    int rank, char *message, unsigned int length, char *payload,
    unsigned int offset, uint64_t size, void (*free_method)(void *));
void arts_remote_set_thread_outbound_queues(unsigned int start,
                                            unsigned int stop);
void arts_remote_thread_outbound_queues_cleanup();
#ifdef __cplusplus
}
#endif

#endif
