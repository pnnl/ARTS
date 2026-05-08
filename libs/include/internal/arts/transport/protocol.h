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
#ifndef ARTS_TRANSPORT_PROTOCOL_H
#define ARTS_TRANSPORT_PROTOCOL_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts/runtime_types.h"
#define SEQUENCENUMBERS 1

enum artsServerMessageType {
  ARTS_REMOTE_SHUTDOWN_MSG,
  ARTS_REMOTE_EDT_SIGNAL_MSG,
  ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG,
  ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG,
  ARTS_REMOTE_ADD_DEPENDENCE_MSG,
  ARTS_REMOTE_EDT_MOVE_MSG,
  ARTS_REMOTE_EVENT_MOVE_MSG,
  ARTS_REMOTE_DB_MOVE_MSG,
  ARTS_REMOTE_GET_FROM_DB_MSG,
  ARTS_REMOTE_PUT_IN_DB_MSG,
  ARTS_REMOTE_SEND_MSG,
  ARTS_EPOCH_INIT_MSG,
  ARTS_EPOCH_INIT_POOL_MSG,
  ARTS_EPOCH_REQ_MSG,
  ARTS_EPOCH_SEND_MSG,
  ARTS_EPOCH_DELETE_MSG,
  ARTS_REMOTE_DB_RENAME_MSG,
  ARTS_REMOTE_TIME_SYNC_REQ_MSG,
  ARTS_REMOTE_TIME_SYNC_RESP_MSG,
  ARTS_REMOTE_SET_DEP_MODE_MSG,
  /* coherence protocol messages.  The dispatcher routes these to
   * arts_coh_handle_* in libs/src/core/memory/coherence_handlers.c.
   * Sequential append (CLAUDE.md rule: NO gaps in this enum). */
  ARTS_REMOTE_LOCK_REQ_MSG,
  ARTS_REMOTE_GRANT_MSG,
  ARTS_REMOTE_WRITEBACK_MSG,
  ARTS_REMOTE_WRITEBACK_ACK_MSG,
  ARTS_REMOTE_INVALIDATE_NOTICE_MSG,
  ARTS_REMOTE_RELEASE_OWNERSHIP_MSG,
  ARTS_REMOTE_GET_DATA_MSG,
  ARTS_REMOTE_DATA_RESPONSE_MSG,
  ARTS_REMOTE_DB_CREATE_COHERENT_MSG,
  ARTS_REMOTE_DESTROY_REQ_MSG,
  ARTS_REMOTE_DESTROY_NOTIFY_MSG,
  /* Event subsystem rewrite: cross-rank
   * arts_event_destroy → mark_delete on the home rank.  Sequential append,
   * no gaps. */
  ARTS_REMOTE_EVENT_DESTROY_MSG,
};

/* WRITEBACK packet flag — selects normal write-back vs. write-back +
 * ownership transfer.  Coherence release path uses these constants
 * (libs/src/core/memory/coherence_release.c lines 166, 174) and the
 * home-side WRITEBACK handler dispatches on the value.  uint8_t in the
 * wire packet to keep the struct layout tight. */
typedef enum {
  ARTS_WB_NORMAL = 0,
  ARTS_WB_AND_TRANSFER = 1,
} arts_writeback_flag_t;

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

struct ARTS_PACKED arts_remote_set_dep_mode_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t edt;
  uint32_t slot;
  arts_db_access_mode_t mode;
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

struct ARTS_PACKED arts_remote_db_rename_s {
  struct arts_remote_packet_s header;
  arts_guid_t old_guid;
  arts_guid_t new_guid;
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

/* ===== coherence wire packets =======================
 * Pad-fields exist to keep the trailing payload (when present) on an
 * 8-byte boundary; coherence_handlers.c fires arts_remote_send_request_
 * payload_async right after sizeof(packet_struct) bytes, so the payload
 * starts at sizeof() — that offset must be 8-aligned.  Header is packed
 * (44 bytes), so each struct's body fields determine the pad. */

struct ARTS_PACKED arts_remote_lock_req_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
};

/* GRANT carries optional trailing buffer payload (data_present == 1).
 * Body: db_guid(8) + version(8) + has_next(4) + data_present(4) = 24,
 * total = 44 + 24 = 68; pad[4] -> 72 (8-aligned). */
struct ARTS_PACKED arts_remote_grant_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint32_t has_next;
  uint32_t data_present;
  uint8_t pad[4];
};

/* WRITEBACK carries optional trailing buffer payload.
 * Body: db_guid(8) + version(8) + seq(8) + flag(1) = 25,
 * total = 44 + 25 = 69; pad[3] -> 72 (8-aligned). */
struct ARTS_PACKED arts_remote_writeback_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint64_t seq;
  uint8_t flag;
  uint8_t pad[3];
};

struct ARTS_PACKED arts_remote_writeback_ack_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  uint64_t seq;
};

struct ARTS_PACKED arts_remote_invalidate_notice_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
};

struct ARTS_PACKED arts_remote_release_ownership_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
};

struct ARTS_PACKED arts_remote_get_data_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  uint64_t waiter_addr;
};

/* DATA_RESPONSE carries optional trailing buffer payload.
 * Body: db_guid(8) + version(8) + waiter_addr(8) + data_present(4) = 28,
 * total = 44 + 28 = 72 (already 8-aligned). pad[0] not portable, use
 * pad[4] for symmetry; payload offset becomes 76 — round up to 80. */
struct ARTS_PACKED arts_remote_data_response_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint64_t waiter_addr;
  uint32_t data_present;
  uint8_t pad[4];
};

/* DB_CREATE_COHERENT — no trailing payload (zero-init buffer at home).
 * Body: db_guid(8) + db_size(8) + flags(2) + db_type(2) = 20,
 * total = 44 + 20 = 64 (already 8-aligned).  Keep pad[4] anyway so any
 * future ABI growth has space without changing on-wire size. */
struct ARTS_PACKED arts_remote_db_create_coherent_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
  uint64_t db_size;
  uint16_t flags;
  uint16_t db_type;
  uint8_t pad[4];
};

struct ARTS_PACKED arts_remote_destroy_req_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
};

struct ARTS_PACKED arts_remote_destroy_notify_packet_s {
  struct arts_remote_packet_s header;
  arts_guid_t db_guid;
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
