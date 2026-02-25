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
#include "arts/network/remote_protocol.h"

#include <string.h>
#include <unistd.h>

#include "arts.h"
#include "arts/network/socket_server.h"
#include "arts/runtime/runtime.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/link_list.h"
#include "arts/utils/malloc.h"

struct out_list_s {
  uint64_t time_stamp;
  unsigned int offset;
  unsigned int length;
  unsigned int rank;
  void *payload;
  uint64_t payloadSize;
  unsigned int offsetPayload;
  void (*free_method)(void *);
};

unsigned int node_list_size;

struct arts_link_list_s *out_head;
extern unsigned int ports;

ARTS_THREAD_LOCAL unsigned int thread_start;
ARTS_THREAD_LOCAL unsigned int thread_stop;

ARTS_THREAD_LOCAL struct out_list_s **out_resend;

#ifdef SEQUENCENUMBERS
unsigned int *seq_num_lock = NULL;
uint64_t *seq_number = NULL;
ARTS_THREAD_LOCAL uint64_t *last_out;
ARTS_THREAD_LOCAL uint64_t *last_sent;
#endif

void partial_send_store(struct out_list_s *out, uint64_t length_remaining) {
  if (out->payload == NULL) {
    out->offset = out->offset + (out->length - length_remaining);
    out->length = length_remaining;
  } else {
    uint64_t sent = out->length + out->payloadSize;
    sent -= length_remaining;
    if (sent >= out->length) {
      out->length = 0;
      out->offsetPayload =
          out->offsetPayload + (out->payloadSize - length_remaining);
      out->payloadSize = length_remaining;

    } else {
      out->offset =
          out->offset + (out->length - (length_remaining - out->payloadSize));
      out->length = length_remaining - out->payloadSize;
    }
  }
}

void arts_remote_set_thread_outbound_queues(unsigned int start,
                                            unsigned int stop) {
  thread_start = start;
  thread_stop = stop;

  unsigned int size = stop - start;
  out_resend =
      (struct out_list_s **)arts_calloc(size, sizeof(struct out_list_s *));
#ifdef SEQUENCENUMBERS
  last_out = (uint64_t *)arts_calloc(arts_global_rank_count, sizeof(uint64_t));
  last_sent = (uint64_t *)arts_calloc(arts_global_rank_count, sizeof(uint64_t));
#endif
}

void arts_remote_thread_outbound_queues_cleanup() {
  if (out_resend) {
    unsigned int size = thread_stop - thread_start;
    for (unsigned int i = 0; i < size; i++) {
      if (out_resend[i]) {
        struct out_list_s *out = out_resend[i];
        if (out->free_method) {
          out->free_method(out->payload);
        }
        arts_link_list_delete_item(out);
      }
    }
    arts_free(out_resend);
    out_resend = NULL;
  }
#ifdef SEQUENCENUMBERS
  if (last_out) {
    arts_free(last_out);
    last_out = NULL;
  }
  if (last_sent) {
    arts_free(last_sent);
    last_sent = NULL;
  }
#endif
}

void out_cleanup(void) {
  if (out_head) {
    for (unsigned int i = 0; i < node_list_size; i++) {
      struct arts_link_list_s *list = arts_link_list_get(out_head, i);
      struct arts_link_list_item_s *cur = list->headPtr;
      while (cur) {
        struct arts_link_list_item_s *next = cur->next;
        struct out_list_s *out = (struct out_list_s *)(cur + 1);
        if (out->payload && out->free_method) {
          out->free_method(out->payload);
        }
        arts_free(cur);
        cur = next;
      }
    }
    arts_free(out_head);
    out_head = NULL;
  }
#ifdef SEQUENCENUMBERS
  arts_free(seq_number);
  seq_number = NULL;
  arts_free(seq_num_lock);
  seq_num_lock = NULL;
#endif
  node_list_size = 0;
}

void out_init(unsigned int size) {
  node_list_size = size;
  out_head = arts_link_list_group_new(size);
#ifdef SEQUENCENUMBERS
  seq_number =
      (uint64_t *)arts_calloc(arts_global_rank_count, sizeof(uint64_t));
  seq_num_lock = (unsigned int *)arts_calloc(size, sizeof(unsigned int));
#endif
}

// Actively flush all outbound queues by directly sending (with timeout)
// This works even if sender threads have stopped, by sending from calling
// thread
void arts_remote_flush_outbound(void) {
  if (!out_head || node_list_size == 0) {
    return;
  }

  uint64_t timeout = arts_get_time_stamp() + 5000000000ULL; // 5 second timeout

  // Track partial sends per queue (not using thread-local out_resend)
  struct out_list_s **pending_sends = (struct out_list_s **)arts_calloc(
      node_list_size, sizeof(struct out_list_s *));

  while (arts_get_time_stamp() < timeout) {
    bool did_work = false;
    bool all_empty = true;

    // Process all outbound queues
    for (unsigned int i = 0; i < node_list_size; i++) {
      struct out_list_s *out = NULL;

      // Check for pending partial send first
      if (pending_sends[i]) {
        out = pending_sends[i];
      } else {
        // Pop new item from queue
        void *free_me;
        struct arts_link_list_s *list = arts_link_list_get(out_head, i);
        out = (struct out_list_s *)arts_link_list_pop_front(list, &free_me);
      }

      if (out) {
        all_empty = false;
        uint64_t length_remaining;

        if (!out->payload) {
          length_remaining = arts_remote_send_request(
              (int)out->rank, i, ((char *)(out + 1)) + out->offset,
              out->length);
        } else {
          length_remaining = arts_remote_send_payload_request(
              (int)out->rank, i, ((char *)(out + 1)) + out->offset, out->length,
              ((char *)out->payload) + out->offsetPayload, out->payloadSize);
          if (out->free_method && !length_remaining) {
            out->free_method(out->payload);
          }
        }

        if (length_remaining == (uint64_t)-1) {
          // Send error, skip this queue for now
          pending_sends[i] = out;
          continue;
        }

        if (length_remaining) {
          // Partial send, store for retry
          partial_send_store(out, length_remaining);
          pending_sends[i] = out;
        } else {
          // Fully sent, free the item
          pending_sends[i] = NULL;
          arts_link_list_delete_item(out);
        }
        did_work = true;
      } else {
        // Check if queue has more items
        struct arts_link_list_s *list = arts_link_list_get(out_head, i);
        if (!arts_link_list_is_empty(list)) {
          all_empty = false;
        }
      }
    }

    if (all_empty) {
      // All queues empty and no pending sends
      bool has_pending = false;
      for (unsigned int i = 0; i < node_list_size; i++) {
        if (pending_sends[i]) {
          has_pending = true;
          break;
        }
      }
      if (!has_pending) {
        break;
      }
    }

    if (!did_work) {
      usleep(100); // Small sleep if no progress
    }
  }

  // Clean up any remaining pending sends (shouldn't happen normally)
  for (unsigned int i = 0; i < node_list_size; i++) {
    if (pending_sends[i]) {
      struct out_list_s *out = pending_sends[i];
      if (out->free_method) {
        out->free_method(out->payload);
      }
      arts_link_list_delete_item(out);
    }
  }
  arts_free(pending_sends);
}

static inline void out_insert_node(struct out_list_s *node,
                                   unsigned int length) {
  (void)length;
  // int list_id = node->rank*ports+arts_thread_info.thread_id%ports;
  long unsigned int list_id;
  // mrand48_r (&arts_thread_info.drand_buf, &list_id);
  list_id = (node->rank * ports) + (arts_thread_info.group_pos % ports);
  struct arts_link_list_s *list = arts_link_list_get(out_head, list_id);
  struct arts_remote_packet_s *packet =
      (struct arts_remote_packet_s *)(node + 1);
#ifdef SEQUENCENUMBERS
  arts_lock(&seq_num_lock[list_id]);
  packet->seq_num = arts_atomic_fetch_add_u64(&seq_number[node->rank], 1U);
  packet->seq_rank = arts_global_rank_id;
#endif
  arts_link_list_push_back(list, node);
#ifdef SEQUENCENUMBERS
  arts_unlock(&seq_num_lock[list_id]);
#endif
}

static inline struct out_list_s *out_pop_node(unsigned int thread_id,
                                              void **free_me) {
  struct out_list_s *out;
  struct arts_link_list_s *list;
  list = arts_link_list_get(out_head, thread_id);
  out = (struct out_list_s *)arts_link_list_pop_front(list, free_me);
  if (out) {
    struct arts_remote_packet_s *packet =
        (struct arts_remote_packet_s *)(out + 1);
#ifdef SEQUENCENUMBERS
    if (last_out[packet->seq_rank] &&
        packet->seq_num != last_out[packet->seq_rank] + 1) {
      ARTS_DEBUG("POP OUT OF ORDER %u -> %u %lu vs %lu %p", packet->seq_rank,
                 packet->rank, last_out[packet->seq_rank], packet->seq_num,
                 list);
    }
    last_out[packet->seq_rank] = packet->seq_num;
#endif
  }
  return out;
}

bool arts_remote_async_send() {
  bool success = false;

  void *free_me;
  uint64_t length_remaining;
  struct out_list_s *out;

  bool sent = true;
  while (sent) {
    sent = false;
    // Loop over our threads
    for (int i = (int)thread_start; i < (int)thread_stop; i++) {
      out = NULL;                              // For looping purposes...
      if (out_resend[i - (int)thread_start]) { // Checking failed sends?
        out = out_resend[i - (int)thread_start];
      } else { // Look for new messages
        out = out_pop_node(i, &free_me);
      }

      if (out) {
#ifdef SEQUENCENUMBERS
        struct arts_remote_packet_s *packet =
            (struct arts_remote_packet_s *)(out + 1);
        if (last_sent[packet->seq_rank] != packet->seq_num &&
            packet->seq_num != last_sent[packet->seq_rank] + 1) {
          ARTS_DEBUG("SENT OUT OF ORDER %lu vs %lu",
                     last_sent[packet->seq_rank], packet->seq_num);
        }
        last_sent[packet->seq_rank] = packet->seq_num;
#endif
        if (!out->payload) {
          length_remaining = arts_remote_send_request(
              (int)out->rank, i, ((char *)(out + 1)) + out->offset,
              out->length);
        } else {
          length_remaining = arts_remote_send_payload_request(
              (int)out->rank, i, ((char *)(out + 1)) + out->offset, out->length,
              ((char *)out->payload) + out->offsetPayload, out->payloadSize);
          if (out->free_method && !length_remaining) {
            out->free_method(out->payload);
          }
        }

        if (length_remaining == (uint64_t)-1) {
          if (out->payload && out->free_method) {
            out->free_method(out->payload);
          }
          out_resend[i - (int)thread_start] = NULL;
          arts_link_list_delete_item(out);
          return false;
        }
        if (length_remaining) {
          partial_send_store(out, length_remaining);
          out_resend[i - (int)thread_start] = out;
        } else {
          struct arts_remote_packet_s *packet =
              (struct arts_remote_packet_s *)(out + 1);
          out_resend[i - (int)thread_start] = NULL;
          arts_link_list_delete_item(out);
        }

        sent = true;
        success = true;
      }
    }
  }
  return success;
}

static inline bool self_send_check(unsigned int rank) {
  if (rank == arts_global_rank_id || rank >= arts_global_rank_count) {
    ARTS_WARN("Cannot send to rank %u (self=%u, total=%u)", rank,
              arts_global_rank_id, arts_global_rank_count);
    return false;
  }
  return true;
}

static inline void size_send_check(uint64_t size) {
  if (size == 0) {
    ARTS_ERROR("Cannot send zero-size message");
  }
}

void arts_remote_send_request_async(int rank, char *message,
                                    unsigned int length) {
  if (!self_send_check(rank)) {
    return;
  }
  struct out_list_s *next = (struct out_list_s *)arts_link_list_new_item(
      length + sizeof(struct out_list_s));
  next->offset = 0;
  next->offsetPayload = 0;
  next->length = length;
  next->rank = rank;
  next->payload = NULL;
  memcpy(next + 1, message, length);
  out_insert_node(next, length + sizeof(struct out_list_s));
}

void arts_remote_send_request_payload_async(int rank, char *message,
                                            unsigned int length, char *payload,
                                            uint64_t size) {
  if (!self_send_check(rank)) {
    return;
  }
  size_send_check(length);
  size_send_check(size);
  struct out_list_s *next = (struct out_list_s *)arts_link_list_new_item(
      length + sizeof(struct out_list_s));
  next->offset = 0;
  next->offsetPayload = 0;
  next->length = length;
  next->rank = rank;
  next->payload = payload;
  next->free_method = NULL;
  next->payloadSize = size;
  memcpy(next + 1, message, length);
  out_insert_node(next, length + sizeof(struct out_list_s));
}

void arts_remote_send_request_payload_async_free(
    int rank, char *message, unsigned int length, char *payload,
    unsigned int offset, uint64_t size, void (*free_method)(void *)) {
  if (!self_send_check(rank)) {
    return;
  }
  size_send_check(length);
  size_send_check(size);
  struct out_list_s *next = (struct out_list_s *)arts_link_list_new_item(
      length + sizeof(struct out_list_s));
  next->offset = 0;
  next->offsetPayload = offset;
  next->length = length;
  next->rank = rank;
  next->payload = payload;
  next->payloadSize = size;
  next->free_method = free_method;
  memcpy(next + 1, message, length);
  out_insert_node(next, length + sizeof(struct out_list_s));
}
