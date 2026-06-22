/* SPDX-License-Identifier: Apache-2.0
 *
 * T174 — outbox MPSC insert/pop under multi-producer push + single-consumer
 * pop with partial-send re-park simulation (transport/outbox.c).
 *
 * The outbound side keeps one lock-free Vyukov MPSC queue per (rank,port)
 * slot.  Any worker/handler thread enqueues via arts_transport_send_async ->
 * the file-local arts_outbox_insert_node (push_back + outbox_pending +1).  A
 * single dedicated sender thread drains a slot via the file-local
 * arts_outbox_pop_node (pop_front).  When the kernel accepts only part of a
 * message the sender PARKS the popped node in a thread-local resend slot and
 * re-sends it on the next pump tick rather than re-queuing it — so a node,
 * once popped, never re-enters the MPSC queue.
 *
 * This test drives the REAL static insert/pop (compiled in by #include'ing
 * outbox.c) and the REAL link_list.c MPSC and asserts:
 *   (1) exactly N*K messages are delivered — none lost, none duplicated;
 *   (2) per-producer FIFO order is preserved on each queue (each producer
 *       pins one group_pos, so all its sends ride one queue in push order);
 *   (3) the slot routing list_id = rank*ports + group_pos%ports partitions
 *       producers across exactly `ports` queues, and the single consumer that
 *       owns those queues never sees a torn / mis-routed node;
 *   (4) outbox_pending == N*K after the drain (insert_node +1 per message;
 *       pop_node does NOT decrement — the matching -1 lives in socket.c's
 *       arts_actual_send, not on the pop path).
 *
 * The partial-send re-park is simulated faithfully: on a configurable
 * fraction of pops the consumer holds the popped node in a per-slot local
 * "resend" pointer and consumes it on the NEXT round before popping anew —
 * mirroring arts_outbox_resend, with the node kept OUT of the queue while
 * parked.  This exercises the consume-parked-before-pop-new ordering the real
 * pump loop uses.
 *
 * Standalone: #includes outbox.c (its statics) + links link_list.c; outbox.c's
 * heavy runtime externs are libc-backed shims; only insert_node / pop_node /
 * the MPSC queue are exercised.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- libc-backed shims for outbox.c's runtime externs ---- */
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void *arts_malloc(size_t s) { return malloc(s); }
void arts_free(void *p) { free(p); }
unsigned int arts_atomic_add(volatile unsigned int *d, unsigned int v) {
  return atomic_fetch_add((_Atomic unsigned int *)d, v) + v;
}
void arts_abort(uint8_t code) { exit(code ? code : 1); }
uint64_t arts_get_time_stamp(void) { return 0; }

#include "arts/runtime_state.h"
#include "arts/system/threads.h"
#include "arts/transport/dispatcher.h"
#include "arts/transport/protocol.h"
#include "arts/transport/socket.h"

uint64_t arts_transport_send(int rank, unsigned int queue, char *message,
                             uint64_t length) {
  (void)rank;
  (void)queue;
  (void)message;
  (void)length;
  return 0;
}
uint64_t arts_transport_send_payload(int rank, unsigned int queue,
                                     char *message, unsigned int length,
                                     char *payload, uint64_t length2) {
  (void)rank;
  (void)queue;
  (void)message;
  (void)length;
  (void)payload;
  (void)length2;
  return 0;
}
void arts_transport_dispatch_body(struct arts_msg_header_s *packet) {
  (void)packet;
}

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;
unsigned int arts_global_rank_count = 8;
unsigned int arts_global_rank_id = 0;
unsigned int ports = 4; /* PORTS — number of queues per rank */

/* Compile the real insert_node / pop_node statics + partial_store. */
#include "../../libs/src/core/transport/outbox.c"

/* ----------------------------------------------------------------------- */

#define PORTS 4
#define PRODUCERS 8 /* > PORTS so some producers share a queue (true MPSC) */
#define PER_PRODUCER 100000
#define TARGET_RANK 1 /* != self(0), < rank_count(8) */
#define TOTAL ((size_t)PRODUCERS * PER_PRODUCER)

/* Each message body (after the wire header) carries this tag so the consumer
 * can verify producer identity + per-producer sequence. */
struct tag {
  uint32_t producer;
  uint32_t seq;
};

static _Atomic(int) g_start;
static _Atomic(int) g_prod_done;

static void *producer(void *arg) {
  uint32_t id = (uint32_t)(uintptr_t)arg;
  /* Each producer pins a distinct group_pos so list_id is deterministic:
   * list_id = TARGET_RANK*ports + (id % ports).  Producers id and id+PORTS
   * therefore share one queue -> exercises real multi-producer contention. */
  arts_thread_info.group_pos = id;

  /* Build a message = wire header + tag.  insert_node casts (node+1) to a
   * header but writes nothing into it without SEQUENCENUMBERS, so the tag we
   * place right after the header survives intact. */
  unsigned int msg_len =
      (unsigned int)(sizeof(struct arts_msg_header_s) + sizeof(struct tag));
  char *buf = (char *)malloc(msg_len);

  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }

  for (uint32_t s = 0; s < PER_PRODUCER; ++s) {
    struct arts_msg_header_s *h = (struct arts_msg_header_s *)buf;
    h->message_type = 0;
    h->size = msg_len;
    h->rank = TARGET_RANK;
    struct tag *t = (struct tag *)(h + 1);
    t->producer = id;
    t->seq = s;
    arts_transport_send_async(TARGET_RANK, buf, msg_len);
  }
  free(buf);
  atomic_fetch_add_explicit(&g_prod_done, 1, memory_order_release);
  return NULL;
}

int main(void) {
  /* Allocate the outbound queue array: rank_count * ports slots. */
  arts_outbox_init(arts_global_rank_count * ports);

  pthread_t prod[PRODUCERS];
  for (int i = 0; i < PRODUCERS; ++i) {
    pthread_create(&prod[i], NULL, producer, (void *)(uintptr_t)i);
  }

  /* Per-producer next-expected sequence (per-producer FIFO check). */
  uint32_t *next_seq = (uint32_t *)calloc(PRODUCERS, sizeof(uint32_t));
  /* Per-producer delivered count (no-loss / no-dup check). */
  size_t *count = (size_t *)calloc(PRODUCERS, sizeof(size_t));

  /* The consumer owns queues [TARGET_RANK*ports .. +ports).  Per-slot parked
   * node mirrors arts_outbox_resend (a popped-but-partially-sent node held out
   * of the queue and consumed next round). */
  unsigned int slot_lo = TARGET_RANK * ports;
  struct arts_outbox_node_s *parked[PORTS];
  for (int i = 0; i < PORTS; ++i) {
    parked[i] = NULL;
  }

  atomic_store_explicit(&g_start, 1, memory_order_release);

  size_t got = 0;
  int failures = 0;
  int idle_after_done = 0;
  unsigned int repark_rotor = 0;

  while (got < TOTAL) {
    size_t before = got;
    for (unsigned int p = 0; p < PORTS; ++p) {
      unsigned int slot = slot_lo + p;
      struct arts_outbox_node_s *out;
      if (parked[p]) {
        /* Consume the previously re-parked node first (pump ordering). */
        out = parked[p];
        parked[p] = NULL;
      } else {
        void *free_me;
        out = arts_outbox_pop_node(slot, &free_me);
        if (!out) {
          continue; /* empty or producer mid-link — re-poll */
        }
        /* Simulate a partial send ~1/8 of the time: hold the node parked and
         * defer consumption to the next round.  The node stays OUT of the MPSC
         * queue while parked — exactly the resend-slot contract. */
        if ((++repark_rotor & 7u) == 0u) {
          parked[p] = out;
          continue;
        }
      }

      struct arts_msg_header_s *h = (struct arts_msg_header_s *)(out + 1);
      struct tag *t = (struct tag *)(h + 1);

      if (t->producer >= PRODUCERS) {
        fprintf(stderr, "FAIL outbox_mpsc: bad producer id %u\n", t->producer);
        ++failures;
      } else {
        /* Per-producer FIFO: a producer pins one group_pos -> one queue, so
         * its sequence must arrive strictly in order. */
        if (t->seq != next_seq[t->producer]) {
          fprintf(stderr,
                  "FAIL outbox_mpsc: producer %u out-of-order: got seq %u "
                  "expected %u (on queue slot %u)\n",
                  t->producer, t->seq, next_seq[t->producer], slot);
          ++failures;
        }
        next_seq[t->producer] = t->seq + 1;
        count[t->producer]++;
      }

      arts_link_list_delete_item(out);
      ++got;
    }

    if (atomic_load_explicit(&g_prod_done, memory_order_acquire) == PRODUCERS) {
      if (got == before) {
        if (++idle_after_done > 100000) {
          /* No progress after all producers finished -> lost messages. */
          break;
        }
      } else {
        idle_after_done = 0;
      }
    }
    if (failures > 16) {
      break; /* don't spam */
    }
  }

  for (int i = 0; i < PRODUCERS; ++i) {
    pthread_join(prod[i], NULL);
  }

  /* No node should remain parked. */
  for (int i = 0; i < PORTS; ++i) {
    if (parked[i]) {
      fprintf(stderr, "FAIL outbox_mpsc: node still parked in slot %d\n", i);
      ++failures;
    }
  }

  /* All queues drained. */
  for (unsigned int p = 0; p < PORTS; ++p) {
    void *free_me;
    if (arts_outbox_pop_node(slot_lo + p, &free_me) != NULL) {
      fprintf(stderr, "FAIL outbox_mpsc: queue slot %u not empty after drain\n",
              slot_lo + p);
      ++failures;
    }
  }

  /* No loss / no dup. */
  if (got != TOTAL) {
    fprintf(stderr, "FAIL outbox_mpsc: delivered %zu of %zu (loss/dup)\n", got,
            TOTAL);
    ++failures;
  }
  for (int i = 0; i < PRODUCERS; ++i) {
    if (count[i] != PER_PRODUCER || next_seq[i] != PER_PRODUCER) {
      fprintf(stderr,
              "FAIL outbox_mpsc: producer %d delivered %zu (next_seq %u), "
              "expected %d\n",
              i, count[i], next_seq[i], PER_PRODUCER);
      ++failures;
    }
  }

  /* outbox_pending accounting: insert_node did +1 per message; pop_node does
   * NOT decrement (the -1 is in socket.c arts_actual_send).  So after the
   * drain the counter must equal exactly the number of messages enqueued. */
  unsigned int pending = atomic_load_explicit(
      (_Atomic unsigned int *)&arts_node_info.outbox_pending,
      memory_order_acquire);
  if (pending != TOTAL) {
    fprintf(stderr,
            "FAIL outbox_mpsc: outbox_pending=%u expected %zu (insert +1/msg, "
            "pop does not decrement)\n",
            pending, TOTAL);
    ++failures;
  }

  /* Quiescent teardown of the queue array. */
  arts_outbox_cleanup();
  free(next_seq);
  free(count);

  if (failures) {
    fprintf(stderr, "FAIL outbox_mpsc: %d failures\n", failures);
    return 1;
  }

  printf("PASS outbox_mpsc: %zu messages, %d producers over %d queues, "
         "per-producer FIFO + no loss/dup + outbox_pending=%u OK\n",
         got, PRODUCERS, PORTS, pending);
  return 0;
}
