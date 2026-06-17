/* SPDX-License-Identifier: Apache-2.0 — MRSW waiter FIFO: FIFO order, pop-one,
 * peek_empty. */
#include "arts/coherence/mrsw/types.h"
#include <assert.h>
#include <stdio.h>

static struct arts_db_rw_waiter_queue_s q;

int main(void) {
  arts_db_rw_waiter_queue_init(&q);
  assert(arts_db_rw_waiter_queue_peek_empty(&q));
  for (unsigned i = 0; i < 100; i++)
    arts_db_rw_waiter_queue_push(&q, (arts_guid_t)(i + 1), i);
  assert(!arts_db_rw_waiter_queue_peek_empty(&q));
  arts_guid_t g;
  unsigned int slot;
  for (unsigned i = 0; i < 100; i++) {
    assert(arts_db_rw_waiter_queue_pop(&q, &g, &slot));
    assert(slot == i); /* FIFO order */
    assert(g ==
           (arts_guid_t)(i + 1)); /* payload round-trips (guid is intptr_t) */
  }
  assert(!arts_db_rw_waiter_queue_pop(&q, &g, &slot)); /* empty */
  assert(arts_db_rw_waiter_queue_peek_empty(&q));
  arts_db_rw_waiter_queue_destroy(&q);
  printf("PASS\n");
  return 0;
}
