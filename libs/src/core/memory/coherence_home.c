/* SPDX-License-Identifier: Apache-2.0
 *
 * Home metadata helper implementations.  See coherence_home.h.
 */

#include "arts/memory/coherence_home.h"

#include <stdlib.h>
#include <string.h>

/*--- pending_rw FIFO ----------------------------------------------------*/

struct arts_lockfree_mpsc_s *arts_pending_rw_create(void) {
  struct arts_lockfree_mpsc_s *q =
      (struct arts_lockfree_mpsc_s *)calloc(1, sizeof(*q));
  return q;
}

void arts_pending_rw_destroy(struct arts_lockfree_mpsc_s *q) {
  if (q == NULL) {
    return;
  }
  struct arts_pending_rank_node_s *n = q->head;
  while (n != NULL) {
    struct arts_pending_rank_node_s *next = n->next;
    free(n);
    n = next;
  }
  free(q);
}

void arts_pending_rw_enqueue(struct arts_lockfree_mpsc_s *q,
                             unsigned int rank) {
  struct arts_pending_rank_node_s *node =
      (struct arts_pending_rank_node_s *)calloc(1, sizeof(*node));
  node->rank = rank;
  node->next = NULL;
  if (q->tail == NULL) {
    q->head = node;
    q->tail = node;
  } else {
    q->tail->next = node;
    q->tail = node;
  }
}

bool arts_pending_rw_dequeue(struct arts_lockfree_mpsc_s *q,
                             unsigned int *out_rank) {
  if (q->head == NULL) {
    return false;
  }
  struct arts_pending_rank_node_s *n = q->head;
  q->head = n->next;
  if (q->head == NULL) {
    q->tail = NULL;
  }
  *out_rank = n->rank;
  free(n);
  return true;
}

bool arts_pending_rw_empty(const struct arts_lockfree_mpsc_s *q) {
  return q->head == NULL;
}

/*--- last_sent_version dense map ---------------------------------------*/

struct arts_rank_to_u64_map_s *arts_rank_u64_map_create(unsigned int nranks) {
  struct arts_rank_to_u64_map_s *m =
      (struct arts_rank_to_u64_map_s *)calloc(1, sizeof(*m));
  m->nranks = nranks;
  m->slots = (uint64_t *)calloc((size_t)nranks, sizeof(uint64_t));
  return m;
}

void arts_rank_u64_map_destroy(struct arts_rank_to_u64_map_s *m) {
  if (m == NULL) {
    return;
  }
  free(m->slots);
  free(m);
}

uint64_t arts_rank_u64_map_get(const struct arts_rank_to_u64_map_s *m,
                               unsigned int rank) {
  if (rank >= m->nranks) {
    return 0;
  }
  return m->slots[rank];
}

void arts_rank_u64_map_set(struct arts_rank_to_u64_map_s *m, unsigned int rank,
                           uint64_t value) {
  if (rank >= m->nranks) {
    return;
  }
  m->slots[rank] = value;
}

bool arts_rank_u64_map_advance(struct arts_rank_to_u64_map_s *m,
                               unsigned int rank, uint64_t value) {
  if (rank >= m->nranks) {
    return false;
  }
  if (value > m->slots[rank]) {
    m->slots[rank] = value;
    return true;
  }
  return false;
}

/*--- arts_db_home_s lifecycle ------------------------------------------*/

struct arts_db_home_s *arts_db_home_create(unsigned int rw_holder,
                                           unsigned int nranks) {
  struct arts_db_home_s *home =
      (struct arts_db_home_s *)calloc(1, sizeof(*home));
  home->rw_holder = rw_holder;
  home->pending_rw = arts_pending_rw_create();
  home->last_sent_version = arts_rank_u64_map_create(nranks);
  return home;
}

void arts_db_home_destroy(struct arts_db_home_s *home) {
  if (home == NULL) {
    return;
  }
  arts_pending_rw_destroy(home->pending_rw);
  arts_rank_u64_map_destroy(home->last_sent_version);
  free(home);
}
