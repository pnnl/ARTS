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
#ifndef ARTS_GAS_ROUTETABLE_H
#define ARTS_GAS_ROUTETABLE_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"
#include "arts/defs.h"
#include "arts/gas/out_of_order_list.h"

struct arts_db_frontier_iterator_s;

// These are for the lock for each item in the RT
#define RESERVED_ITEM 0x8000000000000000
#define AVAILABLE_ITEM 0x4000000000000000
#define DELETE_ITEM 0x2000000000000000
#define STATUS_MASK (RESERVED_ITEM | AVAILABLE_ITEM | DELETE_ITEM)

#define MAX_ITEM 0x1FFFFFFFFFFFFFFF
#define COUNT_MASK ~(RESERVED_ITEM | AVAILABLE_ITEM | DELETE_ITEM)
#define CHECK_MAX_ITEM(x) ((((x) & COUNT_MASK) + 1) < MAX_ITEM)
#define GET_COUNT(x) ((x) & COUNT_MASK)

#define IS_DEL(x) ((x) & DELETE_ITEM)
#define IS_RES(x)                                                              \
  (((x) & RESERVED_ITEM) && !((x) & AVAILABLE_ITEM) && !((x) & DELETE_ITEM))
#define IS_AVAIL(x)                                                            \
  (((x) & AVAILABLE_ITEM) && !((x) & RESERVED_ITEM) && !((x) & DELETE_ITEM))
#define IS_REQ(x)                                                              \
  (((x) & RESERVED_ITEM) && ((x) & AVAILABLE_ITEM) && !((x) & DELETE_ITEM))

#define SHOULD_DELETE(x) (IS_DEL(x) && !GET_COUNT(x))

#define COLLISION_RESOLVES 8

struct arts_route_invalidate_s {
  int size;
  int used;
  struct arts_route_invalidate_s *next;
  unsigned int data[];
};

typedef enum {
  NO_KEY = 0,
  ANY_KEY,
  DELETED_KEY,   // deleted only
  ALLOCATED_KEY, // reserved, available, or requested
  AVAILABLE_KEY, // available only
  REQUESTED_KEY, // available but reserved (means so one else has the valid
                 // copy)
  RESERVED_KEY,  // reserved only
} item_state_t;

struct arts_route_item_s {
  arts_guid_t key;
  void *data;
  volatile uint64_t lock;
  unsigned int rank;
  unsigned int touched;
  struct arts_out_of_order_list_s ooList;
} ARTS_ALIGNED_MAX;

typedef struct arts_route_item_s arts_route_item_t;

typedef struct arts_route_table_s arts_route_table_t;

typedef void (*set_route_item_t)(arts_route_item_t *item, void *data);
typedef void (*free_route_item_t)(arts_route_item_t *item);
typedef arts_route_table_t *(*new_route_table_t)(unsigned int route_table_size,
                                                 unsigned int shift);

// Add padding around locks...
struct arts_route_table_s {
  arts_route_item_t *data;
  unsigned int size;
  unsigned int shift;
  struct arts_route_table_s *next;
  volatile unsigned readerLock;
  volatile unsigned writerLock;
  set_route_item_t setFunc;
  free_route_item_t freeFunc;
  new_route_table_t newFunc;
}; // __attribute__ ((aligned));

typedef struct {
  uint64_t index;
  arts_route_table_t *table;
} arts_route_table_iterator_t;

bool dec_item(arts_route_table_t *route_table, arts_route_item_t *item);

arts_route_table_t *arts_new_route_table(unsigned int route_table_size,
                                         unsigned int shift);

void *arts_route_table_add_item(void *item, arts_guid_t key, unsigned int rank,
                                bool used);
arts_route_item_t *internal_route_table_add_item_race(
    bool *added_item, arts_route_table_t *route_table, void *item,
    arts_guid_t key, unsigned int rank, bool used_res, bool used_avail,
    unsigned int to_add_on_creation);
bool arts_route_table_add_item_race(void *item, arts_guid_t key,
                                    unsigned int route, bool used);
arts_route_item_t *
internal_route_table_add_deleted_item_race(arts_route_table_t *route_table,
                                           void *item, arts_guid_t key,
                                           unsigned int rank);

void *arts_route_table_lookup_item(arts_guid_t key);
int arts_route_table_lookup_rank(arts_guid_t key);
bool internal_route_table_remove_item(arts_route_table_t *route_table,
                                      arts_guid_t key);
bool arts_route_table_remove_item(arts_guid_t key);
bool arts_route_table_mark_delete(arts_guid_t key);
bool arts_route_table_hide_item(arts_guid_t key);
bool arts_route_table_invalidate_item(arts_guid_t key);

arts_route_item_t *
arts_route_table_search_for_key(arts_route_table_t *route_table,
                                arts_guid_t key, item_state_t state);
bool arts_route_table_update_item(arts_guid_t key, void *data,
                                  unsigned int rank, item_state_t state);
bool arts_route_table_add_sent(arts_guid_t key, void *edt, unsigned int slot,
                               bool aggregate);

item_state_t arts_route_table_lookup_item_with_state(arts_guid_t key,
                                                     void ***data,
                                                     item_state_t min,
                                                     bool inc);
item_state_t getitem_state(arts_route_item_t *item);

int arts_route_table_set_rank(arts_guid_t key, int rank);

void **arts_route_table_reserve(arts_guid_t key, bool *dec,
                                item_state_t *state);

void arts_route_table_dec_item(arts_guid_t key, void *data);
arts_route_item_t *get_item_from_data(arts_guid_t key, void *data);

unsigned int internal_inc_db_version(volatile unsigned int *touched);
void *internal_route_table_lookup_db(arts_route_table_t *route_table,
                                     arts_guid_t key, int *rank,
                                     unsigned int **touched);
void *arts_route_table_lookup_db(arts_guid_t key, int *rank, bool touch);
bool internal_route_table_return_db(arts_route_table_t *route_table,
                                    arts_guid_t key, bool mark_to_delete,
                                    bool do_delete);
bool arts_route_table_return_db(arts_guid_t key, bool mark_to_delete);

bool arts_route_table_add_oo(arts_guid_t key, void *data, bool inc);
bool arts_route_table_add_oo_existing(arts_guid_t key, void *data, bool inc);
void arts_route_table_fire_oo(arts_guid_t key,
                              void (*callback_t)(void *, void *));
void arts_route_table_reset_oo(arts_guid_t key);
void **arts_route_table_get_oo_list(arts_guid_t key,
                                    struct arts_out_of_order_list_s **list);

void arts_reset_route_table_iterator(arts_route_table_iterator_t *iter,
                                     arts_route_table_t *table);
arts_route_item_t *arts_route_table_iterate(arts_route_table_iterator_t *iter);
void arts_print_item(arts_route_item_t *item);
void arts_route_table_debug_guid(arts_guid_t key, const char *label);

uint64_t arts_clean_up_route_table(arts_route_table_t *route_table);
void arts_delete_route_table(arts_route_table_t *route_table);
void arts_clean_up_dbs();

#ifdef __cplusplus
}
#endif

#endif
