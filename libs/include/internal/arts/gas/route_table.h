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

/* Portable atomic-void-pointer typedef.  C uses _Atomic; C++/nvcc uses a
 * plain pointer accessed via __atomic_* builtins (which gcc/clang/nvcc all
 * support via the host compiler).  Wire-compatible -- both sides see the
 * same bit layout (sizeof(void *) on both compilers). */
#ifdef __cplusplus
typedef void *arts_atomic_voidp_t;
#else
#include <stdatomic.h>
typedef _Atomic(void *) arts_atomic_voidp_t;
#endif

#define COLLISION_RESOLVES 8

struct arts_route_invalidate_s {
  int size;
  int used;
  struct arts_route_invalidate_s *next;
  unsigned int data[];
};

/* add_oo_ex return enum — caller decides branch.
 *
 * AVAILABLE_NOW: data was non-NULL on the pre-push or push-time check, so the
 *   payload was never inserted into the OO list.  Caller handles inline and
 *   frees the payload itself.
 *
 * ENQUEUED: data was NULL throughout; payload sits in the OO list and a
 *   future installer's fire_oo will dispatch it.  Caller does nothing.
 *
 * FIRED_BY_DRAIN: payload was successfully pushed, but the post-push
 *   recheck found data non-NULL (installer raced past us).  add_oo_ex
 *   then called fire_oo itself, which drained the list and invoked the
 *   handler on every entry, including ours, and freed each payload.
 *   Caller MUST NOT free the payload and MUST NOT re-issue the handler. */
typedef enum {
  OO_RESULT_ENQUEUED,
  OO_RESULT_AVAILABLE_NOW,
  OO_RESULT_FIRED_BY_DRAIN,
} oo_add_result_t;

/* Route_item: 3 fields only. Slot is permanent (init-array, never freed). */
struct arts_route_item_s {
  arts_guid_t key;
  arts_atomic_voidp_t data;     /* NULL = pending OoO, else = AVAILABLE */
  struct arts_oo_list_s ooList; /* lock-free list */
} ARTS_ALIGNED_MAX;

typedef struct arts_route_item_s arts_route_item_t;

typedef struct arts_route_table_s arts_route_table_t;

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
  new_route_table_t newFunc;
}; // __attribute__ ((aligned));

typedef struct {
  uint64_t index;
  arts_route_table_t *table;
} arts_route_table_iterator_t;

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

/* Lookup: returns data ptr directly (atomic_load_acquire). NULL means
 * not-yet-created or destroyed. */
void *arts_route_table_lookup_data(arts_guid_t key);

/* Item lookup — returns data ptr. Replaces legacy lookup_item. */
void *arts_route_table_lookup_item(arts_guid_t key);

/* DB-specific lookup — returns data cast to arts_db_s *. Replaces legacy
 * lookup_db. */
void *arts_route_table_lookup_db(arts_guid_t key, int *rank, bool touch);

int arts_route_table_lookup_rank(arts_guid_t key);
bool arts_route_table_mark_delete(arts_guid_t key);
bool arts_route_table_hide_item(arts_guid_t key);

arts_route_item_t *
arts_route_table_search_for_key(arts_route_table_t *route_table,
                                arts_guid_t key);
int arts_route_table_set_rank(arts_guid_t key, int rank);

/* Slot reserve or lookup — used internally by add_oo_ex. New entries are
 * initialized with data=NULL. */
void arts_route_table_reserve_or_lookup(arts_guid_t key,
                                        arts_route_item_t **out);

arts_route_item_t *get_item_from_data(arts_guid_t key, void *data);

/* OoO-integrated add — returns enum */
oo_add_result_t arts_route_table_add_oo_ex(arts_guid_t key, void *payload);

/* Compatibility wrapper for the 5 existing callers
 * (signal_edt / event_satisfy / add_dep / ready_edt / db_request). */
bool arts_route_table_add_oo(arts_guid_t key, void *payload, bool inc);

/* Compatibility wrapper for the legacy "_existing" variant
 * (route_table.c:786). */
bool arts_route_table_add_oo_existing(arts_guid_t key, void *payload, bool inc);

/* Fire the OO list — called by the installer (e.g. DB_CREATE). Idempotent. */
void arts_route_table_fire_oo(arts_guid_t key,
                              void (*callback)(void *data, void *ctx));

/* Free OO list memory at destroy time — no callback, payload is freed. */
void arts_route_table_drop_oo(arts_guid_t key);

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
