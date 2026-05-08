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
#ifndef ARTS_SYNC_SHARED_H
#define ARTS_SYNC_SHARED_H
#ifdef __cplusplus
extern "C" {
#endif

/* arts_shared_t — per-object deleter holder.
 *
 * Type-erased reference-counted base
 * introduces this struct as the D1-core primitive that every dynamic
 * ARTS object (event, DB, EDT, epoch) will eventually embed as its FIRST
 * member, via the ARTS_SHARED_FIELD macro.
 *
 * Reference counting itself lives in the route_table item's `lock` field
 * (Task 4e — `[DELETE:1 | gen:31 | count:32]`).  arts_shared_t carries
 * only the per-object deleter pointer, keeping the embedded overhead at
 * one pointer and centralising atomic state in the route table.
 *
 * The route_table free_item path (Task 4f) reads `data->shared.deleter`
 * and invokes it once the route_table lock count drops to 0 with the
 * DELETE bit set.  Until each object type is migrated to embed
 * ARTS_SHARED_FIELD (every type that embeds shared
 * 8), `free_item` falls back to the legacy per-type free path — see
 * `object_has_shared_field()` in route_table.c.
 */
typedef struct arts_shared_s {
  void (*deleter)(void *); /* invoked by route_table free_item once
                              count == 0 and DELETE bit is set */
} arts_shared_t;

/* Embed-as-first-member macro.  All migrated dynamic ARTS objects place
 * `ARTS_SHARED_FIELD;` as the very first declaration in their struct so
 * that a `void *data` pointer to the object aliases an `arts_shared_t *`. */
#define ARTS_SHARED_FIELD arts_shared_t shared

static inline void arts_shared_init(arts_shared_t *s, void (*deleter)(void *)) {
  s->deleter = deleter;
}

#ifdef __cplusplus
}
#endif
#endif /* ARTS_SYNC_SHARED_H */
