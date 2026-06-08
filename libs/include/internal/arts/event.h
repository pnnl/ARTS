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
#ifndef ARTS_SYNC_EVENT_H
#define ARTS_SYNC_EVENT_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts/runtime_types.h"
#include "arts/utils/lockfree_lifo.h" /* arts_lf_link_t */
#include <stddef.h>                   /* offsetof */

/** Dep node — one per registered EDT or chained event waiting on this
 *  source event.  Allocated from arts_node_info.event_dep_pool.
 *  After drain, the firing thread releases each dep back to that pool
 *  via arts_lf_pool_release. */
struct arts_event_dep_s {
  arts_lf_link_t link;        /* MUST be first (link = node addr) */
  arts_guid_kind_t kind;      /* ARTS_GUID_EDT or ARTS_GUID_EVENT */
  arts_guid_t target;         /* destination GUID */
  uint32_t slot;              /* destination slot */
  arts_db_access_mode_t mode; /* dep mode (preserved across signal) */
};
/* event.h is included from C (.c) and, via edt.c's #include into edt_gpu.cu,
 * from CUDA C++ (nvcc).  C++ spells the static assertion `static_assert`; C
 * uses `_Static_assert`.  Guard so both front-ends accept this header. */
#ifdef __cplusplus
static_assert(offsetof(struct arts_event_dep_s, link) == 0,
              "link must be first for arts_lf_link_t round-tripping");
#else
_Static_assert(offsetof(struct arts_event_dep_s, link) == 0,
               "link must be first for arts_lf_link_t round-tripping");
#endif

/* arts_event_add_dependence — entity-specific API (src=event): register a
 * dependent on an event source.  arts_add_dependence's event-source branch
 * delegates here. */
void arts_event_add_dependence(arts_guid_t source, arts_guid_t destination,
                               uint32_t slot, arts_db_access_mode_t mode);

/* Mark a simple (non-channel) event single-shot: it marks itself for deletion
 * on fire.  Used for cross-rank finish proxies so they are reclaimed instead of
 * lingering.  No-op for channel events or an absent GUID. */
void arts_event_set_auto_destroy(arts_guid_t guid);

/* OoO replay handlers (g_ooo_table) — operate on the acquired event. */
void arts_handler_event_satisfy_slot(void *item, void *args);
void arts_handler_event_add_dependence(void *item, void *args);

/* Cross-rank wire TX/RX for event ops. */
void arts_send_event_add_dependence(arts_guid_t source, arts_guid_t destination,
                                    uint32_t slot, unsigned int rank,
                                    arts_db_access_mode_t mode);
void arts_send_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                                  uint32_t slot);
void arts_handler_event_create(void *ptr);
/* Cross-rank arts_event_destroy: forwarder + handler.
 * Forwarder serializes the GUID into MSG_EVENT_DESTROY;
 * handler runs arts_route_table_mark_delete on the home rank.  mark_delete
 * is idempotent (DELETE is sticky), so duplicate messages are safe. */
void arts_send_event_destroy(arts_guid_t guid);
void arts_handler_event_destroy(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_SYNC_EVENT_H */
