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
#ifndef ARTS_MEMORY_DB_H
#define ARTS_MEMORY_DB_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime_types.h"

/* Indexed by arts_guid_kind_t value (RESERVED=0, DB=1, EVENT=2, EDT=3).
 * Array size == ARTS_GUID_LAST (4); slot 0 is the reserved/NULL sentinel. */
#define ARTS_TYPE_NAME                                                         \
  const char *const arts_type_name[] = {"ARTS_GUID_RESERVED", "ARTS_GUID_DB",  \
                                        "ARTS_GUID_EVENT", "ARTS_GUID_EDT"}

#define GET_TYPE_NAME(x) arts_type_name[x]

extern const char *const arts_type_name[];

/* Internal-only access modes used by GPU dispatch, in the reserved range
 * arts.h anchors with DB_MODE_INTERNAL_BASE.  Each constant is cast to the
 * public type so it can be stored in arts_edt_dep_t.mode and compared
 * against it same-type; a cast integer constant expression remains valid in
 * case labels.  A GPU-less build has no producer or consumer of these, so
 * they do not exist there at all. */
#ifdef ARTS_USE_GPU
/** GPU LC -> CPU synchronous copy. */
#define DB_MODE_LC_SYNC ((arts_db_access_mode_t)(DB_MODE_INTERNAL_BASE + 0))
/** Allocate on GPU, no host -> GPU copy. */
#define DB_MODE_LC_NO_COPY ((arts_db_access_mode_t)(DB_MODE_INTERNAL_BASE + 1))
/** GPU zero-initialization. */
#define DB_MODE_MEMSET ((arts_db_access_mode_t)(DB_MODE_INTERNAL_BASE + 2))
#endif

#define DB_MODE_NAME                                                           \
  const char *const db_mode_name[] = {"DB_MODE_NULL", "DB_MODE_RO",            \
                                      "DB_MODE_RW", "DB_MODE_VAL"};            \
  const char *const db_mode_internal_name[] = {                                \
      "DB_MODE_LC_SYNC", "DB_MODE_LC_NO_COPY", "DB_MODE_MEMSET"}

#define GET_DB_MODE_NAME(x)                                                    \
  ((x) >= DB_MODE_INTERNAL_BASE                                                \
       ? db_mode_internal_name[(x) - DB_MODE_INTERNAL_BASE]                    \
       : db_mode_name[x])

extern const char *const db_mode_name[];
extern const char *const db_mode_internal_name[];

/* Order MUST match the arts_db_types_t enum in arts.h:
 * ARTS_DB(0), ARTS_DB_PIN(1), ARTS_DB_CXL(2), ARTS_DB_GPU(3),
 * ARTS_DB_GPU_PIN(4). */
#define ARTS_DB_TYPE_NAME                                                      \
  const char *const arts_db_type_name[] = {"ARTS_DB", "ARTS_DB_PIN",           \
                                           "ARTS_DB_CXL", "ARTS_DB_GPU",       \
                                           "ARTS_DB_GPU_PIN"}

#define GET_DB_TYPE_NAME(x) arts_db_type_name[x]

extern const char *const arts_db_type_name[];

void arts_db_acquire_all(struct arts_edt_s *edt);
/* OOO_DB_ACQUIRE replay (table entry): re-dispatch the ONE deferred local dep
 * through the per-dep 3-way (subtype-aware — ARTS_DB → arts_handler_db_acquire,
 * PIN/GPU/CXL → pinned ptr path, still-absent → re-defer).  item = the
 * just-installed db_s (ignored; the 3-way re-looks-it-up under the drain's
 * pinned ref); args = arts_ooo_args_db_acquire_s {edt, db_guid, slot}.  NOT the
 * coherent handler directly — that would mishandle non-coherent subtypes. */
void arts_db_acquire_replay_dep(void *item, void *args);
void release_dbs(unsigned int depc, arts_edt_dep_t *depv, bool gpu);
void arts_release_created_dbs(void);
void prep_dbs(unsigned int depc, arts_edt_dep_t *depv, bool gpu);

void *arts_db_malloc(arts_db_types_t db_type, size_t size);
void arts_db_free(void *ptr);

/* User-visible data pointer for a DB.  For coherent ARTS_DB it is the installed
 * buffer's payload; for every other subtype it is the inline payload after the
 * wrapping struct.  Defined in the per-protocol-compiled runtime so the inline
 * payload offset always matches the protocol's actual struct arts_db_s size —
 * callers (e.g. the OCR shim, linked across protocol variants) must use this
 * instead of `(db + 1)`, whose offset depends on the caller's compile-time view
 * of the protocol-conditional layout.  NULL-safe; NULL if no buffer is yet
 * installed on a coherent DB. */
void *arts_db_user_ptr(struct arts_db_s *db);

/* Internal pre/post-yield helpers used when an EDT yields (e.g.
 * arts_event_wait). Not part of the public ARTS API. */
void arts_wait_release_dbs(void);
void arts_wait_reacquire_dbs(void);

/* Internal: copy a DataBlock to a new GUID with a different DB subtype.
 * (Pure GUID rename — arts_db_rename — was removed as dispensable legacy.) */
arts_guid_t arts_db_copy_to_new_type(arts_guid_t old_guid,
                                     arts_db_types_t new_type);

#ifdef ARTS_USE_CXL
void arts_cxl_producer_flush(arts_guid_t guid);
void arts_cxl_consumer_flush(arts_guid_t guid);
#endif

#ifdef __cplusplus
}
#endif
#endif /* artsDBFUNCTIONS_H */
