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
#ifndef ARTS_RUNTIME_SYNC_RT_H
#define ARTS_RUNTIME_SYNC_RT_H
#ifdef __cplusplus
extern "C" {
#endif

// #define GNU_SOURCE
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>

#include "arts/arts_defs.h"
#include "arts/utils/link_list.h"

/// GUID type
typedef intptr_t arts_guid_t;
#define NULL_GUID ((arts_guid_t)0x0)

/// Pointer type
typedef uintptr_t arts_ptr_t;

// This is the ticker for context switching
typedef uint64_t arts_ticket_t;

typedef enum {
  ARTS_NULL = 0,
  ARTS_EDT,
  ARTS_GPU_EDT,
  ARTS_EVENT,
  ARTS_PERSISTENT_EVENT,
  ARTS_EPOCH,
  ARTS_CALLBACK,
  ARTS_BUFFER,
  // These are the DB modes.  Allocate/cast these types of DBs!

  // ARTS_DB_READ: This mode is write once read many.
  // Create the DB in this mode and write data before signaling the db_guid.
  // This mode aggregates requests, and caches reads in the routing table.
  ARTS_DB_READ,

  // ARTS_DB_WRITE: This mode is used to provide exclusive access.
  // Use this mode by casting ARTS_DB_READ DBs to ARTS_DB_WRITE and signal an
  // EDT. This mode is currently broken!!!
  ARTS_DB_WRITE,

  // ARTS_DB_PIN: This mode bypasses the memory model. The DB is only available
  // on a single node. To interact with it remotely use put/gets
  ARTS_DB_PIN,

  // ARTS_DB_ONCE: This mode will automatically free the DB after it is
  // acquired. This is to help memory management, since we are never reusing the
  // DB.
  ARTS_DB_ONCE,

  // ARTS_DB_ONCE: This mode is the same as ARTS_DB_ONCE except we are
  // guarenteing That the DB is local to the EDT accessing it (i.e. edt_guid and
  // db_guid have the same route).
  ARTS_DB_ONCE_LOCAL,

  ARTS_DB_GPU_READ,
  ARTS_DB_GPU_WRITE,
  ARTS_DB_LC,
  // End DB modes
  ARTS_LAST_TYPE,
  ARTS_SINGLE_VALUE,
  ARTS_PTR,
  ARTS_DB_LC_SYNC,
  ARTS_DB_LC_NO_COPY,
  ARTS_DB_GPU_MEMSET
} arts_type_t;

typedef struct {
  arts_guid_t guid;
  arts_type_t mode;
  void *ptr;
  arts_type_t acquire_mode;
} arts_edt_dep_t;

// Signature of an EDT
// Also signature of an GPU task
typedef void (*arts_edt_t)(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                          arts_edt_dep_t depv[]);

// Signature of an event callback_t.  The data parameter is the value of the
// data_guid used to satisfy the event.
typedef void (*event_callback_t)(arts_edt_dep_t data);

// Signature of the send handler used by arts_remote_send.
typedef void (*send_handler_t)(void *args);

typedef enum {
  ARTS_EVENT_LATCH_DECR_SLOT = 0,
  ARTS_EVENT_LATCH_INCR_SLOT = 1,
  ARTS_EVENT_UPDATE = 2 // Only for persistent events
} arts_latch_event_slot_t;

struct arts_header {
  uint8_t type : 8;
  uint64_t size : 56;
} __attribute__((aligned));

struct arts_db {
  struct arts_header header;
  uint64_t arts_id;           // Unique identifier from compiler (0 if not set)
  arts_guid_t guid;
  arts_guid_t event_guid;
  volatile unsigned int copyCount;
  volatile unsigned int reader;
  volatile unsigned int writer;
  volatile unsigned int version;
  unsigned int time_stamp;
  void *db_list;
} __attribute__((aligned));

struct arts_edt {
  struct arts_header header;
  uint64_t arts_id;           // Unique identifier from compiler (0 if not set)
  arts_edt_t func_ptr;
  uint32_t paramc;
  uint32_t depc;
  arts_guid_t current_edt;
  arts_guid_t output_buffer;
  arts_guid_t epoch_guid;
  unsigned int cluster;
  unsigned int node;
  volatile unsigned int depcNeeded;
  volatile unsigned int invalidateCount;
} __attribute__((aligned));

struct arts_dependent {
  uint8_t type;
  volatile unsigned int slot;
  volatile arts_guid_t addr;
  volatile event_callback_t callback_t;
  volatile bool doneWriting;
  arts_type_t acquire_mode;
  uint64_t byte_offset;
  uint64_t size;
};

struct arts_dependent_list {
  unsigned int size;
  struct arts_dependent_list *volatile next;
  struct arts_dependent dependents[];
};

struct arts_persistent_event_version {
  unsigned int version;
  volatile unsigned int latch_count;
  volatile unsigned int dependent_count;
  struct arts_dependent_list dependent;
};

struct arts_persistent_event {
  volatile unsigned int lock;
  struct arts_header header;
  arts_guid_t data;
  struct arts_link_list *versions;
} __attribute__((aligned));

struct arts_event {
  struct arts_header header;
  volatile bool fired;
  volatile unsigned int destroy_on_fire;
  volatile unsigned int latch_count;
  volatile unsigned int pos;
  volatile unsigned int dependent_count;
  arts_guid_t data;
  struct arts_dependent_list dependent;
} __attribute__((aligned));

struct arts_guid_range {
  unsigned int size;
  unsigned int index;
  arts_guid_t start_guid;
};
typedef struct arts_guid_range arts_guid_range_t;

struct arts_array_db {
  unsigned int element_size;
  unsigned int elements_per_block;
  unsigned int num_blocks;
  char head[];
};
typedef struct arts_array_db arts_array_db_t;

typedef enum { PHASE_1, PHASE_2, PHASE_3 } termination_detection_phase_t;

typedef struct {
  termination_detection_phase_t phase;
  volatile unsigned int activeCount;
  volatile unsigned int finishedCount;
  volatile unsigned int globalActiveCount;
  volatile unsigned int globalFinishedCount;
  volatile unsigned int lastActiveCount;
  volatile unsigned int lastFinishedCount;
  volatile uint64_t queued;
  volatile uint64_t outstanding;
  unsigned int terminationExitSlot;
  arts_guid_t terminationExitGuid;
  arts_guid_t guid;
  arts_guid_t pool_guid;
  volatile unsigned int *wait_ptr;
  volatile uint64_t ticket;
} arts_epoch_t;

typedef struct {
  void *buffer;
  uint32_t *size_to_write;
  unsigned int size;
  arts_guid_t epoch_guid;
  volatile unsigned int uses;
} arts_buffer_t;

void ARTS_PRINTF(const char *format, ...);

#ifdef __cplusplus
}
#endif

#endif /* ARTSRT_H */
