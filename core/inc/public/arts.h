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
#ifndef ARTS_ARTS_H
#define ARTS_ARTS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "arts/runtime/rt.h"

#include "arts/utils/atomics.h"

// This is the entry point to starting the ARTS runtime.  Call from main.
int arts_rt(int argc, char **argv);

// Shuts down the arts runtime.  It is possible to race to shutdown if there are
// multiple calls.
void arts_shutdown();

/*Malloc***********************************************************************/

// Allocates memory of size bytes.  Arts applications should use this allocation
// method to give the runtime a full view of resource utilization.
void *arts_malloc(size_t size);

// Allocates memory of size bytes with given alignment.  Arts applications
// should use this allocation method to give the runtime a full view of resource
// utilization.
void *arts_malloc_align(size_t size, size_t align);

// Allocates memory of size bytes and initializes the memory to zero.  Arts
// applications should use this allocation method to give the runtime a full
// view of resource utilization.
void *arts_calloc(size_t nmemb, size_t size);

// Allocates memory of size bytes with given alignment and initializes the
// memory to zero.  Arts applications should use this allocation method to give
// the runtime a full view of resource utilization.
void *arts_calloc_align(size_t nmemb, size_t size, size_t align);

// Resizes memory that was previously allocated using arts_malloc or arts_calloc.
// This is here for completion but should probably not be used.
void *arts_realloc(void *ptr, size_t size);

// Releases memory allocated by arts_malloc or arts_calloc.
void arts_free(void *ptr);

/*GUID*************************************************************************/

// Reserves a guid of a given type that corresponds to the node given by route.
arts_guid_t arts_reserve_guid_route(arts_type_t type, unsigned int route);

// Indicates if a guid is local to the given node.
bool arts_is_guid_local(arts_guid_t guid);

// Returns the rank of the node who owns guid.
unsigned int arts_guid_get_rank(arts_guid_t guid);

// Returns the type of the guid.
arts_type_t arts_guid_get_type(arts_guid_t guid);

// Returns guid cast to the given type.  This is used to change the access mode
// of a DB
arts_guid_t arts_guid_cast(arts_guid_t guid, arts_type_t type);

// Returns a new guid range of the given size and type.  This allocates size
// consecutive guids on the same node. Since guids are formed by a bit field
// with several fields, the actual value of guids may not be consecutive making
// it cumbersome to handle many guids individually.  Guid ranges provide a way
// of accessing many guids from one the start guid.
arts_guid_range_t *arts_new_guid_range_node(arts_type_t type, unsigned int size,
                                    unsigned int route);

// Gets guid at index away from the start guid of the range.
arts_guid_t arts_get_guid(arts_guid_range_t *range, unsigned int index);

// Guid ranges act as an iterator moving through the guids sequentially.
// If the range is at the last guid in the range, NULL is returned.  Otherwise,
// the next guid in the range is return. Guid ranges are not thread-safe.
arts_guid_t arts_guid_range_next(arts_guid_range_t *range);

// Returns true if there are guids left in the range, false otherwise.  Guid
// Ranges are not thread-safe.
bool arts_guid_range_has_next(arts_guid_range_t *range);

// Resets the iterator in guid ranges.  Guid Ranges are not thread-safe.
void arts_guid_range_reset_iter(arts_guid_range_t *range);

arts_guid_t *arts_reserve_guids_round_robin(unsigned int size, arts_type_t type);

/*EDT**************************************************************************/

// Creates an Event Driven Task (EDT) to run on node route.
// Paramc are the number of static parameters.
// Paramv are the static parameters that are copied into the EDT closure.
// Depc is the number of dependencies required for the EDT to run.  The EDT will
// run with depc slots. Returns a new guid for the new EDT.
arts_guid_t arts_edt_create(arts_edt_t func_ptr, unsigned int route, uint32_t paramc,
                         const uint64_t *paramv, uint32_t depc);

// Creates an EDT with arts_id tracking (for ArtsMate integration).
// Same as arts_edt_create, but also stores the compiler-assigned arts_id
// in the EDT structure for runtime performance tracking.
arts_guid_t arts_edt_create_with_arts_id(arts_edt_t func_ptr, unsigned int route,
                                   uint32_t paramc, const uint64_t *paramv,
                                   uint32_t depc, uint64_t arts_id);

// Creates an EDT associated with a specific epoch and arts_id.
// Mirrors arts_edt_create_with_epoch but preserves the compiler-assigned arts_id
// for runtime introspection.
arts_guid_t arts_edt_create_with_epoch_arts_id(arts_edt_t func_ptr, unsigned int route,
                                        uint32_t paramc, const uint64_t *paramv,
                                        uint32_t depc, arts_guid_t epoch_guid,
                                        uint64_t arts_id);

/// Creates a parallel EDT to run on all the workers on the node. It then waits
/// for all the EDTs to finish before continuing.
// ! Not implemented
// void artsEdtParallel(arts_edt_t func_ptr, unsigned int route, uint32_t paramc,
//                      const uint64_t *paramv, uint32_t depc);

// Creates an EDT with the given guid.  The guid will run on the home node of
// the guid. Paramc are the number of static parameters. Paramv are the static
// parameters that are copied into the EDT closure. Depc is the number of
// dependencies required for the EDT to run.  The EDT will run with depc slots.
// Returns the guid for the new EDT.
arts_guid_t arts_edt_create_with_guid(arts_edt_t func_ptr, arts_guid_t guid,
                                 uint32_t paramc, const uint64_t *paramv,
                                 uint32_t depc);

// Creates an EDT to run on node route that will run in the given epoch.  User
// must ensure the epoch is still live. Paramc are the number of static
// parameters. Paramv are the static parameters that are copied into the EDT
// closure. Depc is the number of dependencies required for the EDT to run.  The
// EDT will run with depc slots. Returns a new guid for the new EDT.
arts_guid_t arts_edt_create_with_epoch(arts_edt_t func_ptr, unsigned int route,
                                  uint32_t paramc, const uint64_t *paramv,
                                  uint32_t depc, arts_guid_t epoch_guid);

// Creates an EDT to run on node route.  The has_depv flag can be used to not
// allocate slots for the EDT, but still use the depc counter. This is useful
// when an EDT has a large dependency count but doesn't require any results from
// them which would otherwise require significant memory. Paramc are the number
// of static parameters. Paramv are the static parameters that are copied into
// the EDT closure. Depc is the number of dependencies required for the EDT to
// run. Returns a new guid for the new EDT.
arts_guid_t arts_edt_create_dep(arts_edt_t func_ptr, unsigned int route,
                            uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                            bool has_depv);

// Creates an EDT with the given guid.  The guid will run on the home node of
// the guid. The has_depv flag can be used to not allocate slots for the EDT, but
// still use the depc counter. This is useful when an EDT has a large dependency
// count but doesn't require any results from them which would otherwise require
// significant memory. Paramc are the number of static parameters. Paramv are
// the static parameters that are copied into the EDT closure. Depc is the
// number of dependencies required for the EDT to run.  The EDT will run with
// depc slots. Returns the guid for the new EDT.
arts_guid_t arts_edt_create_with_guid_dep(arts_edt_t func_ptr, arts_guid_t guid,
                                    uint32_t paramc, const uint64_t *paramv,
                                    uint32_t depc, bool has_depv);

// Creates an EDT to run on node route that will run in the given epoch.  User
// must ensure the epoch is still live. The has_depv flag can be used to not
// allocate slots for the EDT, but still use the depc counter. This is useful
// when an EDT has a large dependency count but doesn't require any results from
// them which would otherwise require significant memory. Paramc are the number
// of static parameters. Paramv are the static parameters that are copied into
// the EDT closure. Depc is the number of dependencies required for the EDT to
// run.  The EDT will run with depc slots. Returns a new guid for the new EDT.
arts_guid_t arts_edt_create_with_epoch_dep(arts_edt_t func_ptr, unsigned int route,
                                     uint32_t paramc, const uint64_t *paramv,
                                     uint32_t depc, arts_guid_t epoch_guid,
                                     bool has_depv);

// Destroys an EDT and removes its guid from the routing table.  EDT are
// automatically destroyed after running.
void arts_edt_destroy(arts_guid_t guid);

// Signals an EDT that a dependency is met.  When all dependencies have been met
// an EDT is scheduled to run. The EDT's arts_edt_dep_t at slot is filled with
// data_guid to a DB and the memory the data_guid/DB points to. The mode the DB is
// acquired in is based on the type of data_guid.  The mode an EDT acquires a DB
// can be changed using arts_guid_cast.
void arts_signal_edt(arts_guid_t edt_guid, uint32_t slot, arts_guid_t data_guid);

// Signals an EDT that a dependency is met.  When all dependencies have been met
// an EDT is scheduled to run. Only signals a signal value not a guid.  The
// value is stored in the guid field of arts_edt_dep_t.
void arts_signal_edt_value(arts_guid_t edt_guid, uint32_t slot, uint64_t value);

// Signals an EDT that a dependency is met.  When all dependencies have been met
// an EDT is scheduled to run. Instead of using a DB, the data of size is copied
// to the ptr field of arts_edt_dep_t.  This memory is freed after the EDT runs.
void arts_signal_edt_ptr(arts_guid_t edt_guid, uint32_t slot, void *ptr,
                      unsigned int size);

// ESD: Signals an EDT with a pointer while preserving the original DB GUID.
// This is used for byte-slice dependencies where we need both the pointer
// to the slice (db_ptr + byte_offset) and the original DB GUID for lifecycle
// management. The EDT's depv[slot].guid will contain db_guid, and
// depv[slot].ptr will contain the computed pointer.
void arts_signal_edt_ptr_with_guid(arts_guid_t edt_guid, uint32_t slot,
                              arts_guid_t db_guid, void *ptr, unsigned int size);

// Signals an EDT slot as satisfied without any data. Used for boundary
// conditions where a dependency should be skipped (e.g., stencil edges).
// The slot is marked satisfied with ARTS_NULL mode - no data is provided.
void arts_signal_edt_null(arts_guid_t edt_guid, uint32_t slot);

// Creates an EDT to run where the db_guid is located.  This is a wrapper around
// EDT create and signal.
arts_guid_t arts_active_message_with_db(arts_edt_t func_ptr, uint32_t paramc,
                                   const uint64_t *paramv, uint32_t depc,
                                   arts_guid_t db_guid);

// Creates an EDT to run on node rank and signals the edt with DB pointed to by
// db_guid.  This is a wrapper around EDT create and signal.
arts_guid_t arts_active_message_with_db_at(arts_edt_t func_ptr, uint32_t paramc,
                                     const uint64_t *paramv, uint32_t depc,
                                     arts_guid_t db_guid, unsigned int rank);

// Creates an EDT and copies data into a buffer.  This is a wrapper around EDT
// create and buffer signal.
arts_guid_t arts_active_message_with_buffer(arts_edt_t func_ptr, unsigned int route,
                                       uint32_t paramc, const uint64_t *paramv,
                                       uint32_t depc, void *data,
                                       unsigned int size);

// This creates a buffer of size bytes, and returns a guid to access the buffer.
// The buffer can only be accessed on the node the buffer was allocated on, but
// can be written to from anywhere. Uses indicate how many accesses to the
// buffer until the entry in the routing table is freed. The epoch_guid is used
// to make sure setBuffer is included in a given round of termination detection.
arts_guid_t arts_allocate_local_buffer(void **buffer, unsigned int size,
                                   unsigned int uses, arts_guid_t epoch_guid);

// Copies the data from buffer of size bytes to the buffer pointed to by
// buffer_guid.  This still counts as an access so the uses in the buffer
// allocation should be > 1.
void *arts_set_buffer(arts_guid_t buffer_guid, void *buffer, unsigned int size);

// Return buffer pointed to by buffer_guid and decrements uses.  If uses == 0 the
// buffer is removed from the guid table. The buffer is only available on a the
// node it was allocated.
void *arts_get_buffer(arts_guid_t buffer_guid);

void *arts_block_for_buffer(arts_guid_t buffer_guid);

/*Event************************************************************************/

// Creates a latch event on node route and returns the new guid to signal.  A
// latch event has a counter that can be incremented and decremented via its
// slots (arts_latch_event_slot_t).  When the counter reaches zero the event fires.
// An event can be used to broadcast a db_guid to EDTs and other events.  Events
// can also be used to execute a callback_t function.  This differs from an EDT in
// that the callback_t is executed immediately when the counter reaches 0.
arts_guid_t arts_event_create(unsigned int route, unsigned int latch_count);

// Creates a latch event on node home to guid and returns the guid to signal.  A
// latch event has a counter that can be incremented and decremented via its
// slots (arts_latch_event_slot_t).  When the counter reaches zero the event fires.
// An event can be used to broadcast a db_guid to EDTs and other events.  Events
// can also be used to execute a callback_t function.  This differs from an EDT in
// that the callback_t is executed immediately when the counter reaches 0.
arts_guid_t arts_event_create_with_guid(arts_guid_t guid, unsigned int latch_count);

// Returns if an event is already fired.
bool arts_is_event_fired(arts_guid_t event);

// Destroys a local event.
void arts_event_destroy(arts_guid_t guid);

// Signals an event with a data_guid.  There are two arts_latch_event_slot_t that
// can be signaled.  ARTS_EVENT_LATCH_INCR_SLOT increments the event's counter.
// ARTS_EVENT_LATCH_DECR_SLOT decrements the events counter.  Once the counter
// reaches 0, the event is fired.  As long as the event exists (hasn't been
// destroyed), any arts_add_dependence or arts_add_local_event_callback will be
// executed immediately.
void arts_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                          uint32_t slot);

// Add a connection between a source event and an EDT or other event.  When the
// event fires, it will signal the destination guid and slot.  If the event has
// already fired, the signal to the destination EDT or event will propagate
// immediately.
void arts_add_dependence(arts_guid_t source, arts_guid_t destination,
                       uint32_t slot);

// Adds a callback_t to be executed when the source event fires.  This differs
// from an EDT in that the callback_t is executed immediately when the counter
// reaches 0.
void arts_add_local_event_callback(arts_guid_t source, event_callback_t callback_t);

/*Persistent Event*************************************************************/
// An arts_persistent_event represents a reusable synchronization point that can
// fire multiple times. Unlike a one-time event, a persistent event remains
// alive after being satisfied and can be triggered repeatedly.
arts_guid_t arts_persistent_event_create(unsigned int route,
                                     unsigned int latch_count,
                                     arts_guid_t data_guid);

/// Satisfy the persistent event
void arts_persistent_event_satisfy(arts_guid_t event_guid, uint32_t action,
                                bool lock);

// Increment the latch count of a persistent event. This is used to indicate
// that a new dependency has been added to the event, allowing it to fire
// again.
void arts_persistent_event_increment_latch(arts_guid_t event_guid);

// Decrement the latch count of a persistent event. This is used to indicate
// that a dependency has been satisfied. If the latch count reaches zero,
// the event will fire, signaling any dependent EDTs.
void arts_persistent_event_decrement_latch(arts_guid_t event_guid);

// Adds a dependence from a source persistent event to a destination EDT slot.
// If the source event latch count is zero, the destination EDT will be signaled
// immediately.
void arts_add_dependence_to_persistent_event(arts_guid_t event_source,
                                        arts_guid_t edt_dest, uint32_t edt_slot);

// Adds a dependence with compiler-inferred acquire mode override
void arts_add_dependence_to_persistent_event_with_mode(arts_guid_t event_source,
                                                arts_guid_t edt_dest,
                                                uint32_t edt_slot,
                                                arts_type_t acquire_mode);

// Adds a dependence with compiler-inferred acquire mode override
void arts_add_dependence_to_persistent_event_with_mode_and_diff(arts_guid_t event_source,
                                                       arts_guid_t edt_dest,
                                                       uint32_t edt_slot,
                                                       arts_type_t acquire_mode);

/// ESD: Adds a dependence with byte offset for slice-based signaling.
/// When byte_offset > 0 or size > 0, the persistent event will signal
/// with a pointer to (db_ptr + byte_offset) while preserving the DB GUID.
void arts_add_dependence_to_persistent_event_with_byte_offset(
    arts_guid_t event_source, arts_guid_t edt_dest, uint32_t edt_slot,
    arts_type_t acquire_mode, uint64_t byte_offset, uint64_t size);

/*DB***************************************************************************/

// A DataBlock (DB) is the main memory abstraction used in ARTS to share data
// between tasks.  A DB can be one of several types (see ARTS_DB* in arts_type_t)
// which will dictate how they are accessed.  DBs are a fixed size and are
// accessed via signaling an EDT with a DBs guid.  When the EDT runs, it will
// have access to the guid, access mode, and raw data via arts_edt_dep_t. This
// creates a DB of size bytes and type mode, stores the pointer to the data in
// addr, and returns a new guid for the created DB.  The DB is created local to
// the calling node.
arts_guid_t arts_db_create(void **addr, uint64_t size, arts_type_t mode);
arts_guid_t arts_db_create_ptr(arts_ptr_t *addr, uint64_t size, arts_type_t mode);

// Creates a DB with arts_id tracking (for ArtsMate integration).
// Same as arts_db_create, but also stores the compiler-assigned arts_id
// in the DB structure for runtime performance tracking.
arts_guid_t arts_db_create_with_arts_id(void **addr, uint64_t size, arts_type_t mode,
                                  uint64_t arts_id);

// Creates a DB using a pre-reserved GUID and assigns the compiler arts_id.
// Keeps existing routing semantics while enabling arts_id tracking.
void *arts_db_create_with_guid_and_arts_id(arts_guid_t guid, uint64_t size,
                                    uint64_t arts_id);

// Creates a DB with a fixed guid of size bytes if the guid is local.  The type
// and route is already fixed by the provided guid, and the pointer to the raw
// data is returned.
void *arts_db_create_with_guid(arts_guid_t guid, uint64_t size);

// Creates a new DB similarly to arts_db_create_with_guid, except the data is
// copyied to the new DB.  This is useful if there are outstanding out-of-order
// requests for the DB which will be satisfied once the DB is created. Otherwise
// there may be a race for the user to write new data to the DB and the EDTs
// acquiring the DB.
void *arts_db_create_with_guid_and_data(arts_guid_t guid, void *data, uint64_t size);

// Creates a DB for a remote node route of size bytes and type mode.  The DB
// will be uninitialized.
arts_guid_t arts_db_create_remote(unsigned int route, uint64_t size,
                              arts_type_t mode);

// Moves a datablock to a remote node.  This can be problematic since the guid
// hasn't changed and that means remote accesses to the DB, will look at the
// home of db_guid. Access from rank will see the DB though.
void arts_db_move(arts_guid_t db_guid, unsigned int rank);

// Destroys all copies of the DB in the system.
void arts_db_destroy(arts_guid_t guid);

// Only removes local copy if found.  If the DB remote, then sends to the DB
// home to destroy.
void arts_db_destroy_safe(arts_guid_t guid, bool remote);

// Writes data in ptr of size bytes to the DB pointed to by db_guid with an
// offest on the home node of db_guid.  The EDT pointed to by edt_guid is signaled
// at slot. The put will be included in whatever epoch is running.
void arts_put_in_db(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                 unsigned int slot, unsigned int offset, unsigned int size);

// Writes data in ptr of size bytes to the DB pointed to by db_guid with an
// offest on node rank.  The EDT pointed to by edt_guid is signaled at slot. The
// put will be included in whatever epoch is running.
void arts_put_in_db_at(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                   unsigned int slot, unsigned int offset, unsigned int size,
                   unsigned int rank);

// Writes data in ptr of size bytes to the DB pointed to by db_guid with an
// offest on the home node of db_guid. The put will be included in whatever epoch
// specified by epoch_guid.
void arts_put_in_db_epoch(void *ptr, arts_guid_t epoch_guid, arts_guid_t db_guid,
                      unsigned int offset, unsigned int size);

// Gets a copy of data in a DB of size bytes and offset from the DB pointed to
// by db_guid.  The data is signaled to the EDT pointed to by edt_guid using
// arts_signal_edt_ptr. Data is copied from the DB found at the home of db_guid.
void arts_get_from_db(arts_guid_t edt_guid, arts_guid_t db_guid, unsigned int slot,
                   unsigned int offset, unsigned int size);

// Gets a copy of data in a DB of size bytes and offset from the DB pointed to
// by db_guid.  The data is signaled to the EDT pointed to by edt_guid using
// arts_signal_edt_ptr. Data is copied from the DB found on node rank.
void arts_get_from_db_at(arts_guid_t edt_guid, arts_guid_t db_guid, unsigned int slot,
                     unsigned int offset, unsigned int size, unsigned int rank);

arts_guid_t arts_db_rename(arts_guid_t guid);

bool arts_db_rename_with_guid(arts_guid_t new_guid, arts_guid_t old_guid);

arts_guid_t arts_db_copy_to_new_type(arts_guid_t old_guid, arts_type_t new_type);

// Increment the latch count associated with the persistent event of the DB
void arts_db_increment_latch(arts_guid_t guid);

// Decrement the latch count associated with the persistent event of the DB
void arts_db_decrement_latch(arts_guid_t guid);

// Adds a dependence from a the persistent event associated with the DB to an
// EDT slot.
void arts_db_add_dependence(arts_guid_t db_src, arts_guid_t edt_dest,
                         uint32_t edt_slot);

// Adds a dependence with compiler-inferred acquire mode override
void arts_db_add_dependence_with_mode(arts_guid_t db_src, arts_guid_t edt_dest,
                                 uint32_t edt_slot, arts_type_t acquire_mode);

// Adds a dependence with compiler-inferred acquire mode override
void arts_db_add_dependence_with_mode_and_diff(arts_guid_t db_src, arts_guid_t edt_dest,
                                        uint32_t edt_slot,
                                        arts_type_t acquire_mode);

// Records a dependency and automatically increments latch when acquire_mode is
// ARTS_DB_WRITE
void arts_record_dep(arts_guid_t db_src, arts_guid_t edt_dest, uint32_t edt_slot,
                   arts_type_t acquire_mode);

/// ESD: Records a dependency at a specific byte offset within the DB.
/// When the DB becomes ready, signals with a pointer to (db_ptr + byte_offset)
/// while preserving the original DB GUID in depv[slot].guid.
/// This is used for stencil halo dependencies.
void arts_record_dep_at(arts_guid_t db_src, arts_guid_t edt_dest, uint32_t edt_slot,
                     arts_type_t acquire_mode, uint64_t byte_offset,
                     uint64_t size);

/*Epoch************************************************************************/

// Returns the current round of termination detection.
arts_guid_t arts_get_current_epoch_guid();

// Makes an EDT part of a specific round of termination detection.  User must
// ensure the EDT hasn't run and the epoch is not over.
void arts_add_edt_to_epoch(arts_guid_t edt_guid, arts_guid_t epoch_guid);

// This creates a new round of termation detection and starts the epoch.  Any
// EDTS created (in the currently running EDT) will correspond to this epoch.
// When the epoch is finished finish_edt_guid will be signaled.  The db_guid field
// of arts_edt_dep_t pointed to by slot will be filled with the number of EDTs,
// buffer alloc/set, get/puts, etc. executed in the epoch.
arts_guid_t arts_initialize_and_start_epoch(arts_guid_t finish_edt_guid,
                                       unsigned int slot);

// This creates a new round of termination detection with the source node of
// rank, but doesn't start it. When the epoch is finished finish_edt_guid will be
// signaled.  The db_guid field of arts_edt_dep_t pointed to by slot will be filled
// with the number of EDTs, buffer alloc/set, get/puts, etc. executed in the
// epoch.
arts_guid_t arts_initialize_epoch(unsigned int rank, arts_guid_t finish_edt_guid,
                               unsigned int slot);

// Starts an epoch created with arts_initialize_epoch.
void arts_start_epoch(arts_guid_t epoch_guid);

// Blocks waiting for epoch to finish.  Only works from an EDT that created the
// epoch.  The current executing thread calls another round of scheduling until
// the epoch finishes.
bool arts_wait_on_handle(arts_guid_t epoch_guid);

// Block current execution and runs another round of scheduling.
void arts_yield();

// Creates a context ticket for arts_signal_context to signal when waiting using
// arts_context_switch
arts_ticket_t arts_get_context_ticket();

// Context switch between threads.  The wait_count is how many signals are
// required Before we can wake this context up.  This must have tmt set in the
// config file.
bool arts_context_switch(unsigned int wait_count);

// Context switch between threads, but does not block the current context.
void arts_open_context_switch();

void arts_next_context();

unsigned int arts_get_context_id();

// Used to wake up a context asleep from a context switch
bool arts_signal_context(arts_ticket_t ticket);

/*ArrayDb************************************************************************/

// ArrayDb is an array that spans all the nodes of the execution.  This returns
// a guid for accessing the arrayDb which can be used anywhere in the system.
// The data is spread equally across all nodes.
arts_guid_t arts_new_array_db(arts_array_db_t **addr, unsigned int element_size,
                          unsigned int num_elements);

// This creates a new arrayDb with a fixed guid.  The guid can be for any node,
// but must be of type ARTS_DB_PIN.
arts_array_db_t *arts_new_array_db_with_guid(arts_guid_t guid, unsigned int element_size,
                                      unsigned int num_elements);

// This ArrayDB is local to the node and is not shared across nodes.
arts_array_db_t *arts_new_local_array_db_with_guid(arts_guid_t guid,
                                           unsigned int element_size,
                                           unsigned int num_elements,
                                           void *data);

void arts_signal_array_db(arts_array_db_t *array, arts_guid_t edt_guid,
                       unsigned int slot);

// Gets an element from an array DB at a specific index.  The results is
// placed in slot for edt_guid using arts_signal_edt_ptr.
void arts_get_from_array_db(arts_guid_t edt_guid, unsigned int slot,
                        arts_array_db_t *array, unsigned int index);

// Puts data in a specific index of an arrayDb and signals edt_guid at slot on
// completion.  This put will also fall in the current epoch.
void arts_put_in_array_db(void *ptr, arts_guid_t edt_guid, unsigned int slot,
                      arts_array_db_t *array, unsigned int index);

// Launches an EDT for each element in the arrayDb.  Data is acquired using
// arts_signal_edt_ptr, thus no element can be changed. EDTs are launched locally.
void arts_for_each_in_array_db(arts_array_db_t *array, arts_edt_t func_ptr,
                          uint32_t paramc, const uint64_t *paramv);

// Launches an EDT for each element in the arrayDb.  Data is acquired using
// arts_signal_edt_ptr, thus no element can be changed. EDTs are launched across
// all nodes.
void arts_for_each_in_array_db_at_data(arts_array_db_t *array, unsigned int stride,
                                arts_edt_t func_ptr, uint32_t paramc,
                                const uint64_t *paramv);

// Gathers all of the chunks of arrayDb on a single node and runs an EDT
// func_ptr.  Each chunk is written using arts_signal_edt_ptr to the EDT's
// arts_edt_dep_t.
void arts_gather_array_db(arts_array_db_t *array, arts_edt_t func_ptr,
                       unsigned int route, uint32_t paramc, const uint64_t *paramv,
                       uint64_t depc);
void arts_gather_array_db_epoch(arts_array_db_t *array, arts_edt_t func_ptr,
                            unsigned int route, uint32_t paramc,
                            const uint64_t *paramv, uint64_t depc,
                            arts_guid_t epoch_guid);
void arts_gather_array_db_in_edt(arts_array_db_t *array, arts_guid_t to_edt_guid,
                            uint64_t slot_offset);

// Performs and atomic add at index, and signals edt_guid at slot upon
// completion.  Also corresponds to the executing epoch.
void arts_atomic_add_in_array_db(arts_array_db_t *array, unsigned int index,
                            unsigned int to_add, arts_guid_t edt_guid,
                            unsigned int slot);

// Performs and atomic compare and swap at index, and signals edt_guid at slot
// upon completion.  Also corresponds to the executing epoch.
void arts_atomic_compare_and_swap_in_array_db(arts_array_db_t *array, unsigned int index,
                                       unsigned int old_value,
                                       unsigned int new_value,
                                       arts_guid_t edt_guid, unsigned int slot);

/*Util*************************************************************************/
/// Returns the guid of the input edt_dep
inline arts_guid_t arts_get_guid_from_edt_dep(arts_edt_dep_t dep) { return dep.guid; }

/// Returns the pointer to the data in the edt_dep
inline void *arts_get_ptr_from_edt_dep(arts_edt_dep_t dep) { return dep.ptr; }

// Returns the guid of the current EDT running.
arts_guid_t arts_get_current_guid();

// Returns the rank of the current node.
unsigned int arts_get_current_node();

// Returns the total number of nodes.
unsigned int arts_get_total_nodes();

// Returns the unique id of the current thread on the current node.
unsigned int arts_get_current_worker();

// Returns the total number of worker threads.  This does not include network
// send/receive threads.
unsigned int arts_get_total_workers();

// Returns the unique id of the current numa domain on the current node.
// Requires HWLOC.
unsigned int arts_get_current_cluster();

// Gets the total number of numa domains.  Requires HWLOC.
unsigned int arts_get_total_clusters();

// Gets the total number of GPUs per node.
unsigned int arts_get_total_gpus();

// Arts timer in nanoseconds.
uint64_t arts_get_time_stamp();

// Gives a threadsafe random number
uint64_t arts_thread_safe_random();

// This is a way to send operations to a specific rank.  If the rank is the
// current node, the function is executed immediately. If the rank is remote,
// the arguments will be packaged, and sent to the appropriate node.  In this
// case the function will be executed by the receiver threads.  The arguments
// are freed if the free flag is set.
void arts_remote_send(unsigned int rank, send_handler_t fun_ptr, void *args,
                    unsigned int size, bool free);
#ifdef __cplusplus
}
#endif
#endif
