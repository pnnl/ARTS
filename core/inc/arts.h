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
#ifndef ARTS_H
#define ARTS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "artsRT.h"

//This is the entry point to starting the ARTS runtime.  Call from main.    
int artsRT(int argc, char **argv);

//Shuts down the arts runtime.  It is possible to race to shutdown if there are multiple calls.
void artsShutdown();

/*Malloc***********************************************************************/

//Allocates memory of size bytes.  Arts applications should use this allocation method to give the runtime a full view of resource utilization.
void *artsMalloc(size_t size);

//Allocates memory of size bytes with given alignment.  Arts applications should use this allocation method to give the runtime a full view of resource utilization.
void *artsMallocAlign(size_t size, size_t align);

//Allocates memory of size bytes and initializes the memory to zero.  Arts applications should use this allocation method to give the runtime a full view of resource utilization.
void *artsCalloc(size_t size);

//Allocates memory of size bytes with given alignment and initializes the memory to zero.  Arts applications should use this allocation method to give the runtime a full view of resource utilization.
void *artsCallocAlign(size_t size, size_t allign);

//Resizes memory that was previously allocated using artsMalloc or artsCalloc.  This is here for completion but should probably not be used.
void * artsRealloc(void *ptr, size_t size);

//Releases memory allocated by artsMalloc or artsCalloc.
void artsFree(void *ptr);

//Releases memory allocated by artsMallocAlign or artsCallocAlign.
void artsFreeAlign(void *ptr);

/*GUID*************************************************************************/

//Reserves a guid of a given type that corresponds to the node given by route.
artsGuid_t artsReserveGuidRoute(artsType_t type, unsigned int route);

//Indicates if a guid is local to the given node. 
bool artsIsGuidLocal(artsGuid_t guid);

//Returns the rank of the node who owns guid.
unsigned int artsGuidGetRank(artsGuid_t guid);

//Returns the type of the guid.
artsType_t artsGuidGetType(artsGuid_t guid);

//Returns guid cast to the given type.  This is used to change the access mode of a DB
artsGuid_t artsGuidCast(artsGuid_t guid, artsType_t type);

//Returns a new guid range of the given size and type.  This allocates size consecutive guids on the same node.  
//Since guids are formed by a bit field with several fields, the actual value of guids may not be consecutive
//making it cumbersome to handle many guids individually.  Guid ranges provide a way of accessing many guids
//from one the start guid.
artsGuidRange * artsNewGuidRangeNode(unsigned int type, unsigned int size, unsigned int route);

//Gets guid at index away from the start guid of the range.
artsGuid_t artsGetGuid(artsGuidRange * range, unsigned int index);

//Guid ranges act as an iterator moving through the guids sequentially.  
//If the range is at the last guid in the range, NULL is returned.  Otherwise, the next guid in the range is return.
//Guid ranges are not thread-safe.
artsGuid_t artsGuidRangeNext(artsGuidRange * range);

//Returns true if there are guids left in the range, false otherwise.  Guid Ranges are not thread-safe.
bool artsGuidRangeHasNext(artsGuidRange * range);

//Resets the iterator in guid ranges.  Guid Ranges are not thread-safe.
void artsGuidRangeResetIter(artsGuidRange * range);

/*EDT**************************************************************************/

//Creates an Event Driven Task (EDT) to run on node route.
//Paramc are the number of static parameters.
//Paramv are the static parameters that are copied into the EDT closure.
//Depc is the number of dependencies required for the EDT to run.  The EDT will run with depc slots.
//Returns a new guid for the new EDT.
artsGuid_t artsEdtCreate(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc);

//Creates an EDT with the given guid.  The guid will run on the home node of the guid.
//Paramc are the number of static parameters.
//Paramv are the static parameters that are copied into the EDT closure.
//Depc is the number of dependencies required for the EDT to run.  The EDT will run with depc slots.
//Returns the guid for the new EDT.
artsGuid_t artsEdtCreateWithGuid(artsEdt_t funcPtr, artsGuid_t guid, uint32_t paramc, uint64_t * paramv, uint32_t depc);

//Creates an EDT to run on node route that will run in the given epoch.  User must ensure the epoch is still live.
//Paramc are the number of static parameters.
//Paramv are the static parameters that are copied into the EDT closure.
//Depc is the number of dependencies required for the EDT to run.  The EDT will run with depc slots.
//Returns a new guid for the new EDT.
artsGuid_t artsEdtCreateWithEpoch(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsGuid_t epochGuid);

//Creates an EDT to run on node route.  The hasDepv flag can be used to not allocate slots for the EDT, but still use the depc counter.
//This is useful when an EDT has a large dependency count but doesn't require any results from them which would otherwise require significant memory.
//Paramc are the number of static parameters.
//Paramv are the static parameters that are copied into the EDT closure.
//Depc is the number of dependencies required for the EDT to run.
//Returns a new guid for the new EDT.
artsGuid_t artsEdtCreateDep(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, bool hasDepv);

//Creates an EDT with the given guid.  The guid will run on the home node of the guid.
//The hasDepv flag can be used to not allocate slots for the EDT, but still use the depc counter.
//This is useful when an EDT has a large dependency count but doesn't require any results from them which would otherwise require significant memory.
//Paramc are the number of static parameters.
//Paramv are the static parameters that are copied into the EDT closure.
//Depc is the number of dependencies required for the EDT to run.  The EDT will run with depc slots.
//Returns the guid for the new EDT.
artsGuid_t artsEdtCreateWithGuidDep(artsEdt_t funcPtr, artsGuid_t guid, uint32_t paramc, uint64_t * paramv, uint32_t depc, bool hasDepv);

//Creates an EDT to run on node route that will run in the given epoch.  User must ensure the epoch is still live.
//The hasDepv flag can be used to not allocate slots for the EDT, but still use the depc counter.
//This is useful when an EDT has a large dependency count but doesn't require any results from them which would otherwise require significant memory.
//Paramc are the number of static parameters.
//Paramv are the static parameters that are copied into the EDT closure.
//Depc is the number of dependencies required for the EDT to run.  The EDT will run with depc slots.
//Returns a new guid for the new EDT.
artsGuid_t artsEdtCreateWithEpochDep(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsGuid_t epochGuid, bool hasDepv);

//Destroys an EDT and removes its guid from the routing table.  EDT are automatically destroyed after running.
void artsEdtDestroy(artsGuid_t guid);

//Signals an EDT that a dependency is met.  When all dependencies have been met an EDT is scheduled to run.
//The EDT's artsEdtDep_t at slot is filled with dataGuid to a DB and the memory the dataGuid/DB points to.
//The mode the DB is acquired in is based on the type of dataGuid.  The mode an EDT acquires a DB can be changed
//using artsGuidCast.
void artsSignalEdt(artsGuid_t edtGuid, uint32_t slot, artsGuid_t dataGuid);

//Signals an EDT that a dependency is met.  When all dependencies have been met an EDT is scheduled to run.
//Only signals a signal value not a guid.  The value is stored in the guid field of artsEdtDep_t.
void artsSignalEdtValue(artsGuid_t edtGuid, uint32_t slot, uint64_t data);

//Signals an EDT that a dependency is met.  When all dependencies have been met an EDT is scheduled to run.
//Instead of using a DB, the data of size is copied to the ptr field of artsEdtDep_t.  This memory is freed after the EDT runs.
void artsSignalEdtPtr(artsGuid_t edtGuid, uint32_t slot, void * ptr, unsigned int size);

//Creates an EDT to run where the dbGuid is located.  This is a wrapper around EDT create and signal.
artsGuid_t artsActiveMessageWithDb(artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsGuid_t dbGuid);

//Creates an EDT to run on node rank and signals the edt with DB pointed to by dbGuid.  This is a wrapper around EDT create and signal.
artsGuid_t artsActiveMessageWithDbAt(artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsGuid_t dbGuid, unsigned int rank);

//Creates an EDT and copies data into a buffer.  This is a wrapper around EDT create and buffer signal.
artsGuid_t artsActiveMessageWithBuffer(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, void * ptr, unsigned int size);

//This creates a buffer of size bytes, and returns a guid to access the buffer.  The buffer can only be accessed on the node
//the buffer was allocated on, but can be written to from anywhere.
//Uses indicate how many accesses to the buffer until the entry in the routing table is freed.
//The epochGuid is used to make sure setBuffer is included in a given round of termination detection.
artsGuid_t artsAllocateLocalBuffer(void ** buffer, unsigned int size, unsigned int uses, artsGuid_t epochGuid);

//Copies the data from buffer of size bytes to the buffer pointed to by bufferGuid.  This still counts as an access so the uses in the buffer allocation should be > 1.
void * artsSetBuffer(artsGuid_t bufferGuid, void * buffer, unsigned int size);

//Return buffer pointed to by bufferGuid and decrements uses.  If uses == 0 the buffer is removed from the guid table.
//The buffer is only available on a the node it was allocated.
void * artsGetBuffer(artsGuid_t bufferGuid);

/*Event************************************************************************/

//Creates a latch event on node route and returns the new guid to signal.  A latch event has a counter that can be incremented and decremented via its slots (artsLatchEventSlot_t).  When the counter reaches zero the
//event fires.  An event can be used to broadcast a dbGuid to EDTs and other events.  Events can also be used to execute a callback function.  This differs from an EDT
//in that the callback is executed immediately when the counter reaches 0.
artsGuid_t artsEventCreate(unsigned int route, unsigned int latchCount);

//Creates a latch event on node home to guid and returns the guid to signal.  A latch event has a counter that can be incremented and decremented via its slots (artsLatchEventSlot_t).  When the counter reaches zero the
//event fires.  An event can be used to broadcast a dbGuid to EDTs and other events.  Events can also be used to execute a callback function.  This differs from an EDT
//in that the callback is executed immediately when the counter reaches 0.
artsGuid_t artsEventCreateWithGuid(artsGuid_t guid, unsigned int latchCount);

//Returns if an event is already fired.
bool artsIsEventFired(artsGuid_t event);

//Destroys a local event.
void artsEventDestroy(artsGuid_t guid);

//Signals an event with a dataGuid.  There are two artsLatchEventSlot_t that can be signaled.  ARTS_EVENT_LATCH_INCR_SLOT increments the event's counter.  
//ARTS_EVENT_LATCH_DECR_SLOT decrements the events counter.  Once the counter reaches 0, the event is fired.  As long as the event exists (hasn't been destroyed), 
//any artsAddDependence or artsAddLocalEventCallback will be executed immediately.
void artsEventSatisfySlot(artsGuid_t eventGuid, artsGuid_t dataGuid, uint32_t slot);

//Add a connection between a source event and an EDT or other event.  When the event fires, it will signal the destination guid and slot.  If the event has already fired, the
//signal to the destination EDT or event will propagate immediately.
void artsAddDependence(artsGuid_t source, artsGuid_t destination, uint32_t slot);

//Adds a callback to be executed when the source event fires.  This differs from an EDT
//in that the callback is executed immediately when the counter reaches 0.
void artsAddLocalEventCallback(artsGuid_t source, eventCallback_t callback);

/*DB***************************************************************************/

//A DataBlock (DB) is the main memory abstraction used in ARTS to share data between tasks.  A DB can be one of several types (see ARTS_DB* in artsType_t) which will dictate how they
//are accessed.  DBs are a fixed size and are accessed via signaling an EDT with a DBs guid.  When the EDT runs, it will have access to the guid, access mode, and raw data via artsEdtDep_t.
//This creates a DB of size bytes and type mode, stores the pointer to the data in addr, and returns a new guid for the created DB.  The DB is created local to the calling node.
artsGuid_t artsDbCreate(void **addr, uint64_t size, artsType_t mode);

//Creates a DB with a fixed guid of size bytes if the guid is local.  The type and route is already fixed by the provided guid, and the pointer to the raw data is returned.
void * artsDbCreateWithGuid(artsGuid_t guid, uint64_t size);

//Creates a new DB similarly to artsDbCreateWithGuid, except the data is copyied to the new DB.  This is useful if there are outstanding out-of-order requests for the DB which will be
//satisfied once the DB is created.  Otherwise there may be a race for the user to write new data to the DB and the EDTs acquiring the DB.
void * artsDbCreateWithGuidAndData(artsGuid_t guid, void * data, uint64_t size);

//Creates a DB for a remote node route of size bytes and type mode.  The DB will be uninitialized.
artsGuid_t artsDbCreateRemote(unsigned int route, uint64_t size, artsType_t mode);

//Moves a datablock to a remote node.  This can be problematic since the guid hasn't changed and that means remote accesses to the DB, will look at the home of dbGuid.
//Access from rank will see the DB though.
void artsDbMove(artsGuid_t dbGuid, unsigned int rank);

//Destroys all copies of the DB in the system.
void artsDbDestroy(artsGuid_t guid);

//Only removes local copy if found.  If the DB remote, then sends to the DB home to destroy.
void artsDbDestroySafe(artsGuid_t guid, bool remote);

//Writes data in ptr of size bytes to the DB pointed to by dbGuid with an offest on the home node of dbGuid.  The EDT pointed to by edtGuid is signaled at slot.
//The put will be included in whatever epoch is running.
void artsPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);

//Writes data in ptr of size bytes to the DB pointed to by dbGuid with an offest on node rank.  The EDT pointed to by edtGuid is signaled at slot.
//The put will be included in whatever epoch is running.
void artsPutInDbAt(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);

//Writes data in ptr of size bytes to the DB pointed to by dbGuid with an offest on the home node of dbGuid.
//The put will be included in whatever epoch specified by epochGuid.
void artsPutInDbEpoch(void * ptr, artsGuid_t epochGuid, artsGuid_t dbGuid, unsigned int offset, unsigned int size);

//Gets a copy of data in a DB of size bytes and offset from the DB pointed to by dbGuid.  The data is signaled to the EDT pointed to by edtGuid using artsSignalEdtPtr.
//Data is copied from the DB found at the home of dbGuid.
void artsGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);

//Gets a copy of data in a DB of size bytes and offset from the DB pointed to by dbGuid.  The data is signaled to the EDT pointed to by edtGuid using artsSignalEdtPtr.
//Data is copied from the DB found on node rank.
void artsGetFromDbAt(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);

/*Epoch************************************************************************/

//Returns the current round of termination detection.
artsGuid_t artsGetCurrentEpochGuid();

//Makes an EDT part of a specific round of termination detection.  User must ensure the EDT hasn't run and the epoch is not over.
void artsAddEdtToEpoch(artsGuid_t edtGuid, artsGuid_t epochGuid);

//This creates a new round of termation detection and starts the epoch.  Any EDTS created (in the currently running EDT) will correspond to this epoch.
//When the epoch is finished finishEdtGuid will be signaled.  The dbGuid field of artsEdtDep_t pointed to by slot will be filled with the 
//number of EDTs, buffer alloc/set, get/puts, etc. executed in the epoch.
artsGuid_t artsInitializeAndStartEpoch(artsGuid_t finishEdtGuid, unsigned int slot);

//This creates a new round of termination detection with the source node of rank, but doesn't start it.
//When the epoch is finished finishEdtGuid will be signaled.  The dbGuid field of artsEdtDep_t pointed to by slot will be filled with the 
//number of EDTs, buffer alloc/set, get/puts, etc. executed in the epoch.
artsGuid_t artsInitializeEpoch(unsigned int rank, artsGuid_t finishEdtGuid, unsigned int slot);

//Starts an epoch created with artsInitializeEpoch.
void artsStartEpoch(artsGuid_t epochGuid);

//Blocks waiting for epoch to finish.  Only works from an EDT that created the epoch.  The current executing thread calls another round of scheduling until the epoch finishes.
bool artsWaitOnHandle(artsGuid_t epochGuid);

//Block current execution and runs another round of scheduling.
void artsYield();

//Creates a context ticket for artsSignalContext to signal when waiting using artsContextSwitch
artsTicket_t artsGetContextTicket();

//Context switch between threads.  The waitCount is how many signals are required
//Before we can wake this context up.  This must have tMT set in the config file.
bool artsContextSwitch(unsigned int waitCount);

//Used to wake up a context asleep from a context switch
bool artsSignalContext(artsTicket_t ticket);


/*ArrayDb************************************************************************/

//ArrayDb is an array that spans all the nodes of the execution.  This returns a guid for accessing the arrayDb which can be used anywhere in the system.
//The data is spread equally across all nodes.
artsGuid_t artsNewArrayDb(artsArrayDb_t **addr, unsigned int elementSize, unsigned int numElements);

//This creates a new arrayDb with a fixed guid.  The guid can be for any node, but must be of type ARTS_DB_PIN.
artsArrayDb_t * artsNewArrayDbWithGuid(artsGuid_t guid, unsigned int elementSize, unsigned int numElements);

//Gets an element from an array DB at a specific index.  The results is placed in slot for edtGuid using artsSignalEdtPtr.
void artsGetFromArrayDb(artsGuid_t edtGuid, unsigned int slot, artsArrayDb_t * array, unsigned int index);

//Puts data in a specific index of an arrayDb and signals edtGuid at slot on completion.  This put will also fall in the current epoch.
void artsPutInArrayDb(void * ptr, artsGuid_t edtGuid, unsigned int slot, artsArrayDb_t * array, unsigned int index);

//Launches an EDT for each element in the arrayDb.  Data is acquired using artsSignalEdtPtr, thus no element can be changed.
//EDTs are launched locally.
void artsForEachInArrayDb(artsArrayDb_t * array, artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv);

//Launches an EDT for each element in the arrayDb.  Data is acquired using artsSignalEdtPtr, thus no element can be changed.
//EDTs are launched across all nodes.
void artsForEachInArrayDbAtData(artsArrayDb_t * array, unsigned int stride, artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv);

//Gathers all of the chunks of arrayDb on a single node and runs an EDT funcPtr.  Each chunk is written using artsSignalEdtPtr to the EDT's artsEdtDep_t.
void artsGatherArrayDb(artsArrayDb_t * array, artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint64_t depc);

//Performs and atomic add at index, and signals edtGuid at slot upon completion.  Also corresponds to the executing epoch.
void artsAtomicAddInArrayDb(artsArrayDb_t * array, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot);

//Performs and atomic compare and swap at index, and signals edtGuid at slot upon completion.  Also corresponds to the executing epoch.
void artsAtomicCompareAndSwapInArrayDb(artsArrayDb_t * array, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot);

/*Util*************************************************************************/

//Returns the guid of the current EDT running.
artsGuid_t artsGetCurrentGuid();

//Returns the rank of the current node.
unsigned int artsGetCurrentNode();

//Returns the total number of nodes.
unsigned int artsGetTotalNodes();

//Returns the unique id of the current thread on the current node.
unsigned int artsGetCurrentWorker();

//Returns the total number of worker threads.  This does not include network send/receive threads.
unsigned int artsGetTotalWorkers();

//Returns the unique id of the current numa domain on the current node.  Requires HWLOC.
unsigned int artsGetCurrentCluster();

//Gets the total number of numa domains.  Requires HWLOC.
unsigned int artsGetTotalClusters();

//Arts timer in nanoseconds.
uint64_t artsGetTimeStamp();

//This is a way to send operations to a specific rank.  If the rank is the current node, the function is executed immediately.
//If the rank is remote, the arguments will be packaged, and sent to the appropriate node.  In this case
//the function will be executed by the receiver threads.  The arguments are freed if the free flag is set.
void artsRemoteSend(unsigned int rank, sendHandler_t funPtr, void * args, unsigned int size, bool free);

#ifdef __cplusplus
}
#endif
#endif
