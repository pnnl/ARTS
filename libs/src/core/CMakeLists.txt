###############################################################################
## This material was prepared as an account of work sponsored by an agency
## of the United States Government.  Neither the United States Government
## nor the United States Department of Energy, nor Battelle, nor any of
## their employees, nor any jurisdiction or organization that has cooperated
## in the development of these materials, makes any warranty, express or
## implied, or assumes any legal liability or responsibility for the accuracy,
## completeness, or usefulness or any information, apparatus, product,
## software, or process disclosed, or represents that its use would not
## infringe privately owned rights.
##
## Reference herein to any specific commercial product, process, or service
## by trade name, trademark, manufacturer, or otherwise does not necessarily
## constitute or imply its endorsement, recommendation, or favoring by the
## United States Government or any agency thereof, or Battelle Memorial
## Institute. The views and opinions of authors expressed herein do not
## necessarily state or reflect those of the United States Government or
## any agency thereof.
##
##                      PACIFIC NORTHWEST NATIONAL LABORATORY
##                                  operated by
##                                    BATTELLE
##                                    for the
##                      UNITED STATES DEPARTMENT OF ENERGY
##                         under Contract DE-AC05-76RL01830
##
## Copyright 2019 Battelle Memorial Institute
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    https://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
###############################################################################

# Core runtime modules — each defines its own OBJECT library target
add_subdirectory(transport)
add_subdirectory(gas)
add_subdirectory(system)
add_subdirectory(utils)
add_subdirectory(counter)

# --- Flattened object libraries (former compute/ memory/ sync/ subdirs) -------
# Object sources now live directly under core/ with object-name prefixes. The
# CPU/GPU build decomposition is preserved exactly: edt.c and db.c carry the
# CUDA paths and are recompiled via NVCC for the GPU library, the coherence TUs
# compile as .c with ARTS_USE_GPU, and event/edt-context stay CPU-only
# (reused as-is in GPU builds).

# EDT compute (edt.c carries CUDA EDT launch paths -> GPU wrapper).
add_library(arts_compute OBJECT edt.c)
target_link_libraries(arts_compute PRIVATE arts_interface)
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_compute PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# DB coherence TU selection: MRNEW and MRSW each compile one timing TU (eager.c
# or lazy.c) plus their single-owner ownership machinery (ownership.c + the home
# OWNERSHIP_REQUEST FIFO home.c). MRSW additionally compiles waiter_queue.c (its
# per-cache RW-waiter Vyukov MPSC FIFO, pop-one). MRMW compiles mrmw.c alone (it
# carries no ownership lease, so the home lockreq queue / home.c are not linked).
# Each TU defines its arts_handler_db_* / arts_db_* bodies directly; the linker
# resolves the single compiled variant.
if(ARTS_COHERENCE_PROTOCOL STREQUAL "MRMW")
    set(DB_COHERENCE_MODEL_TU coherence/mrmw/mrmw.c)
elseif(ARTS_COHERENCE_PROTOCOL STREQUAL "LOCK")
    if(ARTS_PROTOCOL_TIMING STREQUAL "LAZY")
        set(DB_COHERENCE_MODEL_TU coherence/lock/home.c coherence/lock/lazy.c)
    else()
        set(DB_COHERENCE_MODEL_TU coherence/lock/home.c coherence/lock/eager.c)
    endif()
elseif(ARTS_COHERENCE_PROTOCOL STREQUAL "MRSW")
    if(ARTS_PROTOCOL_TIMING STREQUAL "EAGER")
        set(DB_COHERENCE_MODEL_TU coherence/mrsw/eager.c coherence/mrsw/ownership.c coherence/mrsw/home.c coherence/mrsw/waiter_queue.c)
    else() # MRSW + LAZY
        set(DB_COHERENCE_MODEL_TU coherence/mrsw/lazy.c coherence/mrsw/ownership.c coherence/mrsw/home.c coherence/mrsw/waiter_queue.c)
    endif()
elseif(ARTS_PROTOCOL_TIMING STREQUAL "EAGER")
    set(DB_COHERENCE_MODEL_TU coherence/mrnew/eager.c coherence/mrnew/ownership.c coherence/mrnew/home.c)
else() # MRNEW + LAZY (default)
    set(DB_COHERENCE_MODEL_TU coherence/mrnew/lazy.c coherence/mrnew/ownership.c coherence/mrnew/home.c)
endif()

# DB + coherence (db.c carries CUDA staging paths -> GPU wrapper).
set(ARTS_MEMORY_SOURCES
    db.c
    coherence/coherence.c coherence/buffer.c coherence/rank_u64_map.c
    coherence/handlers.c coherence/senders.c
    ${DB_COHERENCE_MODEL_TU})
add_library(arts_memory OBJECT ${ARTS_MEMORY_SOURCES})
target_link_libraries(arts_memory PRIVATE arts_interface)
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_memory PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# Event + EDT execution context (CPU-only; reused as-is in GPU builds).
add_library(arts_sync OBJECT event.c edt_context.c)
target_link_libraries(arts_sync PRIVATE arts_interface)
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_sync PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# Out-of-order deferred-op engine (replays operations that arrived before
# their target object was installed in the route table).  CPU-only; reused
# as-is in GPU builds (plain C, no CUDA paths).
add_library(arts_ooo OBJECT ooo.c)
target_link_libraries(arts_ooo PRIVATE arts_interface)
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_ooo PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# Scheduler (scheduler.c at core/ top level, needs GPU wrapper for CUDA builds)
add_library(arts_scheduler OBJECT scheduler.c)
target_link_libraries(arts_scheduler PRIVATE arts_interface)
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_scheduler PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# GPU module is only built for CUDA library
if(BUILD_CUDA_LIBRARY)
    add_subdirectory(gpu)

    # GPU variant of compute: recompile edt.c as .cu under NVCC with ARTS_USE_GPU.
    set(EDT_WRAPPER ${CMAKE_CURRENT_BINARY_DIR}/edt_gpu.cu)
    file(WRITE ${EDT_WRAPPER} "#include \"${CMAKE_CURRENT_SOURCE_DIR}/edt.c\"\n")
    add_library(arts_compute_gpu OBJECT ${EDT_WRAPPER})
    target_link_libraries(arts_compute_gpu PRIVATE arts_interface)
    if(HWLOC_INCLUDE_DIRS)
        target_include_directories(arts_compute_gpu PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()
    target_compile_definitions(arts_compute_gpu PRIVATE ARTS_USE_GPU)

    # GPU variant of memory: recompile db.c as .cu, coherence TUs as .c, all with
    # ARTS_USE_GPU.
    set(DB_WRAPPER ${CMAKE_CURRENT_BINARY_DIR}/db_gpu.cu)
    file(WRITE ${DB_WRAPPER} "#include \"${CMAKE_CURRENT_SOURCE_DIR}/db.c\"\n")
    set(ARTS_MEMORY_GPU_SOURCES ${ARTS_MEMORY_SOURCES})
    list(REMOVE_ITEM ARTS_MEMORY_GPU_SOURCES db.c)
    list(APPEND ARTS_MEMORY_GPU_SOURCES ${DB_WRAPPER})
    add_library(arts_memory_gpu OBJECT ${ARTS_MEMORY_GPU_SOURCES})
    target_link_libraries(arts_memory_gpu PRIVATE arts_interface)
    if(HWLOC_INCLUDE_DIRS)
        target_include_directories(arts_memory_gpu PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()
    target_compile_definitions(arts_memory_gpu PRIVATE ARTS_USE_GPU)

    # GPU variant of scheduler (wraps scheduler.c as .cu for NVCC)
    set(SCHED_PATH "${CMAKE_CURRENT_SOURCE_DIR}/scheduler.c")
    set(SCHED_WRAPPER ${CMAKE_CURRENT_BINARY_DIR}/scheduler_gpu.cu)
    file(WRITE ${SCHED_WRAPPER} "#include \"${SCHED_PATH}\"\n")
    add_library(arts_scheduler_gpu OBJECT ${SCHED_WRAPPER})
    target_link_libraries(arts_scheduler_gpu PRIVATE arts_interface)
    if(HWLOC_INCLUDE_DIRS)
        target_include_directories(arts_scheduler_gpu PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()
    target_compile_definitions(arts_scheduler_gpu PRIVATE ARTS_USE_GPU)
endif()
