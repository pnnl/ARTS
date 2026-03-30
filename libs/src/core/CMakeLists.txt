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
add_subdirectory(compute)
add_subdirectory(memory)
add_subdirectory(remote)
add_subdirectory(sync)
add_subdirectory(transport)
add_subdirectory(gas)
add_subdirectory(system)
add_subdirectory(utils)
add_subdirectory(counter)

# Scheduler (scheduler.c at core/ top level, needs GPU wrapper for CUDA builds)
add_library(arts_scheduler OBJECT scheduler.c)
target_link_libraries(arts_scheduler PRIVATE arts_interface)
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_scheduler PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# GPU module is only built for CUDA library
if(BUILD_CUDA_LIBRARY)
    add_subdirectory(gpu)

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
