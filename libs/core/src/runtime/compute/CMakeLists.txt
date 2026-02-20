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

# Runtime compute submodule
file(GLOB COMPUTE_SOURCES "*.c")

# CPU-only object library
add_library(arts_runtime_compute OBJECT ${COMPUTE_SOURCES})
target_include_directories(arts_runtime_compute PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_runtime_compute PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# GPU-specific object library: generate a small .cu wrapper so NVCC compiles
# the required translation unit(s) while keeping the original .c files
# available for the CPU object library.
if(BUILD_CUDA_LIBRARY)
  set(EDT_SOURCE "edt_functions.c")
  set(EDT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${EDT_SOURCE}")

  set(EDT_WRAPPER ${CMAKE_CURRENT_BINARY_DIR}/edt_functions_gpu.cu)
  file(WRITE ${EDT_WRAPPER} "#include \"${EDT_PATH}\"\n")

  set(COMPUTE_GPU_SOURCES ${COMPUTE_SOURCES})
  list(REMOVE_ITEM COMPUTE_GPU_SOURCES ${EDT_PATH})
  list(APPEND COMPUTE_GPU_SOURCES ${EDT_WRAPPER})

  add_library(arts_runtime_compute_gpu OBJECT ${COMPUTE_GPU_SOURCES})
  target_include_directories(arts_runtime_compute_gpu PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
  target_compile_definitions(arts_runtime_compute_gpu PRIVATE ARTS_USE_GPU)

  # Add hwloc include directories if available
  if(HWLOC_INCLUDE_DIRS)
      target_include_directories(arts_runtime_compute_gpu PRIVATE ${HWLOC_INCLUDE_DIRS})
  endif()
endif()
