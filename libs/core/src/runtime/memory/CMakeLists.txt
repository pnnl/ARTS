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

# Runtime memory submodule
file(GLOB MEMORY_SOURCES "*.c")

# CPU-only object library
add_library(arts_runtime_memory OBJECT ${MEMORY_SOURCES})
target_include_directories(arts_runtime_memory PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_runtime_memory PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# GPU-specific object library
if(BUILD_CUDA_LIBRARY)
	set(DBF_SOURCE "db_functions.c")
	set(DBF_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${DBF_SOURCE}")

	set(DBF_WRAPPER ${CMAKE_CURRENT_BINARY_DIR}/db_functions_gpu.cu)
	file(WRITE ${DBF_WRAPPER} "#include \"${DBF_PATH}\"\n")

	set(MEMORY_GPU_SOURCES ${MEMORY_SOURCES})
	list(REMOVE_ITEM MEMORY_GPU_SOURCES ${DBF_PATH})
	list(APPEND MEMORY_GPU_SOURCES ${DBF_WRAPPER})

	add_library(arts_runtime_memory_gpu OBJECT ${MEMORY_GPU_SOURCES})
	target_include_directories(arts_runtime_memory_gpu PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
	target_compile_definitions(arts_runtime_memory_gpu PRIVATE ARTS_USE_GPU)

	# Add hwloc include directories if available
	if(HWLOC_INCLUDE_DIRS)
		target_include_directories(arts_runtime_memory_gpu PRIVATE ${HWLOC_INCLUDE_DIRS})
	endif()
endif()

add_subdirectory(array_db)