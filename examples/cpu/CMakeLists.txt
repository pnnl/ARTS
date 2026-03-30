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

# Define CPU example programs (C)
if (ARTS_USE_CXL)
  set(CPU_EXAMPLES
      fib
      fibDB
      fib_yield
      simple_reduction
      tiled_matmul
      bfs
      # graph_search
      triangle_count
      triangle_count2
      triangle_count_promise
      stream_cpu
      cxl_stream
  )
else()
set(CPU_EXAMPLES
    fib
    fibDB
    fib_yield
    simple_reduction
    tiled_matmul
    bfs
    # graph_search
    triangle_count
    triangle_count2
    triangle_count_promise
    stream_cpu
)
endif()

# C++ CPU examples
set(CPU_CXX_EXAMPLES
    scan_statistics
)

# Function to create a CPU example executable
function(add_cpu_example example_name)
    add_executable(${example_name} ${example_name}.c)

    # Set include directories
    target_include_directories(${example_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

    # Use the appropriate ARTS library based on availability
    if(TARGET arts_shared)
        target_link_libraries(${example_name} PRIVATE arts_shared)
    elseif(TARGET arts_static)
        target_link_libraries(${example_name} PRIVATE arts_static)
    endif()

    # Link with runtime libraries
    target_link_libraries(${example_name} PRIVATE Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})

    # Non-PIE: function pointers must match across nodes (multinode EDT dispatch)
    target_compile_options(${example_name} PRIVATE -fno-pie -fno-PIE)
    target_link_options(${example_name} PRIVATE -no-pie -fno-pie -fno-PIE)
endfunction()

# Function to create a C++ CPU example executable
function(add_cpu_cxx_example example_name)
    add_executable(${example_name} ${example_name}.cpp)

    # Set include directories
    target_include_directories(${example_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

    if(ARTS_USE_CXL AND TARGET arts_cxl_lib)
        target_link_libraries(${example_name} PRIVATE arts_cxl_lib)
    endif()
    # Use the appropriate ARTS library based on availability
    if(TARGET arts_shared)
        target_link_libraries(${example_name} PRIVATE arts_shared)
    elseif(TARGET arts_static)
        target_link_libraries(${example_name} PRIVATE arts_static)
    endif()

    # Link with runtime libraries
    target_link_libraries(${example_name} PRIVATE Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})

    # Non-PIE: function pointers must match across nodes (multinode EDT dispatch)
    target_compile_options(${example_name} PRIVATE -fno-pie -fno-PIE)
    target_link_options(${example_name} PRIVATE -no-pie -fno-pie -fno-PIE)
endfunction()

# Create CPU example executables (excluding cxl_stream which needs special handling)
foreach(example ${CPU_EXAMPLES})
    if(NOT example STREQUAL "cxl_stream")
        add_cpu_example(${example})
    endif()
endforeach()

# cxl_stream: needs internal include dir for arts/memory/db.h (CXL flush APIs)
if (ARTS_USE_CXL)
  add_executable(cxl_stream cxl_stream.c)
  target_include_directories(cxl_stream PRIVATE
      ${ARTS_PUBLIC_INCLUDE_DIR}
      ${ARTS_INTERNAL_INCLUDE_DIR}
      ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
)
  if(TARGET arts_shared)
      target_link_libraries(cxl_stream PRIVATE arts_shared)
  elseif(TARGET arts_static)
      target_link_libraries(cxl_stream PRIVATE arts_static)
  endif()
  target_link_libraries(cxl_stream PRIVATE Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})
  if(ARTS_USE_CXL AND TARGET arts_cxl_lib)
      target_link_libraries(cxl_stream PRIVATE arts_cxl_lib)
      target_compile_definitions(cxl_stream PRIVATE ARTS_USE_CXL=1)
  endif()
  # Non-PIE: function pointers must match across nodes (multinode EDT dispatch)
  target_compile_options(cxl_stream PRIVATE -fno-pie -fno-PIE)
  target_link_options(cxl_stream PRIVATE -no-pie -fno-pie -fno-PIE)
endif()

# Create C++ CPU example executables
foreach(example ${CPU_CXX_EXAMPLES})
    add_cpu_cxx_example(${example})
endforeach()

# random_access: needs internal include dir for arts/memory/db.h (CXL flush APIs)
if (ARTS_USE_CXL)
  add_subdirectory(random_access)
endif()
 