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

# Define CPU example programs
set(CPU_EXAMPLES
    fib
    fibYield
    fibContextSwitch
    simpleReduction
)

# Programs that require arts_graph library
set(CPU_GRAPH_EXAMPLES
    bfs
    graphSearch
    triangleCount
    triangleCount2
    triangleCountPromise
)

# C++ programs that require arts_graph library  
set(CPU_CXX_GRAPH_EXAMPLES
    scanStatistics
)

# Function to create a CPU example executable
function(add_cpu_example example_name)
    add_executable(${example_name} ${example_name}.c)
    
    # Set include directories
    target_include_directories(${example_name} PRIVATE ${ARTS_CORE_INCLUDE_DIR} ${ARTS_CORE_BUILD_INCLUDE_DIR})
    
    # Use the appropriate ARTS library based on availability
    if(TARGET arts_shared)
        target_link_libraries(${example_name} PRIVATE arts_shared)
    elseif(TARGET arts_static)
        target_link_libraries(${example_name} PRIVATE arts_static)
    endif()
    
    # Link with runtime libraries
    target_link_libraries(${example_name} PRIVATE Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})

endfunction()

# Function to create a CPU example executable that uses arts_graph
function(add_cpu_graph_example example_name)
    add_executable(${example_name} ${example_name}.c)
    
    # Set include directories
    target_include_directories(${example_name} PRIVATE ${ARTS_CORE_INCLUDE_DIR} ${ARTS_CORE_BUILD_INCLUDE_DIR} ${ARTS_GRAPH_INCLUDE_DIR})
    
    # Link with arts_graph, main ARTS library, and runtime libraries
    target_link_libraries(${example_name} PRIVATE arts_graph Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})
    
    # Link with the appropriate ARTS library based on availability
    if(TARGET arts_shared)
        target_link_libraries(${example_name} PRIVATE arts_shared)
    elseif(TARGET arts_static)
        target_link_libraries(${example_name} PRIVATE arts_static)
    endif()
endfunction()

# Function to create a C++ CPU example executable that uses arts_graph
function(add_cpu_cxx_graph_example example_name)
    add_executable(${example_name} ${example_name}.cpp)
    
    # Set include directories
    target_include_directories(${example_name} PRIVATE ${ARTS_CORE_INCLUDE_DIR} ${ARTS_CORE_BUILD_INCLUDE_DIR} ${ARTS_GRAPH_INCLUDE_DIR})
    
    # Link with arts_graph, main ARTS library, and runtime libraries
    target_link_libraries(${example_name} PRIVATE arts_graph Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})
    
    # Link with the appropriate ARTS library based on availability
    if(TARGET arts_shared)
        target_link_libraries(${example_name} PRIVATE arts_shared)
    elseif(TARGET arts_static)
        target_link_libraries(${example_name} PRIVATE arts_static)
    endif()
endfunction()

# Create CPU example executables
foreach(example ${CPU_EXAMPLES})
    add_cpu_example(${example})
endforeach()

# Create CPU graph example executables
foreach(example ${CPU_GRAPH_EXAMPLES})
    add_cpu_graph_example(${example})
endforeach()

# Create C++ CPU graph example executables
foreach(example ${CPU_CXX_GRAPH_EXAMPLES})
    add_cpu_cxx_graph_example(${example})
endforeach()
