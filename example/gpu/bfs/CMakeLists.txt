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

# Function to create a BFS CUDA example that uses arts_graph
function(add_bfs_cuda_example example_name)
    add_executable(gpu${example_name} ${ARGN})
    
    # Set include directories
    target_include_directories(gpu${example_name} PRIVATE 
        ${ARTS_CORE_INCLUDE_DIR}
        ${ARTS_CORE_BUILD_INCLUDE_DIR}
        ${ARTS_GRAPH_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Link with arts_graph and runtime libraries
    target_link_libraries(gpu${example_name} PRIVATE 
        arts_graph
        Threads::Threads 
        ${RT_LIBRARY} 
        ${M_LIBRARY} 
        ${ANL_LIBRARY} 
        CUDA::cudart 
        CUDA::cuda_driver 
        CUDA::cublas
    )
    
    # Use the CUDA-enabled ARTS library
    if(TARGET arts_cuda_shared)
        target_link_libraries(gpu${example_name} PRIVATE arts_cuda_shared)
    elseif(TARGET arts_cuda_static)
        target_link_libraries(gpu${example_name} PRIVATE arts_cuda_static)
    else()
        message(FATAL_ERROR "No CUDA ARTS library target found for gpu${example_name}")
    endif()
endfunction()

# Create BFS CUDA examples
add_bfs_cuda_example(Bfs bfs.cu graphUtil.cu bins.c)
add_bfs_cuda_example(BfsHybrid bfsHybrid.cu graphUtil.cu bins.c)
add_bfs_cuda_example(BfsHybridBuffer bfsHybridBuffer.cu graphUtil.cu buffer.c bins.c)
