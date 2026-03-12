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

# Matrix multiplication CUDA examples
set(MM_CUDA_EXAMPLES
    mmTileNaive
    mmTile
    mmTileBlas
    mmThrust
    mmTileRed
    mmTileRedBig
)

# Function to create a matrix multiplication CUDA example
function(add_mm_cuda_example example_name)
    add_executable(${example_name} ${example_name}.cu mmUtil.c)
    
    # Set include directories
    target_include_directories(${example_name} PRIVATE
        ${ARTS_CORE_INCLUDE_DIR}
        ${ARTS_CORE_BUILD_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Link with CUDA libraries
    target_link_libraries(${example_name} PRIVATE 
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
        target_link_libraries(${example_name} PRIVATE arts_cuda_shared)
    elseif(TARGET arts_cuda_static)
        target_link_libraries(${example_name} PRIVATE arts_cuda_static)
    else()
        message(FATAL_ERROR "No CUDA ARTS library target found for ${example_name}")
    endif()
endfunction()

# Create standard CUDA matrix multiplication examples
foreach(example ${MM_CUDA_EXAMPLES})
    add_mm_cuda_example(${example})
endforeach()