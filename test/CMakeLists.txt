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

# Define test programs to build
set(TEST_PROGRAMS
    testMain
    testRouteTable
    testRouteTable2
    testCDAG
    testCDAG2
    testOOList
    testPinDB
    testGetFromDb
    testPutInDb
    testArtsSend
    testActiveMessageDb
    testActiveMessageBuffer
    testArrayDb
    testForEach
    testTerminationDetection
    testTerminationDetection2
    testPutArrayDbEpoch
    testPutArrayDbEpoch2
    testAtomicAddInArrayDb
    testAtomicCompareAndSwapInArrayDb
    testDbMove
    testDbOnce
    testRecEpoch
    testEpochPool
    testEpochForShad
    testBuffer
    testEvent
    testShadLock
    testOOBug
    testRouteTableIter
    testRouteTableIter2
    testTMTLite
)

# Programs that require arts_graph library
set(GRAPH_TEST_PROGRAMS
    testCSR
    testDistribution
)

# Function to create a test executable
function(add_arts_test test_name)
    add_executable(${test_name} ${test_name}.c)
    
    # Set include directories
    target_include_directories(${test_name} PRIVATE ${ARTS_CORE_INCLUDE_DIR} ${ARTS_CORE_BUILD_INCLUDE_DIR})
    
    # Link with runtime libraries
    target_link_libraries(${test_name} PRIVATE Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})
    
    # Use the appropriate ARTS library based on availability
    if(TARGET arts_shared)
        target_link_libraries(${test_name} PRIVATE arts_shared)
    elseif(TARGET arts_static)
        target_link_libraries(${test_name} PRIVATE arts_static)
    else()
        message(FATAL_ERROR "No ARTS library target found")
    endif()
endfunction()

# Create standard test executables
foreach(test_program ${TEST_PROGRAMS})
    add_arts_test(${test_program})
endforeach()

# Function to create a test executable that uses arts_graph
function(add_arts_graph_test test_name)
    add_executable(${test_name} ${test_name}.c)
    
    # Set include directories
    target_include_directories(${test_name} PRIVATE ${ARTS_CORE_INCLUDE_DIR} ${ARTS_CORE_BUILD_INCLUDE_DIR} ${ARTS_GRAPH_INCLUDE_DIR})
    
    # Link with arts_graph and runtime libraries
    target_link_libraries(${test_name} PRIVATE arts_graph Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})
    
    # Use the appropriate ARTS library based on availability
    if(TARGET arts_shared)
        target_link_libraries(${test_name} PRIVATE arts_shared)
    elseif(TARGET arts_static)
        target_link_libraries(${test_name} PRIVATE arts_static)
    else()
        message(FATAL_ERROR "No ARTS library target found")
    endif()
endfunction()

# Create graph test executables
foreach(test_program ${GRAPH_TEST_PROGRAMS})
    add_arts_graph_test(${test_program})
endforeach()

# CUDA test programs (only built if CUDA is available)
if(BUILD_CUDA_LIBRARY)
    set(CUDA_TEST_PROGRAMS
        testGpuStream
        testGpuForAll
        testLCSync
        testGpuLCRed
        testGpuLib
        testGpuRing
    )
    
    # Function to create a CUDA test executable
    function(add_cuda_test test_name)
        add_executable(${test_name} ${test_name}.cu)
        
        # Set include directories
        target_include_directories(${test_name} PRIVATE ${ARTS_CORE_INCLUDE_DIR} ${ARTS_CORE_BUILD_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
        
        # Link with CUDA libraries
        target_link_libraries(${test_name} PRIVATE 
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
            target_link_libraries(${test_name} PRIVATE arts_cuda_shared)
        elseif(TARGET arts_cuda_static)
            target_link_libraries(${test_name} PRIVATE arts_cuda_static)
        else()
            message(FATAL_ERROR "No CUDA ARTS library target found")
        endif()
    endfunction()
    
    # Create CUDA test executables
    foreach(cuda_test ${CUDA_TEST_PROGRAMS})
        add_cuda_test(${cuda_test})
    endforeach()
    
    message(STATUS "CUDA test programs will be built")
else()
    message(STATUS "CUDA test programs will not be built (CUDA not available)")
endif()