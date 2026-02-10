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
    test_main
    test_route_table
    test_route_table2
    test_cdag
    test_cdag2
    test_oo_list
    test_pin_db
    test_get_from_db
    test_put_in_db
    test_arts_send
    test_active_message_db
    test_active_message_buffer
    test_array_db
    test_for_each
    test_termination_detection
    test_termination_detection2
    test_put_array_db_epoch
    test_put_array_db_epoch2
    test_atomic_add_in_array_db
    test_atomic_compare_and_swap_in_array_db
    test_db_move
    test_db_once
    test_rec_epoch
    test_epoch_pool
    test_epoch_for_shad
    test_buffer
    test_event
    test_shad_lock
    test_oo_bug
    test_route_table_iter
    test_route_table_iter2
    test_tmt_lite
    test_acquire_mode
    test_arts_id
)

# Programs that require arts_graph library
set(GRAPH_TEST_PROGRAMS
    test_csr
    test_distribution
)

# Function to create a test executable
function(add_arts_test test_name)
    add_executable(${test_name} ${test_name}.c)
    
    # Set include directories
    target_include_directories(${test_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

    # Match CARTS launcher expectations: non-PIE binaries avoid pointer-auth
    # failures when passing raw function pointers across nodes.
    set_property(TARGET ${test_name} PROPERTY POSITION_INDEPENDENT_CODE OFF)
    target_compile_options(${test_name} PRIVATE -fno-pie -fno-PIE)
    target_link_options(${test_name} PRIVATE -no-pie -fno-pie -fno-PIE)
    
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
    target_include_directories(${test_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR} ${ARTS_GRAPH_INCLUDE_DIR})

    set_property(TARGET ${test_name} PROPERTY POSITION_INDEPENDENT_CODE OFF)
    target_compile_options(${test_name} PRIVATE -fno-pie -fno-PIE)
    target_link_options(${test_name} PRIVATE -no-pie -fno-pie -fno-PIE)
    
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
        test_gpu_stream
        test_gpu_for_all
        test_lc_sync
        test_gpu_lc_red
        test_gpu_lib
        test_gpu_ring
    )
    
    # Function to create a CUDA test executable
    function(add_cuda_test test_name)
        add_executable(${test_name} ${test_name}.cu)
        
        # Set include directories
        target_include_directories(${test_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

        set_property(TARGET ${test_name} PROPERTY POSITION_INDEPENDENT_CODE OFF)
        target_compile_options(${test_name} PRIVATE -fno-pie -fno-PIE)
        target_link_options(${test_name} PRIVATE -no-pie -fno-pie -fno-PIE)
        
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
