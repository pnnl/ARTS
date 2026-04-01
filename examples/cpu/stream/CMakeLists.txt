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

# Create the stream CPU example executable
add_executable(stream stream.c stream_util.c)

# Include both public and internal headers (internal needed for arts/memory/db.h)
target_include_directories(stream PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR}
    ${ARTS_INTERNAL_INCLUDE_DIR}
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link with the ARTS runtime library
if(TARGET arts_shared)
    target_link_libraries(stream PRIVATE arts_shared)
elseif(TARGET arts_static)
    target_link_libraries(stream PRIVATE arts_static)
endif()

# Link with runtime libraries
target_link_libraries(stream PRIVATE
    Threads::Threads
    ${RT_LIBRARY}
    ${M_LIBRARY}
    ${ANL_LIBRARY}
)

# Optionally link with the CXL library when available
if(ARTS_USE_CXL AND TARGET arts_cxl_lib)
    target_link_libraries(stream PRIVATE arts_cxl_lib)
    target_compile_definitions(stream PRIVATE ARTS_USE_CXL=1)
endif()

# Non-PIE: function pointers must match across nodes (multinode EDT dispatch)
target_compile_options(stream PRIVATE -fno-pie -fno-PIE)
target_link_options(stream PRIVATE -no-pie -fno-pie -fno-PIE)