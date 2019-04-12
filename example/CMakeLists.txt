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

include (CheckIncludeFiles)
include (CheckSymbolExists)

set(LIBS arts ${M_LIB} ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

CHECK_INCLUDE_FILES (alloca.h HAVE_ALLOCA_H)
check_symbol_exists (alloca alloca.h HAVE_ALLOCA)

if(HAVE_ALLOCA)
  add_definitions(-DHAVE_ALLOCA)
endif()

add_executable(fib fib.c)
target_link_libraries(fib ${LIBS})

add_executable(fibYield fibYield.c)
target_link_libraries(fibYield ${LIBS})


add_executable(fibContextSwitch fibContextSwitch.c)
target_link_libraries(fibContextSwitch ${LIBS})

add_executable(simpleReduction simpleReduction.c)
target_link_libraries(simpleReduction ${LIBS})

add_executable(bfs bfs.c)
target_link_libraries(bfs arts-graph ${LIBS})

add_executable(graphSearch graphSearch.c)
target_link_libraries(graphSearch arts-graph ${LIBS})

#SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})

add_executable(scanStatistics scanStatistics.cpp)
target_link_libraries(scanStatistics arts-graph ${LIBS})

add_executable(triangleCount triangleCount.c)
target_link_libraries(triangleCount arts-graph ${LIBS})

add_executable(triangleCount2 triangleCount2.c)
target_link_libraries(triangleCount2 arts-graph ${LIBS})
