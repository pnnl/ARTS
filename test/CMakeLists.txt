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

set(LIBS arts ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB} ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})

add_executable(testMain testMain.c)
target_link_libraries(testMain ${LIBS})

add_executable(testRouteTable testRouteTable.c)
target_link_libraries(testRouteTable ${LIBS})

add_executable(testRouteTable2 testRouteTable2.c)
target_link_libraries(testRouteTable2 ${LIBS})

add_executable(testCDAG testCDAG.c)
target_link_libraries(testCDAG ${LIBS})

add_executable(testCDAG2 testCDAG2.c)
target_link_libraries(testCDAG2 ${LIBS})

#add_executable(testOOList testOOList.c)
#target_link_libraries(testOOList ${LIBS})

add_executable(testPinDB testPinDB.c)
target_link_libraries(testPinDB ${LIBS})

add_executable(testGetFromDb testGetFromDb.c)
target_link_libraries(testGetFromDb ${LIBS})

add_executable(testPutInDb testPutInDb.c)
target_link_libraries(testPutInDb ${LIBS})

add_executable(testArtsSend testArtsSend.c)
target_link_libraries(testArtsSend ${LIBS})

add_executable(testActiveMessageDb testActiveMessageDb.c)
target_link_libraries(testActiveMessageDb ${LIBS})

add_executable(testActiveMessageBuffer testActiveMessageBuffer.c)
target_link_libraries(testActiveMessageBuffer ${LIBS})

add_executable(testCSR testCSR.c)
target_link_libraries(testCSR arts-graph ${LIBS})

#add_executable(testDistribution testDistribution.c)
#target_link_libraries(testDistribution ${LIBS})

add_executable(testArrayDb testArrayDb.c)
target_link_libraries(testArrayDb ${LIBS})

add_executable(testForEach testForEach.c)
target_link_libraries(testForEach ${LIBS})

add_executable(testTerminationDetection testTerminationDetection.c)
target_link_libraries(testTerminationDetection ${LIBS})

add_executable(testTerminationDetection2 testTerminationDetection2.c)
target_link_libraries(testTerminationDetection2 ${LIBS})

add_executable(testPutArrayDbEpoch testPutArrayDbEpoch.c)
target_link_libraries(testPutArrayDbEpoch ${LIBS})

add_executable(testPutArrayDbEpoch2 testPutArrayDbEpoch2.c)
target_link_libraries(testPutArrayDbEpoch2 ${LIBS})

add_executable(testAtomicAddInArrayDb testAtomicAddInArrayDb.c)
target_link_libraries(testAtomicAddInArrayDb ${LIBS})

add_executable(testAtomicCompareAndSwapInArrayDb testAtomicCompareAndSwapInArrayDb.c)
target_link_libraries(testAtomicCompareAndSwapInArrayDb ${LIBS})

add_executable(testDbMove testDbMove.c)
target_link_libraries(testDbMove ${LIBS})

add_executable(testDbOnce testDbOnce.c)
target_link_libraries(testDbOnce ${LIBS})

add_executable(testRecEpoch testRecEpoch.c)
target_link_libraries(testRecEpoch ${LIBS})

add_executable(testEpochPool testEpochPool.c)
target_link_libraries(testEpochPool ${LIBS})

add_executable(testBuffer testBuffer.c)
target_link_libraries(testBuffer ${LIBS})

add_executable(testEvent testEvent.c)
target_link_libraries(testEvent ${LIBS})

if(CUDA_FOUND)
    set_source_files_properties(testGpuStream.c PROPERTIES LANGUAGE CUDA)
    add_executable(testGpuStream testGpuStream.c)
    target_link_libraries(testGpuStream ${LIBS})
    if(CUBLAS_FOUND)
        set_source_files_properties(testGpuLib.c PROPERTIES LANGUAGE CUDA)
        add_executable(testGpuLib testGpuLib.cu)
        target_link_libraries(testGpuLib ${LIBS})
    endif()
endif()

add_executable(testOOBug testOOBug.c)
target_link_libraries(testOOBug ${LIBS})

add_executable(testRouteTableIter testRouteTableIter.c)
target_link_libraries(testRouteTableIter ${LIBS})

add_executable(testRouteTableIter2 testRouteTableIter2.c)
target_link_libraries(testRouteTableIter2 ${LIBS})