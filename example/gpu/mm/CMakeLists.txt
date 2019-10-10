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

set(LIBS arts ${M_LIB} ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB} ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})

if(CUDA_FOUND)

    add_library(mmUtil mmUtil.h mmUtil.c)
    
    set_source_files_properties(mmTileNaive.cu PROPERTIES LANGUAGE CUDA)
    add_executable(mmTileNaive mmTileNaive.cu)
    target_link_libraries(mmTileNaive mmUtil ${LIBS})

    set_source_files_properties(mmTile.cu PROPERTIES LANGUAGE CUDA)
    add_executable(mmTile mmTile.cu)
    target_link_libraries(mmTile mmUtil ${LIBS})

    

    if(CUBLAS_FOUND)
        set_source_files_properties(mmTileBlas.cu PROPERTIES LANGUAGE CUDA)
        add_executable(mmTileBlas mmTileBlas.cu)
        target_link_libraries(mmTileBlas mmUtil ${LIBS})

        set_source_files_properties(mmThrust.cu PROPERTIES LANGUAGE CUDA)
        add_executable(mmThrust mmThrust.cu)
        target_link_libraries(mmThrust mmUtil ${LIBS})

        set_source_files_properties(mmTileRed.cu PROPERTIES LANGUAGE CUDA)
        add_executable(mmTileRed mmTileRed.cu)
        target_link_libraries(mmTileRed mmUtil ${LIBS})

        set_source_files_properties(mmTileRedBig.cu PROPERTIES LANGUAGE CUDA)
        add_executable(mmTileRedBig mmTileRedBig.cu)
        target_link_libraries(mmTileRedBig mmUtil ${LIBS})
    endif()
endif()
