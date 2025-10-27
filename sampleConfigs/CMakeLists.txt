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

# Define configuration files and scripts to install
set(CONFIG_FILES
    arts.cfg
    artsGpu.cfg
)

set(SCRIPTS
    gpuRun.sh
    gpuRunAllConfig.sh
    generateConfigFiles.py
)

# Copy scripts to build directory (maintaining executable permissions)
foreach(script ${SCRIPTS})
    configure_file(${script} ${script} COPYONLY)
endforeach()

# Function to copy config files to target directories
function(copy_config_to_target source_file target_dir target_name)
    if(IS_DIRECTORY ${CMAKE_BINARY_DIR}/${target_dir})
        configure_file(${source_file} ${CMAKE_BINARY_DIR}/${target_dir}/${target_name} COPYONLY)
    endif()
endfunction()

# Copy configuration files to example directories
copy_config_to_target(arts.cfg example/cpu arts.cfg)

# Copy GPU configuration files to example GPU directories if they exist
if(BUILD_CUDA_LIBRARY)
    set(GPU_EXAMPLE_DIRS
        example/gpu
        example/gpu/mm
        example/gpu/stream
        example/gpu/randomAccess
        example/gpu/bfs
        example/gpu/histogram
    )
    
    foreach(gpu_dir ${GPU_EXAMPLE_DIRS})
        copy_config_to_target(artsGpu.cfg ${gpu_dir} arts.cfg)
    endforeach()
endif()

# Copy configuration files to test directory
copy_config_to_target(arts.cfg test arts.cfg)
if(BUILD_CUDA_LIBRARY)
    copy_config_to_target(artsGpu.cfg test artsGpu.cfg)
endif()

# Install configuration files
install(FILES ${CONFIG_FILES}
    DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/arts
    COMPONENT config
)

# Install scripts
install(PROGRAMS ${SCRIPTS}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT scripts
)
