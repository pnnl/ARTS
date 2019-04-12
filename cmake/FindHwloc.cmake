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

# Try to find the HWLOC runtime system
# Input variables:
#   HWLOC_ROOT   - The HWLOC install directory
# Output variables:
#   HWLOC_FOUND          - System has HWLOC
#   HWLOC_INC        - The HWLOC include directories
#   HWLOC_LIB            - The HWLOC libraries
#   HWLOC_VERSION_STRING - HWLOC version 

include(FindPackageHandleStandardArgs)

if (NOT DEFINED HWLOC_FOUND)

  # Set default search paths
  if (HWLOC_ROOT)
    set(HWLOC_INCLUDE_DIR ${HWLOC_ROOT}/include CACHE PATH "The include directory for HWLOC")
    set(HWLOC_LIBRARY_DIR ${HWLOC_ROOT}/lib CACHE PATH "The library directory for HWLOC")
    set(HWLOC_BINARY_DIR ${HWLOC_ROOT}/bin CACHE PATH "The bin directory for HWLOC")
  elseif(DEFINED ENV{HWLOC_INSTALL_DIR})
    set(HWLOC_INCLUDE_DIR $ENV{HWLOC_INSTALL_DIR}/include CACHE PATH "The include directory for HWLOC")
    set(HWLOC_LIBRARY_DIR $ENV{HWLOC_INSTALL_DIR}/lib CACHE PATH "The library directory for HWLOC")
    set(HWLOC_BINARY_DIR $ENV{HWLOC_INSTALL_DIR}/bin CACHE PATH "The bin directory for HWLOC")
  endif()

  find_path(HWLOC_INC 
    NAMES hwloc.h 
    HINTS ${HWLOC_INCLUDE_DIR})

  if(HWLOC_INC)
    message("Found ${HWLOC_INC}")
  else()
    message("Can't find HWLOC_INC Hint ${HWLOC_INCLUDE_DIR}")
  endif()

  # Search for the HWLOC library
  find_library(HWLOC_LIB 
    NAMES hwloc 
    HINTS ${HWLOC_LIBRARY_DIR})

  if(HWLOC_INC AND HWLOC_LIB)
    message("Found ${HWLOC_LIB}")
  else()
    message("Can't find HWLOC_LIB Hint ${HWLOC_LIBRARY_DIR}")
  endif()

  find_program(HWLOC_INFO_EXECUTABLE 
    NAMES hwloc-info
    HINTS ${HWLOC_BINARY_DIR})
  
  if(HWLOC_LIB AND HWLOC_LIB AND HWLOC_INFO_EXECUTABLE)
    execute_process(
      COMMAND ${HWLOC_INFO_EXECUTABLE} "--version" 
      OUTPUT_VARIABLE HWLOC_VERSION_LINE 
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    string(REGEX MATCH "([0-9]+.[0-9]+.[0-9]+)$" 
      HWLOC_VERSION_STRING "${HWLOC_VERSION_LINE}")
    unset(HWLOC_VERSION_LINE)

    if(${HWLOC_VERSION_STRING} VERSION_LESS "2.0.0")
        message("HWLOC version ${HWLOC_VERSION_STRING}")
    else()
        message("HWLOC version ${HWLOC_VERSION_STRING} -DHWLOC_V2 flag set")
        add_definitions(-DHWLOC_V2)
    endif()
  else()
    message("Can't find hwloc-info Hint ${HWLOC_BINARY_DIR}")
  endif()

  find_package_handle_standard_args(HWLOC
    FOUND_VAR HWLOC_FOUND
    REQUIRED_VARS HWLOC_INC HWLOC_LIB 
    HANDLE_COMPONENTS)

  mark_as_advanced(HWLOC_INC HWLOC_LIB HWLOC_INCLUDE_DIR HWLOC_LIBRARY_DIR HWLOC_BINARY_DIR)

endif()
