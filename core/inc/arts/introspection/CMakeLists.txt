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

# =============================================================================
# Define All Counters - Parse from Counter.h
# =============================================================================

# Read Counter.h and extract counter types from artsCounterType enum
set(COUNTER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Counter.h")
if(NOT EXISTS "${COUNTER_HEADER}")
    message(FATAL_ERROR "Counter.h not found at: ${COUNTER_HEADER}")
endif()

file(READ "${COUNTER_HEADER}" COUNTER_HEADER_CONTENT)

# Extract the enum artsCounterType block
string(REGEX MATCH "typedef enum artsCounterType[^}]*}" ENUM_BLOCK "${COUNTER_HEADER_CONTENT}")

if(NOT ENUM_BLOCK)
    message(FATAL_ERROR "Could not find artsCounterType enum in Counter.h")
endif()

# Extract counter names from the enum - match identifiers at start of line or after whitespace
# This regex matches: optional whitespace, identifier, optional "= digit", followed by comma or newline
string(REGEX MATCHALL "[ \t\n\r]+([a-zA-Z_][a-zA-Z0-9_]*)[ \t]*(=[ \t]*[0-9]+)?[ \t]*," MATCHES "${ENUM_BLOCK}")

set(COUNTER_TYPES)
foreach(MATCH ${MATCHES})
    # Extract just the identifier name (first capture group)
    string(REGEX MATCH "([a-zA-Z_][a-zA-Z0-9_]*)" COUNTER_NAME "${MATCH}")

    # Skip NUM_COUNTER_TYPES
    if(COUNTER_NAME AND NOT COUNTER_NAME STREQUAL "NUM_COUNTER_TYPES")
        list(APPEND COUNTER_TYPES "${COUNTER_NAME}")
    endif()
endforeach()

list(LENGTH COUNTER_TYPES COUNTER_COUNT)
message(STATUS "Parsed ${COUNTER_COUNT} counters from Counter.h: ${COUNTER_TYPES}")

# Set counter configuration
if(NOT EXISTS "${COUNTER_CONFIG_PATH}")
    message(FATAL_ERROR "Counter configuration file not found: ${COUNTER_CONFIG_PATH}")
endif()
message(STATUS "Using counter configuration file: ${COUNTER_CONFIG_PATH}")

# Parse configuration file
file(STRINGS "${COUNTER_CONFIG_PATH}" COUNTER_CONFIG_LINES)
set(ENABLED_COUNT 0)
set(DISABLED_COUNT 0)

# Counter capture mode values: 0=OFF, 1=ONCE, 2=PERIODIC
# Counter capture level values: 0=THREAD, 1=NODE, 2=CLUSTER
# Format: COUNTER=captureMode[,captureLevel]
# Default captureLevel is NODE (1)
foreach(LINE ${COUNTER_CONFIG_LINES})
    if(LINE MATCHES "^[ \\t]*#" OR LINE MATCHES "^[ \\t]*$")
        continue()
    endif()
    # Match: COUNTER=MODE or COUNTER=MODE,LEVEL
    if(LINE MATCHES "^([A-Za-z0-9_]+)[ \\t]*=[ \\t]*(OFF|ONCE|PERIODIC)(,[ \\t]*(THREAD|NODE|CLUSTER))?")
        set(KEY ${CMAKE_MATCH_1})
        set(CAPTURE_MODE ${CMAKE_MATCH_2})
        set(CAPTURE_LEVEL ${CMAKE_MATCH_4})

        # Default captureLevel to NODE if not specified
        if(NOT CAPTURE_LEVEL)
            set(CAPTURE_LEVEL "NODE")
        endif()

        # Set CMake variables for this counter
        if(CAPTURE_MODE STREQUAL "OFF")
            set(ENABLE_${KEY} 0)
            set(CAPTURE_MODE_${KEY} 0) # artsCaptureModeOff
            set(CAPTURE_LEVEL_${KEY} 1) # default to NODE
            math(EXPR DISABLED_COUNT "${DISABLED_COUNT} + 1")
        else()
            set(ENABLE_${KEY} 1)
            math(EXPR ENABLED_COUNT "${ENABLED_COUNT} + 1")

            # Set capture mode
            if(CAPTURE_MODE STREQUAL "ONCE")
                set(CAPTURE_MODE_${KEY} 1) # artsCaptureModeOnce
            elseif(CAPTURE_MODE STREQUAL "PERIODIC")
                set(CAPTURE_MODE_${KEY} 2) # artsCaptureModesPeriodic
            endif()

            # Set capture level
            if(CAPTURE_LEVEL STREQUAL "THREAD")
                set(CAPTURE_LEVEL_${KEY} 0) # artsCaptureLevelThread
            elseif(CAPTURE_LEVEL STREQUAL "NODE")
                set(CAPTURE_LEVEL_${KEY} 1) # artsCaptureLevelNode
            elseif(CAPTURE_LEVEL STREQUAL "CLUSTER")
                set(CAPTURE_LEVEL_${KEY} 2) # artsCaptureLevelCluster
            endif()
        endif()
    endif()
endforeach()

message(STATUS "Counter configuration: ${ENABLED_COUNT} enabled, ${DISABLED_COUNT} disabled")

# Generate counter header file
set(GENERATED_HEADER "${CMAKE_CURRENT_BINARY_DIR}/Preamble.h")

# Start writing the header
file(WRITE ${GENERATED_HEADER}
    "/* Counter.h - AUTOMATICALLY GENERATED - DO NOT EDIT */

#ifndef ARTS_INTROSPECTION_PREAMBLE_H
#define ARTS_INTROSPECTION_PREAMBLE_H

#ifdef __cplusplus
extern \"C\" {
#endif

/*
* This file is automatically generated by CMake
* Source: ${COUNTER_CONFIG_FILE}
*
* To modify: Edit counter.cfg and re-run cmake
*/

#include <stdbool.h>

#include \"arts/introspection/Counter.h\"

/* ============================================================================
* Enable/Disable Flags
* ============================================================================ */

    ")

# Generate enable flags for counters
file(APPEND ${GENERATED_HEADER} "/* Counter Flags */\n")
foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED ENABLE_${COUNTER})
        set(ENABLE_${COUNTER} 0)
    endif()
    file(APPEND ${GENERATED_HEADER} "#define ENABLE_${COUNTER} ${ENABLE_${COUNTER}}\n")
endforeach()

# Generate capture mode flags for counters
file(APPEND ${GENERATED_HEADER} "\n/* Counter Capture Mode Flags (0=OFF, 1=ONCE, 2=PERIODIC) */\n")
foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED CAPTURE_MODE_${COUNTER})
        set(CAPTURE_MODE_${COUNTER} 0)
    endif()
    file(APPEND ${GENERATED_HEADER} "#define CAPTURE_MODE_${COUNTER} ${CAPTURE_MODE_${COUNTER}}\n")
endforeach()

# Generate capture level flags for counters
file(APPEND ${GENERATED_HEADER} "\n/* Counter Capture Level Flags (0=THREAD, 1=NODE, 2=CLUSTER) */\n")
foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED CAPTURE_LEVEL_${COUNTER})
        set(CAPTURE_LEVEL_${COUNTER} 1) # Default to NODE
    endif()
    file(APPEND ${GENERATED_HEADER} "#define CAPTURE_LEVEL_${COUNTER} ${CAPTURE_LEVEL_${COUNTER}}\n")
endforeach()

# Generate counter macros automatically
file(APPEND ${GENERATED_HEADER} "
/* ============================================================================
 * Counter Macros - Automatically Generated
 * ============================================================================ */

")

set(COUNTER_INDEX 0)
foreach(COUNTER ${COUNTER_TYPES})
    # Convert counter name to macro name (e.g., edtCounter -> ARTS_COUNT_EDT_COUNTER)
    string(REGEX REPLACE "([a-z])([A-Z])" "\\1_\\2" MACRO_NAME ${COUNTER})
    string(TOUPPER ${MACRO_NAME} MACRO_NAME)
    set(TIMER_START "${MACRO_NAME}_START")
    set(TIMER_STOP "${MACRO_NAME}_STOP")
    set(INCREMENT "INCREMENT_${MACRO_NAME}_BY")
    set(DECREMENT "DECREMENT_${MACRO_NAME}_BY")

    file(APPEND ${GENERATED_HEADER}
        "#if ENABLE_${COUNTER}
    #define ${TIMER_START}() artsCounterTimerStart(&artsThreadInfo.artsCounters[${COUNTER_INDEX}])
    #define ${TIMER_STOP}() artsCounterTimerEnd(&artsThreadInfo.artsCounters[${COUNTER_INDEX}])
    #define ${INCREMENT}(value) artsCounterIncrementBy(&artsThreadInfo.artsCounters[${COUNTER_INDEX}], value)
    #define ${DECREMENT}(value) artsCounterDecrementBy(&artsThreadInfo.artsCounters[${COUNTER_INDEX}], value)
#else
    #define ${TIMER_START}() ((void)0)
    #define ${TIMER_STOP}() ((void)0)
    #define ${INCREMENT}(value) ((void)0)
    #define ${DECREMENT}(value) ((void)0)
#endif

")
    math(EXPR COUNTER_INDEX "${COUNTER_INDEX} + 1")
endforeach()

# Generate counter enabled array
file(APPEND ${GENERATED_HEADER} "
/* ============================================================================
 * Counter Enabled Array - Automatically Generated
 * ============================================================================ */

")

# Generate counter capture mode array
file(APPEND ${GENERATED_HEADER} "
/* Counter capture mode array (0=OFF, 1=ONCE, 2=PERIODIC) */
static const unsigned int artsCaptureModeArray[] = {
")

foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED CAPTURE_MODE_${COUNTER})
        set(CAPTURE_MODE_${COUNTER} 0)
    endif()
    file(APPEND ${GENERATED_HEADER} "    ${CAPTURE_MODE_${COUNTER}},  /* ${COUNTER} */\n")
endforeach()

file(APPEND ${GENERATED_HEADER} "};\n\n")

# Generate counter capture level array
file(APPEND ${GENERATED_HEADER} "
/* Counter capture level array (0=THREAD, 1=NODE, 2=CLUSTER) */
static const unsigned int artsCaptureLevelArray[] = {
")

foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED CAPTURE_LEVEL_${COUNTER})
        set(CAPTURE_LEVEL_${COUNTER} 1) # Default to NODE
    endif()
    file(APPEND ${GENERATED_HEADER} "    ${CAPTURE_LEVEL_${COUNTER}},  /* ${COUNTER} */\n")
endforeach()

file(APPEND ${GENERATED_HEADER} "};\n\n")

# Function declarations
file(APPEND ${GENERATED_HEADER} "
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* ARTS_INTROSPECTION_PREAMBLE_H */
")

message(STATUS "Generated: ${GENERATED_HEADER}")
message(STATUS "  Counters: ${ENABLED_COUNT} enabled, ${DISABLED_COUNT} disabled")
