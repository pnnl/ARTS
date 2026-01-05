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

# Read Counter.h and extract counter types from ARTS_COUNTER_LIST macro
set(COUNTER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Counter.h")
if(NOT EXISTS "${COUNTER_HEADER}")
    message(FATAL_ERROR "Counter.h not found at: ${COUNTER_HEADER}")
endif()

file(READ "${COUNTER_HEADER}" COUNTER_HEADER_CONTENT)

# Extract counter names from X(counterName) patterns in ARTS_COUNTER_LIST
# This matches all X(identifier) patterns
string(REGEX MATCHALL "X\\(([a-zA-Z_][a-zA-Z0-9_]*)\\)" MATCHES "${COUNTER_HEADER_CONTENT}")

set(COUNTER_TYPES)
foreach(MATCH ${MATCHES})
    # Extract just the identifier name from X(name)
    string(REGEX REPLACE "X\\(([a-zA-Z_][a-zA-Z0-9_]*)\\)" "\\1" COUNTER_NAME "${MATCH}")

    # Skip 'name' and 'counterName' which appear in comments/definitions
    if(COUNTER_NAME AND NOT COUNTER_NAME STREQUAL "name" AND NOT COUNTER_NAME STREQUAL "counterName")
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

# Counter mode values: 0=OFF, 1=ONCE, 2=PERIODIC
# Counter level values: 0=THREAD, 1=NODE, 2=CLUSTER
# Counter reduce method values: 0=SUM, 1=MAX, 2=MIN, 3=MASTER
# Format: COUNTER=counterMode[,counterLevel[,reduceMethod]]
# Default counterLevel is NODE (1), default reduceMethod is SUM (0)
foreach(LINE ${COUNTER_CONFIG_LINES})
    if(LINE MATCHES "^[ \\t]*#" OR LINE MATCHES "^[ \\t]*$")
        continue()
    endif()
    # Match: COUNTER=MODE or COUNTER=MODE,LEVEL or COUNTER=MODE,LEVEL,REDUCE
    if(LINE MATCHES "^([A-Za-z0-9_]+)[ \\t]*=[ \\t]*(OFF|ONCE|PERIODIC)(,[ \\t]*(THREAD|NODE|CLUSTER))?(,[ \\t]*(SUM|MAX|MIN|MASTER))?")
        set(KEY ${CMAKE_MATCH_1})
        set(COUNTER_MODE ${CMAKE_MATCH_2})
        set(COUNTER_LEVEL ${CMAKE_MATCH_4})
        set(REDUCE_METHOD ${CMAKE_MATCH_6})

        # Default counterLevel to NODE if not specified
        if(NOT COUNTER_LEVEL)
            set(COUNTER_LEVEL "NODE")
        endif()

        # Default reduceMethod to SUM if not specified
        if(NOT REDUCE_METHOD)
            set(REDUCE_METHOD "SUM")
        endif()

        # Set CMake variables for this counter
        if(COUNTER_MODE STREQUAL "OFF")
            set(ENABLE_${KEY} 0)
            set(COUNTER_MODE_${KEY} 0) # artsCounterModeOff
            set(COUNTER_LEVEL_${KEY} 1) # default to NODE
            set(REDUCE_METHOD_${KEY} 0) # default to SUM
            math(EXPR DISABLED_COUNT "${DISABLED_COUNT} + 1")
        else()
            set(ENABLE_${KEY} 1)
            math(EXPR ENABLED_COUNT "${ENABLED_COUNT} + 1")

            # Set counter mode
            if(COUNTER_MODE STREQUAL "ONCE")
                set(COUNTER_MODE_${KEY} 1) # artsCounterModeOnce
            elseif(COUNTER_MODE STREQUAL "PERIODIC")
                set(COUNTER_MODE_${KEY} 2) # artsCounterModePeriodic
            endif()

            # Set counter level
            if(COUNTER_LEVEL STREQUAL "THREAD")
                set(COUNTER_LEVEL_${KEY} 0) # artsCounterLevelThread
            elseif(COUNTER_LEVEL STREQUAL "NODE")
                set(COUNTER_LEVEL_${KEY} 1) # artsCounterLevelNode
            elseif(COUNTER_LEVEL STREQUAL "CLUSTER")
                set(COUNTER_LEVEL_${KEY} 2) # artsCounterLevelCluster
            endif()

            # Set reduce method
            if(REDUCE_METHOD STREQUAL "SUM")
                set(REDUCE_METHOD_${KEY} 0) # artsCounterReduceSum
            elseif(REDUCE_METHOD STREQUAL "MAX")
                set(REDUCE_METHOD_${KEY} 1) # artsCounterReduceMax
            elseif(REDUCE_METHOD STREQUAL "MIN")
                set(REDUCE_METHOD_${KEY} 2) # artsCounterReduceMin
            elseif(REDUCE_METHOD STREQUAL "MASTER")
                set(REDUCE_METHOD_${KEY} 3) # artsCounterReduceMaster
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

# Generate counter mode flags for counters
file(APPEND ${GENERATED_HEADER} "\n/* Counter Mode Flags (0=OFF, 1=ONCE, 2=PERIODIC) */\n")
foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED COUNTER_MODE_${COUNTER})
        set(COUNTER_MODE_${COUNTER} 0)
    endif()
    file(APPEND ${GENERATED_HEADER} "#define COUNTER_MODE_${COUNTER} ${COUNTER_MODE_${COUNTER}}\n")
endforeach()

# Generate counter level flags for counters
file(APPEND ${GENERATED_HEADER} "\n/* Counter Level Flags (0=THREAD, 1=NODE, 2=CLUSTER) */\n")
foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED COUNTER_LEVEL_${COUNTER})
        set(COUNTER_LEVEL_${COUNTER} 1) # Default to NODE
    endif()
    file(APPEND ${GENERATED_HEADER} "#define COUNTER_LEVEL_${COUNTER} ${COUNTER_LEVEL_${COUNTER}}\n")
endforeach()

# Generate reduce method flags for counters
file(APPEND ${GENERATED_HEADER} "\n/* Counter Reduce Method Flags (0=SUM, 1=MAX, 2=MIN, 3=MASTER) */\n")
foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED REDUCE_METHOD_${COUNTER})
        set(REDUCE_METHOD_${COUNTER} 0) # Default to SUM
    endif()
    file(APPEND ${GENERATED_HEADER} "#define REDUCE_METHOD_${COUNTER} ${REDUCE_METHOD_${COUNTER}}\n")
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
    #define ${TIMER_START}() artsCounterTimerStart(&artsThreadLocalCounters[${COUNTER_INDEX}])
    #define ${TIMER_STOP}() artsCounterTimerEnd(&artsThreadLocalCounters[${COUNTER_INDEX}])
    #define ${INCREMENT}(value) artsCounterIncrementBy(&artsThreadLocalCounters[${COUNTER_INDEX}], value)
    #define ${DECREMENT}(value) artsCounterDecrementBy(&artsThreadLocalCounters[${COUNTER_INDEX}], value)
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

# Generate counter mode array
file(APPEND ${GENERATED_HEADER} "
/* Counter mode array (0=OFF, 1=ONCE, 2=PERIODIC) */
static const unsigned int artsCounterModeArray[] = {
")

foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED COUNTER_MODE_${COUNTER})
        set(COUNTER_MODE_${COUNTER} 0)
    endif()
    file(APPEND ${GENERATED_HEADER} "    ${COUNTER_MODE_${COUNTER}},  /* ${COUNTER} */\n")
endforeach()

file(APPEND ${GENERATED_HEADER} "};\n\n")

# Generate counter level array
file(APPEND ${GENERATED_HEADER} "
/* Counter level array (0=THREAD, 1=NODE, 2=CLUSTER) */
static const unsigned int artsCounterLevelArray[] = {
")

foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED COUNTER_LEVEL_${COUNTER})
        set(COUNTER_LEVEL_${COUNTER} 1) # Default to NODE
    endif()
    file(APPEND ${GENERATED_HEADER} "    ${COUNTER_LEVEL_${COUNTER}},  /* ${COUNTER} */\n")
endforeach()

file(APPEND ${GENERATED_HEADER} "};\n\n")
# Generate counter reduce method array
file(APPEND ${GENERATED_HEADER} "
/* Counter reduce method array (0=SUM, 1=MAX, 2=MIN, 3=MASTER) */
static const unsigned int artsCounterReduceMethodArray[] = {
")

foreach(COUNTER ${COUNTER_TYPES})
    if(NOT DEFINED REDUCE_METHOD_${COUNTER})
        set(REDUCE_METHOD_${COUNTER} 0) # Default to SUM
    endif()
    file(APPEND ${GENERATED_HEADER} "    ${REDUCE_METHOD_${COUNTER}},  /* ${COUNTER} */\n")
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
