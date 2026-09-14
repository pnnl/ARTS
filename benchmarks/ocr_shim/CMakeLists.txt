# OCR-to-ARTS compatibility shim
# Implements OCR API functions using ARTS v2 primitives
# Output: ocr_arts_objects — apps combine these objects with the coherence
# variant they link (arts_static_<suffix>), so no library target exists here

set(OCR_ROOT "${CMAKE_SOURCE_DIR}/third_party/ocr/ocr")

add_library(ocr_arts_objects OBJECT arts_ocr.c)

# The shim file uses both OCR headers and ARTS internal headers.
# C_EXTENSIONS ON because the shim uses VLAs, __sync builtins, typeof, etc.
set_target_properties(ocr_arts_objects PROPERTIES
    C_STANDARD 11
    C_EXTENSIONS ON
)

target_compile_options(ocr_arts_objects PRIVATE
    ${BENCH_C_OPT}
    -Wall
    ${ARTS_WNO_UNUSED}
)

# OCR extension defines (must come before OCR headers)
target_compile_definitions(ocr_arts_objects PRIVATE
    ENABLE_EXTENSION_AFFINITY
    ENABLE_EXTENSION_PARAMS_EVT
    ENABLE_EXTENSION_MULTI_OUTPUT_SLOT
    ENABLE_EXTENSION_COLLECTIVE_EVT
    ENABLE_EXTENSION_LABELING
    ENABLE_EXTENSION_RTITF
    ENABLE_EXTENSION_CHANNEL_EVT
    ENABLE_EXTENSION_COUNTED_EVT
    ${BENCH_OCR_TYPE_DEFINE}
)

# No-hint DB home policy selector (see top-level ARTS_SHIM_NOHINT_DB_HOME).
target_compile_definitions(ocr_arts_objects PRIVATE
    ARTS_SHIM_NOHINT_DB_HOME_ROUNDROBIN=${_ARTS_SHIM_NOHINT_RR})

# No-hint EDT placement policy selector (see top-level ARTS_SHIM_NOHINT_EDT_PLACE).
target_compile_definitions(ocr_arts_objects PRIVATE
    ARTS_SHIM_NOHINT_EDT_PLACE_ROUNDROBIN=${_ARTS_SHIM_NOHINT_EDT_RR})

# Include: OCR public headers + extension headers
target_include_directories(ocr_arts_objects PRIVATE
    ${OCR_ROOT}/inc                # ocr.h, ocr-types.h, etc.
    ${BENCH_OCR_OPTIONS_DIR}       # generated ocr-options_arts.h
)

# Include: ARTS public + internal headers (shim needs route table, runtime_types)
target_include_directories(ocr_arts_objects PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/include/public
    ${CMAKE_SOURCE_DIR}/libs/include/internal
)

add_executable(ocr_shim_event_payload EXCLUDE_FROM_ALL
    ${CMAKE_SOURCE_DIR}/tests/shim/event_payload.c
    $<TARGET_OBJECTS:ocr_arts_objects>)
target_include_directories(ocr_shim_event_payload PRIVATE
    ${OCR_ROOT}/inc ${BENCH_OCR_OPTIONS_DIR})
target_compile_definitions(ocr_shim_event_payload PRIVATE
    ENABLE_EXTENSION_AFFINITY ENABLE_EXTENSION_PARAMS_EVT ${BENCH_OCR_TYPE_DEFINE})
set_property(TARGET ocr_shim_event_payload PROPERTY POSITION_INDEPENDENT_CODE OFF)
target_compile_options(ocr_shim_event_payload PRIVATE -fno-pie -fno-PIE)
target_link_options(ocr_shim_event_payload PRIVATE -no-pie -fno-pie -fno-PIE)
target_link_libraries(ocr_shim_event_payload PRIVATE arts_static)
if(ARTS_BUILD_TESTS)
    set_target_properties(ocr_shim_event_payload PROPERTIES EXCLUDE_FROM_ALL FALSE)
    register_single_node_test(ocr_shim_event_payload TIMEOUT 15)
    register_multinode_test(ocr_shim_event_payload TIMEOUT 30)
endif()
