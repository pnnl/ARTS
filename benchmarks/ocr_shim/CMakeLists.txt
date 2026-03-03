# OCR-to-ARTS compatibility shim
# Implements OCR API functions using ARTS v2 primitives
# Output: ocr_arts_static (static library containing the shim)

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
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-sign-compare
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

# Static library
add_library(ocr_arts_static STATIC $<TARGET_OBJECTS:ocr_arts_objects>)
target_include_directories(ocr_arts_static PUBLIC ${OCR_ROOT}/inc)
target_link_libraries(ocr_arts_static PUBLIC arts_static pthread)
