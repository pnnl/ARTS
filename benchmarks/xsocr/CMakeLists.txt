# XSOCR (reference OCR x86 implementation) built as a CMake library
#
# All 147 source files are globbed; ocr-config.h #ifdef guards ensure
# only the x86-relevant modules compile to non-empty objects.

set(OCR_ROOT "${CMAKE_SOURCE_DIR}/third_party/ocr/ocr")

file(GLOB_RECURSE XSOCR_SRCS "${OCR_ROOT}/src/*.c")

# Build as OBJECT library first, then assemble into static
add_library(ocr_xsocr_objects OBJECT ${XSOCR_SRCS})

# XSOCR uses GNU extensions (VLAs, typeof, __thread, etc.)
set_target_properties(ocr_xsocr_objects PROPERTIES
    C_STANDARD 11
    C_EXTENSIONS ON
)

target_compile_options(ocr_xsocr_objects PRIVATE
    ${BENCH_C_OPT}
    -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-unused-function
    -Wno-sign-compare
    -Wno-maybe-uninitialized
)

# ocr-config.h lives in this directory
target_include_directories(ocr_xsocr_objects PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}     # ocr_config_x86.h (renamed to ocr-config.h via wrapper)
    ${OCR_ROOT}/src
    ${OCR_ROOT}/src/inc
)

# Public OCR headers for consumers
target_include_directories(ocr_xsocr_objects PUBLIC
    ${OCR_ROOT}/inc
)

# The XSOCR source expects #include "ocr-config.h" but our file is
# named ocr_config_x86.h.  Map it via a thin wrapper.
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/ocr-config.h")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ocr-config.h"
         "#include \"${CMAKE_CURRENT_SOURCE_DIR}/ocr_config_x86.h\"\n")
endif()
target_include_directories(ocr_xsocr_objects PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}     # generated ocr-config.h wrapper
)

target_compile_definitions(ocr_xsocr_objects PRIVATE
    "OCR_TYPE_H=x86.h"
    CACHE_LINE_SZB=64
    GUID_PROVIDER_LOCID_SIZE=10
    ALLOW_EAGER_DB
    ENABLE_LAZY_DB
    OCR_MAX_MULTI_SLOT=32
)

# Static library
add_library(ocr_xsocr_static STATIC $<TARGET_OBJECTS:ocr_xsocr_objects>)
target_include_directories(ocr_xsocr_static PUBLIC ${OCR_ROOT}/inc)
target_link_libraries(ocr_xsocr_static PUBLIC pthread)

# Check for librt (needed on older glibc)
include(CheckLibraryExists)
check_library_exists(rt clock_gettime "" HAVE_LIBRT)
if(HAVE_LIBRT)
    target_link_libraries(ocr_xsocr_static PUBLIC rt)
endif()
