# Benchmarks: OCR apps running on XSOCR (reference) and ARTS (via shim)
#
# Directory structure:
#   xsocr/     - Native XSOCR library (reference OCR x86 implementation)
#   ocr_shim/  - ARTS-backed OCR compatibility shim
#   apps/      - OCR benchmark apps built against both backends

# Benchmark optimization flags — controlled by CMAKE_BUILD_TYPE
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BENCH_C_OPT -O0 -g)
else()
    set(BENCH_C_OPT -O3)
endif()

# Suppress the "#pragma message: OCR_TYPE_H not defined" note from ocr.h.
# Defining OCR_TYPE_H causes ocr.h to #include "ocr-options_<value>.h",
# so we generate a no-op header in the build directory.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ocr-options_arts.h"
     "/* No-op OCR options for ARTS benchmark builds. */\n")
set(BENCH_OCR_TYPE_DEFINE "OCR_TYPE_H=arts.h")
set(BENCH_OCR_OPTIONS_DIR "${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(xsocr)
add_subdirectory(ocrvx)
add_subdirectory(ocr_shim)
add_subdirectory(apps)
add_subdirectory(baseline)
