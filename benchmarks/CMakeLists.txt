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

add_subdirectory(xsocr)
add_subdirectory(ocr_shim)
add_subdirectory(apps)
add_subdirectory(baseline)
