# Benchmarks: OCR apps running on XSOCR (reference) and ARTS (via shim)
#
# Directory structure:
#   xsocr/     - Native XSOCR library (reference OCR x86 implementation)
#   ocr_shim/  - ARTS-backed OCR compatibility shim
#   apps/      - OCR benchmark apps built against both backends

add_subdirectory(xsocr)
add_subdirectory(ocr_shim)
add_subdirectory(apps)
add_subdirectory(baseline)
