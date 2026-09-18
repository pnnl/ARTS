# Fake rapid CMake package.
#
# Provides a no-op rapid::rapid IMPORTED INTERFACE target so that the ARTS
# build system detects the package and defines ARTS_CXL_NATIVE, activating
# the real CXL code paths against the fake shared-memory library.
#
# Usage: cmake -B build -DARTS_CXL_RAPID_INCLUDE_DIR=/path/to/fake_arts_cxl_lib/inc ...
# (ARTS appends /.. to that path when calling find_package(rapid CONFIG))

if(NOT TARGET rapid::rapid)
    add_library(rapid::rapid INTERFACE IMPORTED)
endif()

set(rapid_FOUND TRUE)
