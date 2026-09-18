cmake_minimum_required(VERSION 3.22)
project(fake_arts_cxl_lib VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(arts_cxl_lib SHARED
    src/shm_backend.c
    src/alloc.c
    src/flush_track.c
    src/fake_cxl.c
)

target_include_directories(arts_cxl_lib
    PRIVATE src
    PUBLIC  inc
)

target_link_libraries(arts_cxl_lib PRIVATE rt pthread)

# ARTS CMakeLists expects the .so at ${ARTS_CXL_LIB_DIR}/build/src/libarts_cxl_lib.so
set_target_properties(arts_cxl_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src"
)

target_compile_options(arts_cxl_lib PRIVATE -Wall -Wextra -O2)

# Smoke test
enable_testing()
add_executable(smoke_test test/smoke_test.c)
target_include_directories(smoke_test PRIVATE inc)
target_link_libraries(smoke_test PRIVATE arts_cxl_lib)
set_target_properties(smoke_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME smoke COMMAND smoke_test)
