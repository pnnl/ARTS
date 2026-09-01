cmake_minimum_required(VERSION 3.18)

project(arts_hpx_apps VERSION 1.0 LANGUAGES CXX)

find_package(HPX 1.11.0 EXACT CONFIG REQUIRED)

add_executable(nqueens_core_test tests/nqueens_core_test.cpp)
target_compile_features(nqueens_core_test PRIVATE cxx_std_17)
target_include_directories(nqueens_core_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(nqueens_hpx nqueens.cpp)
target_compile_features(nqueens_hpx PRIVATE cxx_std_17)
target_link_libraries(nqueens_hpx PRIVATE HPX::hpx)
# The embedded hwloc must stay out of the dynamic symbol table: an MPI
# library in the same process loads the host's hwloc for itself, and an
# exported copy would preempt it through interposition.
target_link_options(nqueens_hpx PRIVATE "LINKER:--exclude-libs,libhwloc.a")

if(HPX_APPS_RUNTIME_OUTPUT_DIRECTORY)
    set_target_properties(nqueens_hpx PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${HPX_APPS_RUNTIME_OUTPUT_DIRECTORY}")
endif()
