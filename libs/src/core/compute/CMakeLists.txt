# Compute submodule (EDT creation, scheduling, SHAD adapter)
file(GLOB COMPUTE_SOURCES "*.c")

# CPU-only object library
add_library(arts_compute OBJECT ${COMPUTE_SOURCES})
target_link_libraries(arts_compute PRIVATE arts_interface)

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_compute PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# GPU-specific object library: generate a small .cu wrapper so NVCC compiles
# the required translation unit(s) while keeping the original .c files
# available for the CPU object library.
if(BUILD_CUDA_LIBRARY)
  set(EDT_SOURCE "edt.c")
  set(EDT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${EDT_SOURCE}")
  set(EDT_WRAPPER ${CMAKE_CURRENT_BINARY_DIR}/edt_gpu.cu)
  file(WRITE ${EDT_WRAPPER} "#include \"${EDT_PATH}\"\n")

  set(COMPUTE_GPU_SOURCES ${COMPUTE_SOURCES})
  list(REMOVE_ITEM COMPUTE_GPU_SOURCES ${EDT_PATH})
  list(APPEND COMPUTE_GPU_SOURCES ${EDT_WRAPPER})

  add_library(arts_compute_gpu OBJECT ${COMPUTE_GPU_SOURCES})
  target_link_libraries(arts_compute_gpu PRIVATE arts_interface)
  if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_compute_gpu PRIVATE ${HWLOC_INCLUDE_DIRS})
  endif()
  target_compile_definitions(arts_compute_gpu PRIVATE ARTS_USE_GPU)
endif()
