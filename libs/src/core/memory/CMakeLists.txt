# Memory submodule (DataBlock lifecycle, frontier management)
file(GLOB MEMORY_SOURCES "*.c")

# CPU-only object library
add_library(arts_memory OBJECT ${MEMORY_SOURCES})
target_include_directories(arts_memory PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_memory PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# GPU-specific object library
if(BUILD_CUDA_LIBRARY)
    set(DB_SOURCE "db.c")
    set(DB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${DB_SOURCE}")
    set(DB_WRAPPER ${CMAKE_CURRENT_BINARY_DIR}/db_gpu.cu)
    file(WRITE ${DB_WRAPPER} "#include \"${DB_PATH}\"\n")

    set(MEMORY_GPU_SOURCES ${MEMORY_SOURCES})
    list(REMOVE_ITEM MEMORY_GPU_SOURCES ${DB_PATH})
    list(APPEND MEMORY_GPU_SOURCES ${DB_WRAPPER})

    add_library(arts_memory_gpu OBJECT ${MEMORY_GPU_SOURCES})
    target_include_directories(arts_memory_gpu PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    if(HWLOC_INCLUDE_DIRS)
        target_include_directories(arts_memory_gpu PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()
    target_compile_definitions(arts_memory_gpu PRIVATE ARTS_USE_GPU)
endif()
