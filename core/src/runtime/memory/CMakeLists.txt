# Runtime memory submodule
file(GLOB MEMORY_SOURCES "*.c")

# CPU-only object library
add_library(arts_runtime_memory OBJECT ${MEMORY_SOURCES})
target_include_directories(arts_runtime_memory PRIVATE ${ARTS_CORE_INCLUDE_DIR} ${ARTS_CORE_BUILD_INCLUDE_DIR})

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_runtime_memory PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# GPU-specific object library
if(BUILD_CUDA_LIBRARY)
	set(DBF_SOURCE "DbFunctions.c")
	set(DBF_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${DBF_SOURCE}")

	set(DBF_WRAPPER ${CMAKE_CURRENT_BINARY_DIR}/DbFunctions_gpu.cu)
	file(WRITE ${DBF_WRAPPER} "#include \"${DBF_PATH}\"\n")

	set(MEMORY_GPU_SOURCES ${MEMORY_SOURCES})
	list(REMOVE_ITEM MEMORY_GPU_SOURCES ${DBF_PATH})
	list(APPEND MEMORY_GPU_SOURCES ${DBF_WRAPPER})

	add_library(arts_runtime_memory_gpu OBJECT ${MEMORY_GPU_SOURCES})
	target_include_directories(arts_runtime_memory_gpu PRIVATE ${ARTS_CORE_INCLUDE_DIR} ${ARTS_CORE_BUILD_INCLUDE_DIR})
	target_compile_definitions(arts_runtime_memory_gpu PRIVATE USE_GPU)

	# Add hwloc include directories if available
	if(HWLOC_INCLUDE_DIRS)
		target_include_directories(arts_runtime_memory_gpu PRIVATE ${HWLOC_INCLUDE_DIRS})
	endif()
endif()