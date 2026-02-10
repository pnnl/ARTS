# Runtime network submodule
file(GLOB NETWORK_SOURCES "*.c")
add_library(arts_runtime_network OBJECT ${NETWORK_SOURCES})
target_include_directories(arts_runtime_network PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_runtime_network PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()