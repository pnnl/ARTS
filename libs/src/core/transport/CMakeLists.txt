# Transport submodule (socket I/O, protocol, message dispatch, launcher)
file(GLOB TRANSPORT_SOURCES "*.c")

add_library(arts_transport OBJECT ${TRANSPORT_SOURCES})
target_include_directories(arts_transport PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
)

# Add RDMA include directories if available
if(RDMACM_INCLUDE_DIRS)
    target_include_directories(arts_transport PRIVATE ${RDMACM_INCLUDE_DIRS})
endif()

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_transport PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()
