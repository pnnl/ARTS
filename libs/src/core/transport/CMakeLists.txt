# Transport submodule (socket I/O, protocol, message dispatch, launcher)
file(GLOB TRANSPORT_SOURCES "*.c")

add_library(arts_transport OBJECT ${TRANSPORT_SOURCES})
target_link_libraries(arts_transport PRIVATE arts_interface)

# Add RDMA include directories if available
if(RDMACM_INCLUDE_DIRS)
    target_include_directories(arts_transport PRIVATE ${RDMACM_INCLUDE_DIRS})
endif()

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_transport PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()
