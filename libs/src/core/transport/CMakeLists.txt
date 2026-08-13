# Transport submodule (socket I/O, protocol, message dispatch, launcher)
file(GLOB TRANSPORT_SOURCES "*.c")

add_library(arts_transport OBJECT ${TRANSPORT_SOURCES})
target_link_libraries(arts_transport PRIVATE arts_interface)

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_transport PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# net.c / bootstrap.c include the libfabric headers.  Linking the imported
# target to this OBJECT library propagates its INTERFACE_INCLUDE_DIRECTORIES
# (the fabric headers) at compile time; the actual libfabric.a is linked into
# the final arts_static/arts_shared targets, which already link
# arts::libfabric.
target_link_libraries(arts_transport PRIVATE arts::libfabric)

arts_apply_protocol(arts_transport ${ARTS_COHERENCE_ARM} ${ARTS_WRITE_POLICY} ${ARTS_RELEASE_POLICY})
