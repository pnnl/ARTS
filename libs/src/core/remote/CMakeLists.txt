# Remote submodule (remote DB/EDT operation handlers)
file(GLOB REMOTE_SOURCES "*.c")

add_library(arts_remote OBJECT ${REMOTE_SOURCES})
target_link_libraries(arts_remote PRIVATE arts_interface)

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_remote PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()
