# Remote submodule (remote DB/EDT operation handlers)
file(GLOB REMOTE_SOURCES "*.c")

add_library(arts_remote OBJECT ${REMOTE_SOURCES})
target_include_directories(arts_remote PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_remote PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()
