# Runtime sync submodule
file(GLOB SYNC_SOURCES "*.c")
add_library(arts_runtime_sync OBJECT ${SYNC_SOURCES})
target_include_directories(arts_runtime_sync PRIVATE ${ARTS_CORE_INCLUDE_DIR} ${ARTS_CORE_BUILD_INCLUDE_DIR})