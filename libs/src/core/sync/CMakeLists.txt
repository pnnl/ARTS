# Sync submodule (events, termination detection)
file(GLOB SYNC_SOURCES "*.c")

add_library(arts_sync OBJECT ${SYNC_SOURCES})
target_include_directories(arts_sync PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
