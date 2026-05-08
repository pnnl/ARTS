# Sync submodule (events, epoch / termination detection)
file(GLOB SYNC_SOURCES "*.c")

add_library(arts_sync OBJECT ${SYNC_SOURCES})
target_link_libraries(arts_sync PRIVATE arts_interface)
