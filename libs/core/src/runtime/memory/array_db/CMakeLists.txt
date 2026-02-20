file(GLOB ARRAY_DB_SOURCES "*.c")
add_library(arts_runtime_memory_array_db OBJECT ${ARRAY_DB_SOURCES})
target_include_directories(arts_runtime_memory_array_db PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
