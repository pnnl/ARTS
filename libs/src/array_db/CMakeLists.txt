add_library(arts_array_db OBJECT array_db.c)

target_include_directories(arts_array_db PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR}
    ${ARTS_INTERNAL_INCLUDE_DIR}
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
)
