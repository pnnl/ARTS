add_library(arts_array_db OBJECT array_db.c)
target_link_libraries(arts_array_db PRIVATE arts_interface)
