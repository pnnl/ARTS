add_executable(lulesh_arts lulesh_arts.c)

target_include_directories(lulesh_arts PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR}
    ${ARTS_INTERNAL_INCLUDE_DIR}
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

if(TARGET arts_shared)
    target_link_libraries(lulesh_arts PRIVATE arts_shared)
elseif(TARGET arts_static)
    target_link_libraries(lulesh_arts PRIVATE arts_static)
endif()

target_link_libraries(lulesh_arts PRIVATE
    Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})

# Non-PIE: function pointers must match across nodes (multinode EDT dispatch)
target_compile_options(lulesh_arts PRIVATE -fno-pie -fno-PIE)
target_link_options(lulesh_arts PRIVATE -no-pie -fno-pie -fno-PIE)

# Profiling variant: same source, with -DLULESH_PROFILE for per-iter timing
add_executable(lulesh_arts_prof lulesh_arts.c)
target_include_directories(lulesh_arts_prof PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR}
    ${ARTS_INTERNAL_INCLUDE_DIR}
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
target_compile_definitions(lulesh_arts_prof PRIVATE LULESH_PROFILE)
if(TARGET arts_shared)
    target_link_libraries(lulesh_arts_prof PRIVATE arts_shared)
elseif(TARGET arts_static)
    target_link_libraries(lulesh_arts_prof PRIVATE arts_static)
endif()
target_link_libraries(lulesh_arts_prof PRIVATE
    Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})
target_compile_options(lulesh_arts_prof PRIVATE -fno-pie -fno-PIE)
target_link_options(lulesh_arts_prof PRIVATE -no-pie -fno-pie -fno-PIE)
