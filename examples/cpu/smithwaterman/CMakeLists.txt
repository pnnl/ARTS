add_executable(smithwaterman_arts_native smithwaterman_arts.c)

target_include_directories(smithwaterman_arts_native PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR}
    ${ARTS_INTERNAL_INCLUDE_DIR}
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

if(TARGET arts_shared)
    target_link_libraries(smithwaterman_arts_native PRIVATE arts_shared)
elseif(TARGET arts_static)
    target_link_libraries(smithwaterman_arts_native PRIVATE arts_static)
endif()

target_link_libraries(smithwaterman_arts_native PRIVATE
    Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})

# Non-PIE: function pointers must match across nodes (multinode EDT dispatch)
target_compile_options(smithwaterman_arts_native PRIVATE -fno-pie -fno-PIE)
target_link_options(smithwaterman_arts_native PRIVATE -no-pie -fno-pie -fno-PIE)
