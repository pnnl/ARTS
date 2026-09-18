add_executable(smithwaterman_arts_native smithwaterman_arts.c)

target_include_directories(smithwaterman_arts_native PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR}
    ${ARTS_INTERNAL_INCLUDE_DIR}
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

arts_apply_protocol(smithwaterman_arts_native ${ARTS_COHERENCE_ARM} ${ARTS_WRITE_POLICY} ${ARTS_RELEASE_POLICY})

if(TARGET arts_shared)
    target_link_libraries(smithwaterman_arts_native PRIVATE arts_shared)
elseif(TARGET arts_static)
    target_link_libraries(smithwaterman_arts_native PRIVATE arts_static)
endif()

if(ARTS_USE_CXL AND TARGET arts_cxl_lib)
    target_link_libraries(smithwaterman_arts_native PRIVATE arts_cxl_lib)
endif()

target_link_libraries(smithwaterman_arts_native PRIVATE
    Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})

# Non-PIE: function pointers must match across nodes (multinode EDT dispatch)
target_compile_options(smithwaterman_arts_native PRIVATE -fno-pie -fno-PIE)
target_link_options(smithwaterman_arts_native PRIVATE -no-pie -fno-pie -fno-PIE)
