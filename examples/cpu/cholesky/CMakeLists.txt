add_executable(cholesky_arts_native cholesky_arts.c)
target_include_directories(cholesky_arts_native PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR}
    ${ARTS_INTERNAL_INCLUDE_DIR}
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
if(TARGET arts_shared)
    target_link_libraries(cholesky_arts_native PRIVATE arts_shared)
elseif(TARGET arts_static)
    target_link_libraries(cholesky_arts_native PRIVATE arts_static)
endif()
target_link_libraries(cholesky_arts_native PRIVATE
    Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})
target_compile_options(cholesky_arts_native PRIVATE -fno-pie -fno-PIE)
target_link_options(cholesky_arts_native PRIVATE -no-pie -fno-pie -fno-PIE)
