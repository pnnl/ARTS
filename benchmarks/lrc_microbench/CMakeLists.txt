foreach(app producer_consumer ring_chain wide_reader)
  add_executable(${app} ${CMAKE_CURRENT_SOURCE_DIR}/${app}.c)
  # Non-PIE required: cross-node function pointer passing.
  set_property(TARGET ${app} PROPERTY POSITION_INDEPENDENT_CODE OFF)
  target_compile_options(${app} PRIVATE -fno-pie -fno-PIE)
  target_link_options(${app} PRIVATE -no-pie -fno-pie -fno-PIE)
  if(TARGET arts_shared)
    target_link_libraries(${app} PRIVATE arts_shared)
  elseif(TARGET arts_static)
    target_link_libraries(${app} PRIVATE arts_static)
  else()
    message(FATAL_ERROR "No ARTS library target found for lrc_microbench")
  endif()
  set_target_properties(${app} PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/benchmarks/lrc_microbench")
endforeach()
