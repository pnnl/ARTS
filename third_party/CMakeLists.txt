# Third-party dependencies build configuration

# Build hwloc from source (third_party/hwloc submodule)
function(build_hwloc)
    message(STATUS "Building hwloc from third_party/hwloc")

    # Check if hwloc submodule is properly initialized
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hwloc/autogen.sh")
        message(STATUS "hwloc submodule not initialized, running git submodule update")
        execute_process(
            COMMAND git submodule update --init --recursive third_party/hwloc
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            RESULT_VARIABLE GIT_RESULT
        )
        if(NOT GIT_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to initialize hwloc submodule")
        endif()
    endif()

    # Check if autogen.sh exists and run it if configure doesn't exist
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hwloc/configure")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hwloc/autogen.sh")
            message(STATUS "Running hwloc autogen.sh")
            execute_process(
                COMMAND ./autogen.sh
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hwloc
                RESULT_VARIABLE AUTOGEN_RESULT
            )
            if(NOT AUTOGEN_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to run hwloc autogen.sh")
            endif()
        else()
            message(FATAL_ERROR "hwloc configure script not found and autogen.sh not available")
        endif()
    endif()

    # Configure hwloc
    set(HWLOC_BUILD_DIR "${CMAKE_BINARY_DIR}/third_party/hwloc")
    set(HWLOC_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/hwloc_install")
    file(MAKE_DIRECTORY ${HWLOC_BUILD_DIR})

    message(STATUS "Configuring hwloc with prefix: ${HWLOC_INSTALL_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC ${CMAKE_CURRENT_SOURCE_DIR}/hwloc/configure
            --prefix=${HWLOC_INSTALL_DIR}
            --disable-rsmi
            --disable-cuda
            --disable-nvml
            --disable-libxml2
            --enable-static
            --disable-shared
        WORKING_DIRECTORY ${HWLOC_BUILD_DIR}
        RESULT_VARIABLE CONFIGURE_RESULT
    )
    if(NOT CONFIGURE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to configure hwloc")
    endif()

    # Build hwloc
    message(STATUS "Building hwloc")
    # Set default parallel level if not defined
    if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
        set(CMAKE_BUILD_PARALLEL_LEVEL 4)
    endif()
    execute_process(
        COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL}
        WORKING_DIRECTORY ${HWLOC_BUILD_DIR}
        RESULT_VARIABLE BUILD_RESULT
    )
    if(NOT BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build hwloc")
    endif()

    # Install hwloc to build directory
    message(STATUS "Installing hwloc to ${HWLOC_INSTALL_DIR}")
    execute_process(
        COMMAND make install
        WORKING_DIRECTORY ${HWLOC_BUILD_DIR}
        RESULT_VARIABLE INSTALL_RESULT
    )
    if(NOT INSTALL_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install hwloc")
    endif()

    message(STATUS "hwloc built and installed successfully")
endfunction()

# Build hwloc from source if not already built in the build directory
set(HWLOC_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/hwloc_install")
set(HWLOC_LIB_A "${HWLOC_INSTALL_DIR}/lib/libhwloc.a")
set(HWLOC_INCLUDE_PATH "${HWLOC_INSTALL_DIR}/include/hwloc.h")

if(NOT (EXISTS ${HWLOC_LIB_A} AND EXISTS ${HWLOC_INCLUDE_PATH}))
    build_hwloc()
endif()

if(NOT EXISTS ${HWLOC_LIB_A})
    message(FATAL_ERROR "hwloc static library not found after build at ${HWLOC_LIB_A}")
endif()

# Static library for embedding into libarts
set(HWLOC_LIBRARIES "${HWLOC_LIB_A}")
set(HWLOC_INCLUDE_DIRS "${HWLOC_INSTALL_DIR}/include")
set(HWLOC_FOUND TRUE)

# Determine version from pkgconfig
if(EXISTS "${HWLOC_INSTALL_DIR}/lib/pkgconfig/hwloc.pc")
    file(READ "${HWLOC_INSTALL_DIR}/lib/pkgconfig/hwloc.pc" HWLOC_PC_CONTENT)
    string(REGEX MATCH "Version: ([0-9]+\\.[0-9]+\\.[0-9]+)" HWLOC_VERSION_MATCH "${HWLOC_PC_CONTENT}")
    if(HWLOC_VERSION_MATCH)
        set(HWLOC_VERSION ${CMAKE_MATCH_1})
    endif()
endif()

# Parse hwloc private link dependencies from pkg-config (e.g. -ludev)
set(HWLOC_PRIVATE_LIBS "")
set(HWLOC_PC_FILE "${HWLOC_INSTALL_DIR}/lib/pkgconfig/hwloc.pc")
if(EXISTS "${HWLOC_PC_FILE}")
    file(READ "${HWLOC_PC_FILE}" _hwloc_pc)
    string(REGEX MATCH "Libs\\.private:([^\n]*)" _hwloc_privlibs "${_hwloc_pc}")
    if(_hwloc_privlibs)
        set(_privlibs "${CMAKE_MATCH_1}")
        string(STRIP "${_privlibs}" _privlibs)
        # Extract -l flags, skip -lpthread (already linked via Threads::Threads)
        string(REGEX MATCHALL "-l[^ ]+" _lib_flags "${_privlibs}")
        foreach(_flag ${_lib_flags})
            string(REGEX REPLACE "^-l" "" _libname "${_flag}")
            if(NOT _libname STREQUAL "pthread" AND NOT _libname STREQUAL "m")
                find_library(_hwloc_dep_${_libname} NAMES ${_libname})
                if(_hwloc_dep_${_libname})
                    list(APPEND HWLOC_PRIVATE_LIBS ${_hwloc_dep_${_libname}})
                    message(STATUS "  hwloc transitive dep: ${_libname} -> ${_hwloc_dep_${_libname}}")
                else()
                    message(STATUS "  hwloc transitive dep: ${_libname} (not found, skipping)")
                endif()
            endif()
        endforeach()
    endif()
endif()

# Export variables to parent scope
set(HWLOC_LIBRARIES ${HWLOC_LIBRARIES} PARENT_SCOPE)
set(HWLOC_PRIVATE_LIBS ${HWLOC_PRIVATE_LIBS} PARENT_SCOPE)
set(HWLOC_INCLUDE_DIRS ${HWLOC_INCLUDE_DIRS} PARENT_SCOPE)
set(HWLOC_FOUND ${HWLOC_FOUND} PARENT_SCOPE)
set(HWLOC_VERSION ${HWLOC_VERSION} PARENT_SCOPE)
