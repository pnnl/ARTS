# Third-party dependencies build configuration

# Build hwloc if not found in CMAKE_INSTALL_PREFIX
function(build_hwloc)
    message(STATUS "Building hwloc from third-party/hwloc")

    # Check if hwloc submodule is properly initialized
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hwloc/autogen.sh")
        message(STATUS "hwloc submodule not initialized, running git submodule update")
        execute_process(
            COMMAND git submodule update --init --recursive third-party/hwloc
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
    set(HWLOC_BUILD_DIR "${CMAKE_BINARY_DIR}/third-party/hwloc")
    file(MAKE_DIRECTORY ${HWLOC_BUILD_DIR})

    message(STATUS "Configuring hwloc with prefix: ${CMAKE_INSTALL_PREFIX}")
    execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/hwloc/configure --prefix=${CMAKE_INSTALL_PREFIX} --disable-rsmi --enable-static --enable-shared
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

    # Install hwloc
    message(STATUS "Installing hwloc to ${CMAKE_INSTALL_PREFIX}")
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

# Check if hwloc is needed and build if necessary
if(USE_HWLOC AND NOT HWLOC_FOUND)
    # Check if hwloc is already installed in CMAKE_INSTALL_PREFIX
    # Check for different library extensions
    set(HWLOC_LIB_SO "${CMAKE_INSTALL_PREFIX}/lib/libhwloc.so")
    set(HWLOC_LIB_A "${CMAKE_INSTALL_PREFIX}/lib/libhwloc.a")
    set(HWLOC_INCLUDE_PATH "${CMAKE_INSTALL_PREFIX}/include/hwloc.h")

    if(NOT ((EXISTS ${HWLOC_LIB_SO} OR EXISTS ${HWLOC_LIB_A}) AND EXISTS ${HWLOC_INCLUDE_PATH}))
        message(STATUS "hwloc not found in CMAKE_INSTALL_PREFIX, building from source")
        build_hwloc()
    endif()

    # Prefer shared library over static
    if(EXISTS ${HWLOC_LIB_SO})
        set(HWLOC_LIBRARIES "${HWLOC_LIB_SO}")
    else()
        set(HWLOC_LIBRARIES "${HWLOC_LIB_A}")
    endif()
    set(HWLOC_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")
    set(HWLOC_FOUND TRUE)

    # Try to determine version
    if(EXISTS "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig/hwloc.pc")
        file(READ "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig/hwloc.pc" HWLOC_PC_CONTENT)
        string(REGEX MATCH "Version: ([0-9]+\\.[0-9]+\\.[0-9]+)" HWLOC_VERSION_MATCH "${HWLOC_PC_CONTENT}")
        if(HWLOC_VERSION_MATCH)
            set(HWLOC_VERSION ${CMAKE_MATCH_1})
        endif()
    endif()

    # Export variables to parent scope
    set(HWLOC_LIBRARIES ${HWLOC_LIBRARIES} PARENT_SCOPE)
    set(HWLOC_INCLUDE_DIRS ${HWLOC_INCLUDE_DIRS} PARENT_SCOPE)
    set(HWLOC_FOUND ${HWLOC_FOUND} PARENT_SCOPE)
    set(HWLOC_VERSION ${HWLOC_VERSION} PARENT_SCOPE)
endif()
