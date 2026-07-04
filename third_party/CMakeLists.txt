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

if(ARTS_MALLOC STREQUAL "mimalloc")
  set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_STATIC ON  CACHE BOOL "" FORCE)
  set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
  set(MI_OVERRIDE     OFF CACHE BOOL "" FORCE)  # we call mi_* explicitly, no malloc override
  add_subdirectory(mimalloc EXCLUDE_FROM_ALL)
endif()

# Build libfabric (OFI) from source (third_party/libfabric submodule) via
# autotools, driven through CMake's ExternalProject so it participates in the
# Ninja build graph (rerun on source changes, parallel with other targets).
if(ARTS_TRANSPORT_OFI)
  message(STATUS "Building libfabric from third_party/libfabric")

  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libfabric/autogen.sh")
    message(FATAL_ERROR
        "libfabric submodule not initialized: "
        "${CMAKE_CURRENT_SOURCE_DIR}/libfabric/autogen.sh not found. "
        "Run: git submodule update --init third_party/libfabric")
  endif()

  include(ExternalProject)
  set(OFI_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libfabric")
  set(OFI_SRC_DIR "${CMAKE_BINARY_DIR}/libfabric-src")
  set(OFI_PREFIX "${CMAKE_BINARY_DIR}/libfabric")

  # libfabric ships no `configure` at git-tag checkouts (autotools artifacts
  # are release-tarball-only) — autogen.sh runs `autoreconf` to generate it,
  # writing generated files (configure, aclocal.m4, Makefile.in,
  # autom4te.cache) into its working directory as a side effect. The
  # submodule checkout is a single source tree shared by every build
  # directory that configures this project, so autogen must never run
  # directly on it: two build directories bootstrapping concurrently would
  # race autoreconf on the same files and corrupt each other's output. The
  # download step below instead copies the submodule into a tree private to
  # this build directory; autogen, configure, and build all then operate on
  # that private copy, same as the hwloc build above builds out-of-tree.
  #
  # BUILD_COMMAND parallelism is bounded the same way the hwloc build above
  # bounds it: an unbounded `make -j` spawns one job per source file with no
  # ceiling, which can outstrip available cores/memory on a shared or
  # resource-constrained build host.
  if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL 4)
  endif()
  ExternalProject_Add(libfabric_ep
    DOWNLOAD_COMMAND ${CMAKE_COMMAND} -E remove_directory ${OFI_SRC_DIR}
              COMMAND ${CMAKE_COMMAND} -E copy_directory ${OFI_SUBMODULE_DIR} ${OFI_SRC_DIR}
    SOURCE_DIR ${OFI_SRC_DIR}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC ${OFI_SRC_DIR}/configure
        --prefix=${OFI_PREFIX} --enable-static --disable-shared
        --disable-usnic --disable-efa --disable-psm2 --disable-psm3
        --disable-opx CC=${CMAKE_C_COMPILER}
    BUILD_COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL}
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS ${OFI_PREFIX}/lib/libfabric.a
  )
  ExternalProject_Add_Step(libfabric_ep autogen
    COMMAND ${OFI_SRC_DIR}/autogen.sh
    WORKING_DIRECTORY ${OFI_SRC_DIR}
    DEPENDEES download
    DEPENDERS configure
  )

  # CMake validates INTERFACE_INCLUDE_DIRECTORIES at configure time, before
  # the ExternalProject has run and created the install prefix; pre-create
  # the (empty) include dir as a placeholder so target_link_libraries()
  # consumers configure cleanly on a fresh build tree.
  file(MAKE_DIRECTORY ${OFI_PREFIX}/include)

  add_library(arts::libfabric STATIC IMPORTED GLOBAL)
  set_target_properties(arts::libfabric PROPERTIES
    IMPORTED_LOCATION ${OFI_PREFIX}/lib/libfabric.a
    INTERFACE_INCLUDE_DIRECTORIES ${OFI_PREFIX}/include
    # Static libfabric's core + built-in providers reference dlopen/dlclose
    # (provider load) and shm_open/shm_unlink (shm provider); its pkgconfig
    # Libs.private is "-lrt -latomic -lpthread -ldl".  Declaring the full set as
    # this imported target's usage requirement places them AFTER the archive on
    # every consumer's link line, which single-pass static linking needs to
    # back-resolve these references — otherwise a target that pulls the
    # provider-init objects (fi_getinfo & friends) fails with "undefined
    # reference to dlclose / shm_open".
    INTERFACE_LINK_LIBRARIES "rt;atomic;pthread;${CMAKE_DL_LIBS}")
  add_dependencies(arts::libfabric libfabric_ep)
  # tcp provider needs no extra system libs; verbs pulls ibverbs only if present
endif()
