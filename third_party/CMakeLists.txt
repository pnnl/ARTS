# Third-party dependencies build configuration

# Vendored autotools dependencies are consumed as official release (dist)
# tarballs, never as repository checkouts: a dist tarball ships the
# pre-generated `configure` and auxiliary build files, so building it needs
# only a shell, make, and a C compiler — no autoconf/automake/libtool chain
# on the build host. Each tarball is cached next to this file; a host without
# network access can pre-stage the file there by hand and the build proceeds
# identically. The pinned SHA256 is verified for downloaded and pre-staged
# files alike.
function(arts_fetch_dist_tarball out_var tarball_name url sha256)
    set(_tarball "${CMAKE_CURRENT_SOURCE_DIR}/${tarball_name}")
    if(NOT EXISTS "${_tarball}")
        message(STATUS "Downloading ${tarball_name} from ${url}")
        file(DOWNLOAD "${url}" "${_tarball}" STATUS _dl SHOW_PROGRESS)
        list(GET _dl 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            list(GET _dl 1 _dl_msg)
            file(REMOVE "${_tarball}")
            message(FATAL_ERROR
                "Failed to download ${url}: ${_dl_msg}\n"
                "On a host without network access, fetch the file elsewhere "
                "and place it at ${_tarball}, then re-run cmake.")
        endif()
    endif()
    file(SHA256 "${_tarball}" _have)
    if(NOT _have STREQUAL "${sha256}")
        message(FATAL_ERROR
            "${_tarball} SHA256 mismatch:\n"
            "  have ${_have}\n"
            "  want ${sha256}\n"
            "Delete the file and re-run cmake to re-download.")
    endif()
    set(${out_var} "${_tarball}" PARENT_SCOPE)
endfunction()

set(HWLOC_DIST_VERSION "2.13.0")
set(HWLOC_DIST_URL "https://download.open-mpi.org/release/hwloc/v2.13/hwloc-${HWLOC_DIST_VERSION}.tar.bz2")
set(HWLOC_DIST_SHA256 "52e936afb6ebd80f171f763fcf14f7b1f5ce98b125af5dd2f328b873b1fd0dab")

set(OFI_DIST_VERSION "2.6.0")
set(OFI_DIST_URL "https://github.com/ofiwg/libfabric/releases/download/v${OFI_DIST_VERSION}/libfabric-${OFI_DIST_VERSION}.tar.bz2")
set(OFI_DIST_SHA256 "1ee2ab6eb16462c0fe7a836f9df52b8f12309dfee3f3774aa93a86f521a8e8b1")

# Build hwloc from its vendored dist tarball
function(build_hwloc)
    arts_fetch_dist_tarball(HWLOC_TARBALL
        "hwloc-${HWLOC_DIST_VERSION}.tar.bz2" "${HWLOC_DIST_URL}" "${HWLOC_DIST_SHA256}")

    # Extraction preserves the dist tree's relative timestamps, so make never
    # invokes the maintainer rules that would regenerate configure/Makefile.in
    # (and would require autotools after all).
    set(HWLOC_SRC_DIR "${CMAKE_BINARY_DIR}/third_party/hwloc-${HWLOC_DIST_VERSION}")
    if(NOT EXISTS "${HWLOC_SRC_DIR}/configure")
        message(STATUS "Extracting hwloc ${HWLOC_DIST_VERSION}")
        file(ARCHIVE_EXTRACT INPUT "${HWLOC_TARBALL}"
             DESTINATION "${CMAKE_BINARY_DIR}/third_party")
    endif()

    # Configure hwloc (out-of-tree; the extracted source stays pristine)
    set(HWLOC_BUILD_DIR "${CMAKE_BINARY_DIR}/third_party/hwloc")
    set(HWLOC_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/hwloc_install")
    file(MAKE_DIRECTORY ${HWLOC_BUILD_DIR})

    message(STATUS "Configuring hwloc with prefix: ${HWLOC_INSTALL_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC ${HWLOC_SRC_DIR}/configure
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

# Build libfabric (OFI) from its vendored dist tarball, driven through CMake's
# ExternalProject so it participates in the Ninja build graph (parallel with
# other targets). The tarball is extracted at configure time into a tree
# private to this build directory; configure and make then run out of EP's
# own binary dir, leaving the extracted source pristine.
if(ARTS_TRANSPORT_OFI)
  message(STATUS "Building libfabric ${OFI_DIST_VERSION} from vendored dist tarball")

  arts_fetch_dist_tarball(OFI_TARBALL
      "libfabric-${OFI_DIST_VERSION}.tar.bz2" "${OFI_DIST_URL}" "${OFI_DIST_SHA256}")

  set(OFI_SRC_DIR "${CMAKE_BINARY_DIR}/libfabric-${OFI_DIST_VERSION}")
  if(NOT EXISTS "${OFI_SRC_DIR}/configure")
    message(STATUS "Extracting libfabric ${OFI_DIST_VERSION}")
    file(ARCHIVE_EXTRACT INPUT "${OFI_TARBALL}" DESTINATION "${CMAKE_BINARY_DIR}")
  endif()

  include(ExternalProject)
  set(OFI_PREFIX "${CMAKE_BINARY_DIR}/libfabric")

  # BUILD_COMMAND parallelism is bounded: an unbounded `make -j` spawns one
  # job per source file with no ceiling, which can outstrip available
  # cores/memory on a shared or resource-constrained build host.
  if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL 4)
  endif()
  ExternalProject_Add(libfabric_ep
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${OFI_SRC_DIR}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC ${OFI_SRC_DIR}/configure
        --prefix=${OFI_PREFIX} --enable-static --disable-shared
        --disable-usnic --disable-efa --disable-psm2 --disable-psm3
        --disable-opx CC=${CMAKE_C_COMPILER}
    BUILD_COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL}
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS ${OFI_PREFIX}/lib/libfabric.a
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

# Build OpenBLAS (BLAS + LAPACK/LAPACKE, BSD-3) from the third_party/OpenBLAS
# submodule for benchmark apps that call the standard CBLAS/LAPACKE C
# interfaces.  Built SINGLE-THREADED (USE_THREAD=0): the consuming apps
# parallelize at task granularity (one kernel call per tile task), so a
# threaded BLAS underneath would oversubscribe the machine and serialize on
# its own thread pool.  The submodule tree is shared by every build directory
# and OpenBLAS's make build writes artifacts into its source tree, so each
# build directory builds from a private copy.
if(ARTS_BUILD_BENCHMARKS)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS/Makefile")
    message(FATAL_ERROR
        "OpenBLAS submodule not initialized: "
        "${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS/Makefile not found. "
        "Run: git submodule update --init third_party/OpenBLAS")
  endif()

  include(ExternalProject)
  set(OPENBLAS_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS")
  set(OPENBLAS_SRC_DIR "${CMAKE_BINARY_DIR}/openblas-src")
  set(OPENBLAS_PREFIX "${CMAKE_BINARY_DIR}/openblas")
  if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL 4)
  endif()
  ExternalProject_Add(openblas_ep
    DOWNLOAD_COMMAND ${CMAKE_COMMAND} -E remove_directory ${OPENBLAS_SRC_DIR}
              COMMAND ${CMAKE_COMMAND} -E copy_directory ${OPENBLAS_SUBMODULE_DIR} ${OPENBLAS_SRC_DIR}
    SOURCE_DIR ${OPENBLAS_SRC_DIR}
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL}
        USE_THREAD=0 USE_LOCKING=1 NO_SHARED=1 NO_FORTRAN=0
        libs netlib
    INSTALL_COMMAND make PREFIX=${OPENBLAS_PREFIX} NO_SHARED=1 install
    LOG_BUILD 1 LOG_INSTALL 1
    BUILD_BYPRODUCTS ${OPENBLAS_PREFIX}/lib/libopenblas.a
  )

  file(MAKE_DIRECTORY ${OPENBLAS_PREFIX}/include)

  add_library(arts::openblas STATIC IMPORTED GLOBAL)
  set_target_properties(arts::openblas PROPERTIES
    IMPORTED_LOCATION ${OPENBLAS_PREFIX}/lib/libopenblas.a
    INTERFACE_INCLUDE_DIRECTORIES ${OPENBLAS_PREFIX}/include
    # The bundled LAPACK objects are Fortran; static single-pass linking
    # needs the Fortran runtime after the archive on the consumer link line.
    INTERFACE_LINK_LIBRARIES "gfortran;m;pthread")
  add_dependencies(arts::openblas openblas_ep)
endif()
