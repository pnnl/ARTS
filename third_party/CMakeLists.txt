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

# Submodule sources are fetched on demand: a plain `git clone` (without
# --recurse-submodules) leaves every submodule directory empty, so each
# submodule is ensured here right before it is consumed — marker present →
# use as-is; absent → run `git submodule update --init` for that path (which
# honors per-submodule settings such as shallow). The fallback error names
# the exact command for source trees where auto-init is impossible (no git
# available, or not a git checkout, e.g. an exported source archive).
function(arts_init_submodule path marker)
    if(EXISTS "${CMAKE_SOURCE_DIR}/${path}/${marker}")
        return()
    endif()
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
        message(STATUS "Initializing submodule ${path}")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init ${path}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            RESULT_VARIABLE _rc
        )
        if(_rc EQUAL 0 AND EXISTS "${CMAKE_SOURCE_DIR}/${path}/${marker}")
            return()
        endif()
    endif()
    message(FATAL_ERROR
        "Submodule ${path} is not initialized and could not be fetched "
        "automatically. Run: git submodule update --init ${path}")
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
        # Consumers use CPU/NUMA topology only.  Every I/O discovery backend
    # (accelerators, display, PCI, udev device naming) exists to place I/O
    # devices and, when auto-detected, makes the static library's external
    # references depend on what the build host has installed — the GL
    # backend alone pulls in X11/NVCtrl.  Disabling them keeps the link
    # surface identical on every host.
    COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC ${HWLOC_SRC_DIR}/configure
            --prefix=${HWLOC_INSTALL_DIR}
            --disable-rsmi
            --disable-cuda
            --disable-nvml
            --disable-libxml2
            --disable-opencl
            --disable-levelzero
            --disable-gl
            --disable-libudev
            --disable-pci
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

# Some configure probes address an install prefix through a lib64/ layout;
# the alias keeps such probes resolving inside this install instead of
# falling through to whatever the host has.
if(NOT EXISTS "${HWLOC_INSTALL_DIR}/lib64")
    file(CREATE_LINK "lib" "${HWLOC_INSTALL_DIR}/lib64" SYMBOLIC)
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
  arts_init_submodule(third_party/mimalloc CMakeLists.txt)
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
  # The verbs provider's userspace stack (rdma-core: libibverbs + librdmacm +
  # the per-NIC provider libraries) is vendored so the provider can be
  # compiled in on every build host; at run time the provider simply reports
  # no fabrics on machines without RDMA devices.  rdma-core builds with plain
  # CMake from a git checkout (it bundles its own kernel ABI headers), so it
  # is consumed as a pinned submodule.
  #
  # The build is walled off from the host: ENABLE_RESOLVE_NEIGH=0 drops the
  # userspace neighbour-resolution path — and with it the libnl dependency —
  # leaving device discovery and connection setup on their sysfs / kernel
  # rdma_cm paths.  What that path serves is Ethernet UD address handles on
  # kernels too old to resolve the destination MAC kernel-side; connected
  # (RC) endpoints established through rdma_cm never enter it.
  # PKG_CONFIG_LIBDIR points at an empty directory so optional host packages
  # (systemd, udev) cannot alter what gets built.  INSTALL_RUNDIR must be a
  # short absolute path: the daemon socket path is compiled into librdmacm
  # and a deep build tree would overflow sockaddr_un's sun_path limit.
  # Static consumers must pull hardware-provider objects explicitly
  # (RDMA_STATIC_PROVIDERS in the consumer's CPPFLAGS) — without that a
  # statically linked stack silently discovers zero devices.
  set(RDMA_CORE_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/rdma-core_install")

  function(build_rdma_core)
    arts_init_submodule(third_party/rdma-core CMakeLists.txt)
    set(RDMA_CORE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rdma-core")
    set(RDMA_CORE_BUILD_DIR "${CMAKE_BINARY_DIR}/third_party/rdma-core")
    set(RDMA_CORE_EMPTY_PKGCONFIG "${CMAKE_BINARY_DIR}/third_party/pkgconfig_empty")
    file(MAKE_DIRECTORY ${RDMA_CORE_BUILD_DIR})
    file(MAKE_DIRECTORY ${RDMA_CORE_EMPTY_PKGCONFIG})

    if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
      set(CMAKE_BUILD_PARALLEL_LEVEL 4)
    endif()

    message(STATUS "Configuring rdma-core with prefix: ${RDMA_CORE_INSTALL_DIR}")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env
          PKG_CONFIG_LIBDIR=${RDMA_CORE_EMPTY_PKGCONFIG}
          ${CMAKE_COMMAND} -S ${RDMA_CORE_SRC_DIR} -B ${RDMA_CORE_BUILD_DIR}
          -GNinja
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_INSTALL_PREFIX=${RDMA_CORE_INSTALL_DIR}
          -DCMAKE_INSTALL_LIBDIR=lib
          -DCMAKE_INSTALL_RUNDIR=/var/run
          -DSYSUSERS_DIR=etc/sysusers.d
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
          -DENABLE_STATIC=1
          -DENABLE_RESOLVE_NEIGH=0
          -DNO_MAN_PAGES=1
          -DNO_PYVERBS=1
          -DENABLE_VALGRIND=OFF
      RESULT_VARIABLE CONFIGURE_RESULT
    )
    if(NOT CONFIGURE_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to configure rdma-core")
    endif()

    message(STATUS "Building rdma-core")
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build ${RDMA_CORE_BUILD_DIR}
              -j${CMAKE_BUILD_PARALLEL_LEVEL}
      RESULT_VARIABLE BUILD_RESULT
    )
    if(NOT BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to build rdma-core")
    endif()

    message(STATUS "Installing rdma-core to ${RDMA_CORE_INSTALL_DIR}")
    execute_process(
      COMMAND ${CMAKE_COMMAND} --install ${RDMA_CORE_BUILD_DIR}
      RESULT_VARIABLE INSTALL_RESULT
    )
    if(NOT INSTALL_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to install rdma-core")
    endif()

    # A prefix that offers both .a and .so would let -l resolution pick the
    # shared object; static consumers must only ever see the archives.
    file(GLOB _rdma_shared "${RDMA_CORE_INSTALL_DIR}/lib/*.so*")
    if(_rdma_shared)
      file(REMOVE ${_rdma_shared})
    endif()
  endfunction()

  if(NOT EXISTS "${RDMA_CORE_INSTALL_DIR}/lib/libibverbs.a")
    build_rdma_core()
  endif()

  # Alias layout for probes that address the prefix through lib64/, and a
  # second spelling for archives that must appear twice on one link line
  # (deduplication keys on the path text).
  if(NOT EXISTS "${RDMA_CORE_INSTALL_DIR}/lib64")
    file(CREATE_LINK "lib" "${RDMA_CORE_INSTALL_DIR}/lib64" SYMBOLIC)
  endif()

  # Everything the statically linked verbs provider drags in, as pkg-config
  # describes it (core libraries plus every hardware-provider archive, in
  # link order).  The archives are mutually referential — the static provider
  # table names the providers, the providers call back into the core — so
  # consumers link the set as one group.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
        PKG_CONFIG_LIBDIR=${RDMA_CORE_INSTALL_DIR}/lib/pkgconfig
        pkg-config --static --libs libibverbs librdmacm
    OUTPUT_VARIABLE RDMA_STATIC_LINK_LINE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rdma_pc_rc
  )
  if(NOT _rdma_pc_rc EQUAL 0)
    message(FATAL_ERROR
        "pkg-config could not describe the vendored rdma-core static link line")
  endif()
  separate_arguments(RDMA_STATIC_LINK_FLAGS UNIX_COMMAND "${RDMA_STATIC_LINK_LINE}")
  set(RDMA_VERBS_LINK_SET
      "-Wl,--start-group" ${RDMA_STATIC_LINK_FLAGS} "-Wl,--end-group")
  # The same set for links that pass through libtool, which deduplicates
  # bare -l flags (keeping the last occurrence) and hoists -Wl options in
  # front of the libraries — both of which break a cyclic archive set.
  # Absolute archive paths keep their relative order, and the one archive
  # that must appear twice on the line uses a second spelling (the lib64
  # alias) so text-keyed deduplication cannot collapse it.
  set(RDMA_VERBS_LIBTOOL_LIBS "")
  set(_rdma_seen_ibverbs FALSE)
  foreach(_flag IN LISTS RDMA_STATIC_LINK_FLAGS)
    if(_flag MATCHES "^-l(.+)$")
      set(_name "${CMAKE_MATCH_1}")
      if(_name STREQUAL "pthread" OR _name STREQUAL "m")
        continue()
      endif()
      if(_name STREQUAL "ibverbs" AND _rdma_seen_ibverbs)
        string(APPEND RDMA_VERBS_LIBTOOL_LIBS
               " ${RDMA_CORE_INSTALL_DIR}/lib64/libibverbs.a")
      else()
        string(APPEND RDMA_VERBS_LIBTOOL_LIBS
               " ${RDMA_CORE_INSTALL_DIR}/lib/lib${_name}.a")
        if(_name STREQUAL "ibverbs")
          set(_rdma_seen_ibverbs TRUE)
        endif()
      endif()
    endif()
  endforeach()
  string(STRIP "${RDMA_VERBS_LIBTOOL_LIBS}" RDMA_VERBS_LIBTOOL_LIBS)

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
    # --enable-verbs=<prefix> replaces auto-detection: the provider is always
    # built, always against the vendored stack, so what a build host happens
    # to have installed can no longer change the artifact.
    # RDMA_STATIC_PROVIDERS makes the provider translation units reference
    # every hardware provider so their objects survive static linking.  LIBS
    # carries the static archive set (twice: the set is cyclic and configure
    # probes plus bundled utilities link in a single pass without grouping).
    # HWLOC_DIR pins the verbs provider's optional hwloc probe to the
    # vendored hwloc so its outcome is a pure function of vendored inputs.
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC
        CPPFLAGS=-DRDMA_STATIC_PROVIDERS=all
        "LDFLAGS=-L${RDMA_CORE_INSTALL_DIR}/lib"
        "LIBS=${RDMA_VERBS_LIBTOOL_LIBS} -lpthread -lm"
        HWLOC_DIR=${HWLOC_INSTALL_DIR}
        ${OFI_SRC_DIR}/configure
        --prefix=${OFI_PREFIX} --enable-static --disable-shared
        # Provider allowlist.  With a STATIC libfabric.a every enabled
        # provider's external libraries become link-time dependencies of every
        # consuming binary, so providers auto-enabled by configure probing the
        # build host (cxi, xpmem, psm, ...) break the final link on any
        # cluster that ships their headers.  --enable-only inverts the
        # default: nothing builds unless named here, which makes the build
        # deterministic on every host.  verbs exposes RDM endpoints only as
        # the verbs;ofi_rxm stack, so rxm must stay enabled or every
        # InfiniBand host silently downgrades to tcp.
        --enable-only --enable-tcp --enable-rxm
        --enable-verbs=${RDMA_CORE_INSTALL_DIR}
        CC=${CMAKE_C_COMPILER}
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
    # reference to dlclose / shm_open".  The verbs provider's userspace stack
    # and (when its probe engaged) hwloc are back-resolved the same way.
    INTERFACE_LINK_LIBRARIES "rt;atomic;pthread;${CMAKE_DL_LIBS};${RDMA_VERBS_LINK_SET};${HWLOC_LIB_A}")
  add_dependencies(arts::libfabric libfabric_ep)
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
  # Everything the benchmark tree consumes (reference runtimes, benchmark
  # apps, and their support libraries), ensured here where the other
  # third-party sources are set up rather than at each consumer.
  arts_init_submodule(third_party/ocr ocr/inc/ocr.h)
  arts_init_submodule(third_party/ocr-apps apps)
  arts_init_submodule(third_party/ocr-vx build_one/CMakeLists.txt)
  arts_init_submodule(third_party/tbb CMakeLists.txt)
  arts_init_submodule(third_party/OpenBLAS Makefile)

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
