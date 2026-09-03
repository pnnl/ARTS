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
# Bring one submodule to the commit this repository records.
#
# Two things go wrong without this.  A checkout that never initialised the
# submodule has no source at all.  One that initialised it and later took a
# parent revision recording a newer commit keeps its older checkout, and then
# compiles sources from one revision against a parent that expects another --
# a mismatch nothing announces, and which surfaces later as a missing symbol
# or a missing file rather than as a stale submodule.
#
# A submodule carrying local work is never moved.  Uncommitted changes are the
# caller's to lose, and a checkout ahead of the recorded commit is what
# working on a submodule looks like; both are reported and left alone.  Only a
# clean checkout that is behind, or on an unrelated commit, is moved.
function(arts_init_submodule path marker)
    find_package(Git QUIET)
    set(_git_usable FALSE)
    if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
        set(_git_usable TRUE)
    endif()
    set(_sub "${CMAKE_SOURCE_DIR}/${path}")

    if(NOT EXISTS "${_sub}/${marker}")
        if(_git_usable)
            message(STATUS "Initializing submodule ${path}")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init ${path}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                RESULT_VARIABLE _rc
            )
        endif()
        if(NOT EXISTS "${_sub}/${marker}")
            message(FATAL_ERROR
                "Submodule ${path} is not initialized and could not be fetched "
                "automatically. Run: git submodule update --init ${path}")
        endif()
        # A fresh checkout is at the recorded commit by construction.
        return()
    endif()

    if(NOT _git_usable)
        return()
    endif()

    # The gitlink this revision records, and what is actually checked out.
    execute_process(
        COMMAND ${GIT_EXECUTABLE} ls-tree HEAD -- ${path}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE _entry OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET RESULT_VARIABLE _rc_tree
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${_sub} rev-parse HEAD
        OUTPUT_VARIABLE _actual OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET RESULT_VARIABLE _rc_head
    )
    string(REGEX MATCH "commit ([0-9a-f]+)" _matched "${_entry}")
    set(_recorded "${CMAKE_MATCH_1}")
    # Anything unreadable here (a tarball export, a detached parent, an
    # unborn HEAD) leaves the checkout as it stands rather than guessing.
    if(NOT _rc_tree EQUAL 0 OR NOT _rc_head EQUAL 0
       OR _recorded STREQUAL "" OR _actual STREQUAL ""
       OR _recorded STREQUAL _actual)
        return()
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${_sub} status --porcelain --untracked-files=no
        OUTPUT_VARIABLE _local_changes OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    if(NOT _local_changes STREQUAL "")
        message(WARNING
            "Submodule ${path} is at ${_actual}, this repository records "
            "${_recorded}, and its working tree has uncommitted changes -- "
            "leaving it as it is.  The build uses what is checked out there.  "
            "To take the recorded commit, commit or copy those changes aside "
            "first, then run: git submodule update ${path}")
        return()
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${_sub} merge-base --is-ancestor ${_recorded} ${_actual}
        RESULT_VARIABLE _recorded_is_ancestor ERROR_QUIET
    )
    if(_recorded_is_ancestor EQUAL 0)
        message(WARNING
            "Submodule ${path} is at ${_actual}, ahead of the ${_recorded} "
            "this repository records -- leaving it as it is, since moving it "
            "back would drop the commits it carries.  The build uses sources "
            "no other checkout has: record them with "
            "'git add ${path}' in this repository.")
        return()
    endif()

    message(STATUS "Submodule ${path}: moving to the recorded ${_recorded}")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update ${path}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE _rc_update
    )
    if(NOT _rc_update EQUAL 0)
        message(WARNING
            "Submodule ${path} could not be moved to ${_recorded} and stays at "
            "${_actual}.  Run: git submodule update ${path}")
    endif()
endfunction()

# Apply a pinned upstream commit onto a pristine submodule working tree at
# configure time.  This carries a fix the pinned release lacks without
# forking the upstream repository for one commit: the patch file is the
# upstream commit verbatim (format-patch), so its header preserves the
# provenance.  Application is idempotent — a tree already carrying the
# change verifies in reverse and is left alone; any other state is an error
# rather than a silent partial apply.  A fresh application drops the
# external build's stamps so the next build recompiles what changed;
# a `git submodule update` that reverts the tree is likewise healed on the
# next configure.
function(arts_patch_submodule submodule_dir patch_file stamp_dir)
    execute_process(
        COMMAND git apply --reverse --check "${patch_file}"
        WORKING_DIRECTORY "${submodule_dir}"
        RESULT_VARIABLE _rc_applied
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_rc_applied EQUAL 0)
        return()
    endif()
    execute_process(
        COMMAND git apply --check "${patch_file}"
        WORKING_DIRECTORY "${submodule_dir}"
        RESULT_VARIABLE _rc_appliable
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _rc_appliable EQUAL 0)
        message(FATAL_ERROR
            "${submodule_dir} carries neither a clean base for ${patch_file} "
            "nor its applied result.  Restore the recorded submodule state "
            "(git -C ${submodule_dir} status) and re-run cmake.")
    endif()
    execute_process(
        COMMAND git apply "${patch_file}"
        WORKING_DIRECTORY "${submodule_dir}"
        RESULT_VARIABLE _rc_apply
    )
    if(NOT _rc_apply EQUAL 0)
        message(FATAL_ERROR "Failed to apply ${patch_file} in ${submodule_dir}")
    endif()
    get_filename_component(_patch_name "${patch_file}" NAME)
    message(STATUS "Applied ${_patch_name} to ${submodule_dir}")
    if(EXISTS "${stamp_dir}")
        file(REMOVE_RECURSE "${stamp_dir}")
    endif()
endfunction()

set(HWLOC_DIST_VERSION "2.13.0")
set(HWLOC_DIST_URL "https://download.open-mpi.org/release/hwloc/v2.13/hwloc-${HWLOC_DIST_VERSION}.tar.bz2")
set(HWLOC_DIST_SHA256 "52e936afb6ebd80f171f763fcf14f7b1f5ce98b125af5dd2f328b873b1fd0dab")

# HPX embeds its own hwloc, at the release its pinned version fetches by
# default upstream — the pairing stays exactly as upstream validated it.
set(HPX_HWLOC_DIST_VERSION "2.10.0")
set(HPX_HWLOC_DIST_URL "https://download.open-mpi.org/release/hwloc/v2.10/hwloc-${HPX_HWLOC_DIST_VERSION}.tar.bz2")
set(HPX_HWLOC_DIST_SHA256 "0305dd60c9de2fbe6519fe2a4e8fdc6d3db8de574a0ca7812b92e80c05ae1392")

set(NUMACTL_DIST_VERSION "2.0.19")
set(NUMACTL_DIST_URL "https://github.com/numactl/numactl/releases/download/v${NUMACTL_DIST_VERSION}/numactl-${NUMACTL_DIST_VERSION}.tar.gz")
set(NUMACTL_DIST_SHA256 "f2672a0381cb59196e9c246bf8bcc43d5568bc457700a697f1a1df762b9af884")

set(OFI_DIST_VERSION "2.6.0")
set(OFI_DIST_URL "https://github.com/ofiwg/libfabric/releases/download/v${OFI_DIST_VERSION}/libfabric-${OFI_DIST_VERSION}.tar.bz2")
set(OFI_DIST_SHA256 "1ee2ab6eb16462c0fe7a836f9df52b8f12309dfee3f3774aa93a86f521a8e8b1")

set(LIBNL_DIST_VERSION "3.11.0")
set(LIBNL_DIST_URL "https://github.com/thom311/libnl/releases/download/libnl3_11_0/libnl-${LIBNL_DIST_VERSION}.tar.gz")
set(LIBNL_DIST_SHA256 "2a56e1edefa3e68a7c00879496736fdbf62fc94ed3232c0baba127ecfa76874d")

# Build one hwloc from a pinned dist tarball into its own prefix.  More than
# one consumer embeds hwloc statically, each at the release its own project
# validates against, so version and prefix belong to the caller.
function(build_hwloc version url sha256 build_dir install_dir)
    arts_fetch_dist_tarball(HWLOC_TARBALL
        "hwloc-${version}.tar.bz2" "${url}" "${sha256}")

    # Extraction preserves the dist tree's relative timestamps, so make never
    # invokes the maintainer rules that would regenerate configure/Makefile.in
    # (and would require autotools after all).
    set(HWLOC_SRC_DIR "${CMAKE_BINARY_DIR}/third_party/hwloc-${version}")
    if(NOT EXISTS "${HWLOC_SRC_DIR}/configure")
        message(STATUS "Extracting hwloc ${version}")
        file(ARCHIVE_EXTRACT INPUT "${HWLOC_TARBALL}"
             DESTINATION "${CMAKE_BINARY_DIR}/third_party")
    endif()

    # Configure hwloc (out-of-tree; the extracted source stays pristine)
    set(HWLOC_BUILD_DIR "${build_dir}")
    set(HWLOC_INSTALL_DIR "${install_dir}")
    file(MAKE_DIRECTORY ${HWLOC_BUILD_DIR})

    message(STATUS "Configuring hwloc ${version} with prefix: ${HWLOC_INSTALL_DIR}")
    execute_process(
        # Consumers use CPU/NUMA topology only.  Every I/O discovery backend
    # (accelerators, display, PCI, udev device naming) exists to place I/O
    # devices and, when auto-detected, makes the static library's external
    # references depend on what the build host has installed — the GL
    # backend alone pulls in X11/NVCtrl.  Disabling them keeps the link
    # surface identical on every host.
    # Documentation is never consumed from an embedded build, and generating
    # it costs a full doxygen run on any host that happens to have doxygen.
    COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC ${HWLOC_SRC_DIR}/configure
            --prefix=${HWLOC_INSTALL_DIR}
            --disable-doxygen
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
    # A dist tarball carries prebuilt manuals and installs them whatever
    # --disable-doxygen says, one install line per API page.  Emptying their
    # target directories makes automake skip those rules; the library,
    # headers and hwloc.pc are unaffected.
    execute_process(
        COMMAND make install man3dir= pdfdir=
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
    build_hwloc(${HWLOC_DIST_VERSION} ${HWLOC_DIST_URL} ${HWLOC_DIST_SHA256}
        "${CMAKE_BINARY_DIR}/third_party/hwloc" "${HWLOC_INSTALL_DIR}")
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

# The static archive's own private dependencies, as pkg-config records them.
# Every discovery backend that would need a third-party library is disabled in
# the configure flags above, so the set can name toolchain libraries only —
# and those every consumer already links.  A name outside that set would have
# to be resolved against the build host, which is precisely the dependence on
# what a machine happens to carry that vendoring exists to remove, so it is a
# configure error rather than a silent link item.
set(HWLOC_TOOLCHAIN_PRIVATE_LIBS pthread m rt dl atomic c)
set(HWLOC_PC_FILE "${HWLOC_INSTALL_DIR}/lib/pkgconfig/hwloc.pc")
if(EXISTS "${HWLOC_PC_FILE}")
    file(READ "${HWLOC_PC_FILE}" _hwloc_pc)
    string(REGEX MATCH "Libs\\.private:([^\n]*)" _hwloc_privlibs "${_hwloc_pc}")
    if(_hwloc_privlibs)
        string(REGEX MATCHALL "-l[^ ]+" _lib_flags "${CMAKE_MATCH_1}")
        foreach(_flag ${_lib_flags})
            string(REGEX REPLACE "^-l" "" _libname "${_flag}")
            if(NOT _libname IN_LIST HWLOC_TOOLCHAIN_PRIVATE_LIBS)
                message(FATAL_ERROR
                    "The vendored hwloc records a private dependency on "
                    "'${_libname}', which is neither a toolchain library nor "
                    "vendored here.  Linking it would resolve against the build "
                    "host.  Disable the hwloc backend that pulls it in, or vendor "
                    "the library and give it an imported target.")
            endif()
        endforeach()
    endif()
endif()

# Build libnuma from its vendored dist tarball (numactl project).  Same
# rationale as hwloc: a header the compiler happens to find on one host is
# a build that fails on the next, so the only numa.h and libnuma.a any
# target may use are the ones built here.
#
# libnuma serves only the benchmark side -- xsocr's numa-alloc mem-platform,
# ocr-vx through the shared OCR headers, and the xeonNumaSize app -- never
# the ARTS runtime itself, which is also why the libarts bundle
# (ARTS_VENDORED_ARCHIVES) excludes it.  So the whole section, including the
# imported target those consumers link against, builds only alongside them.
#
# Within that bound it is built on demand rather than while configuring: the
# work is an ExternalProject held out of the default target, so the archive
# is produced when something that links arts::numa is built and at no other
# time.  A dependency added to an imported target is followed in the target's
# place, so each consumer picks the build up through the link edge it already
# declares and nothing has to name the external project.  Downloading is part
# of the same on-demand step, and a tarball already staged next to this file
# under the pinned hash is used as it is -- a host without network access
# stages it by hand exactly as before.
if(ARTS_BUILD_BENCHMARKS)
    include(ExternalProject)

    set(NUMACTL_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/numactl_install")
    set(NUMA_LIB_A "${NUMACTL_INSTALL_DIR}/lib/libnuma.a")
    set(NUMA_INCLUDE_DIRS "${NUMACTL_INSTALL_DIR}/include")

    if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
        set(CMAKE_BUILD_PARALLEL_LEVEL 4)
    endif()

    ExternalProject_Add(numactl_ep
        URL "${NUMACTL_DIST_URL}"
        URL_HASH "SHA256=${NUMACTL_DIST_SHA256}"
        DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
        DOWNLOAD_NO_PROGRESS 1
        # Extraction keeps the dist tree's own relative timestamps, so make
        # never invokes the maintainer rules that would regenerate configure
        # (and would require the autotools chain after all).
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX "${CMAKE_BINARY_DIR}/third_party/numactl_ep"
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC
            <SOURCE_DIR>/configure
            --prefix=${NUMACTL_INSTALL_DIR}
            --enable-static
            --disable-shared
        BUILD_COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL}
        INSTALL_COMMAND make install
        BUILD_BYPRODUCTS ${NUMA_LIB_A}
        EXCLUDE_FROM_ALL TRUE
        LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
    )

    # CMake validates INTERFACE_INCLUDE_DIRECTORIES while configuring, before
    # the external project has created its prefix.
    file(MAKE_DIRECTORY ${NUMA_INCLUDE_DIRS})

    add_library(arts_numa STATIC IMPORTED GLOBAL)
    set_target_properties(arts_numa PROPERTIES
        IMPORTED_LOCATION "${NUMA_LIB_A}"
        INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIRS}")
    add_dependencies(arts_numa numactl_ep)
    add_library(arts::numa ALIAS arts_numa)
endif()

# HPX is configured out-of-tree so its project-wide CMake settings cannot
# alter the ARTS build. Its installed package is consumed by a second,
# standalone benchmark project under benchmarks/hpx.
set(ARTS_HPX_AVAILABLE OFF PARENT_SCOPE)
if(ARTS_BUILD_BENCHMARKS AND ARTS_BUILD_HPX)
    find_package(MPI COMPONENTS CXX QUIET)
    if(MPI_CXX_FOUND)
        arts_init_submodule(third_party/hpx CMakeLists.txt)
        # Backport of upstream 6df14adf09: without it, colocated localities
        # on one host shift each other's thread bindings by their peers'
        # reported core footprints even when an explicit process mask
        # already places every locality.
        arts_patch_submodule(
            "${CMAKE_CURRENT_SOURCE_DIR}/hpx"
            "${CMAKE_CURRENT_SOURCE_DIR}/patches/hpx/0001-Don-t-apply-PU-offset-for-localities-that-explicitly.patch"
            "${CMAKE_CURRENT_BINARY_DIR}/hpx_ep-prefix/src/hpx_ep-stamp")
        include(ExternalProject)

        set(ARTS_HPX_INSTALL_DIR
            "${CMAKE_BINARY_DIR}/third_party/hpx_install")
        set(ARTS_HPX_BUILD_DIR
            "${CMAKE_BINARY_DIR}/third_party/hpx-build")

        set(HPX_HWLOC_INSTALL_DIR
            "${CMAKE_BINARY_DIR}/third_party/hpx_hwloc_install")
        if(NOT EXISTS "${HPX_HWLOC_INSTALL_DIR}/lib/libhwloc.a")
            build_hwloc(${HPX_HWLOC_DIST_VERSION} ${HPX_HWLOC_DIST_URL}
                ${HPX_HWLOC_DIST_SHA256}
                "${CMAKE_BINARY_DIR}/third_party/hpx_hwloc"
                "${HPX_HWLOC_INSTALL_DIR}")
        endif()

        ExternalProject_Add(hpx_ep
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hpx"
            BINARY_DIR "${ARTS_HPX_BUILD_DIR}"
            DOWNLOAD_COMMAND ""
            UPDATE_COMMAND ""
            CMAKE_GENERATOR "${CMAKE_GENERATOR}"
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                # The MPI wrapper is the compiler: a static-library build
                # prefers .a when resolving the wrapper's link line, and a
                # distribution's static MPI archive underlinks its private
                # dependencies.  With the wrapper compiling, MPI detection
                # needs no library resolution at all and MPI itself stays
                # the host's dynamic library — the intended boundary.
                -DCMAKE_CXX_COMPILER=${MPI_CXX_COMPILER}
                -DCMAKE_INSTALL_PREFIX=${ARTS_HPX_INSTALL_DIR}
                -DCMAKE_INSTALL_LIBDIR=lib
                # 128-bit atomics: what the LIFO and ABP thread-queue
                # backends are compiled from.  The macro travels on the
                # command line because this version's
                # hpx_check_for_cxx11_std_atomic_128bit() takes no
                # parameters and never forwards ${ARGN}: the DEFINITIONS
                # its caller passes are dropped, so the feature test can
                # never define the macro on any toolchain and the installed
                # defines.hpp cannot carry it — every consumer of these
                # headers must repeat it, or it compiles a different set of
                # queue backends than the archives hold.  The forced option
                # skips a runtime probe demanding
                # std::atomic<16 bytes>::is_lock_free(), which GCC reports
                # false whatever the hardware (a lock-free 16-byte load
                # would have to write); the 16-byte operations therefore
                # stay calls into libatomic, which selects cmpxchg16b
                # itself, and -mcx16 only permits inlining them.
                "-DCMAKE_CXX_FLAGS=-mcx16 -DHPX_HAVE_CXX11_STD_ATOMIC_128BIT"
                -DCMAKE_C_FLAGS=-mcx16
                -DHPX_WITH_CXX11_ATOMIC_128BIT=ON
                -DHPX_CXX11_STD_ATOMIC_LIBRARIES=atomic
                -DHPX_WITH_DISTRIBUTED_RUNTIME=ON
                -DHPX_WITH_NETWORKING=ON
                -DHPX_WITH_PARCELPORT_MPI=ON
                -DHPX_WITH_PARCELPORT_TCP=OFF
                -DHPX_WITH_FETCH_BOOST=ON
                -DHPX_WITH_FETCH_ASIO=ON
                # HPX's own pinned hwloc as a static archive, named exactly:
                # a find through NAMES/paths can still land on a host copy,
                # and a dynamic hwloc binds at load time to whatever the
                # host ships — per-machine drift, or a loader failure where
                # nothing ships.
                -DHPX_WITH_FETCH_HWLOC=OFF
                -DHwloc_INCLUDE_DIR=${HPX_HWLOC_INSTALL_DIR}/include
                -DHwloc_LIBRARY=${HPX_HWLOC_INSTALL_DIR}/lib/libhwloc.a
                # Static HPX libraries: the apps embed the runtime the same
                # way every other vendored runtime here is embedded, so a
                # binary resolves nothing through the loader beyond the host
                # toolchain and MPI.
                -DHPX_WITH_STATIC_LINKING=ON
                -DHPX_WITH_MALLOC=system
                -DHPX_WITH_DYNAMIC_HPX_MAIN=OFF
                -DHPX_WITH_EXAMPLES=OFF
                -DHPX_WITH_TESTS=OFF
                -DHPX_WITH_TOOLS=OFF
                -DHPX_WITH_DOCUMENTATION=OFF
                -DHPX_WITH_IO_COUNTERS=OFF
                # Parcel and message counts per parcelport: the accounting
                # the cross-model comparison reads.  A module, not a plugin,
                # so it survives static linking.
                -DHPX_WITH_PARCELPORT_COUNTERS=ON
            BUILD_COMMAND
                ${CMAKE_COMMAND} --build <BINARY_DIR>
                --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
            INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
            LOG_CONFIGURE 1
            LOG_BUILD 1
            LOG_INSTALL 1
        )

        set(ARTS_HPX_AVAILABLE ON PARENT_SCOPE)
        set(ARTS_HPX_INSTALL_DIR "${ARTS_HPX_INSTALL_DIR}" PARENT_SCOPE)
        set(HPX_HWLOC_INSTALL_DIR "${HPX_HWLOC_INSTALL_DIR}" PARENT_SCOPE)
    else()
        message(STATUS
            "HPX reference skipped: an MPI C++ implementation is required")
    endif()
endif()

# The vendored archive as a proper target: a consumer names arts::hwloc and
# gets the vendored headers and private link deps with it, so no target can
# reach a host hwloc through a bare -lhwloc or the compiler's default
# include path again.
add_library(arts_hwloc STATIC IMPORTED GLOBAL)
set_target_properties(arts_hwloc PROPERTIES
    IMPORTED_LOCATION "${HWLOC_LIB_A}"
    INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIRS}")
add_library(arts::hwloc ALIAS arts_hwloc)

# Export variables to parent scope
set(HWLOC_LIBRARIES ${HWLOC_LIBRARIES} PARENT_SCOPE)
set(HWLOC_INCLUDE_DIRS ${HWLOC_INCLUDE_DIRS} PARENT_SCOPE)
set(HWLOC_FOUND ${HWLOC_FOUND} PARENT_SCOPE)
set(HWLOC_VERSION ${HWLOC_VERSION} PARENT_SCOPE)

# Every vendored archive the runtime itself needs, collected as the library
# targets are configured and exported at the end of this file.  The static
# library folds them into its own archive and the shared library embeds them,
# so a consumer names the runtime alone.  A dependency that only a benchmark
# or a reference runtime uses (numa, TBB, OpenBLAS) does not belong here: the
# runtime never calls it, and bundling it would put it on every consumer.
set(ARTS_VENDORED_ARCHIVES "${HWLOC_LIB_A}")

if(ARTS_MALLOC STREQUAL "mimalloc")
  arts_init_submodule(third_party/mimalloc CMakeLists.txt)
  set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_STATIC ON  CACHE BOOL "" FORCE)
  set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
  set(MI_OVERRIDE     OFF CACHE BOOL "" FORCE)  # we call mi_* explicitly, no malloc override
  add_subdirectory(mimalloc EXCLUDE_FROM_ALL)
  list(APPEND ARTS_VENDORED_ARCHIVES "$<TARGET_FILE:mimalloc-static>")
endif()

# Build libfabric (OFI) from its vendored dist tarball, driven through CMake's
# ExternalProject so it participates in the Ninja build graph (parallel with
# other targets). The tarball is extracted at configure time into a tree
# private to this build directory; configure and make then run out of EP's
# own binary dir, leaving the extracted source pristine.
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
        # Absolute on purpose: the option is a CACHE PATH, and a relative
        # value would be absolutized against the configuring process's
        # working directory rather than the install prefix.
        -DSYSUSERS_DIR=${RDMA_CORE_INSTALL_DIR}/etc/sysusers.d
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
# The same set once more as plain archive paths, deduplicated: a merge or an
# embed consumes each archive's members once, and the repetition the link
# line needs for a cyclic set has no meaning there.
set(RDMA_VERBS_LIBTOOL_LIBS "")
set(RDMA_VERBS_ARCHIVES "")
set(_rdma_seen_ibverbs FALSE)
foreach(_flag IN LISTS RDMA_STATIC_LINK_FLAGS)
  if(_flag MATCHES "^-l(.+)$")
    set(_name "${CMAKE_MATCH_1}")
    if(_name STREQUAL "pthread" OR _name STREQUAL "m")
      continue()
    endif()
    list(APPEND RDMA_VERBS_ARCHIVES "${RDMA_CORE_INSTALL_DIR}/lib/lib${_name}.a")
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
list(REMOVE_DUPLICATES RDMA_VERBS_ARCHIVES)

# ---------------------------------------------------------------------------
# cxi provider input stack (HPE Slingshot / Cassini).
#
# The cxi provider hard-requires: the Cassini hardware headers and the cxi
# kernel UAPI headers (header-only), libcxi (the userspace device library),
# libcurl and json-c (FI_CHECK_PACKAGE link checks — the provider's
# collectives/REST path), and libnl-3 (libcxi's netlink transport).  All of
# it is vendored so the provider compiles identically on every build host
# and simply reports no fabrics where there is no Cassini NIC — the same
# posture as the verbs provider above.  Everything is built static at
# configure time into private prefixes, so the libfabric configure below
# consumes plain absolute archive paths (single-pass linking, libtool-safe),
# and the archives join the runtime bundle like the rdma-core set.
#
# libcxi is compiled directly rather than through its autotools build: its
# upstream configure hard-demands daemon-and-utility-only dependencies
# (libconfig/libuv/fuse/yaml) that the library proper never uses.  The
# library is three translation units plus a one-line config.h; its only
# library-side link dependency is libnl-3 core.
if(ARTS_OFI_CXI)
  # libnl's dist tarball ships grammar sources (.l/.y) without their
  # generated scanners, so flex and bison are genuine build-host
  # prerequisites for this stack.  A repo-local ./install/bin provisioning
  # is honored ahead of the host toolchain.
  find_program(ARTS_FLEX flex HINTS "${CMAKE_SOURCE_DIR}/install/bin")
  find_program(ARTS_BISON bison HINTS "${CMAKE_SOURCE_DIR}/install/bin")
  if(NOT ARTS_FLEX OR NOT ARTS_BISON)
    message(FATAL_ERROR
        "ARTS_OFI_CXI=ON requires flex and bison to build the vendored "
        "libnl (its dist tarball ships no generated scanners).  Provision "
        "them (e.g. under ${CMAKE_SOURCE_DIR}/install/bin) or configure "
        "with -DARTS_OFI_CXI=OFF.")
  endif()

  set(LIBNL_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/libnl_install")
  set(LIBNL_LIB_A "${LIBNL_INSTALL_DIR}/lib/libnl-3.a")
  if(NOT EXISTS "${LIBNL_LIB_A}")
    arts_fetch_dist_tarball(LIBNL_TARBALL
        "libnl-${LIBNL_DIST_VERSION}.tar.gz"
        "${LIBNL_DIST_URL}" "${LIBNL_DIST_SHA256}")
    set(LIBNL_SRC_DIR "${CMAKE_BINARY_DIR}/third_party/libnl-${LIBNL_DIST_VERSION}")
    if(NOT EXISTS "${LIBNL_SRC_DIR}/configure")
      message(STATUS "Extracting libnl ${LIBNL_DIST_VERSION}")
      file(ARCHIVE_EXTRACT INPUT "${LIBNL_TARBALL}"
           DESTINATION "${CMAKE_BINARY_DIR}/third_party")
    endif()
    set(LIBNL_BUILD_DIR "${CMAKE_BINARY_DIR}/third_party/libnl")
    file(MAKE_DIRECTORY "${LIBNL_BUILD_DIR}")
    if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
      set(CMAKE_BUILD_PARALLEL_LEVEL 4)
    endif()
    message(STATUS "Configuring libnl ${LIBNL_DIST_VERSION}")
    # Static only: libtool hoists a shared object's absolute path ahead of
    # --as-needed on consumer link lines, where it is then discarded;
    # archives keep their position.  LEX/YACC are pinned to the resolved
    # tools so configure cannot silently pick a different host pair.
    execute_process(
      # libnl's configure resolves the tools through FLEX/YACC (its own
      # prerequisite check tests $FLEX, not $LEX); LEX rides along for the
      # generated automake rules.
      COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC
          "FLEX=${ARTS_FLEX}" "LEX=${ARTS_FLEX}" "YACC=${ARTS_BISON} -y"
          ${LIBNL_SRC_DIR}/configure
          --prefix=${LIBNL_INSTALL_DIR}
          --enable-static --disable-shared --disable-cli
      WORKING_DIRECTORY ${LIBNL_BUILD_DIR}
      RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Failed to configure libnl")
    endif()
    execute_process(COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL}
      WORKING_DIRECTORY ${LIBNL_BUILD_DIR} RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Failed to build libnl")
    endif()
    execute_process(COMMAND make install
      WORKING_DIRECTORY ${LIBNL_BUILD_DIR} RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Failed to install libnl")
    endif()
  endif()

  arts_init_submodule(third_party/cassini-headers include/cxi_prov_hw.h)
  arts_init_submodule(third_party/cxi-driver include/uapi/misc/cxi.h)
  arts_init_submodule(third_party/libcxi src/libcxi.c)
  set(CASSINI_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cassini-headers")
  set(CXI_DRIVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cxi-driver")

  set(LIBCXI_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/libcxi_install")
  set(LIBCXI_LIB_A "${LIBCXI_INSTALL_DIR}/lib/libcxi.a")
  if(NOT EXISTS "${LIBCXI_LIB_A}")
    set(LIBCXI_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libcxi")
    set(LIBCXI_BUILD_DIR "${CMAKE_BINARY_DIR}/third_party/libcxi_build")
    file(MAKE_DIRECTORY "${LIBCXI_BUILD_DIR}")
    # The one config.h fact the sources consult: the vendored rdma-core's
    # rdma_netlink.h does declare RDMA_NLDEV_SYS_ATTR_COPY_ON_FORK.
    file(WRITE "${LIBCXI_BUILD_DIR}/config.h"
        "#define HAVE_DECL_RDMA_NLDEV_SYS_ATTR_COPY_ON_FORK 1\n")
    set(_libcxi_objs "")
    foreach(_src libcxi devinfo memory)
      message(STATUS "Compiling libcxi: ${_src}.c")
      execute_process(
        COMMAND ${CMAKE_C_COMPILER} -O2 -fPIC
            -I${LIBCXI_BUILD_DIR}
            -I${LIBCXI_SRC_DIR}/include
            -I${LIBCXI_SRC_DIR}/src
            -I${CASSINI_HEADERS_DIR}/include
            -I${CXI_DRIVER_DIR}/include
            -I${CMAKE_CURRENT_SOURCE_DIR}/rdma-core/kernel-headers
            -I${LIBNL_INSTALL_DIR}/include/libnl3
            -c ${LIBCXI_SRC_DIR}/src/${_src}.c
            -o ${LIBCXI_BUILD_DIR}/${_src}.o
        RESULT_VARIABLE _rc)
      if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "Failed to compile libcxi ${_src}.c")
      endif()
      list(APPEND _libcxi_objs "${LIBCXI_BUILD_DIR}/${_src}.o")
    endforeach()
    file(MAKE_DIRECTORY "${LIBCXI_INSTALL_DIR}/lib")
    file(MAKE_DIRECTORY "${LIBCXI_INSTALL_DIR}/include/libcxi")
    execute_process(
      COMMAND ${CMAKE_AR} rcs "${LIBCXI_LIB_A}" ${_libcxi_objs}
      RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Failed to archive libcxi.a")
    endif()
    # Staged as <prefix>/include/libcxi/libcxi.h — the path the provider's
    # FI_CHECK_PACKAGE probes (libcxi/libcxi.h).
    file(COPY_FILE "${LIBCXI_SRC_DIR}/include/libcxi.h"
         "${LIBCXI_INSTALL_DIR}/include/libcxi/libcxi.h")
  endif()

  set(THIRD_PARTY_EMPTY_PKGCONFIG "${CMAKE_BINARY_DIR}/third_party/pkgconfig_empty")
  file(MAKE_DIRECTORY "${THIRD_PARTY_EMPTY_PKGCONFIG}")

  # Minimal static curl: an HTTP client for the provider's REST path and
  # nothing else — every TLS/compression/protocol backend is pinned off so
  # the archive's external references cannot depend on what the build host
  # ships.  PKG_CONFIG_LIBDIR points at an empty directory for the same
  # reason.  Unknown -D options are warnings, so this flag set stays valid
  # across curl's own option renames.
  set(CURL_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/curl_install")
  set(CURL_LIB_A "${CURL_INSTALL_DIR}/lib/libcurl.a")
  if(NOT EXISTS "${CURL_LIB_A}")
    arts_init_submodule(third_party/curl CMakeLists.txt)
    set(CURL_BUILD_DIR "${CMAKE_BINARY_DIR}/third_party/curl-build")
    message(STATUS "Configuring vendored curl")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env
          PKG_CONFIG_LIBDIR=${THIRD_PARTY_EMPTY_PKGCONFIG}
          ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_SOURCE_DIR}/curl -B ${CURL_BUILD_DIR}
          -GNinja
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_INSTALL_PREFIX=${CURL_INSTALL_DIR}
          -DCMAKE_INSTALL_LIBDIR=lib
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
          -DBUILD_SHARED_LIBS=OFF
          -DBUILD_STATIC_LIBS=ON
          -DBUILD_CURL_EXE=OFF
          -DBUILD_EXAMPLES=OFF
          -DBUILD_TESTING=OFF
          -DBUILD_LIBCURL_DOCS=OFF
          -DBUILD_MISC_DOCS=OFF
          -DENABLE_CURL_MANUAL=OFF
          -DCURL_ENABLE_SSL=OFF
          -DCURL_USE_LIBPSL=OFF
          -DCURL_USE_LIBSSH2=OFF
          -DCURL_ZLIB=OFF
          -DCURL_BROTLI=OFF
          -DCURL_ZSTD=OFF
          -DUSE_NGHTTP2=OFF
          -DUSE_LIBIDN2=OFF
          -DCURL_DISABLE_LDAP=ON
          -DHTTP_ONLY=ON
      RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Failed to configure vendored curl")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${CURL_BUILD_DIR}
        -j${CMAKE_BUILD_PARALLEL_LEVEL} RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Failed to build vendored curl")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --install ${CURL_BUILD_DIR}
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Failed to install vendored curl")
    endif()
  endif()

  set(JSONC_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/json-c_install")
  set(JSONC_LIB_A "${JSONC_INSTALL_DIR}/lib/libjson-c.a")
  if(NOT EXISTS "${JSONC_LIB_A}")
    arts_init_submodule(third_party/json-c CMakeLists.txt)
    set(JSONC_BUILD_DIR "${CMAKE_BINARY_DIR}/third_party/json-c-build")
    message(STATUS "Configuring vendored json-c")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env
          PKG_CONFIG_LIBDIR=${THIRD_PARTY_EMPTY_PKGCONFIG}
          ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_SOURCE_DIR}/json-c -B ${JSONC_BUILD_DIR}
          -GNinja
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_INSTALL_PREFIX=${JSONC_INSTALL_DIR}
          -DCMAKE_INSTALL_LIBDIR=lib
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
          -DBUILD_SHARED_LIBS=OFF
          -DBUILD_STATIC_LIBS=ON
          -DBUILD_TESTING=OFF
          -DBUILD_APPS=OFF
          -DDISABLE_WERROR=ON
      RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Failed to configure vendored json-c")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${JSONC_BUILD_DIR}
        -j${CMAKE_BUILD_PARALLEL_LEVEL} RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Failed to build vendored json-c")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --install ${JSONC_BUILD_DIR}
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Failed to install vendored json-c")
    endif()
  endif()

  # What the libfabric configure below consumes.  The provider prefix flags
  # pin every probe to the vendored trees; the archives ride the global LIBS
  # (absolute paths — libtool-safe) so the cxi probe's link test, which
  # names no dependent libraries of its own, back-resolves libcxi's libnl
  # references and the provider's curl/json-c references in one pass.
  set(OFI_CXI_CONFIGURE_ARGS
      --enable-cxi=${LIBCXI_INSTALL_DIR}
      --with-cassini-headers=${CASSINI_HEADERS_DIR}
      --with-cxi-uapi-headers=${CXI_DRIVER_DIR}
      --with-curl=${CURL_INSTALL_DIR}
      --with-json-c=${JSONC_INSTALL_DIR})
  set(OFI_CXI_LIBTOOL_LIBS
      " ${LIBCXI_LIB_A} ${LIBNL_LIB_A} ${CURL_LIB_A} ${JSONC_LIB_A}")
  set(OFI_CXI_ARCHIVES
      "${LIBCXI_LIB_A}" "${LIBNL_LIB_A}" "${CURL_LIB_A}" "${JSONC_LIB_A}")
else()
  set(OFI_CXI_CONFIGURE_ARGS "")
  set(OFI_CXI_LIBTOOL_LIBS "")
  set(OFI_CXI_ARCHIVES "")
endif()

message(STATUS "Building libfabric ${OFI_DIST_VERSION} from vendored dist tarball")

arts_fetch_dist_tarball(OFI_TARBALL
    "libfabric-${OFI_DIST_VERSION}.tar.bz2" "${OFI_DIST_URL}" "${OFI_DIST_SHA256}")

set(OFI_SRC_DIR "${CMAKE_BINARY_DIR}/libfabric-${OFI_DIST_VERSION}")
if(NOT EXISTS "${OFI_SRC_DIR}/configure")
  message(STATUS "Extracting libfabric ${OFI_DIST_VERSION}")
  file(ARCHIVE_EXTRACT INPUT "${OFI_TARBALL}" DESTINATION "${CMAKE_BINARY_DIR}")
endif()

# libfabric tags its exported symbols with ELF symbol versions
# (fi_getinfo@@FABRIC_1.9 and friends), and a version-tagged definition can
# only enter a shared object whose link declares the matching version node --
# otherwise the link fails with "version node not found".  A consumer that
# embeds the static archive cannot reuse libfabric's own script verbatim: its
# `local: *` would hide the consumer's own exports too.  Derive a
# declarations-only script -- the node names and their inheritance, none of
# the symbol bodies -- from the same source, so unmatched symbols keep the
# base version and stay exported, and a libfabric that adds a version node
# needs no second edit here.
set(OFI_VERSION_NODES_MAP
    "${CMAKE_BINARY_DIR}/third_party/libfabric_version_nodes.map")
file(STRINGS "${OFI_SRC_DIR}/libfabric.map.in" _ofi_map_lines)
set(_ofi_nodes "")
foreach(_line IN LISTS _ofi_map_lines)
  if(_line MATCHES "^([A-Za-z_][A-Za-z_0-9.]*)[ \t]*{")
    string(APPEND _ofi_nodes "${CMAKE_MATCH_1} {")
  elseif(_line MATCHES "^}[ \t]*([A-Za-z_][A-Za-z_0-9.]*)[ \t]*;")
    string(APPEND _ofi_nodes "} ${CMAKE_MATCH_1};\n")
  elseif(_line MATCHES "^}[ \t]*;")
    string(APPEND _ofi_nodes "};\n")
  endif()
endforeach()
if(NOT _ofi_nodes)
  message(FATAL_ERROR
      "No symbol-version nodes found in ${OFI_SRC_DIR}/libfabric.map.in")
endif()
file(WRITE "${OFI_VERSION_NODES_MAP}" "${_ofi_nodes}")

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
      "LIBS=${RDMA_VERBS_LIBTOOL_LIBS}${OFI_CXI_LIBTOOL_LIBS} -lpthread -lm"
      HWLOC_DIR=${HWLOC_INSTALL_DIR}
      ${OFI_SRC_DIR}/configure
      --prefix=${OFI_PREFIX} --enable-static --disable-shared
      # Provider allowlist.  With a STATIC libfabric.a every enabled
      # provider's external libraries become link-time dependencies of every
      # consuming binary, so a provider auto-enabled by configure probing
      # the build host (psm, ...) breaks the final link on any
      # cluster that ships its headers.  --enable-only inverts the
      # default: nothing builds unless named here, which makes the build
      # deterministic on every host.  verbs exposes RDM endpoints only as
      # the verbs;ofi_rxm stack, so rxm must stay enabled or every
      # InfiniBand host silently downgrades to tcp.  cxi joins the
      # allowlist when ARTS_OFI_CXI is ON, built strictly from the vendored
      # Cassini/libcxi/curl/json-c/libnl stack above (empty args when OFF).
      --enable-only --enable-tcp --enable-rxm
      --enable-verbs=${RDMA_CORE_INSTALL_DIR}
      ${OFI_CXI_CONFIGURE_ARGS}
      # --enable-only governs providers (FI_PROVIDER_SETUP entries) only.
      # xpmem and io_uring are standalone feature toggles whose default-on
      # host probes append to the global LIBS, so each must be pinned off
      # explicitly or a host that ships their headers changes the
      # artifact's link requirements.
      --disable-xpmem --without-uring
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

# libatomic enters through the already-resolved link item instead of the
# bare name libfabric's pkgconfig records: a host whose libatomic ships only
# as the runtime SONAME offers the linker no `-latomic` to resolve, and an
# unresolvable name breaks every consumer that inherits this interface.
# m is load-bearing at the end: the cxi provider's collectives reference the
# C99 floating-point environment (fegetround) and json-c does too, and
# single-pass static links resolve those only from a libm placed after the
# archives.
set(OFI_LINK_INTERFACE rt ${ARTS_ATOMIC_LINK} pthread ${CMAKE_DL_LIBS}
    ${RDMA_VERBS_LINK_SET} ${OFI_CXI_ARCHIVES} ${HWLOC_LIB_A} m)

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
  INTERFACE_LINK_LIBRARIES "${OFI_LINK_INTERFACE}")
add_dependencies(arts::libfabric libfabric_ep)

list(APPEND ARTS_VENDORED_ARCHIVES
     "${OFI_PREFIX}/lib/libfabric.a" ${RDMA_VERBS_ARCHIVES}
     ${OFI_CXI_ARCHIVES})

# Bundling these into a library makes their symbols part of it; naming each
# archive here keeps them out of its dynamic exports, so nothing outside can
# bind to a vendored symbol and interpose on the library's own use of it.
set(ARTS_VENDORED_EXCLUDE_LIBS "")
foreach(_archive IN LISTS ARTS_VENDORED_ARCHIVES)
  if(NOT _archive MATCHES "^\\$<")
    get_filename_component(_archive_name "${_archive}" NAME)
    list(APPEND ARTS_VENDORED_EXCLUDE_LIBS "${_archive_name}")
  endif()
endforeach()
if(ARTS_MALLOC STREQUAL "mimalloc")
  list(APPEND ARTS_VENDORED_EXCLUDE_LIBS "$<TARGET_FILE_NAME:mimalloc-static>")
endif()

set(ARTS_VENDORED_ARCHIVES ${ARTS_VENDORED_ARCHIVES} PARENT_SCOPE)
set(ARTS_VENDORED_EXCLUDE_LIBS ${ARTS_VENDORED_EXCLUDE_LIBS} PARENT_SCOPE)
set(OFI_VERSION_NODES_MAP ${OFI_VERSION_NODES_MAP} PARENT_SCOPE)

# Build OpenBLAS (BLAS + LAPACK/LAPACKE, BSD-3) from the third_party/OpenBLAS
# submodule for benchmark apps that call the standard CBLAS/LAPACKE C
# interfaces.  Built SINGLE-THREADED (USE_THREAD=0): the consuming apps
# parallelize at task granularity (one kernel call per tile task), so a
# threaded BLAS underneath would oversubscribe the machine and serialize on
# its own thread pool.  The submodule tree is shared by every build directory
# and OpenBLAS's make build writes artifacts into its source tree, so each
# build directory builds from a private copy.
#
# Its one consumer is the tiled Cholesky whose kernels are CBLAS/LAPACKE
# calls, so the external project stays out of the default target: building it
# is a consequence of building that application, not of building the tree.
if(ARTS_BUILD_BENCHMARKS)
  # The benchmark sources every configuration of this tree consumes: the OCR
  # headers (the shim and every app compile against them), the applications
  # themselves, and OpenBLAS.  Fetching a submodule costs network and disk and
  # fails outright where neither is available, so a submodule whose consumer
  # is conditional is ensured at that consumer instead -- ocr-vx and TBB are
  # fetched in benchmarks/ocrvx, which alone knows whether the host has the
  # MPI that reference runtime is built on.
  arts_init_submodule(third_party/ocr ocr/inc/ocr.h)
  arts_init_submodule(third_party/ocr-apps apps)
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
    EXCLUDE_FROM_ALL TRUE
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
