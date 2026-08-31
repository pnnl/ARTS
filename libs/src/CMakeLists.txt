###############################################################################
## This material was prepared as an account of work sponsored by an agency
## of the United States Government.  Neither the United States Government
## nor the United States Department of Energy, nor Battelle, nor any of
## their employees, nor any jurisdiction or organization that has cooperated
## in the development of these materials, makes any warranty, express or
## implied, or assumes any legal liability or responsibility for the accuracy,
## completeness, or usefulness or any information, apparatus, product,
## software, or process disclosed, or represents that its use would not
## infringe privately owned rights.
##
## Reference herein to any specific commercial product, process, or service
## by trade name, trademark, manufacturer, or otherwise does not necessarily
## constitute or imply its endorsement, recommendation, or favoring by the
## United States Government or any agency thereof, or Battelle Memorial
## Institute. The views and opinions of authors expressed herein do not
## necessarily state or reflect those of the United States Government or
## any agency thereof.
##
##                      PACIFIC NORTHWEST NATIONAL LABORATORY
##                                  operated by
##                                    BATTELLE
##                                    for the
##                      UNITED STATES DEPARTMENT OF ENERGY
##                         under Contract DE-AC05-76RL01830
##
## Copyright 2019 Battelle Memorial Institute
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    https://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
###############################################################################

# Create the main ARTS library interface target
add_library(arts_interface INTERFACE)
target_include_directories(arts_interface INTERFACE
    $<BUILD_INTERFACE:${ARTS_PUBLIC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ARTS_INTERNAL_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ARTS_BUILD_INTERNAL_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(ARTS_USE_CXL)
    target_link_libraries(arts_interface INTERFACE arts_cxl_lib)
endif()

# --- Coherence-protocol helpers (shared by main lib + benchmark variants) -----
# Sets ${out} (caller scope) to the coherence model TU paths for
# (proto,placement), RELATIVE to libs/src/core/.  Mirrors
# libs/src/core/CMakeLists.txt selection.
# The grant plane's release-policy half: WHO initiates a grant's return.  Only
# a write-through arm can hand the permission back without the payload, so a
# write-back grant is always demand-revoked.
function(arts_grant_release_tu out write_policy release_policy)
    if(write_policy STREQUAL "WT" AND release_policy STREQUAL "PURGE")
        set(${out} coherence/grant_purge.c PARENT_SCOPE)
    else()
        set(${out} coherence/grant_retain.c PARENT_SCOPE)
    endif()
endfunction()

function(arts_coherence_model_tus out arm write_policy release_policy)
    if(arm STREQUAL "WRF_VAL")
        set(tus coherence/wrf_val/wrf_val.c)
    elseif(arm STREQUAL "INV")
        if(write_policy STREQUAL "WB")
            set(tus coherence/inv/directory.c coherence/inv/engine.c coherence/inv/wb.c coherence/grant.c coherence/grant_queue.c coherence/grant_wb.c)
        else()
            set(tus coherence/inv/directory.c coherence/inv/engine.c coherence/inv/wt.c coherence/grant.c coherence/grant_queue.c coherence/grant_wt.c)
        endif()
        arts_grant_release_tu(rel "${write_policy}" "${release_policy}")
        list(APPEND tus ${rel})
    elseif(arm STREQUAL "EXCL")
        if(release_policy STREQUAL "RETAIN")
            set(tus coherence/excl/directory.c coherence/excl/retain.c)
        else()
            set(tus coherence/excl/directory.c coherence/excl/purge.c)
        endif()
    else() # VAL
        if(write_policy STREQUAL "WT")
            set(tus coherence/val/wt.c coherence/grant.c coherence/grant_queue.c coherence/grant_wt.c coherence/val/directory.c)
        else()
            set(tus coherence/val/wb.c coherence/grant.c coherence/grant_queue.c coherence/grant_wb.c coherence/val/directory.c)
        endif()
        arts_grant_release_tu(rel "${write_policy}" "${release_policy}")
        list(APPEND tus ${rel})
    endif()
    set(${out} ${tus} PARENT_SCOPE)
endfunction()

# Applies the family + second-axis compile definitions to <target> (PRIVATE).
# All three macro groups are always emitted: ARTS_PROTOCOL_<arm>, one of
# ARTS_WRITE_POLICY_WT/WB, and one of ARTS_RELEASE_PURGE/RETAIN.  Each arm's
# sources test only its live axis (EXCL: ARTS_RELEASE_*; INV/VAL and the
# grant plane: ARTS_WRITE_POLICY_*).
function(arts_apply_protocol target arm write_policy release_policy)
    # A call with no declaration in scope is an error, not a warning.  Each arm
    # compiles a different subset of the sources, so a declaration guarded to
    # the wrong set of arms is invisible in the arm that was tested and leaves
    # the others calling through an implicit int -- which silently narrows a
    # pointer or a bool return.  Compilers that still default this to a warning
    # let it accumulate unseen; the ones that do not simply refuse to build.
    target_compile_options(${target} PRIVATE -Werror=implicit-function-declaration)
    # An optional 5th argument pins read combining OFF for this target (the
    # ablation twins); without it the target follows the directory-wide
    # disable option (default off, so combining is compiled in).  Targets that
    # name it are self-describing: nothing outside their own definition
    # decides what they compile, which is what lets several configurations
    # coexist in one build.
    if(ARGC GREATER 4)
        set(nocomb "${ARGV4}")
    else()
        set(nocomb "${ARTS_NO_RO_COMBINING}")
    endif()
    if(nocomb)
        target_compile_definitions(${target} PRIVATE ARTS_NO_RO_COMBINING=1)
    endif()
    target_compile_definitions(${target} PRIVATE ARTS_PROTOCOL_${arm}=1)
    if(arm STREQUAL "WRF_VAL")
        # WRF_VAL is WT by construction and links no grant plane; it defines
        # neither second-axis macro so the shared selection chains fall through
        # to their ARTS_PROTOCOL_WRF_VAL branches.
        return()
    endif()
    if(write_policy STREQUAL "WT")
        target_compile_definitions(${target} PRIVATE ARTS_WRITE_POLICY_WT=1)
    else()
        target_compile_definitions(${target} PRIVATE ARTS_WRITE_POLICY_WB=1)
    endif()
    if(release_policy STREQUAL "PURGE")
        target_compile_definitions(${target} PRIVATE ARTS_RELEASE_PURGE=1)
    else()
        target_compile_definitions(${target} PRIVATE ARTS_RELEASE_RETAIN=1)
    endif()
endfunction()

# --- Vendored dependencies inside the installed libraries --------------------
# An application links the runtime and nothing else: `-larts` alone must
# resolve, from either the archive or the shared object.  The vendored
# dependencies are therefore part of the artifact rather than part of a
# consumer's link line, which also makes the artifact's behaviour independent
# of whatever the consumer's host happens to offer under the same names.
#
# Ordering: the archives are produced by an ExternalProject and a subdirectory
# target.  A link edge already implies both, but the archive merge below runs
# as a POST_BUILD step whose inputs the build system cannot see, so each is
# named explicitly.
function(arts_depend_on_vendored target)
    if(TARGET libfabric_ep)
        add_dependencies(${target} libfabric_ep)
    endif()
    if(TARGET mimalloc-static)
        add_dependencies(${target} mimalloc-static)
    endif()
endfunction()

# Folds the vendored archives into <target>'s own archive and localizes their
# symbols, so a consumer that links its own copy of one of them keeps it.  The
# static-library rule recreates the archive from its objects before POST_BUILD
# runs, so the merge starts from the same inputs on every build instead of
# compounding -- and the archive holds exactly this target's own objects at that
# moment, which is what the script reads its keep-global allowlist from.
function(arts_bundle_vendored_archives target)
    if(NOT ARTS_VENDORED_ARCHIVES)
        return()
    endif()
    arts_depend_on_vendored(${target})
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DMERGE_AR=${CMAKE_AR}
            -DMERGE_RANLIB=${CMAKE_RANLIB}
            -DMERGE_NM=${CMAKE_NM}
            -DMERGE_LD=${CMAKE_LINKER}
            -DMERGE_OBJCOPY=${CMAKE_OBJCOPY}
            -DMERGE_ARCHIVE=$<TARGET_FILE:${target}>
            -P ${PROJECT_SOURCE_DIR}/cmake/MergeStaticArchives.cmake
            -- ${ARTS_VENDORED_ARCHIVES}
        COMMENT "Bundling vendored archives into $<TARGET_FILE_NAME:${target}>"
        VERBATIM)
endfunction()

# Embeds the same archives in a shared <target> and keeps their symbols out of
# its dynamic exports: a vendored symbol that stays exported can be interposed
# by a copy in the executable, which would split a library's state between two
# instances of itself.
function(arts_embed_vendored_archives target)
    arts_depend_on_vendored(${target})
    # PRIVATE, not INTERFACE: the archive belongs inside the shared object, not
    # on the line of everything that links it.
    target_link_libraries(${target} PRIVATE arts::libfabric)
    # Both options below are GNU ld / lld spellings, and ELF symbol versioning
    # is the problem only one of them solves.
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        return()
    endif()
    foreach(name IN LISTS ARTS_VENDORED_EXCLUDE_LIBS)
        target_link_options(${target} PRIVATE "LINKER:--exclude-libs,${name}")
    endforeach()
    # libfabric's symbols are version-tagged (fi_getinfo@@FABRIC_1.x) and a
    # version-tagged definition needs its version node declared at this link,
    # which the derived script supplies without hiding anything of ours the way
    # libfabric's own script would.
    target_link_options(${target} PRIVATE
        "LINKER:--version-script=${OFI_VERSION_NODES_MAP}")
endfunction()

# No-hint placement policies, visible to every libarts compile in this
# directory and its subdirectories (edt.c / db.c gate their no-hint branches
# on these; see ARTS_NOHINT_EDT_PLACEMENT / ARTS_NOHINT_DB_HOME at top level).
add_compile_definitions(
    ARTS_NOHINT_EDT_ROUNDROBIN=${_ARTS_NOHINT_EDT_RR}
    ARTS_NOHINT_DB_ROUNDROBIN=${_ARTS_NOHINT_DB_RR})

# Module subdirectories — each defines OBJECT library targets
add_subdirectory(core)
add_subdirectory(graph)

# Main source file (main.c lives in core/; scheduler.c is an OBJECT library)
set(ARTS_MAIN_SOURCES "core/main.c")

# Some platforms do not provide a standalone libatomic.
# ARTS uses compiler atomic builtins, so link libatomic only when present.
# Reuse the top-level ATOMIC_LIBRARY (probed for DWCAS support); fall
# back to a local find_library if it wasn't set (e.g., older configurations).
if(ATOMIC_LIBRARY)
    set(ARTS_ATOMIC_LIBRARY "${ATOMIC_LIBRARY}" CACHE FILEPATH
        "libatomic path (shared with top-level DWCAS probe)" FORCE)
else()
    find_library(ARTS_ATOMIC_LIBRARY atomic)
endif()
if(ARTS_ATOMIC_LIBRARY)
    message(STATUS "Using libatomic: ${ARTS_ATOMIC_LIBRARY}")
else()
    message(STATUS "libatomic not found; relying on compiler/runtime atomics")
endif()

#
# Standard ARTS Library (always built, no CUDA)
#

# The static library is the substrate every other artifact derives from.
add_library(arts_static STATIC ${ARTS_MAIN_SOURCES})

target_include_directories(arts_static BEFORE PRIVATE
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
    ${ARTS_PUBLIC_INCLUDE_DIR}
    ${ARTS_INTERNAL_INCLUDE_DIR}
)

# Add hwloc include directories if available
if(HWLOC_INCLUDE_DIRS)
    target_include_directories(arts_static PRIVATE ${HWLOC_INCLUDE_DIRS})
endif()

# Add object library contents directly to avoid export issues
target_sources(arts_static PRIVATE
    $<TARGET_OBJECTS:arts_gas>
    $<TARGET_OBJECTS:arts_transport>
    $<TARGET_OBJECTS:arts_counter>
    $<TARGET_OBJECTS:arts_scheduler>
    $<TARGET_OBJECTS:arts_compute>
    $<TARGET_OBJECTS:arts_memory>
    $<TARGET_OBJECTS:arts_sync>
    $<TARGET_OBJECTS:arts_ooo>
    $<TARGET_OBJECTS:arts_regpool>
    $<TARGET_OBJECTS:arts_system>
    $<TARGET_OBJECTS:arts_utils>
    $<TARGET_OBJECTS:arts_graph>
)

target_link_libraries(arts_static PUBLIC arts_interface)

# Link external dependencies statically
target_link_libraries(arts_static PRIVATE
    Threads::Threads
    ${RT_LIBRARY}
    ${M_LIBRARY}
    ${ANL_LIBRARY}
    $<BUILD_INTERFACE:${HWLOC_LIBRARIES}>
)

if(ARTS_ATOMIC_LIBRARY)
    target_link_libraries(arts_static PRIVATE ${ARTS_ATOMIC_LIBRARY})
endif()

# Add macOS framework dependencies for hwloc
if(APPLE AND HWLOC_FOUND)
    target_link_libraries(arts_static PRIVATE "-framework CoreFoundation" "-framework IOKit" "-framework OpenCL")
endif()

if(ARTS_USE_CXL)
    target_link_libraries(arts_static PUBLIC arts_cxl_lib)
endif()

target_link_libraries(arts_static PRIVATE arts::libfabric)

arts_apply_protocol(arts_static ${ARTS_COHERENCE_ARM} ${ARTS_WRITE_POLICY} ${ARTS_RELEASE_POLICY})

# The vendored archives above stay on this target's PRIVATE link list even
# though the merge makes them redundant there: the entries carry the
# vendored include directories the sources in this directory compile
# against, and the toolchain libraries (dl, rt) that back-resolve the
# embedded objects on a consumer's line, which no merge can supply.
arts_bundle_vendored_archives(arts_static)

# Set library properties
set_target_properties(arts_static PROPERTIES
    OUTPUT_NAME arts
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME static
)

# Create shared library if requested
if(ARTS_BUILD_SHARED)
    add_library(arts_shared SHARED ${ARTS_MAIN_SOURCES})

    target_include_directories(arts_shared BEFORE PRIVATE
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
        ${ARTS_PUBLIC_INCLUDE_DIR}
        ${ARTS_INTERNAL_INCLUDE_DIR}
    )

    # Add hwloc include directories if available
    if(HWLOC_INCLUDE_DIRS)
        target_include_directories(arts_shared PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()

    # Add object library contents directly to avoid export issues
    target_sources(arts_shared PRIVATE
        $<TARGET_OBJECTS:arts_gas>
        $<TARGET_OBJECTS:arts_transport>
        $<TARGET_OBJECTS:arts_counter>
        $<TARGET_OBJECTS:arts_scheduler>
        $<TARGET_OBJECTS:arts_compute>
        $<TARGET_OBJECTS:arts_memory>
        $<TARGET_OBJECTS:arts_sync>
        $<TARGET_OBJECTS:arts_ooo>
        $<TARGET_OBJECTS:arts_regpool>
        $<TARGET_OBJECTS:arts_system>
        $<TARGET_OBJECTS:arts_utils>
        $<TARGET_OBJECTS:arts_graph>
    )

    target_link_libraries(arts_shared PUBLIC arts_interface)

    # Link external dependencies statically into the shared library
    target_link_libraries(arts_shared PRIVATE
        Threads::Threads
        ${RT_LIBRARY}
        ${M_LIBRARY}
        ${ANL_LIBRARY}
        $<BUILD_INTERFACE:${HWLOC_LIBRARIES}>
    )

    arts_embed_vendored_archives(arts_shared)

    if(ARTS_ATOMIC_LIBRARY)
        target_link_libraries(arts_shared PRIVATE ${ARTS_ATOMIC_LIBRARY})
    endif()

    # Add macOS framework dependencies for hwloc
    if(APPLE AND HWLOC_FOUND)
        target_link_libraries(arts_shared PRIVATE "-framework CoreFoundation" "-framework IOKit" "-framework OpenCL")
    endif()

    if(ARTS_USE_CXL)
        target_link_libraries(arts_shared PUBLIC arts_cxl_lib)
    endif()

    arts_apply_protocol(arts_shared ${ARTS_COHERENCE_ARM} ${ARTS_WRITE_POLICY} ${ARTS_RELEASE_POLICY})

    # Set library properties
    set_target_properties(arts_shared PROPERTIES
        OUTPUT_NAME arts
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        EXPORT_NAME shared
    )
endif()

#
# CUDA-enabled ARTS Library (built when CUDA available)
#

if(BUILD_CUDA_LIBRARY)
    add_library(arts_cuda_static STATIC ${ARTS_MAIN_SOURCES})

    target_include_directories(arts_cuda_static BEFORE PRIVATE
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
        ${ARTS_PUBLIC_INCLUDE_DIR}
        ${ARTS_INTERNAL_INCLUDE_DIR}
    )

    # Add hwloc include directories if available
    if(HWLOC_INCLUDE_DIRS)
        target_include_directories(arts_cuda_static PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()

    # Add object library contents directly to avoid export issues
    target_sources(arts_cuda_static PRIVATE
        $<TARGET_OBJECTS:arts_gas>
        $<TARGET_OBJECTS:arts_transport>
        $<TARGET_OBJECTS:arts_counter>
        $<TARGET_OBJECTS:arts_scheduler_gpu>
        $<TARGET_OBJECTS:arts_compute_gpu>
        $<TARGET_OBJECTS:arts_memory_gpu>
        $<TARGET_OBJECTS:arts_sync>
        $<TARGET_OBJECTS:arts_ooo>
        $<TARGET_OBJECTS:arts_regpool>
        $<TARGET_OBJECTS:arts_system>
        $<TARGET_OBJECTS:arts_utils>
        $<TARGET_OBJECTS:arts_gpu>
        $<TARGET_OBJECTS:arts_graph>
    )

    target_link_libraries(arts_cuda_static PUBLIC arts_interface)

    # Add CUDA compile definition for this target
    target_compile_definitions(arts_cuda_static PRIVATE ARTS_USE_GPU)

    # Link external dependencies and CUDA libraries
    target_link_libraries(arts_cuda_static PRIVATE
        Threads::Threads
        ${RT_LIBRARY}
        ${M_LIBRARY}
        ${ANL_LIBRARY}
        $<BUILD_INTERFACE:${HWLOC_LIBRARIES}>
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
    )

    if(ARTS_ATOMIC_LIBRARY)
        target_link_libraries(arts_cuda_static PRIVATE ${ARTS_ATOMIC_LIBRARY})
    endif()

    # Add macOS framework dependencies for hwloc
    if(APPLE AND HWLOC_FOUND)
        target_link_libraries(arts_cuda_static PRIVATE "-framework CoreFoundation" "-framework IOKit" "-framework OpenCL")
    endif()

    arts_bundle_vendored_archives(arts_cuda_static)

    arts_apply_protocol(arts_cuda_static ${ARTS_COHERENCE_ARM} ${ARTS_WRITE_POLICY} ${ARTS_RELEASE_POLICY})

    # Set library properties
    set_target_properties(arts_cuda_static PROPERTIES
        OUTPUT_NAME arts_cuda
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        EXPORT_NAME cuda_static
    )

    # Create shared CUDA library if requested
    if(ARTS_BUILD_SHARED)
        add_library(arts_cuda_shared SHARED ${ARTS_MAIN_SOURCES})

        target_include_directories(arts_cuda_shared BEFORE PRIVATE
            ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
            ${ARTS_PUBLIC_INCLUDE_DIR}
            ${ARTS_INTERNAL_INCLUDE_DIR}
        )

        # Add hwloc include directories if available
        if(HWLOC_INCLUDE_DIRS)
            target_include_directories(arts_cuda_shared PRIVATE ${HWLOC_INCLUDE_DIRS})
        endif()

        # Add object library contents directly to avoid export issues
        target_sources(arts_cuda_shared PRIVATE
            $<TARGET_OBJECTS:arts_gas>
            $<TARGET_OBJECTS:arts_transport>
            $<TARGET_OBJECTS:arts_counter>
            $<TARGET_OBJECTS:arts_scheduler_gpu>
            $<TARGET_OBJECTS:arts_compute_gpu>
            $<TARGET_OBJECTS:arts_memory_gpu>
            $<TARGET_OBJECTS:arts_sync>
            $<TARGET_OBJECTS:arts_ooo>
            $<TARGET_OBJECTS:arts_regpool>
            $<TARGET_OBJECTS:arts_system>
            $<TARGET_OBJECTS:arts_utils>
            $<TARGET_OBJECTS:arts_gpu>
            $<TARGET_OBJECTS:arts_graph>
        )

        target_link_libraries(arts_cuda_shared PUBLIC arts_interface)

        # Add CUDA compile definition for this target
        target_compile_definitions(arts_cuda_shared PRIVATE ARTS_USE_GPU)

        # Link external dependencies and CUDA libraries
        target_link_libraries(arts_cuda_shared PRIVATE
            Threads::Threads
            ${RT_LIBRARY}
            ${M_LIBRARY}
            ${ANL_LIBRARY}
            $<BUILD_INTERFACE:${HWLOC_LIBRARIES}>
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::cublas
        )

        arts_embed_vendored_archives(arts_cuda_shared)

        if(ARTS_ATOMIC_LIBRARY)
            target_link_libraries(arts_cuda_shared PRIVATE ${ARTS_ATOMIC_LIBRARY})
        endif()

        # Add macOS framework dependencies for hwloc
        if(APPLE AND HWLOC_FOUND)
            target_link_libraries(arts_cuda_shared PRIVATE "-framework CoreFoundation" "-framework IOKit" "-framework OpenCL")
        endif()

        arts_apply_protocol(arts_cuda_shared ${ARTS_COHERENCE_ARM} ${ARTS_WRITE_POLICY} ${ARTS_RELEASE_POLICY})

        # Set library properties
        set_target_properties(arts_cuda_shared PROPERTIES
            OUTPUT_NAME arts_cuda
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
            EXPORT_NAME cuda_shared
        )
    endif()
endif()

# --- Per-protocol variant libraries (benchmarks/harness only) -----------------
# Build a full arts static lib for one (proto,placement). The protocol-independent
# sources are gathered from the CPU object-lib targets (their compiled objects
# cannot be reused — each variant needs its own protocol defines), the coherence
# MODEL TUs are swapped for this variant's, and the defines are applied per-target.
if(ARTS_BUILD_BENCHMARKS)
    set(_ARTS_VARIANT_OBJLIBS
        arts_compute arts_memory arts_sync arts_ooo arts_regpool arts_scheduler
        arts_transport arts_gas arts_system arts_utils arts_counter arts_graph)

    function(arts_variant suffix arm write_policy release_policy nocomb)
        # Gather all protocol-independent sources (absolute paths) from the
        # object libs, dropping any coherence MODEL TU (path contains
        # /coherence/<model>/), which is variant-specific.
        set(srcs "${CMAKE_CURRENT_SOURCE_DIR}/core/main.c")
        foreach(lib ${_ARTS_VARIANT_OBJLIBS})
            get_target_property(lsrcs ${lib} SOURCES)
            get_target_property(ldir  ${lib} SOURCE_DIR)
            foreach(s ${lsrcs})
                if(IS_ABSOLUTE "${s}")
                    set(abs "${s}")
                else()
                    set(abs "${ldir}/${s}")
                endif()
                # Skip variant-specific coherence TUs: the per-arm subdirs, and
                # the grant plane (linked only by arms whose ownership migrates).
                if(NOT abs MATCHES "/coherence/(val|excl|wrf_val|inv)/|/coherence/grant(_wt|_wb|_queue|_purge|_retain)?\\.c$")
                    list(APPEND srcs "${abs}")
                endif()
            endforeach()
        endforeach()
        # add this variant's coherence model TUs
        arts_coherence_model_tus(model "${arm}" "${write_policy}" "${release_policy}")
        foreach(tu ${model})
            list(APPEND srcs "${CMAKE_CURRENT_SOURCE_DIR}/core/${tu}")
        endforeach()

        add_library(arts_static_${suffix} STATIC ${srcs})
        target_include_directories(arts_static_${suffix} BEFORE PRIVATE
            ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
            ${ARTS_PUBLIC_INCLUDE_DIR}
            ${ARTS_INTERNAL_INCLUDE_DIR})
        if(HWLOC_INCLUDE_DIRS)
            target_include_directories(arts_static_${suffix} PRIVATE ${HWLOC_INCLUDE_DIRS})
        endif()
        target_link_libraries(arts_static_${suffix} PUBLIC arts_interface)
        target_link_libraries(arts_static_${suffix} PRIVATE
            Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY}
            $<BUILD_INTERFACE:${HWLOC_LIBRARIES}>)
        if(ARTS_ATOMIC_LIBRARY)
            target_link_libraries(arts_static_${suffix} PRIVATE ${ARTS_ATOMIC_LIBRARY})
        endif()
        if(ARTS_MALLOC STREQUAL "mimalloc")
            target_link_libraries(arts_static_${suffix} PRIVATE mimalloc-static)
        endif()
        # regpool.c (gathered from arts_regpool) includes fabric headers and
        # the static lib resolves fi_* symbols from libfabric.
        target_link_libraries(arts_static_${suffix} PRIVATE arts::libfabric)
        arts_apply_protocol(arts_static_${suffix} "${arm}" "${write_policy}"
                            "${release_policy}" "${nocomb}")
    endfunction()

    # Suffix = <memory model>_<family>_<live second axis>; the second argument
    # is the protocol ARM (the TU family): VAL/INV/EXCL live under the OCR
    # model, WRF_VAL is the VAL family under the DB-WRF model.  The dead axis
    # sits at its required value (WB for EXCL, RETAIN for INV/VAL).  The last
    # argument DISABLES read combining: OFF for every plane configuration
    # (combining is part of the validation family's definition — every other
    # message class already suppresses duplicates structurally), ON only for
    # the *_nocomb ablation twins that measure what the combining window buys.
    arts_variant(ocr_val_wt        VAL     WT RETAIN OFF)
    arts_variant(ocr_val_wb        VAL     WB RETAIN OFF)
    arts_variant(ocr_val_wt_purge  VAL     WT PURGE  OFF)
    arts_variant(ocr_val_wt_nocomb VAL     WT RETAIN ON)
    arts_variant(ocr_val_wb_nocomb VAL     WB RETAIN ON)
    arts_variant(ocr_val_wt_purge_nocomb VAL WT PURGE ON)
    arts_variant(ocr_excl_purge    EXCL    WB PURGE  OFF)
    arts_variant(ocr_excl_retain   EXCL    WB RETAIN OFF)
    arts_variant(wrf_val_wt        WRF_VAL WT RETAIN OFF)
    arts_variant(ocr_inv_wt        INV     WT RETAIN OFF)
    arts_variant(ocr_inv_wb        INV     WB RETAIN OFF)
    arts_variant(ocr_inv_wt_purge  INV     WT PURGE  OFF)
endif()

# Link mimalloc-static when selected.  The OBJECT library arts_utils needs
# the include dir at compile time (for malloc.c's #include <mimalloc.h>);
# target_link_libraries on an OBJECT target propagates INTERFACE properties
# (include dirs) to its compilation unit without introducing a link dep.
if(ARTS_MALLOC STREQUAL "mimalloc")
    target_link_libraries(arts_utils PRIVATE mimalloc-static)
    target_link_libraries(arts_static PRIVATE mimalloc-static)
    if(ARTS_BUILD_SHARED)
        target_link_libraries(arts_shared PRIVATE mimalloc-static)
    endif()
    if(BUILD_CUDA_LIBRARY)
        target_link_libraries(arts_cuda_static PRIVATE mimalloc-static)
        if(ARTS_BUILD_SHARED)
            target_link_libraries(arts_cuda_shared PRIVATE mimalloc-static)
        endif()
    endif()
endif()

# Create alias targets for convenience (prefer shared over static)
if(ARTS_BUILD_SHARED)
    add_library(ARTS::arts ALIAS arts_shared)
    add_library(ARTS::shared ALIAS arts_shared)
    if(BUILD_CUDA_LIBRARY)
        add_library(ARTS::arts_cuda ALIAS arts_cuda_shared)
        add_library(ARTS::cuda_shared ALIAS arts_cuda_shared)
    endif()
endif()

if(NOT TARGET ARTS::arts)
    add_library(ARTS::arts ALIAS arts_static)
endif()
add_library(ARTS::static ALIAS arts_static)
if(BUILD_CUDA_LIBRARY)
    if(NOT TARGET ARTS::arts_cuda)
        add_library(ARTS::arts_cuda ALIAS arts_cuda_static)
    endif()
    add_library(ARTS::cuda_static ALIAS arts_cuda_static)
endif()
