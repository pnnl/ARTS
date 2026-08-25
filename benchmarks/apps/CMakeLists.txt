# OCR benchmark applications
# Each app is built against the ARTS shim backend only.
#
# Helper libraries (reduction, timer, ocrAppUtils) are compiled from
# third_party/ocr-apps/apps/libs/src/ and linked into apps that need them.

# Every coherence configuration the app factory builds an executable for.  One
# list, referenced everywhere: a per-app copy of it silently omits new
# configurations from whichever apps forgot to grow.
set(ARTS_VARIANT_SUFFIXES
    ocr_val_wt ocr_val_wb ocr_val_wt_comb ocr_val_wb_comb
    ocr_excl_purge ocr_excl_retain wrf_val_wt ocr_inv_wt ocr_inv_wb)

set(OCR_APPS_ROOT "${CMAKE_SOURCE_DIR}/third_party/ocr-apps/apps")
set(OCR_LIBS_ROOT "${OCR_APPS_ROOT}/libs/src")
set(OCR_INC_DIR "${CMAKE_SOURCE_DIR}/third_party/ocr/ocr/inc")

# Common OCR extension defines (needed by most apps via OCR headers)
set(OCR_EXT_DEFINES
    ENABLE_EXTENSION_AFFINITY
    ENABLE_EXTENSION_PARAMS_EVT
    ENABLE_EXTENSION_LABELING
    ENABLE_EXTENSION_RTITF
    ENABLE_EXTENSION_CHANNEL_EVT
    ENABLE_EXTENSION_COUNTED_EVT
    # Gates the sharedDB benchmark's PD-distributed per-thread EDT affinity
    # (#if defined(...) in its main.c); the app's own Makefile defines it but
    # ARTS's CMake had omitted it, pinning every thread-EDT to rank0.
    # Referenced by RSBench_intel_sharedDB only (xsbench_dist distributes via
    # its SPMD fork instead).
    SINGLE_RUN_ACROSS_PD
)

# ==========================================================================
# Helper libraries (compiled as OBJECT libraries, linked into both backends)
# ==========================================================================

# --- reduction library ---
add_library(ocr_lib_reduction OBJECT
    ${OCR_LIBS_ROOT}/reduction/reduction.c
)
target_include_directories(ocr_lib_reduction PUBLIC
    ${OCR_LIBS_ROOT}/reduction
)
target_include_directories(ocr_lib_reduction PRIVATE
    ${OCR_LIBS_ROOT}/macros/macros
    ${OCR_INC_DIR}
)
# target_link_libraries(ocr_lib_reduction PRIVATE ocr_xsocr_static)
set_target_properties(ocr_lib_reduction PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(ocr_lib_reduction PRIVATE ${BENCH_C_OPT})
target_compile_definitions(ocr_lib_reduction PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
target_include_directories(ocr_lib_reduction PRIVATE ${BENCH_OCR_OPTIONS_DIR})

# --- reductionEager library ---
add_library(ocr_lib_reductionEager OBJECT
    ${OCR_LIBS_ROOT}/reductionEager/reductionEager.c
)
target_include_directories(ocr_lib_reductionEager PUBLIC
    ${OCR_LIBS_ROOT}/reductionEager
)
target_include_directories(ocr_lib_reductionEager PRIVATE
    ${OCR_LIBS_ROOT}/macros/macros
    ${OCR_INC_DIR}
)
# target_link_libraries(ocr_lib_reductionEager PRIVATE ocr_xsocr_static)
set_target_properties(ocr_lib_reductionEager PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(ocr_lib_reductionEager PRIVATE ${BENCH_C_OPT})
target_compile_definitions(ocr_lib_reductionEager PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
target_include_directories(ocr_lib_reductionEager PRIVATE ${BENCH_OCR_OPTIONS_DIR})

# --- timer library (x86) ---
add_library(ocr_lib_timer OBJECT
    ${OCR_LIBS_ROOT}/timer/x86/timer.c
)
target_include_directories(ocr_lib_timer PUBLIC
    ${OCR_LIBS_ROOT}/timer/x86
)
target_include_directories(ocr_lib_timer PRIVATE
    ${OCR_INC_DIR}
)
# target_link_libraries(ocr_lib_timer PRIVATE ocr_xsocr_static)
set_target_properties(ocr_lib_timer PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(ocr_lib_timer PRIVATE ${BENCH_C_OPT})
target_compile_definitions(ocr_lib_timer PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
target_include_directories(ocr_lib_timer PRIVATE ${BENCH_OCR_OPTIONS_DIR})

# --- reduction_v1 library (used by reduction/intel-chandra) ---
add_library(ocr_lib_reduction_v1 OBJECT
    ${OCR_LIBS_ROOT}/reduction/reduction_v1.c
)
target_include_directories(ocr_lib_reduction_v1 PUBLIC
    ${OCR_LIBS_ROOT}/reduction
)
target_include_directories(ocr_lib_reduction_v1 PRIVATE
    ${OCR_INC_DIR}
)
# target_link_libraries(ocr_lib_reduction_v1 PRIVATE ocr_xsocr_static)
set_target_properties(ocr_lib_reduction_v1 PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(ocr_lib_reduction_v1 PRIVATE ${BENCH_C_OPT})
target_compile_definitions(ocr_lib_reduction_v1 PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
target_include_directories(ocr_lib_reduction_v1 PRIVATE ${BENCH_OCR_OPTIONS_DIR})

# --- ocrAppUtils library ---
add_library(ocr_lib_ocrAppUtils OBJECT
    ${OCR_LIBS_ROOT}/ocrAppUtils/ocrAppUtils.c
)
target_include_directories(ocr_lib_ocrAppUtils PUBLIC
    ${OCR_LIBS_ROOT}/ocrAppUtils
)
target_include_directories(ocr_lib_ocrAppUtils PRIVATE
    ${OCR_INC_DIR}
)
# target_link_libraries(ocr_lib_ocrAppUtils PRIVATE ocr_xsocr_static)
set_target_properties(ocr_lib_ocrAppUtils PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(ocr_lib_ocrAppUtils PRIVATE ${BENCH_C_OPT})
target_compile_definitions(ocr_lib_ocrAppUtils PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
target_include_directories(ocr_lib_ocrAppUtils PRIVATE ${BENCH_OCR_OPTIONS_DIR})

# ==========================================================================
# OCR-VX helper library variants (compiled against OCR-VX headers / vdm.h)
# Only created when the ocr_vdm_mpi target exists.
# ==========================================================================
if(TARGET ocr_vdm_mpi)

add_library(ocr_lib_reduction_ocrvx OBJECT
    ${OCR_LIBS_ROOT}/reduction/reduction.c
)
target_include_directories(ocr_lib_reduction_ocrvx PUBLIC
    ${OCR_LIBS_ROOT}/reduction
)
target_include_directories(ocr_lib_reduction_ocrvx PRIVATE
    ${OCR_LIBS_ROOT}/macros/macros
    ${OCRVX_OCR_INC}
)
set_target_properties(ocr_lib_reduction_ocrvx PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(ocr_lib_reduction_ocrvx PRIVATE ${BENCH_C_OPT}
    "-include" "${OCRVX_COMPAT_H}")
target_compile_definitions(ocr_lib_reduction_ocrvx PRIVATE ${OCR_EXT_DEFINES} "OCR_TYPE_H=vdm.h")

add_library(ocr_lib_reductionEager_ocrvx OBJECT
    ${OCR_LIBS_ROOT}/reductionEager/reductionEager.c
)
target_include_directories(ocr_lib_reductionEager_ocrvx PUBLIC
    ${OCR_LIBS_ROOT}/reductionEager
)
target_include_directories(ocr_lib_reductionEager_ocrvx PRIVATE
    ${OCR_LIBS_ROOT}/macros/macros
    ${OCRVX_OCR_INC}
)
set_target_properties(ocr_lib_reductionEager_ocrvx PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(ocr_lib_reductionEager_ocrvx PRIVATE ${BENCH_C_OPT}
    "-include" "${OCRVX_COMPAT_H}")
target_compile_definitions(ocr_lib_reductionEager_ocrvx PRIVATE ${OCR_EXT_DEFINES} "OCR_TYPE_H=vdm.h")

add_library(ocr_lib_timer_ocrvx OBJECT
    ${OCR_LIBS_ROOT}/timer/x86/timer.c
)
target_include_directories(ocr_lib_timer_ocrvx PUBLIC
    ${OCR_LIBS_ROOT}/timer/x86
)
target_include_directories(ocr_lib_timer_ocrvx PRIVATE
    ${OCRVX_OCR_INC}
)
set_target_properties(ocr_lib_timer_ocrvx PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(ocr_lib_timer_ocrvx PRIVATE ${BENCH_C_OPT}
    "-include" "${OCRVX_COMPAT_H}")
target_compile_definitions(ocr_lib_timer_ocrvx PRIVATE ${OCR_EXT_DEFINES} "OCR_TYPE_H=vdm.h")

add_library(ocr_lib_reduction_v1_ocrvx OBJECT
    ${OCR_LIBS_ROOT}/reduction/reduction_v1.c
)
target_include_directories(ocr_lib_reduction_v1_ocrvx PUBLIC
    ${OCR_LIBS_ROOT}/reduction
)
target_include_directories(ocr_lib_reduction_v1_ocrvx PRIVATE
    ${OCRVX_OCR_INC}
)
set_target_properties(ocr_lib_reduction_v1_ocrvx PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(ocr_lib_reduction_v1_ocrvx PRIVATE ${BENCH_C_OPT}
    "-include" "${OCRVX_COMPAT_H}")
target_compile_definitions(ocr_lib_reduction_v1_ocrvx PRIVATE ${OCR_EXT_DEFINES} "OCR_TYPE_H=vdm.h")

add_library(ocr_lib_ocrAppUtils_ocrvx OBJECT
    ${OCR_LIBS_ROOT}/ocrAppUtils/ocrAppUtils.c
)
target_include_directories(ocr_lib_ocrAppUtils_ocrvx PUBLIC
    ${OCR_LIBS_ROOT}/ocrAppUtils
)
target_include_directories(ocr_lib_ocrAppUtils_ocrvx PRIVATE
    ${OCRVX_OCR_INC}
)
set_target_properties(ocr_lib_ocrAppUtils_ocrvx PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(ocr_lib_ocrAppUtils_ocrvx PRIVATE ${BENCH_C_OPT}
    "-include" "${OCRVX_COMPAT_H}")
target_compile_definitions(ocr_lib_ocrAppUtils_ocrvx PRIVATE ${OCR_EXT_DEFINES} "OCR_TYPE_H=vdm.h")

endif() # TARGET ocr_vdm_mpi

# ==========================================================================
# add_benchmark_app: builds an OCR app against both XSOCR and ARTS backends
#
# Usage:
#   add_benchmark_app(NAME <name>
#       DIR <relative-path-under-ocr-apps/apps>
#       SOURCES <file1.c> [file2.c ...]
#       [SRC_SUBDIR <subdir>]           # source files are in DIR/subdir/
#       [LIBS reduction timer ...]      # OCR helper library OBJECT targets
#       [EXTRA_INCLUDES <dir> ...]      # additional include dirs
#       [EXTRA_DEFINES <def> ...]       # additional compile definitions
#       [EXTRA_LINK_LIBS <lib> ...]     # extra link libraries (e.g. m)
#       [OCRVX_SKIP]                    # suppress _ocrvx target (app uses unsupported OCR extensions)
#       [HINTED_PLACEMENT]           # source carries a placement hint layer
#   )
#
# Creates, per backend, <name>_arts_<suffix> (one per coherence configuration),
# <name>_xsocr and (when compatible) <name>_ocrvx.  With HINTED_PLACEMENT the
# same set is created a second time under the name <name>_hinted, compiled with the
# hint layer enabled — so both placement settings coexist in one build tree and
# neither is a property of the build directory.
# ==========================================================================
function(add_benchmark_app)
    cmake_parse_arguments(APP "OCRVX_SKIP;HINTED_PLACEMENT" "NAME;DIR;SRC_SUBDIR" "SOURCES;LIBS;EXTRA_INCLUDES;EXTRA_DEFINES;EXTRA_LINK_LIBS" ${ARGN})

    set(APP_DIR "${OCR_APPS_ROOT}/${APP_DIR}")
    if(APP_SRC_SUBDIR)
        set(SRC_DIR "${APP_DIR}/${APP_SRC_SUBDIR}")
    else()
        set(SRC_DIR "${APP_DIR}")
    endif()

    # Build absolute source paths
    set(ABS_SOURCES "")
    foreach(src ${APP_SOURCES})
        list(APPEND ABS_SOURCES "${SRC_DIR}/${src}")
    endforeach()

    # Collect OBJECT library objects and include dirs for helper libs
    set(LIB_OBJECTS "")
    set(LIB_INCLUDES "")
    foreach(lib ${APP_LIBS})
        list(APPEND LIB_OBJECTS $<TARGET_OBJECTS:ocr_lib_${lib}>)
        get_target_property(inc ocr_lib_${lib} INTERFACE_INCLUDE_DIRECTORIES)
        if(inc)
            list(APPEND LIB_INCLUDES ${inc})
        endif()
    endforeach()

    # Common compile settings
    set(COMMON_INCLUDES
        ${APP_DIR}
        ${SRC_DIR}
        ${OCR_LIBS_ROOT}/macros/macros
        ${APP_EXTRA_INCLUDES}
        ${LIB_INCLUDES}
        ${BENCH_OCR_OPTIONS_DIR}
    )
    set(COMMON_DEFINES ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE} ${APP_EXTRA_DEFINES})

    # Placement flavors.  The base name compiles the application as born, with
    # its hint layer left out; the _hinted name compiles that layer in.  A flavor
    # differs only by a define in the shared application source, so every
    # backend below sees the same setting — placement is a property of the
    # program, never of one runtime.
    set(_flavors base)
    if(APP_HINTED_PLACEMENT)
        list(APPEND _flavors hinted)
    endif()

    foreach(_flavor ${_flavors})
        if(_flavor STREQUAL hinted)
            set(_app "${APP_NAME}_hinted")
            set(_flavor_define OCR_APP_OPTIMIZED_PLACEMENT)
        else()
            set(_app "${APP_NAME}")
            set(_flavor_define "")
        endif()

        # --- XSOCR variant (only when the ocr_xsocr_static target exists) ---
        if(TARGET ocr_xsocr_static)
            add_executable(${_app}_xsocr ${ABS_SOURCES} ${LIB_OBJECTS})
            target_include_directories(${_app}_xsocr PRIVATE ${COMMON_INCLUDES})
            target_compile_definitions(${_app}_xsocr PRIVATE ${COMMON_DEFINES} ${_flavor_define})
            target_link_libraries(${_app}_xsocr PRIVATE ocr_xsocr_static pthread ${APP_EXTRA_LINK_LIBS})
            # Non-PIE: OCR's distributed mode serializes function pointers as raw
            # addresses across MPI ranks.  With PIE + ASLR each rank gets a
            # different base address and deserialized funcPtrs become invalid.
            target_link_options(${_app}_xsocr PRIVATE -no-pie -fno-pie)
            set_target_properties(${_app}_xsocr PROPERTIES
                C_STANDARD 11 C_EXTENSIONS ON
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            )
            target_compile_options(${_app}_xsocr PRIVATE ${BENCH_C_OPT})
        endif()

        # --- ARTS variants (one per coherence configuration) ---
        foreach(sfx ${ARTS_VARIANT_SUFFIXES})
            add_executable(${_app}_arts_${sfx} ${ABS_SOURCES} ${LIB_OBJECTS}
                           $<TARGET_OBJECTS:ocr_arts_objects>)
            target_include_directories(${_app}_arts_${sfx} PRIVATE
                ${COMMON_INCLUDES} ${OCR_INC_DIR})
            target_compile_definitions(${_app}_arts_${sfx} PRIVATE
                ${COMMON_DEFINES} ${_flavor_define})
            target_link_libraries(${_app}_arts_${sfx} PRIVATE
                arts_static_${sfx} pthread ${APP_EXTRA_LINK_LIBS})
            target_link_options(${_app}_arts_${sfx} PRIVATE -no-pie -fno-pie)
            set_target_properties(${_app}_arts_${sfx} PROPERTIES
                C_STANDARD 11 C_EXTENSIONS ON
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
            target_compile_options(${_app}_arts_${sfx} PRIVATE ${BENCH_C_OPT})
            set_property(GLOBAL APPEND PROPERTY BENCHMARK_ARTS_TARGETS ${_app}_arts_${sfx})
        endforeach()

        # --- OCR-VX variant (only when ocr_vdm_mpi target exists, OCRVX_SKIP not set,
        #     and all required helper libs have an _ocrvx variant compiled against
        #     vdm.h headers) ---
        if(TARGET ocr_vdm_mpi AND NOT APP_OCRVX_SKIP)
            set(_ocrvx_ok TRUE)
            foreach(lib ${APP_LIBS})
                if(NOT TARGET ocr_lib_${lib}_ocrvx)
                    set(_ocrvx_ok FALSE)
                endif()
            endforeach()
            if(_ocrvx_ok)
                set(LIB_OBJECTS_OCRVX "")
                foreach(lib ${APP_LIBS})
                    list(APPEND LIB_OBJECTS_OCRVX $<TARGET_OBJECTS:ocr_lib_${lib}_ocrvx>)
                endforeach()

                add_executable(${_app}_ocrvx ${ABS_SOURCES} ${LIB_OBJECTS_OCRVX})
                target_include_directories(${_app}_ocrvx PRIVATE
                    ${APP_DIR}
                    ${SRC_DIR}
                    ${OCR_LIBS_ROOT}/macros/macros
                    ${APP_EXTRA_INCLUDES}
                    ${LIB_INCLUDES}
                    ${OCRVX_OCR_INC}
                )
                target_compile_definitions(${_app}_ocrvx PRIVATE
                    ${OCR_EXT_DEFINES}
                    "OCR_TYPE_H=vdm.h"
                    OCR_ENABLE_EDT_NAMING
                    ${APP_EXTRA_DEFINES}
                    ${_flavor_define}
                )
                target_link_libraries(${_app}_ocrvx PRIVATE
                    ocr_vdm_mpi ${APP_EXTRA_LINK_LIBS})
                target_link_options(${_app}_ocrvx PRIVATE -no-pie -fno-pie)
                set_target_properties(${_app}_ocrvx PROPERTIES
                    C_STANDARD 11 C_EXTENSIONS ON
                    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                )
                target_compile_options(${_app}_ocrvx PRIVATE ${BENCH_C_OPT}
                    "-include" "${OCRVX_COMPAT_H}")
            endif()
        endif()
    endforeach()
endfunction()

# ==========================================================================
# Tier 1: Simple single-file apps
# ==========================================================================

add_benchmark_app(NAME fibonacci
    DIR fibonacci/ocr
    SOURCES fib.c
    HINTED_PLACEMENT
)

add_benchmark_app(NAME printf
    DIR printf/ocr
    SOURCES printf.c
)

add_benchmark_app(NAME quicksort
    DIR quicksort/ocr
    SOURCES quicksort.c
)

# Registration held back: this program's source is not carried by the
# application submodule yet, so a checkout that has not produced it
# locally cannot configure.  Restore this block, and the application's
# restructured version in the driver's catalog, once the submodule
# carries the source.
#add_benchmark_app(NAME quicksort_dist
#    DIR quicksort/ocr
#    SOURCES quicksort_dist.c
#)

add_benchmark_app(NAME nqueens
    DIR nqueens/refactored/ocr
    SOURCES nqueens.c
    LIBS timer
    HINTED_PLACEMENT
)

add_benchmark_app(NAME smithwaterman
    DIR smithwaterman/ocr
    SOURCES smithwaterman.c
    HINTED_PLACEMENT
)

add_benchmark_app(NAME basicIO
    DIR basicIO/ocr
    SOURCES basicIO.c
)

# ==========================================================================
# Tier 2: Examples
# ==========================================================================

add_benchmark_app(NAME cache_offset
    DIR examples/cache-offset
    SOURCES cacheOffset.c
)

add_benchmark_app(NAME highbw
    DIR examples/highbw
    SOURCES highbw.c
    EXTRA_DEFINES NTHREADS=2 DBSIZE=1024 ITER=2
)

add_benchmark_app(NAME multigen
    DIR examples/multigen
    SOURCES multigen.c
)

add_benchmark_app(NAME multigen_2
    DIR examples/multigen_2
    SOURCES multigen.c
)

add_benchmark_app(NAME task_priorities
    DIR examples/task-priorities
    SOURCES task-priorities.c
)

add_benchmark_app(NAME testlibs
    DIR examples/testlibs
    SOURCES testlibs.c
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME xeonNumaSize
    DIR examples/xeonNumaSize
    SOURCES numaHelpers.c actions.c utils.c xeonNumaSize.c myOcrHelpers.c
    EXTRA_LINK_LIBS arts::numa
)

# ==========================================================================
# Tier 3: Kernels and utilities
# ==========================================================================

add_benchmark_app(NAME dbctrl
    DIR kernels/dbctrl/ocr
    SOURCES dbctrl.c
)

add_benchmark_app(NAME prodcon
    DIR kernels/prodcon/ocr
    SOURCES prodcon.c
    EXTRA_DEFINES NUM_THREADS=4 ARRAY_SIZE=4096 NTIMES=3
)

add_benchmark_app(NAME curvefit
    DIR curvefit/ocr/intel
    SOURCES curveFit.c
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME triangle
    DIR triangle/refactored/ocr/intel
    SOURCES triangle.c
    HINTED_PLACEMENT
    # COUNTED_EVENTS states the completion events' true consumer counts (the
    # program knows both before creating them).  The OCR spec destroys a ONCE
    # once its consumers are satisfied, so declaring the count changes nothing
    # a spec-conforming runtime does -- it hands a runtime that keeps fired
    # events for late binders the reclaim point the spec's wording assumes.
    # (The board destroys are unconditional in the source: freeing an
    # unreachable block is the program's own obligation, not an option.)
    EXTRA_DEFINES OCR_APP_COUNTED_EVENTS
)

add_benchmark_app(NAME tempest
    DIR tempest/refactored/ocr/intel-bryan
    SOURCES tempestCommunication.c
    EXTRA_DEFINES DURATION=4400
    HINTED_PLACEMENT
)

# ==========================================================================
# Tier 4: Medium-complexity apps
# ==========================================================================

add_benchmark_app(NAME cholesky
    DIR cholesky/ocr
    SOURCES cholesky.c
    EXTRA_LINK_LIBS m
    HINTED_PLACEMENT
)

# The app tree's standalone input/output converters — plain host C, no OCR.
# convertData turns a whitespace-text matrix into the tile-stream binary the
# app's --fib path (and the TG build) reads; convertOut turns the app's
# binary result file back into readable text.
add_executable(cholesky_convertData
    "${OCR_APPS_ROOT}/cholesky/ocr/convertData.c")
add_executable(cholesky_convertOut
    "${OCR_APPS_ROOT}/cholesky/ocr/convertOut.c")
set_target_properties(cholesky_convertData cholesky_convertOut PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Tiled Cholesky whose per-tile kernels are the standard CBLAS/LAPACKE calls
# (dpotrf/dtrsm/dsyrk/dgemm), served by the third_party OpenBLAS submodule.
add_benchmark_app(NAME cholesky_blas
    DIR cholesky/ocr-mkl
    SOURCES ocr_mkl_cholesky.c
    EXTRA_LINK_LIBS arts::openblas m
    HINTED_PLACEMENT
)

add_benchmark_app(NAME fft
    DIR fft/ocr
    SOURCES fft.c verify.c
    EXTRA_LINK_LIBS m
)

# Registration held back: this program's source is not carried by the
# application submodule yet, so a checkout that has not produced it
# locally cannot configure.  Restore this block, and the application's
# restructured version in the driver's catalog, once the submodule
# carries the source.
#add_benchmark_app(NAME fft_dist
#    DIR fft/ocr
#    SOURCES fft_dist.c
#    EXTRA_LINK_LIBS m
#)

add_benchmark_app(NAME graph500
    DIR graph500
    SOURCES graph500.c
    EXTRA_DEFINES NO_FILES NO_MAP
    EXTRA_LINK_LIBS m
)

# GlobalSum - 3 variants (different main files, shared gsLib.c)
add_benchmark_app(NAME globalsum_cgShim
    DIR globalsum/refactored/ocr/intel
    SOURCES cgShim.c gsLib.c
)

add_benchmark_app(NAME globalsum_cgNoShim
    DIR globalsum/refactored/ocr/intel
    SOURCES cgNoShim.c gsLib.c
)

add_benchmark_app(NAME globalsum_pcg
    DIR globalsum/refactored/ocr/intel
    SOURCES pcg.c gsLib.c
)

# ==========================================================================
# Tier 5: CoMD variants
# ==========================================================================

add_benchmark_app(NAME CoMD_intel_chandra
    DIR CoMD/refactored/ocr/intel-chandra
    SOURCES CoMD.c wtime.c command.c lj.c eam.c timers.c
    LIBS reduction
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME CoMD_intel_chandra_tiled
    DIR CoMD/refactored/ocr/intel-chandra-tiled
    EXTRA_DEFINES OCR_APP_COUNTED_OEVT
    SOURCES CoMD.c mycommand.c cmdLineParser.c eam.c ljForce.c decomposition.c
    linkCells.c initAtoms.c random.c timestep.c performanceTimers.c
    haloExchange.c
    LIBS reduction ocrAppUtils
    EXTRA_DEFINES WITH_COUNTED_EVT CHANNEL_EVENTS_AT_RECEIVER DOUBLE_BUFFERED_EVTS
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME CoMD_sdsc
    DIR CoMD/refactored/ocr/sdsc
    SOURCES atoms.c cells.c comd.c command.c eam.c lj.c simulation.c timers.c
    timestep.c
    LIBS ocrAppUtils
    EXTRA_DEFINES OCR_APP_COUNTED_OEVT
    EXTRA_LINK_LIBS m
    HINTED_PLACEMENT
)

add_benchmark_app(NAME CoMD_sdsc2
    DIR CoMD/refactored/ocr/sdsc2
    SOURCES comd.c command.c eam.c lj.c reductions.c simulation.c timers.c
    timestep.c
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# Tier 6: HPCG variants
# ==========================================================================

add_benchmark_app(NAME hpcg_intel
    DIR hpcg/refactored/ocr/intel
    SOURCES hpcg.c timers.c
    LIBS reduction
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME hpcg_intel_Eager
    DIR hpcg/refactored/ocr/intel-Eager
    SOURCES hpcgEager.c timers.c
    LIBS reductionEager timer
    EXTRA_LINK_LIBS m
)

# hpcg Eager-Collective uses ocrAddDependenceSlot and ocrEventCollectiveSatisfySlot.
# These rely on MULTI_OUTPUT_SLOT + COLLECTIVE_EVT OCR extensions.  Both are
# now enabled in the xsocr config (ocr_config_x86.h) and the oddContribs
# vendor bug in hc-event.c is fixed, so both backends build and run this app.
add_benchmark_app(NAME hpcg_intel_Eager_Collective
    DIR hpcg/refactored/ocr/intel-Eager-Collective
    SOURCES hpcgEagerRedevt.c timers.c
    LIBS timer
    EXTRA_DEFINES ENABLE_EXTENSION_COLLECTIVE_EVT ENABLE_EXTENSION_MULTI_OUTPUT_SLOT
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# Tier 7: Stencil variants
# ==========================================================================

# The chandra stencil source is dimension-parameterized (PROBLEM_TYPE 1|2)
# and the upstream Stencil1D port left the header's default (2) in place, so
# this target used to build the SAME 2-D problem as Stencil2D_intel_chandra.
# Select the real 1-D problem here.
add_benchmark_app(NAME Stencil1D_intel_chandra
    DIR Stencil1D/refactored/ocr/intel-chandra
    SOURCES stencil_1d.c timers.c
    LIBS reduction
    EXTRA_DEFINES PROBLEM_TYPE=1
    EXTRA_LINK_LIBS m
)

# Stencil1D/intel-david: multiple code variants (each is a standalone app),
# one per event/dependence idiom (sticky / stickyLG labeled-guid / channel /
# once / oncePI / guid / guidPI).  All ported to the current OCR API
# (ocrGuidIsEq/ocrGuidIsNull instead of struct ==/!=, hint-bearing
# ocrEdtCreate/ocrDbCreate signatures).
add_benchmark_app(NAME stencil1D_sticky
    DIR Stencil1D/refactored/ocr/intel-david
    SOURCES stencil1Dsticky.c
)


add_benchmark_app(NAME stencil1D_channel
    DIR Stencil1D/refactored/ocr/intel-david
    SOURCES stencil1Dchannel.c
)

add_benchmark_app(NAME stencil1D_stickyLG
    DIR Stencil1D/refactored/ocr/intel-david
    SOURCES stencil1DstickyLG.c
)

add_benchmark_app(NAME stencil1D_guid
    DIR Stencil1D/refactored/ocr/intel-david
    SOURCES stencil1Dguid.c
)

add_benchmark_app(NAME stencil1D_guidPI
    DIR Stencil1D/refactored/ocr/intel-david
    SOURCES stencil1DguidPI.c
)

add_benchmark_app(NAME stencil1D_once
    DIR Stencil1D/refactored/ocr/intel-david
    SOURCES stencil1Donce.c
)

add_benchmark_app(NAME stencil1D_oncePI
    DIR Stencil1D/refactored/ocr/intel-david
    SOURCES stencil1DoncePI.c
)





add_benchmark_app(NAME Stencil2D_intel_chandra
    DIR Stencil2D/refactored/ocr/intel-chandra
    SOURCES stencil_2d.c timers.c
    LIBS reduction
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME Stencil2D_intel_channelEVTs
    DIR Stencil2D/refactored/ocr/intel-channelEVTs
    SOURCES stencil_2d.c timers.c
    LIBS reduction ocrAppUtils timer
    EXTRA_DEFINES STENCIL_WITH_DBUF_CHRECV CHANNEL_EVENTS_AT_RECEIVER
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# Tier 8: P2P and Reduction
# ==========================================================================

add_benchmark_app(NAME p2p
    DIR p2p/refactored/ocr/intel
    SOURCES p2p.c wtime.c
    LIBS reduction
    HINTED_PLACEMENT
)

add_benchmark_app(NAME reduction_intel
    DIR reduction/refactored/ocr/intel
    SOURCES driver.c
    LIBS reduction timer
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME reduction_intel_chandra
    DIR reduction/refactored/ocr/intel-chandra
    SOURCES driver.c
    LIBS reduction_v1
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# Tier 9: HPGMG
# ==========================================================================

add_benchmark_app(NAME hpgmg
    DIR hpgmg/refactored/ocr/sdsc
    SOURCES hpgmg.c init.c utils.c mg.c mg_edt.c solve_edt.c test_edt.c
    residual_edt.c restrict_edt.c smooth_edt.c interpolate_edt.c
    EXTRA_LINK_LIBS m
    HINTED_PLACEMENT
)

# Restructured decomposition: a separate program (its own main, lattice
# driver and face-slab exchange) that links the base port's kernel and
# per-box task sources unchanged — rank-persistent data plane, point-to-point
# phase chains, distributed initialization.  Placement is part of the design,
# so the layer is compiled in unconditionally (no _hinted family).
add_benchmark_app(NAME hpgmg_dist
    DIR hpgmg/refactored/ocr/sdsc
    SOURCES hpgmg_dist_main.c mg_dist.c exchange_dist.c
    init.c utils.c mg_edt.c solve_edt.c
    residual_edt.c restrict_edt.c smooth_edt.c interpolate_edt.c
    EXTRA_DEFINES OCR_APP_OPTIMIZED_PLACEMENT
    EXTRA_LINK_LIBS m
)


# ==========================================================================
# Tier 10: MiniAMR variants
# ==========================================================================

add_benchmark_app(NAME miniAMR_forkbomb
    DIR miniAMR/refactored/ocr/forkbomb
    SOURCES mainOCR.c
    EXTRA_DEFINES FORKBOMB_MAX_TIMESTEPS=3
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME miniAMR_intel
    DIR miniAMR/refactored/ocr/intel
    SOURCES block.c chksum.c comm.c clone.c init.c move.c parent.c plot.c
    stencil.c refine.c root.c util.c wtime.c
    EXTRA_LINK_LIBS m
    HINTED_PLACEMENT
)

add_benchmark_app(NAME miniAMR_intel_bryan
    DIR miniAMR/refactored/ocr/intel-bryan
    SOURCES mainOCR.c
    LIBS ocrAppUtils
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME miniAMR_intel_chandra
    EXTRA_DEFINES OCR_APP_DESTROY_REDUCTION_TEMPS OCR_APP_COUNTED_OEVT
    DIR miniAMR/refactored/ocr/intel-chandra
    SOURCES main.c init.c check_sum.c refine.c move.c block.c comm_refine.c
    comm_parent.c driver.c comm.c comm_utils.c stencil.c util.c
    load_balance.c profile.c
    LIBS reduction ocrAppUtils timer
    # VERIFICATION_RUN: the app's own deterministic-init mode.  The default
    # init fills cells from one global unseeded rand() stream that worker
    # threads consume concurrently, so every run (same binary, same input)
    # produces different checksums — unusable for cross-runtime consensus.
    # The verification init is data-identical on every runtime and run while
    # the stencil work (data-independent FLOPs) is unchanged.
    EXTRA_DEFINES VERIFICATION_RUN
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# Tier 11: Nekbone, NPB-CG
# ==========================================================================

add_benchmark_app(NAME nekbone
    DIR nekbone/refactored/ocr_src
    SOURCES fastfcn.c neko_reduction.c timing.c nekbone_cg.c ax.c local_grad.c
    set_f.c blas1.c blas3.c polybasis.c set_vert_box.c cubic.c
    init_mesh.c nekbone_setup.c app_ocr_util.c integer_log2.c
    neko_globals.c nekos_triplet.c nekos_cubiclattice.c nekos_tools.c
    neko_halo_util.c neko_halo.c spmd_global_data.c tailrecursion.c
    treeForkJoin.c z_nekbone_inOcr.c
    LIBS reduction
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME npb_cg
    DIR npb-cg/sdsc-ocr
    SOURCES cg_edt.c cg_graph.c cg_ocr.c main_edt.c makea_ocr.c util_ocr.c
    EXTRA_LINK_LIBS m
)

# Restructured decomposition: a separate program (row-band SPMD chains,
# channel-event fragment exchange, library reductions) sharing the matrix
# generator and utilities; its placement is internal, so no _hinted family.
add_benchmark_app(NAME npb_cg_dist
    DIR npb-cg/sdsc-ocr
    SOURCES cg_dist_edt.c cg_dist_makea.c makea_ocr.c util_ocr.c
    LIBS reduction
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# Tier 12: RSBench, XSBench
# ==========================================================================

add_benchmark_app(NAME RSBench_intel
    HINTED_PLACEMENT
    DIR RSBench/refactored/ocr/intel
    SRC_SUBDIR src
    SOURCES main.c init.c material.c io.c timers.c utils.c rs_kernel.c
    LIBS timer
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME RSBench_intel_sharedDB
    DIR RSBench/refactored/ocr/intel-sharedDB
    SRC_SUBDIR src
    SOURCES main.c init.c material.c io.c utils.c rs_kernel.c timers.c
    LIBS timer reduction ocrAppUtils
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME XSBench_intel
    HINTED_PLACEMENT
    DIR XSBench/refactored/ocr/intel
    SRC_SUBDIR src
    SOURCES Main.c io.c CalculateXS.c XSutils.c Materials.c qsort.c timers.c
    LIBS timer
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME XSBench_intel_sharedDB
    DIR XSBench/refactored/ocr/intel-sharedDB
    SRC_SUBDIR src
    SOURCES Main.c io.c GridInit.c CalculateXS.c XSutils.c Materials.c timers.c
    LIBS timer reduction ocrAppUtils
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# Tier 13: LCS variants
# ==========================================================================

add_benchmark_app(NAME LCS_distributed_ST
    DIR LCS/refactored/ocr/intel-jesmin-lcs_distributed_ST_datablocks
    SOURCES lcs_ST_distributed.c
    LIBS reduction timer
    EXTRA_LINK_LIBS m
)
# LCS uses bare "inline" (C11 = no external def) → fails at -O0; force GNU89 semantics
if(TARGET LCS_distributed_ST_xsocr)
    target_compile_options(LCS_distributed_ST_xsocr PRIVATE -fgnu89-inline)
endif()
foreach(sfx ${ARTS_VARIANT_SUFFIXES})
    target_compile_options(LCS_distributed_ST_arts_${sfx} PRIVATE -fgnu89-inline)
endforeach()
if(TARGET LCS_distributed_ST_ocrvx)
    target_compile_options(LCS_distributed_ST_ocrvx PRIVATE -fgnu89-inline)
endif()

add_benchmark_app(NAME LCS_shared
    DIR LCS/refactored/ocr/intel-jesmin-lcs_shared_datablocks
    SOURCES lcs.c
    LIBS reduction timer
    EXTRA_LINK_LIBS m
)
if(TARGET LCS_shared_xsocr)
    target_compile_options(LCS_shared_xsocr PRIVATE -fgnu89-inline)
endif()
foreach(sfx ${ARTS_VARIANT_SUFFIXES})
    target_compile_options(LCS_shared_arts_${sfx} PRIVATE -fgnu89-inline)
endforeach()
if(TARGET LCS_shared_ocrvx)
    target_compile_options(LCS_shared_ocrvx PRIVATE -fgnu89-inline)
endif()

# LCS C++ variant: standard OCR API written in C++.  mainEdt is declared
# extern "C" and ocr.h is extern "C"-guarded, so CMake compiles the .cpp with
# the C++ toolchain (project enables CXX17) and links the C OCR libraries.
# cilktime.h here is a plain gettimeofday timer, not the Cilk runtime.
add_benchmark_app(NAME LCS_all_db_distributed
    DIR LCS/refactored/ocr/intel-jesmin-lcs_all_db_distributed
    SOURCES lcs_distributed.cpp
    EXTRA_LINK_LIBS m
    HINTED_PLACEMENT
)

# Wavefront restructuring of the same computation: boundary strips travel on
# labeled counted events, leaves spawn per row and run in anti-diagonal
# order, memory follows the frontier.
add_benchmark_app(NAME LCS_wavefront
    DIR LCS/refactored/ocr/intel-jesmin-lcs_all_db_distributed
    SOURCES lcs_wavefront.c
    LIBS timer
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# Tier 14: SAR (problem_size_scaling, over sources shared from sar/ocr/src/)
# ==========================================================================
#
#   1. crlibm: extract tarball → configure → RAG patches → compile
#   2. Link the SAR executable against crlibm
#
# The one variant reads its data files and its parameter file at runtime, so
# the problem size is an argument rather than a build.

set(SAR_ROOT "${OCR_APPS_ROOT}/sar")
set(SAR_SRC_DIR "${SAR_ROOT}/ocr/src")
set(CRLIBM_PKG_DIR "${OCR_LIBS_ROOT}/crlibm-1.0beta5")
set(CRLIBM_TARBALL "${CRLIBM_PKG_DIR}/tar/crlibm-1.0beta5.tgz")
set(CRLIBM_DIR "${CMAKE_CURRENT_BINARY_DIR}/crlibm")

# ---------------------------------------------------------------------------
# Step 1: Extract and configure crlibm at configure time
# ---------------------------------------------------------------------------
# The tarball is in the submodule; x86_64/ only has RAG-install.sh tracked.
# We extract into the build directory so the source tree stays clean.
if(EXISTS "${CRLIBM_TARBALL}")
    if(NOT EXISTS "${CRLIBM_DIR}/crlibm.h")
        message(STATUS "SAR: Extracting crlibm tarball...")
        file(MAKE_DIRECTORY "${CRLIBM_DIR}")
        execute_process(
            COMMAND tar --gzip --extract --strip-components=1
            --file=${CRLIBM_TARBALL}
            WORKING_DIRECTORY "${CRLIBM_DIR}"
            RESULT_VARIABLE _tar_result
        )
        if(NOT _tar_result EQUAL 0)
            message(FATAL_ERROR "SAR: Failed to extract crlibm tarball")
        endif()

        # Run autotools configure (generates crlibm_config.h)
        message(STATUS "SAR: Configuring crlibm...")
        execute_process(
            COMMAND ./configure
            --without-gmp --without-mpfr --without-ultim --without-mcr
            --disable-gmp --disable-mpfr --disable-ultim --disable-mcr
            --enable-sse2 --disable-dependency-tracking
            --disable-double-extended --prefix=/dev/null
            WORKING_DIRECTORY "${CRLIBM_DIR}"
            RESULT_VARIABLE _cfg_result
        )
        if(NOT _cfg_result EQUAL 0)
            message(FATAL_ERROR "SAR: Failed to configure crlibm")
        endif()

        # Apply RAG patches to crlibm_config.h (disable system headers for
        # portability — matches x86_64/RAG-install.sh)
        message(STATUS "SAR: Applying RAG patches to crlibm_config.h...")
        set(_config_h "${CRLIBM_DIR}/crlibm_config.h")
        file(READ "${_config_h}" _cfg_content)
        string(REPLACE "#define CRLIBM_HAS_FPU_CONTROL 1" "/* #undef CRLIBM_HAS_FPU_CONTROL -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAS_SSE2 1" "/* #undef HAS_SSE2 -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_FENV_H 1" "/* #undef HAVE_FENV_H -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_FLOAT_H 1" "/* #undef HAVE_FLOAT_H -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_LIBM 1" "/* #undef HAVE_LIBM -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_LONG_DOUBLE 1" "/* #undef HAVE_LONG_DOUBLE -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_LONG_DOUBLE_WIDER 1" "/* #undef HAVE_LONG_DOUBLE_WIDER -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_MEMORY_H 1" "/* #undef HAVE_MEMORY_H -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_STRINGS_H 1" "/* #undef HAVE_STRINGS_H -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_STRING_H 1" "/* #undef HAVE_STRING_H -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_SYS_STAT_H 1" "/* #undef HAVE_SYS_STAT_H -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_SYS_TYPES_H 1" "/* #undef HAVE_SYS_TYPES_H -- RAG */" _cfg_content "${_cfg_content}")
        string(REPLACE "#define HAVE_UNISTD_H 1" "/* #undef HAVE_UNISTD_H -- RAG */" _cfg_content "${_cfg_content}")
        file(WRITE "${_config_h}" "${_cfg_content}")

        # Patch crlibm_private.c: guard FPU control with CRLIBM_HAS_FPU_CONTROL
        set(_priv_c "${CRLIBM_DIR}/crlibm_private.c")
        file(READ "${_priv_c}" _priv_content)
        string(REPLACE
            "#if (defined(CRLIBM_TYPECPU_X86) || defined(CRLIBM_TYPECPU_AMD64))"
            "#if defined(CRLIBM_HAS_FPU_CONTROL) && (defined(CRLIBM_TYPECPU_X86) || defined(CRLIBM_TYPECPU_AMD64)) // RAG -- added CRLIBM_HAS_FPU_CONTROL"
            _priv_content "${_priv_content}")
        file(WRITE "${_priv_c}" "${_priv_content}")

        # Patch asincos.c: silence unused variable warning
        set(_asin_c "${CRLIBM_DIR}/asincos.c")
        file(READ "${_asin_c}" _asin_content)
        string(REPLACE
            "  double deltatemp, deltah, deltal;"
            "  double deltatemp,deltah, __attribute__ ((unused)) deltal; // RAG -- avoid fatal warning"
            _asin_content "${_asin_content}")
        file(WRITE "${_asin_c}" "${_asin_content}")

        message(STATUS "SAR: crlibm extraction and patching complete")
    endif()
endif()

# ---------------------------------------------------------------------------
# Step 1b: Build crlibm static library
# ---------------------------------------------------------------------------
if(EXISTS "${CRLIBM_DIR}/crlibm.h")
    # Explicit source list — file(GLOB) picks up Itanium/double-extended files
    set(CRLIBM_SRCS
        ${CRLIBM_DIR}/crlibm_private.c
        ${CRLIBM_DIR}/triple-double.c
        ${CRLIBM_DIR}/exp-td.c
        ${CRLIBM_DIR}/exp-td-standalone.c
        ${CRLIBM_DIR}/expm1-standalone.c
        ${CRLIBM_DIR}/expm1.c
        ${CRLIBM_DIR}/log.c
        ${CRLIBM_DIR}/log1p.c
        ${CRLIBM_DIR}/rem_pio2_accurate.c
        ${CRLIBM_DIR}/trigo_fast.c
        ${CRLIBM_DIR}/trigo_accurate.c
        ${CRLIBM_DIR}/trigpi.c
        ${CRLIBM_DIR}/asincos.c
        ${CRLIBM_DIR}/pow.c
        ${CRLIBM_DIR}/atan_fast.c
        ${CRLIBM_DIR}/atan_accurate.c
        ${CRLIBM_DIR}/csh_fast.c
    )
    add_library(ocr_lib_crlibm STATIC ${CRLIBM_SRCS})
    target_include_directories(ocr_lib_crlibm PUBLIC ${CRLIBM_DIR})
    set_target_properties(ocr_lib_crlibm PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
    target_compile_definitions(ocr_lib_crlibm PRIVATE HAVE_CONFIG_H)
    target_compile_options(ocr_lib_crlibm PRIVATE ${BENCH_C_OPT})
    # crlibm needs scs_lib
    set(SCS_DIR "${CRLIBM_DIR}/scs_lib")
    if(EXISTS "${SCS_DIR}")
        file(GLOB SCS_SRCS "${SCS_DIR}/*.c")
        target_sources(ocr_lib_crlibm PRIVATE ${SCS_SRCS})
        target_include_directories(ocr_lib_crlibm PRIVATE ${SCS_DIR})
    endif()
    set(HAS_CRLIBM TRUE)
else()
    message(WARNING "SAR: crlibm tarball not found at ${CRLIBM_TARBALL} — "
        "SAR benchmarks will be skipped")
    set(HAS_CRLIBM FALSE)
endif()

# ---------------------------------------------------------------------------
# The RAG_IMPLICIT_INPUTS sizes are gone (2026-08-24).  tiny/small/medium/large
# compiled their datasets into the binary through a datagen -> dataImage ->
# .incbin pipeline, which is why they took no arguments: each was one rung of
# the problem_size_scaling ladder frozen at build time.  The ladder itself is
# Parameter files differing only in the image grid (Ix = 400*(n+1)), so the pss
# target with a parameter file covers every one of them and the four binaries,
# their host utilities and their generated data objects buy nothing.  The same
# rule extends the ladder: Parameter41 is 16800 a side.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 4: Build SAR executables
# ---------------------------------------------------------------------------
set(SAR_SOURCES
    main.c inputs.c digspot.c registration.c cfar.c sinc_interp.c sinc.c
    image_formation.c back_proj.c ccd.c fftw3.c rag_ocr.c
)

# DEBUG_SSCP (per-variant upstream Makefile default for every size but
# problem_size_scaling) is left out for ALL sizes: it dumps the full
# image/correlation planes to cwd-relative .bin files (hundreds of MB per
# run) and routes shutdown through the dump EDT — debug-only output that has
# no place in measured runs.  Define it manually when the dumps are needed.
set(SAR_DEFINES
    RAG_TG_ARCH_NULL_GUID_WORKAROUND
    RAG_NEW_BLK_SIZE=32
    TRACE
    RAG_DIG_SPOT_ON=0
    RAG_AFFINE_ON=1
    RAG_QSORT_ON=0
    RAG_PURE_FLOAT
    RAG_SINCOS
    RAG_CRLIBM
    RAG_PETER_DIST_AND_TRIG
    RAG_SPAD
    RAG_DRAM
    GANESH_STRENGTH_RED_OPT
)

if(HAS_CRLIBM)
    set(SAR_DIST_DEFINES ${SAR_DEFINES})
    list(REMOVE_ITEM SAR_DIST_DEFINES RAG_NEW_BLK_SIZE=32)
    list(APPEND SAR_DIST_DEFINES RAG_NEW_BLK_SIZE=50)
    # The detection pass computes the correlation values its own windows need
    # instead of reading them from a whole-image map, so its blocking factor
    # is its own: the halo it recomputes is (blk+Ncfar-1)^2/blk^2.
    list(APPEND SAR_DIST_DEFINES RAG_CFAR_BLK_SIZE=100)

    # The resample interpolates one pixel per iteration, so a projection-sized
    # block holds microseconds of work; this phase blocks coarser.
    list(APPEND SAR_DIST_DEFINES RAG_AFFINE_BLK_SIZE=250)

    # --- Restructured: sar_dist ---
    # It owns only the files whose decomposition it changes; the rest of the
    # pipeline is the shared source, compiled once more.  ccd.c is absent
    # because the correlation pass is fused into the detection pass.
    set(SAR_DIST_SOURCES
        main.c cfar.c back_proj.c registration.c image_formation.c blocks.c
        ../src/inputs.c ../src/digspot.c ../src/sinc_interp.c ../src/sinc.c
        ../src/fftw3.c ../src/rag_ocr.c
    )
    add_benchmark_app(NAME sar_dist
        DIR sar/ocr/dist
        SRC_SUBDIR .
        SOURCES ${SAR_DIST_SOURCES}
        EXTRA_INCLUDES ${SAR_ROOT}/ocr/problem_size_scaling ${SAR_SRC_DIR} ${CRLIBM_DIR}
        EXTRA_DEFINES ${SAR_DIST_DEFINES}
        EXTRA_LINK_LIBS m ocr_lib_crlibm
    )

    # --- Base: the one runtime-input variant ---
    # sar/ocr/huge carries byte-identical sources (main.c, common.h, argv.h) and
    # differs solely in the Parameters.txt it ships, whose grid is this ladder's
    # tenth rung, so it is not a target of its own either.
    # This variant's own Makefile.x86 sets the blocking factor to 50 (the
    # shared list carries the other variants' 32).
    set(SAR_RT_DEFINES ${SAR_DEFINES})
    list(REMOVE_ITEM SAR_RT_DEFINES RAG_NEW_BLK_SIZE=32)
    list(APPEND SAR_RT_DEFINES RAG_NEW_BLK_SIZE=50)
    # No HINTED_PLACEMENT: measured, a placement layer for this program either
    # collapses the pipeline onto one rank (97% of the work on one of four) or,
    # written honestly, runs 23-36% SLOWER than no hints at all.  The
    # serialisation is on a shared whole image block, which affinity cannot
    # address.
    add_benchmark_app(NAME sar_problem_size_scaling
        DIR sar/ocr/problem_size_scaling
        SRC_SUBDIR ../src
        SOURCES ${SAR_SOURCES}
        EXTRA_INCLUDES ${SAR_SRC_DIR} ${CRLIBM_DIR}
        EXTRA_DEFINES ${SAR_RT_DEFINES}
        EXTRA_LINK_LIBS m ocr_lib_crlibm
    )
endif()


# ==========================================================================
# Tier 15: STREAM memory bandwidth, UTS tree search
# ==========================================================================

add_benchmark_app(NAME stream
    DIR stream/ocr
    SOURCES stream.c
    # No NTIMES here: stream.c hardcodes it unconditionally, so a -D value
    # never took effect — its measured scalar (a[0] after NTIMES sweeps)
    # was always the source default's.
    EXTRA_DEFINES STREAM_ARRAY_SIZE=1000 NUM_THREADS=2
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME stream_dist
    DIR stream/ocr
    SOURCES stream_dist.c
    EXTRA_DEFINES STREAM_ARRAY_SIZE=1000 NUM_THREADS=2 NTIMES=3
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME stream_org
    DIR stream/ocr
    SOURCES stream_org.c
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME stream_sa
    DIR stream/ocr
    SOURCES stream_sa.c
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME uts
    DIR uts-1.1/ocr
    SOURCES uts.c brg_sha1.c
    EXTRA_DEFINES BRG_RNG STATS
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# release_sync_stress  (bespoke: source lives in this repo, not in ocr-apps)
# Probes whether ocrDbRelease is a synchronizing operation across runtimes.
# ==========================================================================
set(RSS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/release_sync_stress.c")

if(TARGET ocr_xsocr_static)
    add_executable(release_sync_stress_xsocr ${RSS_SRC})
    target_include_directories(release_sync_stress_xsocr PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR})
    target_compile_definitions(release_sync_stress_xsocr PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
    target_link_libraries(release_sync_stress_xsocr PRIVATE ocr_xsocr_static pthread)
    target_link_options(release_sync_stress_xsocr PRIVATE -no-pie -fno-pie)
    set_target_properties(release_sync_stress_xsocr PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(release_sync_stress_xsocr PRIVATE ${BENCH_C_OPT})
endif()

foreach(sfx ${ARTS_VARIANT_SUFFIXES})
    add_executable(release_sync_stress_arts_${sfx} ${RSS_SRC}
                   $<TARGET_OBJECTS:ocr_arts_objects>)
    target_include_directories(release_sync_stress_arts_${sfx} PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR} ${OCR_INC_DIR})
    target_compile_definitions(release_sync_stress_arts_${sfx} PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
    target_link_libraries(release_sync_stress_arts_${sfx} PRIVATE arts_static_${sfx} pthread)
    target_link_options(release_sync_stress_arts_${sfx} PRIVATE -no-pie -fno-pie)
    set_target_properties(release_sync_stress_arts_${sfx} PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(release_sync_stress_arts_${sfx} PRIVATE ${BENCH_C_OPT})
endforeach()

if(TARGET ocr_vdm_mpi)
    add_executable(release_sync_stress_ocrvx ${RSS_SRC})
    target_include_directories(release_sync_stress_ocrvx PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${OCRVX_OCR_INC})
    target_compile_definitions(release_sync_stress_ocrvx PRIVATE
        ${OCR_EXT_DEFINES} "OCR_TYPE_H=vdm.h" OCR_ENABLE_EDT_NAMING)
    target_link_libraries(release_sync_stress_ocrvx PRIVATE ocr_vdm_mpi)
    target_link_options(release_sync_stress_ocrvx PRIVATE -no-pie -fno-pie)
    set_target_properties(release_sync_stress_ocrvx PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(release_sync_stress_ocrvx PRIVATE ${BENCH_C_OPT}
        "-include" "${OCRVX_COMPAT_H}")
endif()

# ==========================================================================
# dbcreate_matrix  (bespoke: source lives in this repo, not in ocr-apps)
# Probes the DB-create capability matrix -- (creator-acquisition) x (home-
# determination: affinity / labeled / no-hint) x (locality) -- through the
# portable OCR API, so the SAME source builds for every runtime and the
# correctness harness can compare their create-path capability.  Prints
# CELLS_OK=<k>/12.
# ==========================================================================
set(DBCM_SRC "${CMAKE_CURRENT_SOURCE_DIR}/dbcreate_matrix.c")

if(TARGET ocr_xsocr_static)
    add_executable(dbcreate_matrix_xsocr ${DBCM_SRC})
    target_include_directories(dbcreate_matrix_xsocr PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR})
    target_compile_definitions(dbcreate_matrix_xsocr PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
    target_link_libraries(dbcreate_matrix_xsocr PRIVATE ocr_xsocr_static pthread)
    target_link_options(dbcreate_matrix_xsocr PRIVATE -no-pie -fno-pie)
    set_target_properties(dbcreate_matrix_xsocr PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(dbcreate_matrix_xsocr PRIVATE ${BENCH_C_OPT})
endif()

foreach(sfx ${ARTS_VARIANT_SUFFIXES})
    add_executable(dbcreate_matrix_arts_${sfx} ${DBCM_SRC}
                   $<TARGET_OBJECTS:ocr_arts_objects>)
    target_include_directories(dbcreate_matrix_arts_${sfx} PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR} ${OCR_INC_DIR})
    target_compile_definitions(dbcreate_matrix_arts_${sfx} PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
    target_link_libraries(dbcreate_matrix_arts_${sfx} PRIVATE arts_static_${sfx} pthread)
    target_link_options(dbcreate_matrix_arts_${sfx} PRIVATE -no-pie -fno-pie)
    set_target_properties(dbcreate_matrix_arts_${sfx} PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(dbcreate_matrix_arts_${sfx} PRIVATE ${BENCH_C_OPT})
endforeach()

if(TARGET ocr_vdm_mpi)
    add_executable(dbcreate_matrix_ocrvx ${DBCM_SRC})
    target_include_directories(dbcreate_matrix_ocrvx PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${OCRVX_OCR_INC})
    target_compile_definitions(dbcreate_matrix_ocrvx PRIVATE
        ${OCR_EXT_DEFINES} "OCR_TYPE_H=vdm.h" OCR_ENABLE_EDT_NAMING)
    target_link_libraries(dbcreate_matrix_ocrvx PRIVATE ocr_vdm_mpi)
    target_link_options(dbcreate_matrix_ocrvx PRIVATE -no-pie -fno-pie)
    set_target_properties(dbcreate_matrix_ocrvx PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(dbcreate_matrix_ocrvx PRIVATE ${BENCH_C_OPT}
        "-include" "${OCRVX_COMPAT_H}")
endif()

# ==========================================================================
# Meta targets
# ==========================================================================

# get_property(ALL_XSOCR GLOBAL PROPERTY BENCHMARK_XSOCR_TARGETS)
get_property(ALL_ARTS GLOBAL PROPERTY BENCHMARK_ARTS_TARGETS)

# add_custom_target(benchmark_xsocr DEPENDS ${ALL_XSOCR})
add_custom_target(benchmark_arts DEPENDS ${ALL_ARTS})
add_custom_target(benchmark_apps DEPENDS benchmark_arts)

# ==========================================================================
# rwmix  (bespoke: source lives in this repo, not in ocr-apps)
# Racy single-DB reader/writer contention probe: N gated task EDTs over one
# data block, W of them RW, roles uniformly interleaved in creation order.
# Prints ELAPSED_US from trigger to global completion.
# ==========================================================================
set(RWMIX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/rwmix.c")

if(TARGET ocr_xsocr_static)
    add_executable(rwmix_xsocr ${RWMIX_SRC})
    target_include_directories(rwmix_xsocr PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR})
    target_compile_definitions(rwmix_xsocr PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
    target_link_libraries(rwmix_xsocr PRIVATE ocr_xsocr_static pthread)
    target_link_options(rwmix_xsocr PRIVATE -no-pie -fno-pie)
    set_target_properties(rwmix_xsocr PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(rwmix_xsocr PRIVATE ${BENCH_C_OPT})
endif()

foreach(sfx ${ARTS_VARIANT_SUFFIXES})
    add_executable(rwmix_arts_${sfx} ${RWMIX_SRC}
                   $<TARGET_OBJECTS:ocr_arts_objects>)
    target_include_directories(rwmix_arts_${sfx} PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR} ${OCR_INC_DIR})
    target_compile_definitions(rwmix_arts_${sfx} PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
    target_link_libraries(rwmix_arts_${sfx} PRIVATE arts_static_${sfx} pthread)
    target_link_options(rwmix_arts_${sfx} PRIVATE -no-pie -fno-pie)
    set_target_properties(rwmix_arts_${sfx} PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(rwmix_arts_${sfx} PRIVATE ${BENCH_C_OPT})
endforeach()

if(TARGET ocr_vdm_mpi)
    add_executable(rwmix_ocrvx ${RWMIX_SRC})
    target_include_directories(rwmix_ocrvx PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${OCRVX_OCR_INC})
    target_compile_definitions(rwmix_ocrvx PRIVATE
        ${OCR_EXT_DEFINES} "OCR_TYPE_H=vdm.h" OCR_ENABLE_EDT_NAMING)
    target_link_libraries(rwmix_ocrvx PRIVATE ocr_vdm_mpi)
    target_link_options(rwmix_ocrvx PRIVATE -no-pie -fno-pie)
    set_target_properties(rwmix_ocrvx PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(rwmix_ocrvx PRIVATE ${BENCH_C_OPT}
        "-include" "${OCRVX_COMPAT_H}")
endif()

# ==========================================================================
# rwrounds  (bespoke: source lives in this repo, not in ocr-apps)
# Racy single-DB reader/writer contention probe: N gated task EDTs over one
# data block, W of them RW, roles uniformly interleaved in creation order.
# Prints ELAPSED_US from trigger to global completion.
# ==========================================================================
set(RWROUNDS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/rwrounds.c")

if(TARGET ocr_xsocr_static)
    add_executable(rwrounds_xsocr ${RWROUNDS_SRC})
    target_include_directories(rwrounds_xsocr PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR})
    target_compile_definitions(rwrounds_xsocr PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
    target_link_libraries(rwrounds_xsocr PRIVATE ocr_xsocr_static pthread)
    target_link_options(rwrounds_xsocr PRIVATE -no-pie -fno-pie)
    set_target_properties(rwrounds_xsocr PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(rwrounds_xsocr PRIVATE ${BENCH_C_OPT})
endif()

foreach(sfx ${ARTS_VARIANT_SUFFIXES})
    add_executable(rwrounds_arts_${sfx} ${RWROUNDS_SRC}
                   $<TARGET_OBJECTS:ocr_arts_objects>)
    target_include_directories(rwrounds_arts_${sfx} PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR} ${OCR_INC_DIR})
    target_compile_definitions(rwrounds_arts_${sfx} PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
    target_link_libraries(rwrounds_arts_${sfx} PRIVATE arts_static_${sfx} pthread)
    target_link_options(rwrounds_arts_${sfx} PRIVATE -no-pie -fno-pie)
    set_target_properties(rwrounds_arts_${sfx} PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(rwrounds_arts_${sfx} PRIVATE ${BENCH_C_OPT})
endforeach()

if(TARGET ocr_vdm_mpi)
    add_executable(rwrounds_ocrvx ${RWROUNDS_SRC})
    target_include_directories(rwrounds_ocrvx PRIVATE
        ${OCR_LIBS_ROOT}/macros/macros ${OCRVX_OCR_INC})
    target_compile_definitions(rwrounds_ocrvx PRIVATE
        ${OCR_EXT_DEFINES} "OCR_TYPE_H=vdm.h" OCR_ENABLE_EDT_NAMING)
    target_link_libraries(rwrounds_ocrvx PRIVATE ocr_vdm_mpi)
    target_link_options(rwrounds_ocrvx PRIVATE -no-pie -fno-pie)
    set_target_properties(rwrounds_ocrvx PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(rwrounds_ocrvx PRIVATE ${BENCH_C_OPT}
        "-include" "${OCRVX_COMPAT_H}")
endif()

