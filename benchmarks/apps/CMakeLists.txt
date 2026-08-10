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
target_compile_options(ocr_lib_reduction PRIVATE ${BENCH_C_OPT} -w)
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
target_compile_options(ocr_lib_reductionEager PRIVATE ${BENCH_C_OPT} -w)
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
target_compile_options(ocr_lib_timer PRIVATE ${BENCH_C_OPT} -w)
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
target_compile_options(ocr_lib_reduction_v1 PRIVATE ${BENCH_C_OPT} -w)
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
target_compile_options(ocr_lib_ocrAppUtils PRIVATE ${BENCH_C_OPT} -w)
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
target_compile_options(ocr_lib_reduction_ocrvx PRIVATE ${BENCH_C_OPT} -w
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
target_compile_options(ocr_lib_reductionEager_ocrvx PRIVATE ${BENCH_C_OPT} -w
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
target_compile_options(ocr_lib_timer_ocrvx PRIVATE ${BENCH_C_OPT} -w
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
target_compile_options(ocr_lib_reduction_v1_ocrvx PRIVATE ${BENCH_C_OPT} -w
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
target_compile_options(ocr_lib_ocrAppUtils_ocrvx PRIVATE ${BENCH_C_OPT} -w
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
#       [OPTIMIZED_PLACEMENT]           # source carries a placement hint layer
#   )
#
# Creates, per backend, <name>_arts_<suffix> (one per coherence configuration),
# <name>_xsocr and (when compatible) <name>_ocrvx.  With OPTIMIZED_PLACEMENT the
# same set is created a second time under the name <name>_opt, compiled with the
# hint layer enabled — so both placement settings coexist in one build tree and
# neither is a property of the build directory.
# ==========================================================================
function(add_benchmark_app)
    cmake_parse_arguments(APP "OCRVX_SKIP;OPTIMIZED_PLACEMENT" "NAME;DIR;SRC_SUBDIR" "SOURCES;LIBS;EXTRA_INCLUDES;EXTRA_DEFINES;EXTRA_LINK_LIBS" ${ARGN})

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
    # its hint layer left out; the _opt name compiles that layer in.  A flavor
    # differs only by a define in the shared application source, so every
    # backend below sees the same setting — placement is a property of the
    # program, never of one runtime.
    set(_flavors base)
    if(APP_OPTIMIZED_PLACEMENT)
        list(APPEND _flavors opt)
    endif()

    foreach(_flavor ${_flavors})
        if(_flavor STREQUAL opt)
            set(_app "${APP_NAME}_opt")
            set(_flavor_define OCR_APP_OPTIMIZED_PLACEMENT)
        else()
            set(_app "${APP_NAME}")
            set(_flavor_define "")
        endif()

        # --- XSOCR variant ---
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
        target_compile_options(${_app}_xsocr PRIVATE ${BENCH_C_OPT} -w)

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
            target_compile_options(${_app}_arts_${sfx} PRIVATE ${BENCH_C_OPT} -w)
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
                target_compile_options(${_app}_ocrvx PRIVATE ${BENCH_C_OPT} -w
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
    OPTIMIZED_PLACEMENT
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
    OPTIMIZED_PLACEMENT
)

add_benchmark_app(NAME smithwaterman
    DIR smithwaterman/ocr
    SOURCES smithwaterman.c
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
    EXTRA_LINK_LIBS numa
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
    OPTIMIZED_PLACEMENT
)

add_benchmark_app(NAME tempest
    DIR tempest/refactored/ocr/intel-bryan
    SOURCES tempestCommunication.c
    OPTIMIZED_PLACEMENT
)

# ==========================================================================
# Tier 4: Medium-complexity apps
# ==========================================================================

add_benchmark_app(NAME cholesky
    DIR cholesky/ocr
    SOURCES cholesky.c
    EXTRA_LINK_LIBS m
    OPTIMIZED_PLACEMENT
)

# Tiled Cholesky whose per-tile kernels are the standard CBLAS/LAPACKE calls
# (dpotrf/dtrsm/dsyrk/dgemm), served by the third_party OpenBLAS submodule.
add_benchmark_app(NAME cholesky_blas
    DIR cholesky/ocr-mkl
    SOURCES ocr_mkl_cholesky.c
    EXTRA_LINK_LIBS arts::openblas m
    OPTIMIZED_PLACEMENT
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
    EXTRA_LINK_LIBS m
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
    OPTIMIZED_PLACEMENT
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
    OPTIMIZED_PLACEMENT
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
    OPTIMIZED_PLACEMENT
)

add_benchmark_app(NAME miniAMR_intel_bryan
    DIR miniAMR/refactored/ocr/intel-bryan
    SOURCES mainOCR.c
    LIBS ocrAppUtils
    EXTRA_LINK_LIBS m
)

add_benchmark_app(NAME miniAMR_intel_chandra
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

# Registration held back: this program's source is not carried by the
# application submodule yet, so a checkout that has not produced it
# locally cannot configure.  Restore this block, and the application's
# restructured version in the driver's catalog, once the submodule
# carries the source.
#add_benchmark_app(NAME npb_cg_dist
#    DIR npb-cg/sdsc-ocr
#    SOURCES cg_dist_edt.c makea_ocr.c util_ocr.c
#    LIBS reduction
#    EXTRA_LINK_LIBS m
#)

# ==========================================================================
# Tier 12: RSBench, XSBench
# ==========================================================================

add_benchmark_app(NAME RSBench_intel
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

# xsbench_dist: distributed sibling driver in the same source dir (the
# quicksort_dist/fft_dist pattern) — true SPMD with one deterministic table
# replica per rank; reuses the pristine physics/io sources unchanged.
# Registration held back: this program's source is not carried by the
# application submodule yet, so a checkout that has not produced it
# locally cannot configure.  Restore this block, and the application's
# restructured version in the driver's catalog, once the submodule
# carries the source.
#add_benchmark_app(NAME xsbench_dist
#    DIR XSBench/refactored/ocr/intel-sharedDB
#    SRC_SUBDIR src
#    SOURCES Main_dist.c io.c GridInit.c CalculateXS.c XSutils.c Materials.c timers.c
#    LIBS timer ocrAppUtils
#    EXTRA_LINK_LIBS m
#)

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
target_compile_options(LCS_distributed_ST_xsocr PRIVATE -fgnu89-inline)
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
target_compile_options(LCS_shared_xsocr PRIVATE -fgnu89-inline)
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
    OPTIMIZED_PLACEMENT
)

# ==========================================================================
# Tier 14: SAR (all sizes share same sources from sar/ocr/src/)
# ==========================================================================
#
# SAR build pipeline (ported from original Makefiles):
#
#   1. crlibm: extract tarball → configure → RAG patches → compile
#   2. datagen: build host utility → generate .bin datasets
#   3. dataImage: build host utility → convert .bin → .c
#   4. Compile generated .c → .o (for RAG_IMPLICIT_INPUTS sizes)
#   5. Link SAR executables with data objects + crlibm
#
# Sizes tiny/small/medium/large use RAG_IMPLICIT_INPUTS (compile-time data).
# Sizes huge/problem_size_scaling read data files at runtime.

set(SAR_ROOT "${OCR_APPS_ROOT}/sar")
set(SAR_SRC_DIR "${SAR_ROOT}/ocr/src")
set(SAR_DATASETS_DIR "${SAR_ROOT}/datasets")
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
    target_compile_options(ocr_lib_crlibm PRIVATE ${BENCH_C_OPT} -w)
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
# Step 2: Build host utilities (datagen + dataImage)
# ---------------------------------------------------------------------------
if(HAS_CRLIBM)
    # datagen: generates .bin datasets from Parameters.txt + Targets.txt
    set(DATAGEN_SRC_DIR "${SAR_ROOT}/datagen-tsml/src")
    add_executable(sar_datagen
        ${DATAGEN_SRC_DIR}/main.c
        ${DATAGEN_SRC_DIR}/inputs.c
        ${DATAGEN_SRC_DIR}/fir_filter.c
        ${DATAGEN_SRC_DIR}/gen_rand.c
        ${DATAGEN_SRC_DIR}/fftw3.c
    )
    target_include_directories(sar_datagen PRIVATE ${DATAGEN_SRC_DIR})
    target_compile_definitions(sar_datagen PRIVATE RAG_ADD_ERROR=1)
    target_compile_options(sar_datagen PRIVATE -O2 -w -fopenmp)
    target_link_libraries(sar_datagen PRIVATE m gomp)
    set_target_properties(sar_datagen PROPERTIES
        C_STANDARD 99 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # dataImage: converts .bin → .c source with embedded data arrays
    set(DATAIMAGE_SRC_DIR "${SAR_ROOT}/utils")
    add_executable(sar_dataImage
        ${DATAIMAGE_SRC_DIR}/dataImage.c
        ${DATAIMAGE_SRC_DIR}/inputs.c
    )
    target_include_directories(sar_dataImage PRIVATE
        ${DATAIMAGE_SRC_DIR}
        ${SAR_SRC_DIR}
    )
    target_compile_definitions(sar_dataImage PRIVATE RAG)
    target_compile_options(sar_dataImage PRIVATE -O2 -w)
    target_link_libraries(sar_dataImage PRIVATE m)
    set_target_properties(sar_dataImage PROPERTIES
        C_STANDARD 99 C_EXTENSIONS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

# ---------------------------------------------------------------------------
# Step 3: Data generation pipeline for RAG_IMPLICIT_INPUTS sizes
# ---------------------------------------------------------------------------
# For each size (tiny/small/medium/large):
#   datagen → .bin files → dataImage → .c files → compiled into data objects
# All generated artifacts go into the build directory.
#
# Per-file data object resolution for RAG_IMPLICIT_INPUTS sizes:
#   1. .o exists in datasets/<size>/x86/ → copy to build dir (fastest)
#   2. .c exists but .o missing → compile once, cache .o back to datasets dir
#   3. Neither → full pipeline: .bin → dataImage → .c → compile .o
if(HAS_CRLIBM)
    set(SAR_IMPLICIT_SIZES tiny small medium large)

    # Helper script for dataImage invocation (handles stdin/stdout redirection)
    set(_dataimage_script "${CMAKE_CURRENT_BINARY_DIR}/sar_run_dataimage.sh")
    file(WRITE "${_dataimage_script}"
        "#!/bin/sh\nset -e\n"
        "# Usage: sar_run_dataimage.sh <dataImage> <type_index> <input.bin> <output.c>\n"
        "\"$1\" \"$2\" < \"$3\" > \"$4\"\n"
    )

    # dataImage type indices per data file (legacy Case 2 path)
    set(_dataimage_idx_Data 1)
    set(_dataimage_idx_PlatformPosition 2)
    set(_dataimage_idx_PulseTransmissionTime 3)

    # Symbol prefix per data file — used by the .incbin wrapper script
    # (Case 3 path).  These match what app code extern-declares in
    # third_party/ocr-apps/apps/sar/ocr/src/inputs.c.
    set(_sym_prefix_Data image)
    set(_sym_prefix_PlatformPosition platform)
    set(_sym_prefix_PulseTransmissionTime pulse)

    # .incbin wrapper script — bypasses the cc1 OOM that sar_dataImage's
    # giant C-source output causes on medium/large datasets (~10 GB+ peak
    # compiler memory).  The assembler's .incbin copies bytes directly
    # in O(1) memory regardless of dataset size.
    set(_sar_incbin_script
        "${CMAKE_SOURCE_DIR}/benchmarks/apps/sar_make_incbin_wrapper.sh")

    # datagen wrapper — enters the size's input directory (the generator reads
    # Parameters.txt / Targets.txt from the working directory) and stages its
    # outputs before moving them into place.
    set(_sar_datagen_script
        "${CMAKE_SOURCE_DIR}/benchmarks/apps/sar_run_datagen.sh")

    foreach(sar_size ${SAR_IMPLICIT_SIZES})
        set(_ocr_size_dir "${SAR_ROOT}/ocr/${sar_size}")
        set(_data_build_dir "${CMAKE_CURRENT_BINARY_DIR}/sar_data/${sar_size}")
        set(_datasets_dir "${SAR_DATASETS_DIR}/${sar_size}")
        set(_prebuilt_dir "${_datasets_dir}/x86")

        file(MAKE_DIRECTORY "${_data_build_dir}")

        set(_data_names Data PlatformPosition PulseTransmissionTime)
        set(_data_objects "")

        # The .bin datasets are a deterministic function of two tracked files,
        # and the datasets directory itself is not tracked — so a checkout that
        # has never staged them has no .bin at all.  Generating them here makes
        # that the build's job: without a rule the build would instead stop on
        # a missing file it calls a source, naming neither SAR nor the two
        # inputs that produce it.  Only the sizes embedded at build time are
        # generated; the runtime-input size costs orders of magnitude more to
        # produce and is staged onto a host rather than rebuilt on it.
        set(_datagen_inputs "${SAR_ROOT}/datagen-tsml/${sar_size}")
        set(_datagen_outputs "")
        foreach(_name ${_data_names})
            list(APPEND _datagen_outputs "${_datasets_dir}/${_name}.bin")
        endforeach()
        add_custom_command(
            OUTPUT ${_datagen_outputs} "${_datasets_dir}/Radar.txt"
            COMMAND sh "${_sar_datagen_script}" "$<TARGET_FILE:sar_datagen>"
                    "${_datagen_inputs}" "${_datasets_dir}"
            DEPENDS sar_datagen "${_sar_datagen_script}"
                    "${_datagen_inputs}/Parameters.txt"
                    "${_datagen_inputs}/Targets.txt"
            COMMENT "SAR: Generating ${sar_size} dataset from datagen inputs..."
            VERBATIM
        )

        foreach(_name ${_data_names})
            set(_build_o "${_data_build_dir}/${_name}.o")

            if(EXISTS "${_prebuilt_dir}/${_name}.o")
                # Case 1: Pre-built .o exists → copy to build directory
                add_custom_command(
                    OUTPUT "${_build_o}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_prebuilt_dir}/${_name}.o" "${_build_o}"
                    DEPENDS "${_prebuilt_dir}/${_name}.o"
                    COMMENT "SAR: Copying pre-built ${sar_size}/${_name}.o"
                )
            elseif(EXISTS "${_prebuilt_dir}/${_name}.c")
                # Case 2: .c exists but .o missing → compile once, cache to datasets dir
                message(STATUS "SAR: ${sar_size}/${_name}.o missing — will compile from .c (one-time)")
                add_custom_command(
                    OUTPUT "${_build_o}"
                    COMMAND ${CMAKE_C_COMPILER} -std=c11 -O0 -w -c
                        -o "${_build_o}" "${_prebuilt_dir}/${_name}.c"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_build_o}" "${_prebuilt_dir}/${_name}.o"
                    DEPENDS "${_prebuilt_dir}/${_name}.c"
                    COMMENT "SAR: Compiling ${sar_size}/${_name}.c → .o (one-time, caching to datasets dir)..."
                )
            else()
                # Case 3: No prebuilt files → emit .S via .incbin and
                # assemble.  This path uses ~0 KB of compiler memory
                # because the assembler embeds the bytes directly,
                # bypassing the cc1 OOM that sar_dataImage's gigantic
                # C-source output triggers on medium/large datasets.
                set(_build_S "${_data_build_dir}/${_name}.S")
                add_custom_command(
                    OUTPUT "${_build_S}"
                    COMMAND sh "${_sar_incbin_script}"
                        "${_datasets_dir}/${_name}.bin"
                        "${_build_S}"
                        "${_sym_prefix_${_name}}"
                    DEPENDS "${_sar_incbin_script}"
                            "${_datasets_dir}/${_name}.bin"
                    COMMENT "SAR: Generating ${sar_size}/${_name}.S (.incbin wrapper)..."
                )
                add_custom_command(
                    OUTPUT "${_build_o}"
                    COMMAND ${CMAKE_C_COMPILER} -c -o "${_build_o}" "${_build_S}"
                    DEPENDS "${_build_S}"
                    COMMENT "SAR: Assembling ${sar_size}/${_name}.S → .o..."
                )
            endif()

            list(APPEND _data_objects "${_build_o}")
        endforeach()

        # Custom target to trigger all data object builds for this size
        add_custom_target(sar_data_${sar_size}_objs DEPENDS ${_data_objects})
        set(SAR_DATA_${sar_size}_OBJECTS ${_data_objects})
    endforeach()
endif()

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
    # --- RAG_IMPLICIT_INPUTS sizes (tiny/small/medium/large) ---
    foreach(sar_size ${SAR_IMPLICIT_SIZES})
        set(SAR_SIZE_DEFINES ${SAR_DEFINES} RAG_IMPLICIT_INPUTS)

        add_benchmark_app(NAME sar_${sar_size}
            DIR sar/ocr/${sar_size}
            SRC_SUBDIR ../src
            SOURCES ${SAR_SOURCES}
            EXTRA_INCLUDES ${SAR_SRC_DIR} ${CRLIBM_DIR}
            EXTRA_DEFINES ${SAR_SIZE_DEFINES}
            EXTRA_LINK_LIBS m ocr_lib_crlibm
            OPTIMIZED_PLACEMENT
        )
        # Link the per-size data objects (image_*, platform_*, pulse_*
        # symbols emitted from the sar_dataImage / .incbin pipeline) into
        # the executable.  Without this the link fails with "undefined
        # reference to image_0/platform_0/pulse_0/...". */
        foreach(_sa sar_${sar_size} sar_${sar_size}_opt)
            target_link_libraries(${_sa}_xsocr PRIVATE
                ${SAR_DATA_${sar_size}_OBJECTS})
            add_dependencies(${_sa}_xsocr sar_data_${sar_size}_objs)
            foreach(sfx ${ARTS_VARIANT_SUFFIXES})
                target_link_libraries(${_sa}_arts_${sfx} PRIVATE
                    ${SAR_DATA_${sar_size}_OBJECTS})
                add_dependencies(${_sa}_arts_${sfx} sar_data_${sar_size}_objs)
            endforeach()
            if(TARGET ${_sa}_ocrvx)
                target_link_libraries(${_sa}_ocrvx PRIVATE
                    ${SAR_DATA_${sar_size}_OBJECTS})
                add_dependencies(${_sa}_ocrvx sar_data_${sar_size}_objs)
            endif()
        endforeach()
    endforeach()

    # --- Runtime-input sizes (huge, problem_size_scaling) ---
    foreach(sar_size huge problem_size_scaling)
        if(EXISTS "${SAR_ROOT}/ocr/${sar_size}")
            set(SAR_RT_DEFINES ${SAR_DEFINES})
            if(sar_size STREQUAL "problem_size_scaling")
                # This variant's own Makefile.x86 sets the blocking factor to
                # 50 (the shared list carries the other variants' 32).
                list(REMOVE_ITEM SAR_RT_DEFINES RAG_NEW_BLK_SIZE=32)
                list(APPEND SAR_RT_DEFINES RAG_NEW_BLK_SIZE=50)
            endif()
            add_benchmark_app(NAME sar_${sar_size}
                DIR sar/ocr/${sar_size}
                SRC_SUBDIR ../src
                SOURCES ${SAR_SOURCES}
                EXTRA_INCLUDES ${SAR_SRC_DIR} ${CRLIBM_DIR}
                EXTRA_DEFINES ${SAR_RT_DEFINES}
                EXTRA_LINK_LIBS m ocr_lib_crlibm
                OPTIMIZED_PLACEMENT
            )
        endif()
    endforeach()
endif()


# ==========================================================================
# Tier 15: STREAM memory bandwidth, UTS tree search
# ==========================================================================

add_benchmark_app(NAME stream
    DIR stream/ocr
    SOURCES stream.c
    EXTRA_DEFINES STREAM_ARRAY_SIZE=1000 NUM_THREADS=2 NTIMES=3
    EXTRA_LINK_LIBS pthread m
)
# stream.c has functions used before declaration (C11 error) — allow it
foreach(sfx ${ARTS_VARIANT_SUFFIXES})
    target_compile_options(stream_arts_${sfx} PRIVATE -Wno-error=implicit-function-declaration)
endforeach()
# target_compile_options(stream_xsocr PRIVATE -Wno-error=implicit-function-declaration)
if(TARGET stream_ocrvx)
    target_compile_options(stream_ocrvx PRIVATE -Wno-error=implicit-function-declaration)
endif()

add_benchmark_app(NAME stream_dist
    DIR stream/ocr
    SOURCES stream_dist.c
    EXTRA_DEFINES STREAM_ARRAY_SIZE=1000 NUM_THREADS=2 NTIMES=3
    EXTRA_LINK_LIBS pthread m
)
foreach(sfx ${ARTS_VARIANT_SUFFIXES})
    target_compile_options(stream_dist_arts_${sfx} PRIVATE -Wno-error=implicit-function-declaration)
endforeach()
# target_compile_options(stream_dist_xsocr PRIVATE -Wno-error=implicit-function-declaration)
if(TARGET stream_dist_ocrvx)
    target_compile_options(stream_dist_ocrvx PRIVATE -Wno-error=implicit-function-declaration)
endif()

add_benchmark_app(NAME stream_org
    DIR stream/ocr
    SOURCES stream_org.c
    EXTRA_LINK_LIBS pthread m
)
foreach(sfx ${ARTS_VARIANT_SUFFIXES})
    target_compile_options(stream_org_arts_${sfx} PRIVATE -Wno-error=implicit-function-declaration)
endforeach()
if(TARGET stream_org_ocrvx)
    target_compile_options(stream_org_ocrvx PRIVATE -Wno-error=implicit-function-declaration)
endif()

add_benchmark_app(NAME stream_sa
    DIR stream/ocr
    SOURCES stream_sa.c
    EXTRA_LINK_LIBS pthread m
)
foreach(sfx ${ARTS_VARIANT_SUFFIXES})
    target_compile_options(stream_sa_arts_${sfx} PRIVATE -Wno-error=implicit-function-declaration)
endforeach()
if(TARGET stream_sa_ocrvx)
    target_compile_options(stream_sa_ocrvx PRIVATE -Wno-error=implicit-function-declaration)
endif()

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

add_executable(release_sync_stress_xsocr ${RSS_SRC})
target_include_directories(release_sync_stress_xsocr PRIVATE
    ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR})
target_compile_definitions(release_sync_stress_xsocr PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
target_link_libraries(release_sync_stress_xsocr PRIVATE ocr_xsocr_static pthread)
target_link_options(release_sync_stress_xsocr PRIVATE -no-pie -fno-pie)
set_target_properties(release_sync_stress_xsocr PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(release_sync_stress_xsocr PRIVATE ${BENCH_C_OPT} -w)

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
    target_compile_options(release_sync_stress_arts_${sfx} PRIVATE ${BENCH_C_OPT} -w)
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
    target_compile_options(release_sync_stress_ocrvx PRIVATE ${BENCH_C_OPT} -w
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

add_executable(dbcreate_matrix_xsocr ${DBCM_SRC})
target_include_directories(dbcreate_matrix_xsocr PRIVATE
    ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR})
target_compile_definitions(dbcreate_matrix_xsocr PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
target_link_libraries(dbcreate_matrix_xsocr PRIVATE ocr_xsocr_static pthread)
target_link_options(dbcreate_matrix_xsocr PRIVATE -no-pie -fno-pie)
set_target_properties(dbcreate_matrix_xsocr PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(dbcreate_matrix_xsocr PRIVATE ${BENCH_C_OPT} -w)

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
    target_compile_options(dbcreate_matrix_arts_${sfx} PRIVATE ${BENCH_C_OPT} -w)
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
    target_compile_options(dbcreate_matrix_ocrvx PRIVATE ${BENCH_C_OPT} -w
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

add_executable(rwmix_xsocr ${RWMIX_SRC})
target_include_directories(rwmix_xsocr PRIVATE
    ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR})
target_compile_definitions(rwmix_xsocr PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
target_link_libraries(rwmix_xsocr PRIVATE ocr_xsocr_static pthread)
target_link_options(rwmix_xsocr PRIVATE -no-pie -fno-pie)
set_target_properties(rwmix_xsocr PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(rwmix_xsocr PRIVATE ${BENCH_C_OPT} -w)

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
    target_compile_options(rwmix_arts_${sfx} PRIVATE ${BENCH_C_OPT} -w)
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
    target_compile_options(rwmix_ocrvx PRIVATE ${BENCH_C_OPT} -w
        "-include" "${OCRVX_COMPAT_H}")
endif()

# ==========================================================================
# rwrounds  (bespoke: source lives in this repo, not in ocr-apps)
# Racy single-DB reader/writer contention probe: N gated task EDTs over one
# data block, W of them RW, roles uniformly interleaved in creation order.
# Prints ELAPSED_US from trigger to global completion.
# ==========================================================================
set(RWROUNDS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/rwrounds.c")

add_executable(rwrounds_xsocr ${RWROUNDS_SRC})
target_include_directories(rwrounds_xsocr PRIVATE
    ${OCR_LIBS_ROOT}/macros/macros ${BENCH_OCR_OPTIONS_DIR})
target_compile_definitions(rwrounds_xsocr PRIVATE ${OCR_EXT_DEFINES} ${BENCH_OCR_TYPE_DEFINE})
target_link_libraries(rwrounds_xsocr PRIVATE ocr_xsocr_static pthread)
target_link_options(rwrounds_xsocr PRIVATE -no-pie -fno-pie)
set_target_properties(rwrounds_xsocr PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(rwrounds_xsocr PRIVATE ${BENCH_C_OPT} -w)

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
    target_compile_options(rwrounds_arts_${sfx} PRIVATE ${BENCH_C_OPT} -w)
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
    target_compile_options(rwrounds_ocrvx PRIVATE ${BENCH_C_OPT} -w
        "-include" "${OCRVX_COMPAT_H}")
endif()

