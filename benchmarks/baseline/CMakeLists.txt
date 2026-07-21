# Baseline benchmark applications (OpenMP and/or MPI)
#
# These are the original (non-OCR) implementations of benchmark apps,
# compiled with native OpenMP and MPI for performance comparison against
# the ARTS and XSOCR backends.

find_package(OpenMP)
find_package(MPI)

set(BASELINE_APPS_ROOT "${CMAKE_SOURCE_DIR}/third_party/ocr-apps/apps")

# ==========================================================================
# add_baseline_app: builds a native OMP/MPI baseline executable
#
# Usage:
#   add_baseline_app(NAME <name>
#       DIR <relative-path-under-ocr-apps/apps>
#       SOURCES <file1.c> [file2.c ...]
#       [SRC_SUBDIR <subdir>]           # source files are in DIR/subdir/
#       [LANGUAGE <C|CXX>]             # default: C
#       [NEEDS_OMP]                    # requires OpenMP
#       [NEEDS_MPI]                    # requires MPI
#       [EXTRA_INCLUDES <dir> ...]     # additional include dirs
#       [EXTRA_DEFINES <def> ...]      # additional compile definitions
#       [EXTRA_LINK_LIBS <lib> ...]    # extra link libraries (e.g. m)
#   )
# ==========================================================================
function(add_baseline_app)
    cmake_parse_arguments(APP "NEEDS_OMP;NEEDS_MPI" "NAME;DIR;SRC_SUBDIR;LANGUAGE"
        "SOURCES;EXTRA_INCLUDES;EXTRA_DEFINES;EXTRA_LINK_LIBS" ${ARGN})

    # Skip if required dependencies are missing
    if(APP_NEEDS_OMP AND NOT OpenMP_C_FOUND AND NOT OpenMP_CXX_FOUND)
        message(STATUS "Skipping ${APP_NAME}: OpenMP not found")
        return()
    endif()
    if(APP_NEEDS_MPI AND NOT MPI_FOUND)
        message(STATUS "Skipping ${APP_NAME}: MPI not found")
        return()
    endif()

    # Default language is C
    if(NOT APP_LANGUAGE)
        set(APP_LANGUAGE "C")
    endif()

    set(APP_DIR "${BASELINE_APPS_ROOT}/${APP_DIR}")
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

    add_executable(${APP_NAME} ${ABS_SOURCES})
    target_include_directories(${APP_NAME} PRIVATE ${APP_DIR} ${SRC_DIR} ${APP_EXTRA_INCLUDES})
    target_compile_definitions(${APP_NAME} PRIVATE ${APP_EXTRA_DEFINES})
    target_compile_options(${APP_NAME} PRIVATE ${BENCH_C_OPT} -w)
    set_target_properties(${APP_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    if(APP_LANGUAGE STREQUAL "CXX")
        set_target_properties(${APP_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    else()
        set_target_properties(${APP_NAME} PROPERTIES
            C_STANDARD 11
            C_EXTENSIONS ON
        )
    endif()

    # Link OpenMP
    if(APP_NEEDS_OMP)
        if(APP_LANGUAGE STREQUAL "CXX")
            target_link_libraries(${APP_NAME} PRIVATE OpenMP::OpenMP_CXX)
        else()
            target_link_libraries(${APP_NAME} PRIVATE OpenMP::OpenMP_C)
        endif()
    endif()

    # Link MPI
    if(APP_NEEDS_MPI)
        if(APP_LANGUAGE STREQUAL "CXX")
            target_link_libraries(${APP_NAME} PRIVATE MPI::MPI_CXX)
        else()
            target_link_libraries(${APP_NAME} PRIVATE MPI::MPI_C)
        endif()
    endif()

    target_link_libraries(${APP_NAME} PRIVATE ${APP_EXTRA_LINK_LIBS})

    # Track for meta targets
    if(APP_NEEDS_MPI)
        set_property(GLOBAL APPEND PROPERTY BASELINE_MPI_TARGETS ${APP_NAME})
    else()
        set_property(GLOBAL APPEND PROPERTY BASELINE_OMP_TARGETS ${APP_NAME})
    endif()
endfunction()

# ==========================================================================
# Pure OpenMP apps
# ==========================================================================

add_baseline_app(NAME XSBench_omp
    DIR XSBench/baseline/mpi-omp-serial/doe-original
    SRC_SUBDIR src
    SOURCES Main.c io.c CalculateXS.c XSutils.c Materials.c GridInit.c
    NEEDS_OMP
    EXTRA_LINK_LIBS m
)

add_baseline_app(NAME RSBench_omp
    DIR RSBench/baseline/omp-serial/doe-original
    SRC_SUBDIR src
    SOURCES main.c init.c material.c io.c utils.c xs_kernel.c
    NEEDS_OMP
    EXTRA_LINK_LIBS m
)

set(STENCIL2D_INCLUDE "${BASELINE_APPS_ROOT}/Stencil2D/baseline/include")

set(STENCIL2D_COMMON "${BASELINE_APPS_ROOT}/Stencil2D/baseline/common")

add_baseline_app(NAME Stencil2D_omp
    DIR Stencil2D/baseline/prk-original/omp
    SOURCES stencil.c
    NEEDS_OMP
    EXTRA_INCLUDES "${STENCIL2D_INCLUDE}" "${STENCIL2D_COMMON}"
    EXTRA_DEFINES STAR DOUBLE
    EXTRA_LINK_LIBS m
)
# PRK helper functions (bail_out, wtime) live in common/
target_sources(Stencil2D_omp PRIVATE
    "${STENCIL2D_COMMON}/OPENMP_bail_out.c"
    "${STENCIL2D_COMMON}/wtime.c"
)

add_baseline_app(NAME nqueens_omp
    DIR nqueens/baseline/omp
    SOURCES nqueens.c
    NEEDS_OMP
)

# cg_omp.c replaces cg.c (has OpenMP-parallelized versions of the same functions)
add_baseline_app(NAME npb_cg_omp
    DIR npb-cg/sdsc-omp
    SOURCES cg_omp.c main.c makea.c util.c
    NEEDS_OMP
    EXTRA_LINK_LIBS m
)

# Tile Cholesky, hand-coded OpenMP tasks (no BLAS) — the reference for the
# cholesky / cholesky_blas OCR ports (run at both problem sizes).
add_baseline_app(NAME cholesky_omp
    DIR cholesky/omp-no-mkl
    SOURCES omp_cholesky.c
    NEEDS_OMP
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# MPI + OpenMP apps (C)
# ==========================================================================

# CoMD needs CoMD_info.h (normally generated by its Makefile). Generate a stub.
set(COMD_DIR "${BASELINE_APPS_ROOT}/CoMD/baseline/mpi-omp/sdsc")
set(COMD_INFO_DIR "${CMAKE_CURRENT_BINARY_DIR}/comd_generated")
file(MAKE_DIRECTORY "${COMD_INFO_DIR}")
file(WRITE "${COMD_INFO_DIR}/CoMD_info.h"
"#ifndef COMD_INFO_H\n"
"#define COMD_INFO_H\n"
"#define CoMD_VARIANT \"CoMD-mpi-omp\"\n"
"#define CoMD_HOSTNAME \"benchmark\"\n"
"#define CoMD_KERNEL_NAME \"Linux\"\n"
"#define CoMD_KERNEL_RELEASE \"\"\n"
"#define CoMD_PROCESSOR \"x86_64\"\n"
"#define CoMD_COMPILER \"${CMAKE_C_COMPILER}\"\n"
"#define CoMD_COMPILER_VERSION \"${CMAKE_C_COMPILER_VERSION}\"\n"
"#define CoMD_CFLAGS \"-O3\"\n"
"#define CoMD_LDFLAGS \"\"\n"
"#endif\n"
)

add_baseline_app(NAME CoMD_mpi_omp
    DIR CoMD/baseline/mpi-omp/sdsc
    SOURCES CoMD.c mycommand.c cmdLineParser.c eam.c ljForce.c decomposition.c
            linkCells.c initAtoms.c random.c timestep.c performanceTimers.c
            haloExchange.c parallel.c yamlOutput.c
    NEEDS_OMP
    NEEDS_MPI
    EXTRA_INCLUDES "${COMD_INFO_DIR}"
    EXTRA_DEFINES DOUBLE DO_MPI
    EXTRA_LINK_LIBS m
)

# miniAMR uses MPI_Errhandler_set (removed in MPI-3.0). Force-include a
# compatibility header that suppresses the removal error in Open MPI 5.x.
add_baseline_app(NAME miniAMR_mpi
    DIR miniAMR/baseline/mpi-serial/doe-original
    SOURCES block.c check_sum.c comm_block.c comm.c comm_parent.c comm_refine.c
            comm_util.c driver.c init.c main.c move.c pack.c plot.c profile.c
            rcb.c refine.c stencil.c target.c util.c
    NEEDS_MPI
    EXTRA_LINK_LIBS m
)
target_compile_options(miniAMR_mpi PRIVATE
    -include "${CMAKE_CURRENT_SOURCE_DIR}/mpi3_compat.h"
    -fcommon
)

add_baseline_app(NAME Stencil1D_mpi
    DIR Stencil1D/baseline/mpi/intel
    SOURCES stencil.c
    NEEDS_MPI
    EXTRA_LINK_LIBS m
)

add_baseline_app(NAME Stencil2D_mpi
    DIR Stencil2D/baseline/prk-original/mpi
    SOURCES stencil.c
    NEEDS_MPI
    EXTRA_INCLUDES "${STENCIL2D_INCLUDE}" "${STENCIL2D_COMMON}"
    EXTRA_DEFINES STAR DOUBLE_USER
    EXTRA_LINK_LIBS m
)
target_sources(Stencil2D_mpi PRIVATE
    "${STENCIL2D_COMMON}/MPI_bail_out.c"
    "${STENCIL2D_COMMON}/wtime.c"
)

# SNAP uses old MPI_Comm-as-int convention, incompatible with Open MPI 5.
# Downgrade type errors to warnings (then suppressed by -w).
add_baseline_app(NAME SNAP_mpi_omp
    DIR SNAP/baseline/mpi-omp/doe-original-intel-cport/snap-c
    SOURCES snap_main.c control.c data.c dealloc.c dim1_sweep.c dim3_sweep.c
            expxs.c geom.c inner.c input.c mms.c octsweep.c outer.c output.c
            plib.c setup.c sn.c solvar.c sweep.c time.c translv.c utils.c
            version.c
    NEEDS_OMP
    NEEDS_MPI
    EXTRA_LINK_LIBS m
)
target_compile_options(SNAP_mpi_omp PRIVATE
    -Wno-error=incompatible-pointer-types -Wno-error=int-conversion
)

add_baseline_app(NAME hpgmg_mpi_omp
    DIR hpgmg/baseline/mpi-omp/doe-original/finite-volume/source
    SOURCES timers.c level.c operators.7pt.c mg.c solvers.c hpgmg.c
    NEEDS_OMP
    NEEDS_MPI
    EXTRA_INCLUDES
        "${BASELINE_APPS_ROOT}/hpgmg/baseline/mpi-omp/doe-original/finite-volume/source/operators"
    EXTRA_DEFINES USE_MPI=1 USE_SUBCOMM=1 USE_FCYCLES=1 USE_BICGSTAB=1 USE_CHEBY=1
    EXTRA_LINK_LIBS m
)

# ==========================================================================
# C++ baselines
# ==========================================================================

# LULESH 2.0 — serial OpenMP version (USE_MPI=0)
add_baseline_app(NAME lulesh_omp
    DIR lulesh-2.0.3/baseline/mpi-omp/doe-original
    SOURCES lulesh.cc lulesh-comm.cc lulesh-viz.cc lulesh-util.cc lulesh-init.cc
    LANGUAGE CXX
    NEEDS_OMP
    EXTRA_DEFINES USE_MPI=0
    EXTRA_LINK_LIBS m
)

# LULESH 2.0 — MPI+OpenMP version
add_baseline_app(NAME lulesh_mpi_omp
    DIR lulesh-2.0.3/baseline/mpi-omp/doe-original
    SOURCES lulesh.cc lulesh-comm.cc lulesh-viz.cc lulesh-util.cc lulesh-init.cc
    LANGUAGE CXX
    NEEDS_OMP
    NEEDS_MPI
    EXTRA_DEFINES USE_MPI=1
    EXTRA_LINK_LIBS m
)

# HPCG 3.0 — official source (OpenMP+MPI hybrid).  Two targets from the same
# sources: hpcg_mpi is MPI-only (HPCG_NO_OPENMP), hpcg_mpi_omp is the hybrid
# build (OpenMP threads within each MPI rank), for the MPI+OMP ceiling.
set(HPCG_SOURCES
    CG.cpp CG_ref.cpp CheckAspectRatio.cpp CheckProblem.cpp
    ComputeDotProduct.cpp ComputeDotProduct_ref.cpp
    ComputeMG.cpp ComputeMG_ref.cpp ComputeOptimalShapeXYZ.cpp
    ComputeProlongation_ref.cpp ComputeResidual.cpp
    ComputeRestriction_ref.cpp ComputeSPMV.cpp ComputeSPMV_ref.cpp
    ComputeSYMGS.cpp ComputeSYMGS_ref.cpp
    ComputeWAXPBY.cpp ComputeWAXPBY_ref.cpp
    ExchangeHalo.cpp GenerateCoarseProblem.cpp GenerateGeometry.cpp
    GenerateProblem.cpp GenerateProblem_ref.cpp
    SetupHalo.cpp SetupHalo_ref.cpp
    MixedBaseCounter.cpp OptimizeProblem.cpp ReadHpcgDat.cpp
    ReportResults.cpp TestCG.cpp TestNorms.cpp TestSymmetry.cpp
    WriteProblem.cpp YAML_Doc.cpp YAML_Element.cpp
    finalize.cpp init.cpp main.cpp mytimer.cpp)

add_baseline_app(NAME hpcg_mpi
    DIR hpcg/baseline/mpi/sandia/hpcg-3.0/src
    SOURCES ${HPCG_SOURCES}
    LANGUAGE CXX
    NEEDS_OMP
    NEEDS_MPI
    EXTRA_DEFINES HPCG_NO_OPENMP
)

add_baseline_app(NAME hpcg_mpi_omp
    DIR hpcg/baseline/mpi/sandia/hpcg-3.0/src
    SOURCES ${HPCG_SOURCES}
    LANGUAGE CXX
    NEEDS_OMP
    NEEDS_MPI
)

# ==========================================================================
# SAR OpenMP baseline (requires crlibm from OCR libs)
# ==========================================================================

# SAR OMP needs both crlibm (extracted by the benchmarks/apps SAR tier into
# ${CMAKE_BINARY_DIR}/benchmarks/apps/crlibm) AND a single-precision FFTW3
# (fftwf_*) implementation.  crlibm is now reachable, but the only in-repo
# fftw3.h/fftw3.c is an OCR-runtime-coupled shim (it #includes ocr.h), so it
# cannot back a native OpenMP executable, and no standalone FFTW3 dev header
# is installed on this box.  Skip until a real FFTW3 dev package is available.
set(CRLIBM_GEN_DIR "${CMAKE_BINARY_DIR}/benchmarks/apps/crlibm")
if(EXISTS "/usr/include/fftw3.h" AND EXISTS "${CRLIBM_GEN_DIR}/crlibm.h" AND TARGET ocr_lib_crlibm)
    add_baseline_app(NAME sar_omp
        DIR sar/omp/src
        SOURCES main.c common.c ccd.c cfar.c sinc.c sinc_interp.c backproj.c
                downsample.c registration.c digspot.c inputs.c image_formation.c
        NEEDS_OMP
        EXTRA_INCLUDES "${CRLIBM_GEN_DIR}"
        EXTRA_LINK_LIBS m ocr_lib_crlibm fftw3f
    )
else()
    message(STATUS "Skipping sar_omp: standalone FFTW3 (fftw3.h) not available")
endif()

# ==========================================================================
# Meta targets
# ==========================================================================

get_property(ALL_OMP GLOBAL PROPERTY BASELINE_OMP_TARGETS)
get_property(ALL_MPI GLOBAL PROPERTY BASELINE_MPI_TARGETS)

if(ALL_OMP)
    add_custom_target(benchmark_baseline_omp DEPENDS ${ALL_OMP})
else()
    add_custom_target(benchmark_baseline_omp)
endif()

if(ALL_MPI)
    add_custom_target(benchmark_baseline_mpi DEPENDS ${ALL_MPI})
else()
    add_custom_target(benchmark_baseline_mpi)
endif()

add_custom_target(benchmark_baseline DEPENDS benchmark_baseline_omp benchmark_baseline_mpi)
