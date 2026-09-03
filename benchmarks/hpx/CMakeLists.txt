cmake_minimum_required(VERSION 3.18)

project(arts_hpx_apps VERSION 1.0 LANGUAGES CXX)

find_package(HPX 1.11.0 EXACT CONFIG REQUIRED)
enable_testing()

# The 16-byte operations the 128-bit queue backends compile to are resolved
# from libatomic, and a static link resolves a library only against what
# precedes it — so the entry has to follow every HPX archive, which is what
# the standard-libraries list is.
set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -latomic")

# Settings every consumer of this HPX install needs, whatever it is, and
# every one of them stated here: this project is configurable on its own
# against the install, and inherits nothing from the tree that built it.
# The macro selects which thread-queue backends the headers declare, and
# the installed defines.hpp cannot carry it (see third_party/CMakeLists.txt),
# so a consumer that omits it declares different backends than the archives
# hold.
function(arts_hpx_configure target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE -mcx16)
    target_compile_definitions(${target} PRIVATE HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${target} PRIVATE HPX::hpx)
    # The embedded hwloc must stay out of the dynamic symbol table: an
    # MPI library in the same process loads the host's hwloc for
    # itself, and an exported copy would preempt it.
    target_link_options(${target} PRIVATE "LINKER:--exclude-libs,libhwloc.a")
endfunction()

# One port is one source under one placement guard: the base tier is the
# runtime's blind policy carried in the program, the hinted tier the
# application's own placement key.  Both targets land in the output
# directory the driver reads; the core tests do not.
function(add_hpx_port)
    cmake_parse_arguments(P "HINTED" "NAME" "SOURCES" ${ARGN})
    if(P_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "add_hpx_port: unexpected arguments: ${P_UNPARSED_ARGUMENTS}")
    endif()
    set(_targets ${P_NAME}_hpx)
    if(P_HINTED)
        list(APPEND _targets ${P_NAME}_hinted_hpx)
    endif()
    foreach(_t IN LISTS _targets)
        add_executable(${_t} ${P_SOURCES})
        arts_hpx_configure(${_t})
        if(HPX_APPS_RUNTIME_OUTPUT_DIRECTORY)
            set_target_properties(${_t} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${HPX_APPS_RUNTIME_OUTPUT_DIRECTORY}")
        endif()
    endforeach()
    if(P_HINTED)
        target_compile_definitions(${P_NAME}_hinted_hpx PRIVATE HPX_APP_HINTED_PLACEMENT)
    endif()
endfunction()

# A core test is plain C++ over the algorithm's non-runtime part; it runs
# on every build of this project and must be sub-second.
function(add_core_test NAME)
    add_executable(${NAME} tests/${NAME}.cpp)
    target_compile_features(${NAME} PRIVATE cxx_std_17)
    target_include_directories(${NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

add_core_test(nqueens_core_test)
add_hpx_port(NAME nqueens SOURCES nqueens.cpp HINTED)

add_core_test(p2p_core_test)
add_hpx_port(NAME p2p SOURCES p2p.cpp)

add_core_test(stencil2d_core_test)
add_hpx_port(NAME Stencil2D_intel_channelEVTs SOURCES stencil2d.cpp)

add_core_test(smithwaterman_core_test)
add_hpx_port(NAME smithwaterman SOURCES smithwaterman.cpp HINTED)

add_core_test(triangle_core_test)
add_hpx_port(NAME triangle SOURCES triangle.cpp HINTED)

add_core_test(tempest_core_test)
add_hpx_port(NAME tempest SOURCES tempest.cpp HINTED)

# A probe is not a port: it measures one runtime primitive, carries no
# application, and is never a campaign cell.  It keeps its own output
# directory so a driver enumerating the ports cannot pick it up.
function(add_hpx_probe NAME)
    add_executable(${NAME} probes/${NAME}.cpp)
    arts_hpx_configure(${NAME})
    if(HPX_APPS_RUNTIME_OUTPUT_DIRECTORY)
        set_target_properties(${NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${HPX_APPS_RUNTIME_OUTPUT_DIRECTORY}/probes")
    endif()
endfunction()

add_hpx_probe(action_rtt)

# An allocator probe: one application source built a second time against a
# substitute malloc, so the pair differs in nothing but the allocator.  It is
# a probe, not a port — same isolated output directory, never a catalog
# target — and it exists only when a path to the override archive is given.
# Whole-archive is mandatory rather than cosmetic: an override provides the
# standard entry points, which the C library already defines, so on-demand
# member extraction pulls in nothing at all.
set(ARTS_HPX_MIMALLOC_OVERRIDE "" CACHE FILEPATH
    "Path to an override-mode static mimalloc archive; when set, the HPX app project also builds its allocator probe (empty by default)")
if(ARTS_HPX_MIMALLOC_OVERRIDE)
    add_executable(Stencil2D_intel_channelEVTs_mimalloc_hpx stencil2d.cpp)
    target_link_options(Stencil2D_intel_channelEVTs_mimalloc_hpx PRIVATE
        "LINKER:--whole-archive,${ARTS_HPX_MIMALLOC_OVERRIDE},--no-whole-archive")
    arts_hpx_configure(Stencil2D_intel_channelEVTs_mimalloc_hpx)
    if(HPX_APPS_RUNTIME_OUTPUT_DIRECTORY)
        set_target_properties(Stencil2D_intel_channelEVTs_mimalloc_hpx PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${HPX_APPS_RUNTIME_OUTPUT_DIRECTORY}/probes")
    endif()
endif()

# Two-locality smoke of the static build: components (collectives), a plain
# action into a receive buffer, the counter registry, and the hpx_main
# shape.  Needs an MPI launcher, which the outer project hands in together
# with the switch that leaves the smoke out on a host whose launcher cannot
# start a two-rank job.
if(ARTS_MPIEXEC AND ARTS_HPX_SMOKE)
    add_executable(static_smoke tests/static_smoke.cpp)
    arts_hpx_configure(static_smoke)
    add_test(NAME static_smoke
        COMMAND ${ARTS_MPIEXEC} -n 2 $<TARGET_FILE:static_smoke> --hpx:threads=2)
    # The exit status is the verdict; a pass expression would make ctest
    # ignore it and call a crash after the printed line a pass.
    set_tests_properties(static_smoke PROPERTIES
        FAIL_REGULAR_EXPRESSION "SMOKE FAIL" LABELS mpi
        ENVIRONMENT "ARTS_E2E_MARKER=1;ARTS_STRUCT_MARKER=1" TIMEOUT 120)
    # The same smoke choosing a policy on the command line.  The scheduling
    # entry the ports install is deliberately non-forcing, and only a run
    # that overrides it proves that: a forcing entry passes the first test
    # exactly as well as this one fails.  Naming a policy also cannot
    # pass quietly on a build whose thread-queue backends were not compiled
    # in: the runtime reports the policy as unconfigured and then dies.
    add_test(NAME static_smoke_queuing_override
        COMMAND ${ARTS_MPIEXEC} -n 2 $<TARGET_FILE:static_smoke> --hpx:threads=2
            --hpx:queuing=local-priority-fifo)
    set_tests_properties(static_smoke_queuing_override PROPERTIES
        FAIL_REGULAR_EXPRESSION "SMOKE FAIL" LABELS mpi
        ENVIRONMENT "ARTS_E2E_MARKER=1;ARTS_STRUCT_MARKER=1;ARTS_HPX_EXPECT_SCHEDULER=local-priority-fifo"
        TIMEOUT 120)
endif()
