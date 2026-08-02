cmake_minimum_required(VERSION 3.21)
project(lindblad VERSION 1.1.20.0 DESCRIPTION "Lindblad Quantum Computing Framework" LANGUAGES CXX)
# Lindblad version labeling (independent of semantic versioning)
set(LINDBLAD_VERSION_LABEL "R.1.20.0")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Compiler flags
# ==============================================================================
option(LINDBLAD_MARCH_NATIVE "Compile with -march=native (non-distributable, max perf)" OFF)

# Coverage instrumentation for gcovr/lcov reports (test infrastructure; the
# measurement workflow is documented in docs/BuildAndTest.md). OFF for normal
# builds: it overrides optimisation with -O0 so line attribution is exact.
option(LINDBLAD_BUILD_COVERAGE "Compile with --coverage for gcovr/lcov reports" OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(
        -O3
        -fopenmp
        -funroll-loops
        -ffast-math
        -Wall
        -Wextra
        -Wpedantic
    )
    # -march is an x86 option. A non-x86 target (Apple Silicon, and any arm64
    # host generally) rejects BOTH spellings outright — clang reports
    # "unsupported argument 'x86-64-v3' to option '-march='" and the build stops
    # — so the flag is probed rather than assumed. On a target that does not
    # take it, the baseline is simply the compiler's default, which is correct:
    # -march only ever widened the instruction set, it was never load-bearing.
    include(CheckCXXCompilerFlag)
    if(LINDBLAD_MARCH_NATIVE)
        check_cxx_compiler_flag("-march=native" LINDBLAD_HAS_MARCH_NATIVE)
        if(LINDBLAD_HAS_MARCH_NATIVE)
            add_compile_options(-march=native)
        else()
            message(STATUS "lindblad: -march=native unsupported here; using compiler default")
        endif()
    else()
        # AVX2 + FMA baseline (Intel Haswell+ / AMD Zen2+, covers >95% of hardware since 2014)
        check_cxx_compiler_flag("-march=x86-64-v3" LINDBLAD_HAS_MARCH_X86_64_V3)
        if(LINDBLAD_HAS_MARCH_X86_64_V3)
            add_compile_options(-march=x86-64-v3)
        else()
            message(STATUS "lindblad: -march=x86-64-v3 unsupported here (non-x86 target); using compiler default")
        endif()
    endif()
    add_link_options(-fopenmp)
    if(LINDBLAD_BUILD_COVERAGE)
        add_compile_options(--coverage -O0 -g)
        add_link_options(--coverage)
        message(STATUS "lindblad: coverage instrumentation enabled (--coverage -O0 -g)")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(
        /O2
        /arch:AVX2
        /openmp
        /fp:fast
        /W4
    )
    if(LINDBLAD_BUILD_COVERAGE)
        message(WARNING
            "LINDBLAD_BUILD_COVERAGE is only supported with GCC/Clang; ignored for MSVC")
    endif()
endif()

# ==============================================================================
# FetchContent dependencies
# ==============================================================================
include(FetchContent)

# CMake 4.0 REMOVED (not deprecated) compatibility with a declared
# cmake_minimum_required below 3.5, so any dependency still declaring an older
# floor is a hard configure error there. NLopt v2.7.1 declares VERSION 3.2 and
# fails outright on a CMake 4 host; Homebrew already ships CMake 4, so this hit
# the macOS build first. This variable tells cmake_minimum_required to treat a
# too-old declared floor as 3.5, and is read by the fetched subprojects below.
# CMake versions before 3.31 ignore it, so the setting is safe everywhere.
#
# The variable only has an effect where a declared floor is below 3.5, which
# here is the fetched dependencies alone; Lindblad declares 3.21 above and is
# unaffected. The durable fix is upgrading the NLopt pin, deliberately NOT
# bundled here: NLopt is quarantined with -fno-fast-math below because its
# COBYLA convergence is numerically fragile, so a version change needs its own
# optimiser-focused test cycle.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)

# Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
# Google Benchmark promotes its own warnings to errors by default
# (BENCHMARK_ENABLE_WERROR), and it applies that -Werror to its own sources only.
# The pinned release predates several clang diagnostics, so a sufficiently new
# compiler fails the dependency build on code we do not own and cannot patch:
# clang 20 reports -Winvalid-offsetof inside its State cache-line assertions, and
# clang 22 reports -Wc2y-extensions on its __COUNTER__ probe. Neither is a defect
# we can act on, and neither says anything about this project's own sources, which
# keep -Wall -Wextra -Wpedantic. Turning the dependency's -Werror off is the
# upstream-provided knob for exactly this situation.
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

# pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
)

# NLopt
FetchContent_Declare(
    nlopt
    GIT_REPOSITORY https://github.com/stevengj/nlopt.git
    GIT_TAG        v2.7.1
)
set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE)
set(NLOPT_OCTAVE OFF CACHE BOOL "" FORCE)
set(NLOPT_MATLAB OFF CACHE BOOL "" FORCE)
set(NLOPT_GUILE OFF CACHE BOOL "" FORCE)
set(NLOPT_SWIG OFF CACHE BOOL "" FORCE)
set(NLOPT_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)  # NLopt respects NLOPT_BUILD_SHARED; kept for other deps

# Eigen3 — header-only, used for robust SVD and eigendecomposition
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
# Unless this is ON, Eigen 3.4.0 injects -std=c++03 into its own configure
# checks (its CMakeLists.txt adds the flag, then probes for the standard math
# library). MSVC's standard library headers cannot be parsed as C++03 — they
# declare `noexcept` — so on a Windows Clang toolchain every Eigen probe that
# includes <cmath> fails to COMPILE, and Eigen reports the misleading
# "Can't link to the standard math library" and halts configuration. Turning
# this ON only stops the stale flag being injected: Lindblad is C++23 and never
# builds Eigen's own tests (BUILD_TESTING is forced OFF below).
set(EIGEN_TEST_CXX11 ON CACHE BOOL "" FORCE)

# Options declared before FetchContent_MakeAvailable so each fetch is conditional
option(LINDBLAD_BUILD_TESTS      "Build test suite (ON when building Lindblad directly, OFF when consumed as a dependency)" ${PROJECT_IS_TOP_LEVEL})
option(LINDBLAD_BUILD_BENCHMARKS "Build benchmarks (ON when building Lindblad directly, OFF when consumed as a dependency)" ${PROJECT_IS_TOP_LEVEL})
option(LINDBLAD_BUILD_PYTHON     "Build Python bindings" OFF)
option(LINDBLAD_BUILD_APPS       "Build user-facing CLI apps (lindblad_draw, ...)" ${PROJECT_IS_TOP_LEVEL})

# Eigen 3.4.0 gates its own test registration on the standard BUILD_TESTING
# variable (its option() defaults it ON); the EIGEN_BUILD_TESTING name below
# is only honoured by newer Eigen. Without this, Eigen silently registers
# ~915 ctest entries for test binaries that are never built. Lindblad's own
# suite uses enable_testing() + LINDBLAD_BUILD_TESTS and never consults
# BUILD_TESTING, so forcing it OFF affects dependencies only.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlopt eigen)

# Strict FP for the MPS simulator TU (R.1.16.0, issue #44). Eigen is
# header-only, so mps_sim.cpp's JacobiSVD instantiation inherits this
# directory's -ffast-math; under that codegen, on degenerate rank-deficient
# MPS thetas (13-qubit Shor IQFT), Eigen 3.4.0 returned internally
# inconsistent factorizations whose damage reached the KEPT singular
# channels as finite-but-wrong data — undetectable by any truncation-side
# check. Under strict FP the same input yields an exact kept slice with
# garbage confined to discardable null-space columns (proven by the
# DiagR1160StrictFP suite), which svd_truncate's hardened selection then
# handles. Same pattern as the NLopt -fno-fast-math quarantine below:
# numerical ALGORITHMS need IEEE semantics; kernel arithmetic keeps
# fast-math. Later flags override earlier ones, so this TU keeps -O3/-march.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_source_files_properties(src/simulators/mps_sim.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math")
endif()

# Third-party hygiene for the vendored NLopt build:
#   -w              its own sources (stogo) trip warnings; the project's
#                   warning inventory must stay signal-only.
#   -fno-fast-math  NLopt is a numerical optimiser whose convergence logic
#                   assumes IEEE semantics; inheriting the directory-wide
#                   -ffast-math broke COBYLA outright under Clang with
#                   -march=native (immediate bail-out, observed as VQE
#                   returning the initial energy). Later flags override, so
#                   NLopt keeps -O3/-march but regains strict FP behaviour.
if(TARGET nlopt AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    target_compile_options(nlopt PRIVATE -w -fno-fast-math)
endif()

if(LINDBLAD_BUILD_TESTS)
    FetchContent_MakeAvailable(googletest)
endif()
if(LINDBLAD_BUILD_BENCHMARKS)
    FetchContent_MakeAvailable(benchmark)
    # Google Benchmark compiles its own sources with -pedantic-errors, which it
    # applies unconditionally — turning BENCHMARK_ENABLE_WERROR off above does
    # not remove it. Clang classifies invalid-offsetof as an extension warning,
    # and -pedantic-errors promotes extension warnings to hard errors, so
    # Clang 20 rejects the static assertions Benchmark makes about its own State
    # layout. The diagnostic is reported as [-Werror,-Winvalid-offsetof] even
    # with no -Werror on the command line, which makes the real cause easy to
    # misread. Disabling the group outright is what actually clears it: -w only
    # silences warnings, and these are errors by the time it would apply.
    if(TARGET benchmark AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
        target_compile_options(benchmark PRIVATE -Wno-invalid-offsetof)
    endif()
endif()
if(LINDBLAD_BUILD_PYTHON)
    FetchContent_MakeAvailable(pybind11)
endif()

# Runtime CLI banner object (linked into executables explicitly)
add_library(lindblad_banner_obj OBJECT
    src/banner.cpp
)
# banner.cpp prints LINDBLAD_VERSION_LABEL directly. It is an OBJECT lib that does
# not link lindblad_core, so it does not inherit that target's PUBLIC definition —
# give it the same compile definition here so the banner never drifts.
target_compile_definitions(lindblad_banner_obj PRIVATE
    LINDBLAD_VERSION_LABEL="${LINDBLAD_VERSION_LABEL}"
)

# ==============================================================================
# Core library
# ==============================================================================
add_library(lindblad_core STATIC
    src/hw_info.cpp
    src/statevector.cpp
    src/gates/single_qubit.cpp
    src/gates/two_qubit.cpp
    src/gates/multi_qubit.cpp
    src/circuit.cpp
    src/dag.cpp
    src/simulators/statevector_sim.cpp
    src/simulators/density_matrix_sim.cpp
    src/simulators/clifford_sim.cpp
    src/simulators/mps_sim.cpp
    src/noise/kraus_channel.cpp
    src/noise/channels.cpp
    src/noise/noise_model.cpp
    src/quantum_info/operators.cpp
    src/quantum_info/states.cpp
    src/quantum_info/metrics.cpp
    src/transpiler/basis_translator.cpp
    src/transpiler/high_level_decompose.cpp
    src/transpiler/preset_pass_manager.cpp
    src/transpiler/layout/sabre_layout.cpp
    src/transpiler/layout/trivial_layout.cpp
    src/transpiler/routing/sabre_swap.cpp
    src/transpiler/routing/stochastic_swap.cpp
    src/transpiler/optimisation/optimize_1q.cpp
    src/transpiler/optimisation/cx_cancellation.cpp
    src/transpiler/optimisation/consolidate_blocks.cpp
    src/transpiler/optimisation/commutative_cancellation.cpp
    src/transpiler/optimisation/remove_diagonal.cpp
    src/transpiler/scheduling/scheduling.cpp
    src/primitives/estimator.cpp
    src/primitives/sampler.cpp
    src/algorithms/vqe.cpp
    src/algorithms/qaoa.cpp
    src/algorithms/maqaoa.cpp
    src/algorithms/qpe.cpp
    src/algorithms/grover.cpp
    src/algorithms/deutsch_jozsa.cpp
    src/algorithms/bernstein_vazirani.cpp
    src/algorithms/simon.cpp
    src/algorithms/qft.cpp
    src/algorithms/ising.cpp
    src/algorithms/dispatch.cpp
    src/algorithms/shor.cpp
    src/qudit/qudit_statevector.cpp
    src/qudit/qudit_gates.cpp
    src/qudit/qudit_simulator.cpp
    src/qudit/qudit_noise_model.cpp
    src/qudit/qudit_density_matrix.cpp
    src/qudit/qudit_clifford.cpp
    src/qudit/qudit_mps.cpp
    src/qasm/qasm2_parser.cpp
    src/qasm/qasm3_parser.cpp
    src/backends/local_backend.cpp
    src/visualisation/gate_symbols.cpp
    src/visualisation/composite_catalogue.cpp
    src/visualisation/gate_builders.cpp
    src/visualisation/format_params.cpp
    src/visualisation/layout.cpp
    src/visualisation/render_ascii.cpp
    src/visualisation/render_svg.cpp
    src/visualisation/render_latex.cpp
    src/visualisation/render_html.cpp
)

target_include_directories(lindblad_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(lindblad::core ALIAS lindblad_core)

# Expose version label as a compile definition so version() never drifts from CMakeLists.txt
# EIGEN_MPL2_ONLY restricts inclusion to MPL-2.0-only headers; Eigen's LGPL files are excluded
target_compile_definitions(lindblad_core PUBLIC
    LINDBLAD_VERSION_LABEL="${LINDBLAD_VERSION_LABEL}"
    EIGEN_MPL2_ONLY
)

# Link OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(lindblad_core PUBLIC OpenMP::OpenMP_CXX)

# Link NLopt
target_link_libraries(lindblad_core PUBLIC nlopt)

# Link Eigen3 (header-only)
target_link_libraries(lindblad_core PUBLIC Eigen3::Eigen)

# ==============================================================================
# Python bindings
# ==============================================================================
if(LINDBLAD_BUILD_PYTHON)
    pybind11_add_module(lindblad_python bindings/python_bindings.cpp)
    target_link_libraries(lindblad_python PRIVATE lindblad_core)
    set_target_properties(lindblad_python PROPERTIES OUTPUT_NAME "lindblad")
endif()

# ==============================================================================
# Tests
# ==============================================================================
if(LINDBLAD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ==============================================================================
# Benchmarks
# ==============================================================================
if(LINDBLAD_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ==============================================================================
# CLI apps (R.1.10.3: lindblad_draw)
# ==============================================================================
if(LINDBLAD_BUILD_APPS)
    add_subdirectory(apps)
endif()