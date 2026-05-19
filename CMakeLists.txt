cmake_minimum_required(VERSION 3.21)
project(lindblad VERSION 1.1.7.5 LANGUAGES CXX)
set(LINDBLAD_VERSION_LABEL "R.1.7.5")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Compiler flags
# ==============================================================================
option(LINDBLAD_MARCH_NATIVE "Compile with -march=native (non-distributable, max perf)" OFF)

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
    if(LINDBLAD_MARCH_NATIVE)
        add_compile_options(-march=native)
    else()
        # AVX2 + FMA baseline (Intel Haswell+ / AMD Zen2+, covers >95% of hardware since 2014)
        add_compile_options(-march=x86-64-v3)
    endif()
    add_link_options(-fopenmp)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(
        /O2
        /arch:AVX2
        /openmp
        /fp:fast
        /W4
    )
endif()

# ==============================================================================
# FetchContent dependencies
# ==============================================================================
include(FetchContent)

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

# Options declared before FetchContent_MakeAvailable so each fetch is conditional
option(LINDBLAD_BUILD_TESTS      "Build test suite (ON when building Lindblad directly, OFF when consumed as a dependency)" ${PROJECT_IS_TOP_LEVEL})
option(LINDBLAD_BUILD_BENCHMARKS "Build benchmarks (ON when building Lindblad directly, OFF when consumed as a dependency)" ${PROJECT_IS_TOP_LEVEL})
option(LINDBLAD_BUILD_PYTHON     "Build Python bindings" OFF)

FetchContent_MakeAvailable(nlopt eigen)

if(LINDBLAD_BUILD_TESTS)
    FetchContent_MakeAvailable(googletest)
endif()
if(LINDBLAD_BUILD_BENCHMARKS)
    FetchContent_MakeAvailable(benchmark)
endif()
if(LINDBLAD_BUILD_PYTHON)
    FetchContent_MakeAvailable(pybind11)
endif()

# Runtime CLI banner object (linked into executables explicitly)
add_library(lindblad_banner_obj OBJECT
    src/banner.cpp
)

# ==============================================================================
# Core library
# ==============================================================================
add_library(lindblad_core STATIC
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
