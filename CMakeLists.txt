cmake_minimum_required(VERSION 3.20)
project(qpp VERSION 1.5.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Compiler flags
# ==============================================================================
option(QPP_MARCH_NATIVE "Compile with -march=native (non-distributable, max perf)" OFF)

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
    if(QPP_MARCH_NATIVE)
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

FetchContent_MakeAvailable(googletest nlopt eigen)

# Benchmark and pybind11 are optional — only fetch if requested
option(QPP_BUILD_BENCHMARKS "Build benchmarks" ON)
option(QPP_BUILD_PYTHON "Build Python bindings" OFF)

if(QPP_BUILD_BENCHMARKS)
    FetchContent_MakeAvailable(benchmark)
endif()
if(QPP_BUILD_PYTHON)
    FetchContent_MakeAvailable(pybind11)
endif()

# ==============================================================================
# Core library
# ==============================================================================
add_library(qpp_core STATIC
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
    src/qasm/qasm2_parser.cpp
    src/qasm/qasm3_parser.cpp
    src/backends/local_backend.cpp
)

target_include_directories(qpp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(qpp_core PUBLIC OpenMP::OpenMP_CXX)

# Link NLopt
target_link_libraries(qpp_core PUBLIC nlopt)

# Link Eigen3 (header-only)
target_link_libraries(qpp_core PUBLIC Eigen3::Eigen)

# ==============================================================================
# Python bindings
# ==============================================================================
if(QPP_BUILD_PYTHON)
    pybind11_add_module(qpp_python bindings/python_bindings.cpp)
    target_link_libraries(qpp_python PRIVATE qpp_core)
    set_target_properties(qpp_python PROPERTIES OUTPUT_NAME "qpp")
endif()

# ==============================================================================
# Tests
# ==============================================================================
enable_testing()
add_subdirectory(tests)

# ==============================================================================
# Benchmarks
# ==============================================================================
if(QPP_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
