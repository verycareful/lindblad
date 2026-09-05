cmake_minimum_required(VERSION 3.21)
project(lindblad VERSION 1.1.28.0 DESCRIPTION "Lindblad Quantum Computing Framework" LANGUAGES CXX)
# Lindblad version labeling (independent of semantic versioning)
set(LINDBLAD_VERSION_LABEL "1.1.28.0")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Compiler flags
# ==============================================================================
option(LINDBLAD_MARCH_NATIVE "Compile with -march=native (non-distributable, max perf)" ON)

# Coverage instrumentation for gcovr/lcov reports (test infrastructure; the
# measurement workflow is documented in docs/BuildAndTest.md). OFF for normal
# builds: it overrides optimisation with -O0 so line attribution is exact.
option(LINDBLAD_BUILD_COVERAGE "Compile with --coverage for gcovr/lcov reports" OFF)

# Optimisation, warnings and instruction-set policy, one branch per compiler.
# LINDBLAD_MARCH_NATIVE and LINDBLAD_BUILD_COVERAGE above are read there.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(LindbladCompilerFlags)
lindblad_apply_compiler_flags()

# ==============================================================================
# FetchContent dependencies
# ==============================================================================
include(FetchContent)

# CMake 4.0 REMOVED (not deprecated) compatibility with a declared
# cmake_minimum_required below 3.5, so any dependency still declaring an older
# floor is a hard configure error there. NLopt v2.7.1 declares VERSION 3.2 and
# fails outright on a CMake 4 host. This variable tells cmake_minimum_required
# to treat a too-old declared floor as 3.5, and is read by the fetched
# subprojects below.
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
# 5.0.0 is the only supported version. It is pinned for the accuracy of its
# self-adjoint eigensolver on degenerate Hermitian input, which is what the Gram
# rescue route factorises: the null-space tail there lands at 2.086188e-16
# against 3.4.1's 1.059506e-08, the latter being sqrt of an eigenvalue at eps
# and the signature of that route's squared condition number. It is also the
# version on which BDCSVD is correct for degenerate complex input, which is what
# makes BDC the MPS default (see SVDMethod in types.hpp).
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        5.0.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
# Options declared before FetchContent_MakeAvailable so each fetch is conditional
option(LINDBLAD_BUILD_TESTS      "Build test suite (ON when building Lindblad directly, OFF when consumed as a dependency)" ${PROJECT_IS_TOP_LEVEL})
option(LINDBLAD_BUILD_BENCHMARKS "Build benchmarks (ON when building Lindblad directly, OFF when consumed as a dependency)" ${PROJECT_IS_TOP_LEVEL})
option(LINDBLAD_BUILD_PYTHON     "Build Python bindings" OFF)
# Dumps one representative two-site block per distinct shape, plus a shape
# histogram, so an SVD kernel can be benchmarked against the matrices a
# simulation actually produces rather than synthetic ones. OFF everywhere but a
# deliberate harvesting build: when off, src/detail/theta_harvest.cpp is not
# compiled and the call site in mps_sim.cpp is preprocessed away, so neither the
# cost nor the API surface exists.
option(LINDBLAD_MPS_THETA_HARVEST "Dump MPS two-site blocks for external SVD benchmarking" OFF)

# Links the autonne kernels and makes SVDMethod::AutonneJacobi selectable. OFF
# by default: autonne publishes no tag or release, so a consumer has to name the
# commit it wants and there is no defensible default to name for them.
option(LINDBLAD_WITH_AUTONNE "Link autonne and enable SVDMethod::AutonneJacobi" OFF)
set(LINDBLAD_AUTONNE_REPOSITORY "https://github.com/shek014/autonne.git"
    CACHE STRING "Git remote the autonne backend is fetched from")
# A bare commit SHA, not the branch name it currently heads. The branch moves as
# its pull request is revised, so naming it would change what this builds
# against without anything here changing.
#
# This commit is the tip of a four-deep stack: shek014/autonne#2 adds the
# kernels, #3 the corpus, install rules and CI on top of it, and #4 the
# floating-point cross-check on top of that. Pinning the tip rather than the
# kernels alone takes the install and export rules with it, and takes the
# harness scaling fix that came with the cross-check. autonne's main is not a
# usable pin: its svd_thin is still a stub returning false unconditionally.
#
# Re-pin once that repository tags a release. Until it does, a SHA is the only
# thing that identifies a build: project(VERSION) there has read 0.1.0 since the
# initial commit, so the stub and the finished kernel report the same version.
set(LINDBLAD_AUTONNE_GIT_TAG "7af77d7bb6c8633b809cdabd8a260870cde17e8d"
    CACHE STRING "Commit SHA, tag or branch of autonne to build against")
option(LINDBLAD_BUILD_APPS       "Build user-facing CLI apps (lindblad_draw, ...)" ${PROJECT_IS_TOP_LEVEL})

# Belt and braces alongside EIGEN_BUILD_TESTING above: a dependency that gates
# its own test registration on the standard BUILD_TESTING name would otherwise
# register ctest entries for binaries that are never built. Lindblad's own suite
# uses enable_testing() + LINDBLAD_BUILD_TESTS and never consults BUILD_TESTING,
# so forcing it OFF affects dependencies only.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlopt eigen)

# The three MPS translation units take the project-wide -ffast-math. Every
# factorisation they reach is performed in eigen_backend.cpp and checked by the
# reconstruction residual in svd_verify.cpp, both strict unconditionally below,
# so what these files hold is contraction, environment and selection arithmetic.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Unconditionally strict, and NOT under any of the options above. These two
    # TUs are what a released configuration is measured AGAINST: eigen_backend.cpp
    # is the single emitter that makes any FP flag reach Eigen at all, and
    # svd_verify.cpp holds the reconstruction residual that decides whether a
    # factorisation is accepted. Releasing either would remove the instrument
    # rather than the thing being measured.
    set_source_files_properties(
        src/eigen_backend.cpp
        src/svd_verify.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-fast-math")


    # Strict FP for the Class C residuals, same reasoning one level down: these
    # are measurements rather than kernels. Each evaluates a near-cancellation
    # (U†U - I), the shape fast-math perturbs hardest, and the number it
    # produces is compared against the caller's atol and reported in the failure
    # message for a caller to pick a tolerance from. Under -ffast-math the same
    # Hadamard measured 1.2386923780787705e-16 at -march=native against the
    # correct 2.2204460492503131e-16, so the verdict on a matrix sitting near
    # atol depended on the target rather than on the matrix. The check costs a
    # fraction of a percent of the kernel it guards, so buying reproducibility
    # here is cheap.
    set_source_files_properties(src/validate_physical.cpp PROPERTIES
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
    src/eigen_backend.cpp
    src/autonne_backend.cpp
    src/svd_verify.cpp
    src/unitary_repair.cpp
    src/hw_info.cpp
    src/validation.cpp
    src/validate_physical.cpp
    src/svd_truncate.cpp
    src/statevector.cpp
    src/observation.cpp
    src/observers.cpp
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

target_compile_definitions(lindblad_core PUBLIC
    LINDBLAD_VERSION_LABEL="${LINDBLAD_VERSION_LABEL}"
)

# Theta harvest: the source joins the target and the macro is defined only when
# the option is on. PUBLIC because the call site lives in a header-visible
# detail interface, so a consumer compiling against these headers has to agree
# with the library about whether the entry point exists.
if(LINDBLAD_MPS_THETA_HARVEST)
    target_sources(lindblad_core PRIVATE src/detail/theta_harvest.cpp)
    target_compile_definitions(lindblad_core PUBLIC LINDBLAD_MPS_THETA_HARVEST)
endif()

# autonne backend. src/autonne_backend.cpp is in the target either way, so
# SVDMethod::AutonneJacobi resolves in every build; only its body changes.
if(LINDBLAD_WITH_AUTONNE)
    # Refused rather than resolved to whatever the remote's default branch is.
    # autonne carries project(VERSION 0.1.0) unchanged since its initial commit
    # and publishes no tags, so every commit from the stub that returned false
    # to a working kernel reports the same version. An empty pin would leave
    # the choice to the remote, and those two commits differ by whether the
    # backend computes anything at all.
    if(NOT LINDBLAD_AUTONNE_GIT_TAG)
        message(FATAL_ERROR
            "LINDBLAD_WITH_AUTONNE=ON requires a non-empty "
            "LINDBLAD_AUTONNE_GIT_TAG. It has a default; you have cleared it.")
    endif()
    include(FetchContent)
    FetchContent_Declare(autonne
        GIT_REPOSITORY "${LINDBLAD_AUTONNE_REPOSITORY}"
        GIT_TAG        "${LINDBLAD_AUTONNE_GIT_TAG}"
    )
    FetchContent_MakeAvailable(autonne)
    target_link_libraries(lindblad_core PUBLIC autonne::autonne)
    target_compile_definitions(lindblad_core PUBLIC LINDBLAD_WITH_AUTONNE)
endif()

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
