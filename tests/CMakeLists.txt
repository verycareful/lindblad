# Tests CMakeLists.txt

# Register all test source files
set(LINDBLAD_TEST_SOURCES
    test_gates.cpp
    test_circuit.cpp
    test_simulators.cpp
    test_noise.cpp
    test_operators.cpp
    test_transpiler.cpp
    test_maqaoa.cpp
    test_maqaoa_noisy.cpp
    test_maqaoa_5qubit.cpp
    test_maqaoa_microgrid.cpp
    test_integration.cpp
    test_classic_algorithms.cpp
    test_simulators_r130.cpp
    test_mps_sim.cpp
    test_gates_extended.cpp
    test_operators_qi.cpp
    test_ising.cpp
    test_feedforward_qft.cpp
    test_qudit_bv.cpp
    test_qudit_algorithms.cpp
    test_qudit_simulators.cpp
    test_bug_regression.cpp
    test_primitives.cpp
    test_dag.cpp
    test_qasm_parser.cpp
    test_shor.cpp
    test_qft_convention.cpp
    test_qasm3_parser.cpp
    test_visualiser_layout.cpp
    test_visualiser_catalogue.cpp
    test_visualiser_format.cpp
    test_visualiser_palette.cpp
    test_visualiser_folding.cpp
    test_visualiser_ascii_fixtures.cpp
    test_visualiser_svg_fixtures.cpp
    test_visualiser_latex_fixtures.cpp
    test_visualiser_html_fixtures.cpp
    test_visualiser_file_io.cpp
    test_qudit_r1111.cpp
    test_audit_2026_06_11.cpp
    test_r1121_types.cpp
    test_r1121_statevector.cpp
    test_r1121_gates_matrices.cpp
    test_r1121_circuit_api.cpp
    test_r1121_circuit_serialization.cpp
    test_r1121_dag_api.cpp
    test_r1121_exec_semantics.cpp
    test_r1121_noise_channels.cpp
    test_r1121_noise_model.cpp
    test_r1121_operators.cpp
    test_r1121_quantum_info.cpp
    test_r1121_clifford_tableau.cpp
    test_r1121_mps_state.cpp
    test_r1121_coupling_topologies.cpp
    test_r1121_passes_individual.cpp
    test_r1121_primitives.cpp
    test_r1121_backend.cpp
    test_r1121_qudit_core.cpp
    test_r1121_qudit_dm_mps.cpp
    test_r1121_qudit_noise_clifford.cpp
    test_r1121_algorithms_e2e.cpp
    test_r1121_viz_options.cpp
    test_r1121_passes_more.cpp
    test_r1121_symbol_gaps.cpp
    test_r1122_fill_simulators.cpp
    test_r1122_fill_transpiler.cpp
    test_r1122_fill_algorithms.cpp
    test_r1122_fill_frontends.cpp
    test_r1131_structured_ops.cpp
    test_r1131_cow_matrix.cpp
    test_r1131_estimator_grouping.cpp
    test_r1131_mps.cpp
    test_r1131_dm.cpp
    test_r1131_kernels.cpp
    test_r1131_clifford.cpp
    test_r1131_qudit.cpp
    test_r1131_algorithms.cpp
    test_r1131_transpiler.cpp
    test_r1141_corpus.cpp
    test_r1141_corpus_semantics.cpp
    test_r1151_routing_regression.cpp
    test_diag_r1160_mps_shor.cpp
    test_diag_r1160_strictfp.cpp
    test_r1161_qudit_mps_stress.cpp
    test_r1171_sv_fusion.cpp
    test_r1171_hw_info.cpp
    test_r1171_dm_superop.cpp
    test_r1181_structured_lowering.cpp
    test_r1181_export_roundtrip.cpp
    test_r1182_ccx_floor.cpp
    test_r1191_sv_gate_validation.cpp
    test_r1191_dm_validation.cpp
    test_r1191_mps_validation.cpp
    test_r1191_clifford_validation.cpp
    test_r1191_qudit_validation.cpp
    test_r1191_preflight.cpp
    test_r1201_constants.cpp
    test_r1201_constants_tu2.cpp
    test_r1201_strict_fp.cpp
    test_r1201_strict_fp_nofast.cpp
    test_r1201_truncation.cpp
    test_r1201_qaoa_validity.cpp
)

# Strict-FP twin of the R.1.16.0 diagnostic: Eigen is header-only, so this
# TU's JacobiSVD/BDCSVD instantiations get IEEE semantics while the rest of
# the suite keeps the project-wide -ffast-math. Mirrors the vendored-NLopt
# precedent (see the nlopt target_compile_options in the top CMakeLists).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_source_files_properties(test_diag_r1160_strictfp.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math")

    # The strict-FP leg of the non-finite sentinel suite. Same mechanism, same
    # reason. r1201_fp_checks.hpp is header-only so this TU compiles the shared
    # checks under IEEE semantics while its twin (test_r1201_strict_fp.cpp)
    # compiles the identical source under the project-wide -ffast-math. The pair
    # is the experiment: quiet_nan_strict and is_finite_strict claim to be
    # independent of the FP model, and two legs that agree are what makes that
    # claim evidence rather than assertion.
    #
    # The non-inline round_trip() both legs call is defined once, in the
    # fast-math twin; this TU links against it, which is deliberate — the call
    # must cross a boundary the optimiser cannot see through.
    set_source_files_properties(test_r1201_strict_fp_nofast.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math")
endif()

# Create test executable
add_executable(lindblad_tests ${LINDBLAD_TEST_SOURCES} $<TARGET_OBJECTS:lindblad_banner_obj>)

target_link_libraries(lindblad_tests
    PRIVATE
        lindblad_core
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(lindblad_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/benchmarks  # compare_common.hpp (R.1.14.1)
)

# R.1.10.1 : path to the visualiser golden-file tree (absolute, set at
# configure time so the test binary finds the files regardless of cwd).
# R.1.14.1 : path to the committed comparison-benchmark corpus, mirroring the
# definition the bench_compare_* targets receive (benchmarks/CMakeLists.txt).
target_compile_definitions(lindblad_tests
    PRIVATE
        LINDBLAD_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
        LINDBLAD_BENCH_QASM_DIR="${PROJECT_SOURCE_DIR}/benchmarks/compare/circuits"
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(lindblad_tests PROPERTIES LABELS "lindblad")

# R.1.12.1 Batch 5 : Python bindings test. Registered ONLY when the pybind
# module is built (LINDBLAD_BUILD_PYTHON=ON); otherwise it is not added, so the
# default C++-only build is unaffected. The test file itself also skips every
# case when the module cannot be imported.
if(LINDBLAD_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME python_bindings
            COMMAND ${Python3_EXECUTABLE} -m unittest discover
                    -s "${CMAKE_CURRENT_SOURCE_DIR}/python" -p "test_*.py"
        )
        set_tests_properties(python_bindings PROPERTIES
            LABELS "lindblad"
            ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:lindblad_python>")
    endif()
endif()

# Python tool tests, discovered from tests/tools/: bench_report.py (unit +
# end-to-end) and the line-ending scanner (classification rules + a scan of the
# real tree). Pure stdlib; registered whenever a Python3 interpreter exists,
# independent of LINDBLAD_BUILD_PYTHON, which gates the bindings test above.
# LINDBLAD_REPO_ROOT is how a case locates the tree it is checking, since ctest
# runs it from the build directory.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME python_tools
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s "${CMAKE_CURRENT_SOURCE_DIR}/tools" -p "test_*.py"
    )
    set_tests_properties(python_tools PROPERTIES
        LABELS "lindblad"
        ENVIRONMENT "LINDBLAD_REPO_ROOT=${PROJECT_SOURCE_DIR}")
endif()

# R.1.14.1 : bench_validate smoke test -- runs the result-parity validator and
# checks it reports a written output. Guarded on the benchmarks option (the
# benchmarks directory is added after tests/, so if(TARGET ...) would be false
# here at configure time); $<TARGET_FILE:...> resolves at generation time.
if(LINDBLAD_BUILD_BENCHMARKS)
    add_test(
        NAME bench_validate_smoke
        COMMAND $<TARGET_FILE:bench_validate>
                "${CMAKE_CURRENT_BINARY_DIR}/bench_validate_smoke.json"
    )
    set_tests_properties(bench_validate_smoke PROPERTIES
        LABELS "lindblad"
        PASS_REGULAR_EXPRESSION "wrote")
endif()

# Standalone diagnostic executable (no gtest)
add_executable(diagnose diagnose.cpp)
target_link_libraries(diagnose PRIVATE lindblad_core)
target_include_directories(diagnose PRIVATE ${PROJECT_SOURCE_DIR}/include)

# R.1.10.1 : standalone golden-file regeneration tool for the visualiser
# tests. Not wired into ctest; run by hand when the rendering format
# changes intentionally.
add_executable(lindblad_visualiser_regen visualiser_regen.cpp)
target_link_libraries(lindblad_visualiser_regen PRIVATE lindblad_core)
target_include_directories(lindblad_visualiser_regen
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(lindblad_visualiser_regen
    PRIVATE
        LINDBLAD_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
)
