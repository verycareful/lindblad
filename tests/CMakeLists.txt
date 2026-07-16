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
    # R.1.3.1 test suite
    test_simulators_r130.cpp
    test_mps_sim.cpp
    test_gates_extended.cpp
    test_operators_qi.cpp
    test_ising.cpp
    # R.1.5.1 test suite
    test_feedforward_qft.cpp
    # Qudit BV
    test_qudit_bv.cpp
    # Qudit algorithms (DJ, Grover, QPE, Simon)
    test_qudit_algorithms.cpp
    # R.1.7.1 test suite — qudit backend simulators (DM, MPS, Clifford, noise); fixes in R.1.7.2
    test_qudit_simulators.cpp
    # Bug regression suite — black-box tests for the 8 open TODOs at R.1.7.4
    test_bug_regression.cpp
    # R.1.7.8 test suite — Primitives, DAG, and QASM parser
    test_primitives.cpp
    test_dag.cpp
    test_qasm_parser.cpp
    # R.1.8.1 test suite — Shor's factoring algorithm
    test_shor.cpp
    # Diagnostic suite — pins down gate-based QFT/IQFT convention (R.1.10.5 WIP)
    test_qft_convention.cpp
    # R.1.9.1 test suite — QASM3 Parser tests
    test_qasm3_parser.cpp
    # R.1.10.1 test suite — Circuit visualiser
    test_visualiser_layout.cpp
    test_visualiser_catalogue.cpp
    test_visualiser_format.cpp
    test_visualiser_palette.cpp
    test_visualiser_folding.cpp
    test_visualiser_ascii_fixtures.cpp
    test_visualiser_svg_fixtures.cpp
    test_visualiser_latex_fixtures.cpp
    test_visualiser_html_fixtures.cpp
    # R.1.10.3 additions
    test_visualiser_file_io.cpp
    # R.1.11.1 test suite — qudit completeness (apply_P odd-d, affine oracles, composite-d Simon)
    test_qudit_r1111.cpp
    # 2026-06-11 audit verification suite: asserts correct behaviour per audit
    # findings C1..C17; each test FAILS while its bug is open and becomes a
    # regression once fixed (same convention as test_bug_regression.cpp)
    test_audit_2026_06_11.cpp
    # R.1.12.1 total-coverage suite, Batch 1 (foundations): every public symbol
    # in types/statevector/gates/circuit asserted against its contract; gate
    # formulas pinned numerically; serialization and DAG round-trips.
    test_r1121_types.cpp
    test_r1121_statevector.cpp
    test_r1121_gates_matrices.cpp
    test_r1121_circuit_api.cpp
    test_r1121_circuit_serialization.cpp
    test_r1121_dag_api.cpp
    # R.1.12.1 Batch 2 (engines): execution semantics, noise, operators,
    # quantum info, Clifford tableau, MPS.
    test_r1121_exec_semantics.cpp
    test_r1121_noise_channels.cpp
    test_r1121_noise_model.cpp
    test_r1121_operators.cpp
    test_r1121_quantum_info.cpp
    test_r1121_clifford_tableau.cpp
    test_r1121_mps_state.cpp
    # R.1.12.1 Batch 3 (toolchain): coupling maps, passes, primitives, backend.
    test_r1121_coupling_topologies.cpp
    test_r1121_passes_individual.cpp
    test_r1121_primitives.cpp
    test_r1121_backend.cpp
    # R.1.12.1 Batch 4 (qudit layer + algorithms).
    test_r1121_qudit_core.cpp
    test_r1121_qudit_dm_mps.cpp
    test_r1121_qudit_noise_clifford.cpp
    test_r1121_algorithms_e2e.cpp
    # R.1.12.1 Batch 5 (visualisation options).
    test_r1121_viz_options.cpp
    # R.1.12.1 gap-closure: drive the static symbol matrix to zero unreferenced
    # public symbols (passes + assorted previously-untested public methods).
    test_r1121_passes_more.cpp
    test_r1121_symbol_gaps.cpp
    # R.1.12.2 coverage-fill suite: closes the line-coverage gaps measured on
    # the instrumented R.1.12.1 run (gcovr). F1: simulator engines.
    # F2: transpiler. F3: algorithms. F4: front ends, primitives, backend,
    # noise factories, qudit MPS, circuit/DAG utilities, viz catalogues.
    test_r1122_fill_simulators.cpp
    test_r1122_fill_transpiler.cpp
    test_r1122_fill_algorithms.cpp
    test_r1122_fill_frontends.cpp
    # R.1.13.1 test suite — R.1.13.0 performance-wave coverage: structured ops
    # (MCX/MCP/PERMUTATION), CowMatrix, estimator QWC grouping, MPS SVD toggle +
    # BDC broken-warning, density-matrix rework, shared kernels, Clifford
    # terminal fast path, qudit OpenMP kernels + MPS sampler, algorithm e2e,
    # transpiler BFS distance + SABRE routing legality.
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
    # R.1.14.1 test suite — comparison-benchmark corpus and tooling: every
    # committed QASM2 corpus file parses gate-only with the named qubit count,
    # coupling/observable loaders (structure + failure modes), and behavioural
    # anchors (Grover non-symmetric marked state with restored ancillas, QFT
    # closed-form product state, Clifford-vs-statevector agreement).
    test_r1141_corpus.cpp
    test_r1141_corpus_semantics.cpp
    # R.1.15.1 test suite — routing regression for the R.1.15.0 transpiler
    # correctness wave (#47/#48/#49): smaller-than-map expansion (legality,
    # unitary-level permutation contract at levels 0-1, state relabeling at
    # all levels, non-symmetric end-to-end clbit keys), per-stage preset
    # composition + level validation, unroutable-throws-from-the-right-stage,
    # initial_layout validation/completion, basis contract (basis-only
    # output, u/u3 alias, untranslatable throws, condition propagation,
    # feedforward end-to-end), run-to-run determinism.
    test_r1151_routing_regression.cpp
)

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

# R.1.14.1 : tools/bench_report.py unit + end-to-end tests. Pure stdlib;
# registered whenever a Python3 interpreter exists (independent of
# LINDBLAD_BUILD_PYTHON, which gates the bindings test above).
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME bench_report_tools
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s "${CMAKE_CURRENT_SOURCE_DIR}/tools" -p "test_*.py"
    )
    set_tests_properties(bench_report_tools PROPERTIES
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
