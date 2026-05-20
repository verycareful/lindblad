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
    # R.1.9.1 test suite — QASM3 Parser tests
    test_qasm3_parser.cpp
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
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(lindblad_tests PROPERTIES LABELS "lindblad")

# Standalone diagnostic executable (no gtest)
add_executable(diagnose diagnose.cpp)
target_link_libraries(diagnose PRIVATE lindblad_core)
target_include_directories(diagnose PRIVATE ${PROJECT_SOURCE_DIR}/include)
