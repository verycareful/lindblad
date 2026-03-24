# Tests CMakeLists.txt

# Register all test source files
set(QPP_TEST_SOURCES
    test_gates.cpp
    test_circuit.cpp
    test_simulators.cpp
    test_noise.cpp
    test_operators.cpp
    test_transpiler.cpp
    test_maqaoa.cpp
    test_maqaoa_5qubit.cpp
    test_maqaoa_microgrid.cpp
    test_maqaoa_20qubit.cpp
    test_integration.cpp
)

# Create test executable
add_executable(qpp_tests ${QPP_TEST_SOURCES})

target_link_libraries(qpp_tests
    PRIVATE
        qpp_core
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(qpp_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(qpp_tests)
