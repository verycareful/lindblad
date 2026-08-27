# Benchmarks CMakeLists.txt
#
# Two families:
#   - internal micro-benchmarks (bench_gates .. bench_scaling)
#   - the Lindblad vs Qiskit/Aer comparison suite (bench_compare_*), which
#     loads the shared QASM2 corpus from compare/circuits/ (path baked in via
#     LINDBLAD_BENCH_QASM_DIR). The Qiskit half lives in compare/aer_bench.py;
#     tools/bench_report.py merges both into docs/Benchmarks.md.

set(LINDBLAD_BENCH_SOURCES
    bench_gates.cpp
    bench_statevector.cpp
    bench_maqaoa.cpp
    bench_qudit_sv.cpp
    bench_qudit_dm.cpp
    bench_scaling.cpp
    bench_validation_cost.cpp
    bench_compare_sv.cpp
    bench_compare_dm.cpp
    bench_compare_mps.cpp
    bench_compare_clifford.cpp
    bench_compare_transpiler.cpp
    bench_compare_estimator.cpp
)

set(LINDBLAD_BENCH_QASM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/compare/circuits")

foreach(bench_source ${LINDBLAD_BENCH_SOURCES})
    get_filename_component(bench_name ${bench_source} NAME_WE)
    add_executable(${bench_name} ${bench_source} $<TARGET_OBJECTS:lindblad_banner_obj>)
    target_link_libraries(${bench_name}
        PRIVATE
            lindblad_core
            benchmark::benchmark
    )
    target_include_directories(${bench_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_definitions(${bench_name}
        PRIVATE
            LINDBLAD_BENCH_QASM_DIR="${LINDBLAD_BENCH_QASM_DIR}"
    )
endforeach()

# Result-parity validator (plain executable, no Google Benchmark): the
# Lindblad half of the cross-engine correctness gate. See bench_validate.cpp.
add_executable(bench_validate bench_validate.cpp $<TARGET_OBJECTS:lindblad_banner_obj>)
target_link_libraries(bench_validate PRIVATE lindblad_core)
target_include_directories(bench_validate PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(bench_validate
    PRIVATE
        LINDBLAD_BENCH_QASM_DIR="${LINDBLAD_BENCH_QASM_DIR}"
)
