# Benchmarks CMakeLists.txt

set(LINDBLAD_BENCH_SOURCES
    bench_gates.cpp
    bench_statevector.cpp
    bench_maqaoa.cpp
    bench_qudit_sv.cpp
    bench_qudit_dm.cpp
    bench_scaling.cpp
)

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
endforeach()
