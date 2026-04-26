# Benchmarks CMakeLists.txt

set(QPP_BENCH_SOURCES
    bench_gates.cpp
    bench_statevector.cpp
    bench_maqaoa.cpp
)

foreach(bench_source ${QPP_BENCH_SOURCES})
    get_filename_component(bench_name ${bench_source} NAME_WE)
    add_executable(${bench_name} ${bench_source} $<TARGET_OBJECTS:qpp_banner_obj>)
    target_link_libraries(${bench_name}
        PRIVATE
            qpp_core
            benchmark::benchmark
    )
    target_include_directories(${bench_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )
endforeach()
