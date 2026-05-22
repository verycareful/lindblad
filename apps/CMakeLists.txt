# apps/CMakeLists.txt
#
# Public-facing executables built on top of lindblad_core. Each app gets its
# own add_executable; do not bundle multiple apps into one target.

add_executable(lindblad_draw lindblad_draw.cpp $<TARGET_OBJECTS:lindblad_banner_obj>)
target_link_libraries(lindblad_draw PRIVATE lindblad_core)
target_include_directories(lindblad_draw PRIVATE ${PROJECT_SOURCE_DIR}/include)
