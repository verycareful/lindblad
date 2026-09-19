# Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
# SPDX-License-Identifier: LicenseRef-Lindblad-2.3
#
# This file is part of the Lindblad Quantum Computing Framework and is
# licensed under the Lindblad Software License Agreement, Version 2.3. The
# full text is in the LICENSE file at the root of the repository. Free for
# non-commercial and academic use; commercial use requires a separate
# Commercial License Agreement with the Author.

# apps/CMakeLists.txt
#
# Public-facing executables built on top of lindblad_core. Each app gets its
# own add_executable; do not bundle multiple apps into one target.

add_executable(lindblad_draw lindblad_draw.cpp $<TARGET_OBJECTS:lindblad_banner_obj>)
target_link_libraries(lindblad_draw PRIVATE lindblad_core)
target_include_directories(lindblad_draw PRIVATE ${PROJECT_SOURCE_DIR}/include)
