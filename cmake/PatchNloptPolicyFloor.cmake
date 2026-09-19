# Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
# SPDX-License-Identifier: LicenseRef-Lindblad-2.3
#
# This file is part of the Lindblad Quantum Computing Framework and is
# licensed under the Lindblad Software License Agreement, Version 2.3. The
# full text is in the LICENSE file at the root of the repository. Free for
# non-commercial and academic use; commercial use requires a separate
# Commercial License Agreement with the Author.

# ==============================================================================
# PatchNloptPolicyFloor - raise the CMake floor NLopt's helper scripts declare
# ==============================================================================
# Run as the PATCH_COMMAND of the NLopt FetchContent_Declare in the top-level
# CMakeLists.txt, in script mode, with NLOPT_SOURCE_DIR pointing at the fetched
# copy. It rewrites the cmake_minimum_required floor in the two scripts NLopt
# runs through `cmake -P` at build time to generate its C++ and Fortran
# bindings.
#
# Those scripts are fresh CMake processes: the CMAKE_POLICY_VERSION_MINIMUM the
# project sets for its fetched dependencies covers configure only, and a script
# run through -P reads the variable from the environment or not at all. A
# floor below 3.5 is a hard error on CMake 4, so without this patch every
# build on such a host needs the variable exported by hand before every build
# command. Rewriting the declaration in the fetched copy removes that need
# on every host and every generator.
#
# Confined to the fetched copy under the build tree; the NLopt pin, its
# sources and its numerics are untouched. A replacement that finds nothing to
# do is a no-op, which is what makes the step safe to run twice (ExternalProject
# calls it again after an update). The postcondition is checked rather than
# assumed: the script fails the fetch if either file still declares a floor
# below 3.5 afterwards, so a pin that moves the declaration cannot leave the
# patch silently ineffective.

cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED NLOPT_SOURCE_DIR)
    message(FATAL_ERROR
        "PatchNloptPolicyFloor: NLOPT_SOURCE_DIR must name the fetched NLopt "
        "source directory (pass -DNLOPT_SOURCE_DIR=<dir>)")
endif()

set(_floor "3.5")
set(_scripts
    "${NLOPT_SOURCE_DIR}/cmake/generate-cpp.cmake"
    "${NLOPT_SOURCE_DIR}/cmake/generate-fortran.cmake")

foreach(_script IN LISTS _scripts)
    if(NOT EXISTS "${_script}")
        message(FATAL_ERROR
            "PatchNloptPolicyFloor: ${_script} does not exist; the NLopt "
            "checkout is not where NLOPT_SOURCE_DIR says, or its layout changed")
    endif()

    file(READ "${_script}" _text)

    # NLopt writes the call with a space before the parenthesis; both spellings
    # are matched so a reformatted pin still patches.
    string(REGEX REPLACE
        "cmake_minimum_required[ \t]*\\([ \t]*VERSION[ \t]+3\\.[0-4](\\.[0-9]+)?[ \t]*\\)"
        "cmake_minimum_required(VERSION ${_floor})"
        _patched "${_text}")

    if(NOT _patched STREQUAL _text)
        file(WRITE "${_script}" "${_patched}")
    endif()

    # Postcondition: whatever the file declares now must be at or above the
    # floor CMake 4 accepts.
    string(REGEX MATCH
        "cmake_minimum_required[ \t]*\\([ \t]*VERSION[ \t]+([0-9]+\\.[0-9]+)"
        _decl "${_patched}")
    if(NOT _decl)
        message(FATAL_ERROR
            "PatchNloptPolicyFloor: ${_script} declares no "
            "cmake_minimum_required floor; the NLopt layout changed")
    endif()
    if(CMAKE_MATCH_1 VERSION_LESS _floor)
        message(FATAL_ERROR
            "PatchNloptPolicyFloor: ${_script} still declares VERSION "
            "${CMAKE_MATCH_1}, below ${_floor}")
    endif()
endforeach()
