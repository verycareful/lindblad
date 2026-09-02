# ==============================================================================
# LindbladCompilerFlags - optimisation, warning and instruction-set policy
# ==============================================================================
# lindblad_apply_compiler_flags() is called once from the top-level
# CMakeLists.txt and applies directory-wide. It owns three policies that share a
# compiler branch and nothing else: the optimisation and floating-point model,
# the warning level, and the instruction-set baseline.
#
# INSTRUCTION SET. LINDBLAD_MARCH_NATIVE defaults ON, because -march=native is
# what the README build line, the CI workflow and the comparison benchmarks all
# use: it is the configuration every published number describes, so it is the
# one a plain configure should produce. The binary is specific to the machine
# that compiled it, which is the documented trade and is why the option help
# says non-distributable.
#
# OFF selects the portable x86-64-v3 baseline (AVX2 + FMA). That baseline is a
# second configuration to TEST against, not a target to ship. Under the
# project-wide -ffast-math, a near-cancellation evaluated at two instruction
# sets can produce two different numbers, and the only reason anyone found that
# happening is that two configurations disagreed. Keeping both buildable is what
# makes the next such defect visible. The arithmetic whose result must not
# depend on this choice is quarantined -fno-fast-math at its own translation
# units; this flag is not what protects it.
#
# PROBING. -march is an x86 option, and a non-x86 target rejects both spellings
# outright with "unsupported argument to option '-march='", stopping the build.
# Each spelling is therefore probed rather than assumed. Where neither is taken,
# the compiler's own default applies and nothing is lost: -march only ever
# widened the instruction set.
#
# MSVC. There is no /arch:native. LINDBLAD_MARCH_NATIVE cannot mean on MSVC what
# it means elsewhere, so it maps to /arch:AVX512, which carries the same intent
# (widest available, not portable) while being a fixed target rather than a
# probed one. It says so at configure time instead of passing silently, matching
# how LINDBLAD_BUILD_COVERAGE reports being unavailable there.

include(CheckCXXCompilerFlag)

function(lindblad_apply_compiler_flags)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(
            -O3
            -fopenmp
            -funroll-loops
            -ffast-math
            # -ffast-math bundles -ffinite-math-only, under which clang marks
            # floating-point parameters and return values nofpclass(nan inf). A
            # NaN crossing any function boundary is then poison, and every
            # non-finite guard in the tree reads it back as finite. This must
            # follow -ffast-math to override that one member of the bundle.
            -fno-finite-math-only
            -Wall
            -Wextra
            -Wpedantic
        )

        if(LINDBLAD_MARCH_NATIVE)
            check_cxx_compiler_flag("-march=native" LINDBLAD_HAS_MARCH_NATIVE)
            if(LINDBLAD_HAS_MARCH_NATIVE)
                add_compile_options(-march=native)
            else()
                message(STATUS "lindblad: -march=native unsupported here; using compiler default")
            endif()
        else()
            check_cxx_compiler_flag("-march=x86-64-v3" LINDBLAD_HAS_MARCH_X86_64_V3)
            if(LINDBLAD_HAS_MARCH_X86_64_V3)
                add_compile_options(-march=x86-64-v3)
            else()
                message(STATUS "lindblad: -march=x86-64-v3 unsupported here (non-x86 target); using compiler default")
            endif()
        endif()

        add_link_options(-fopenmp)

        if(LINDBLAD_BUILD_COVERAGE)
            add_compile_options(--coverage -O0 -g)
            add_link_options(--coverage)
            message(STATUS "lindblad: coverage instrumentation enabled (--coverage -O0 -g)")
        endif()

    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(
            /O2
            /openmp
            /fp:fast
            /W4
        )

        if(LINDBLAD_MARCH_NATIVE)
            message(STATUS
                "lindblad: MSVC has no /arch:native; LINDBLAD_MARCH_NATIVE selects "
                "/arch:AVX512, a fixed target rather than the host's own")
            add_compile_options(/arch:AVX512)
        else()
            add_compile_options(/arch:AVX2)
        endif()

        if(LINDBLAD_BUILD_COVERAGE)
            message(WARNING
                "LINDBLAD_BUILD_COVERAGE is only supported with GCC/Clang; ignored for MSVC")
        endif()
    endif()
endfunction()
