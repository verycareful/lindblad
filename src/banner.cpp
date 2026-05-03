#include <cstdio>
#include <iostream>
#include <mutex>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

bool is_interactive_stdout() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

void print_qpp_banner_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        if (!is_interactive_stdout()) {
            return;
        }

        std::cout
            << "\n"
            << "Welcome to Q++ Quantum Toolkit R.1.0.1 | CLI startup\n"
            << "_________________________________________________________________________\n\n"
            << "     .^.    (`-')                      \n"
            << "     '|` ___( OO)                      \n"
            << "      \_/-'---\\_)   ,-.     ,-.       \n"
            << "      |  .-.  |  ,-| |-. ,-| |-.     \n"
            << "      |  | |  |  '-| |-' '-| |-'     \n"
            << "      |  | |  |    `-'     `-'       \n"
            << "      |  '-' .'-.                    \n"
            << "       `-----'--'                    \n"
            << "_________________________________________________________________________\n\n"
            << "Made with no love (jk lol) and too much coffee by Sricharan (verycareful)!\n"
            << "Ready to run tests and simulations.\n\n";
    });
}

void print_qpp_exit_banner() {
    if (!is_interactive_stdout()) return;
    std::cout
        << "\n"
        << "_________________________________________________________________________\n\n"
        << "   Q++ simulation finished. May your qubits stay coherent!\n"
        << "_________________________________________________________________________\n\n";
}

struct BannerInitializer {
    BannerInitializer() { print_qpp_banner_once(); }
    ~BannerInitializer() { print_qpp_exit_banner(); }
};

BannerInitializer g_banner_initializer;

} // namespace
