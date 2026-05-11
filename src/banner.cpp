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

void print_lindblad_banner_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        if (!is_interactive_stdout()) {
            return;
        }

        std::cout
            << "\n"
            << "    Welcome to Lindblad Quantum Toolkit R.1.3.2 | CLI startup\n"
            << "__________________________________________________________________________________________________________________________________________________\n\n"
            << "              _     <─. (`─')_  _(`─')   <─.(`─')           (`─')  _ _(`─')             \n"
            << "      <─.    (_)       ╲( OO) )( (OO ).─> __( OO)    <─.    (OO ).─╱( (OO ).─>          \n"
            << "    ,──. )   ,─(`─'),──.╱ ,──╱  ╲    .'_ '─'───.╲  ,──. )   ╱ ,───.  ╲    .'_           \n"
            << "    │  (`─') │ ( OO)│   ╲ │  │  '`'─..__)│ .─. (╱  │  (`─') │ ╲ ╱`.╲ '`'─..__)          \n"
            << "    │  │OO ) │  │  )│  . '│  │) │  │  ' ││ '─' `.) │  │OO ) '─'│_.' ││  │  ' │          \n"
            << "   (│  '__ │(│  │_╱ │  │╲    │  │  │  ╱ :│ ╱`'.  │(│  '__ │(│  .─.  ││  │  ╱ :          \n"
            << "    │     │' │  │'─>│  │ ╲   │  │  '─'  ╱│ '──'  ╱ │     │' │  │ │  ││  '─'  ╱          \n"
            << "    `─────'  `──'   `──'  `──'  `──────' `──────'  `─────'  `──' `──'`──────'           \n"
            << "__________________________________________________________________________________________________________________________________________________\n\n"
            << "    Made with no love (jk lol) and too much coffee (Or are those monster energy drinks?) by Sricharan (verycareful)!\n"
            << "    License Notice: This software is proprietary and source-available. Free for non-commercial and academic use only.\n\nCommercial use of any kind requires a separate written license agreement. \nRedistribution in any form — including forks, copies, and derivative works — is strictly prohibited without \nexplicit written authorization from the author, regardless of whether the use is commercial or non-commercial.\nPublic GitHub forks are technically permitted by GitHub's platform but are not licensed under this agreement \nfor any purpose other than reviewing or submitting contributions via pull request; any other use of a fork constitutes a violation.\nBy submitting any contribution (pull request, code snippet, bug fix, or similar) you irrevocably assign full \ncopyright ownership of that contribution to the author — see 6.3 of LICENSE.\nSee LICENSE for full terms — `qpp.support@proton.me` for licensing inquiries.\n"
            << "    Ready to run tests and simulations.\n\n";
    });
}

void print_lindblad_exit_banner() {
    if (!is_interactive_stdout()) return;
    std::cout
        << "\n"

        << "    Thank you for using Lindblad Quantum Toolkit R.1.3.2\n"
        << "__________________________________________________________________________________________________________________________________________________\n\n"
        << "              _     <─. (`─')_  _(`─')   <─.(`─')           (`─')  _ _(`─')             \n"
        << "      <─.    (_)       ╲( OO) )( (OO ).─> __( OO)    <─.    (OO ).─╱( (OO ).─>          \n"
        << "    ,──. )   ,─(`─'),──.╱ ,──╱  ╲    .'_ '─'───.╲  ,──. )   ╱ ,───.  ╲    .'_           \n"
        << "    │  (`─') │ ( OO)│   ╲ │  │  '`'─..__)│ .─. (╱  │  (`─') │ ╲ ╱`.╲ '`'─..__)          \n"
        << "    │  │OO ) │  │  )│  . '│  │) │  │  ' ││ '─' `.) │  │OO ) '─'│_.' ││  │  ' │          \n"
        << "   (│  '__ │(│  │_╱ │  │╲    │  │  │  ╱ :│ ╱`'.  │(│  '__ │(│  .─.  ││  │  ╱ :          \n"
        << "    │     │' │  │'─>│  │ ╲   │  │  '─'  ╱│ '──'  ╱ │     │' │  │ │  ││  '─'  ╱          \n"
        << "    `─────'  `──'   `──'  `──'  `──────' `──────'  `─────'  `──' `──'`──────'           \n"
        << "__________________________________________________________________________________________________________________________________________________\n\n"
        << "    License Notice: This software is proprietary and source-available. Free for non-commercial and academic use only.\n\nCommercial use of any kind requires a separate written license agreement. \nRedistribution in any form — including forks, copies, and derivative works — is strictly prohibited without \nexplicit written authorization from the author, regardless of whether the use is commercial or non-commercial.\nPublic GitHub forks are technically permitted by GitHub's platform but are not licensed under this agreement \nfor any purpose other than reviewing or submitting contributions via pull request; any other use of a fork constitutes a violation.\nBy submitting any contribution (pull request, code snippet, bug fix, or similar) you irrevocably assign full \ncopyright ownership of that contribution to the author — see 6.3 of LICENSE.\nSee LICENSE for full terms — `qpp.support@proton.me` for licensing inquiries.\n"
        << "__________________________________________________________________________________________________________________________________________________\n\n"
        << "    Lindblad simulation finished. May your qubits stay coherent!\n"
        << "__________________________________________________________________________________________________________________________________________________\n\n";
}

struct BannerInitializer {
    BannerInitializer() { print_lindblad_banner_once(); }
    ~BannerInitializer() { print_lindblad_exit_banner(); }
};

BannerInitializer g_banner_initializer;

} // namespace
