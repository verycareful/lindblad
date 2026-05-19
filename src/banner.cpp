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
            << "    Welcome to Lindblad Quantum Toolkit R.1.7.5 | CLI startup\n"
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
            << "\n"
            << "    License Notice: This software is proprietary and source-available (Lindblad SLA v2.1).\n"
            << "    Free for non-commercial and academic use only. Commercial use requires a separate written license agreement.\n"
            << "    Private non-commercial redistribution of unmodified copies to peers/collaborators is permitted\n"
            << "    under the same license terms (see LICENSE §3.1). Public redistribution in any form — including\n"
            << "    forks, mirrors, and package registries — is prohibited without explicit written authorization.\n"
            << "    Public GitHub forks are not licensed under this agreement for any purpose other than PR review.\n"
            << "    By submitting any contribution you irrevocably assign full copyright ownership to the author\n"
            << "    — see LICENSE §6.3. Full terms: LICENSE | Inquiries: qpp.support@proton.me\n"
            << "\n"
            << "    Ready to run tests and simulations.\n\n";
    });
}

void print_lindblad_exit_banner() {
    if (!is_interactive_stdout()) return;
    std::cout
        << "\n"

        << "    Thank you for using Lindblad Quantum Toolkit R.1.7.5\n"
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
        << "    License Notice: This software is proprietary and source-available (Lindblad SLA v2.1).\n"
        << "    Free for non-commercial and academic use only. Commercial use requires a separate written license agreement.\n"
        << "    Private non-commercial redistribution of unmodified copies to peers/collaborators is permitted\n"
        << "    under the same license terms (see LICENSE §3.1). Public redistribution in any form — including\n"
        << "    forks, mirrors, and package registries — is prohibited without explicit written authorization.\n"
        << "    Public GitHub forks are not licensed under this agreement for any purpose other than PR review.\n"
        << "    By submitting any contribution you irrevocably assign full copyright ownership to the author\n"
        << "    — see LICENSE §6.3. Full terms: LICENSE | Inquiries: qpp.support@proton.me\n"
        << "\n"
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
