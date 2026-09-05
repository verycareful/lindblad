// =============================================================================
// autonne_backend - the SVDMethod::AutonneJacobi entry point
// =============================================================================
// Why this is its own translation unit, and why the entry point exists whether
// or not autonne is linked, is in
// include/lindblad/detail/autonne_backend.hpp.

#include "lindblad/detail/autonne_backend.hpp"

#include <stdexcept>

#ifdef LINDBLAD_WITH_AUTONNE
#include <autonne/autonne.hpp>
#endif

namespace lindblad {
namespace detail {

#ifdef LINDBLAD_WITH_AUTONNE

// autonne's own MatrixOrder is a distinct type with the same meaning, so the
// two are mapped explicitly. Casting between them because the enumerators
// happen to line up would survive exactly until one of them gained a value.
namespace {

autonne::MatrixOrder to_autonne_order(MatrixOrder order) {
    return order == MatrixOrder::RowMajor ? autonne::MatrixOrder::RowMajor
                                          : autonne::MatrixOrder::ColMajor;
}

}  // namespace

bool autonne_svd_thin(const std::complex<double>* data, int rows, int cols,
                      MatrixOrder order,
                      std::complex<double>* U_out, double* S_out,
                      std::complex<double>* V_out) {
    if (rows <= 0 || cols <= 0) return false;
    // autonne::svd_thin takes no method parameter by design: the library
    // exposes one algorithm per entry point rather than a switch. The
    // translation lives here so no call site has to know that.
    return autonne::svd_thin(data, rows, cols, to_autonne_order(order),
                             U_out, S_out, V_out);
}

#else

bool autonne_svd_thin(const std::complex<double>*, int, int, MatrixOrder,
                      std::complex<double>*, double*, std::complex<double>*) {
    // Throws rather than returning false. False means the factorisation was
    // attempted and did not converge, which sends the ladder into its Gram
    // rescue and yields a valid answer from a kernel the caller did not ask
    // for. Asking for a backend the build does not contain is a different
    // failure and gets a different signal.
    throw std::runtime_error(
        "SVDMethod::AutonneJacobi was requested, but this build did not link "
        "autonne. Reconfigure with -DLINDBLAD_WITH_AUTONNE=ON, or select "
        "SVDMethod::BDC or SVDMethod::Jacobi.");
}

#endif  // LINDBLAD_WITH_AUTONNE

} // namespace detail
} // namespace lindblad
