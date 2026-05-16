#pragma once

namespace lindblad {

enum class QuditBackend {
    STATEVECTOR,     // QuditStatevector (default; exact, any d)
    DENSITY_MATRIX,  // QuditDensityMatrix (mixed states + Kraus/Lindblad noise, any d)
    MPS,             // QuditMPS (low-entanglement tensor network, any d)
    CLIFFORD,        // QuditCliffordSimulator (prime d only; Clifford circuits only)
};

} // namespace lindblad
