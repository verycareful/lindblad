#include "lindblad/qudit/qudit_simulator.hpp"

#include <chrono>
#include <stdexcept>

namespace lindblad {

QuditSimulator::Result QuditSimulator::run(
    QuditStatevector& sv,
    const std::vector<QuditGateOp>& ops,
    uint64_t seed)
{
    const auto t_start = std::chrono::high_resolution_clock::now();

    for (const auto& op : ops) {
        if (op.type == QuditGateOp::Type::SINGLE) {
            sv.apply_1qudit(op.q0, op.matrix);
        } else {
            if (op.q1 < 0)
                throw std::invalid_argument(
                    "QuditSimulator: TWO gate requires q1 >= 0");
            sv.apply_2qudit(op.q0, op.q1, op.matrix);
        }
    }

    auto outcome = sv.measure(seed);

    const auto t_end = std::chrono::high_resolution_clock::now();
    const double dt = std::chrono::duration<double>(t_end - t_start).count();
    return {outcome, dt};
}

} // namespace lindblad
