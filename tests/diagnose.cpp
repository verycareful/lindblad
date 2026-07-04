#include <iostream>
#include <string>
#include <cmath>

#include "lindblad/algorithms.hpp"
#include "lindblad/circuit.hpp"
#include "lindblad/gates.hpp"
#include "lindblad/operators.hpp"
#include "lindblad/statevector.hpp"
#include "lindblad/simulators/statevector_sim.hpp"
#include "lindblad/simulators/density_matrix_sim.hpp"
#include "lindblad/simulators/clifford_sim.hpp"
#include "lindblad/simulators/mps_sim.hpp"
#include "lindblad/noise.hpp"

using namespace lindblad;
using namespace lindblad::algorithms;
using namespace lindblad::QuantumInfo;

static QuantumCircuit dj_constant_oracle(int n) { return QuantumCircuit(n + 1); }
static QuantumCircuit dj_balanced_oracle(int n) {
    QuantumCircuit qc(n + 1); qc.cx(0, n); return qc;
}
static QuantumCircuit bv_oracle(const std::string& s) {
    int n = s.size();
    QuantumCircuit qc(n + 1);
    for (int i = 0; i < n; ++i) if (s[i] == '1') qc.cx(i, n);
    return qc;
}
static double fidelity(const Statevector& a, const Statevector& b) {
    auto ip = a.inner_product(b); return ip.real*ip.real + ip.imag*ip.imag;
}

int main() {
    // =========================================================================
    std::cout << "=== DeutschJozsa ===\n";
    {
        auto r1 = DeutschJozsa::solve(dj_constant_oracle(3), 3);
        std::cout << "  constant(3): " << (r1.type == DeutschJozsa::Result::CONSTANT ? "CONSTANT" : "BALANCED")
                  << " (expect CONSTANT)\n";
        auto r2 = DeutschJozsa::solve(dj_balanced_oracle(3), 3);
        std::cout << "  balanced(3): " << (r2.type == DeutschJozsa::Result::CONSTANT ? "CONSTANT" : "BALANCED")
                  << " (expect BALANCED)\n";
        auto r3 = DeutschJozsa::solve(dj_constant_oracle(1), 1);
        std::cout << "  constant(1): " << (r3.type == DeutschJozsa::Result::CONSTANT ? "CONSTANT" : "BALANCED")
                  << " (expect CONSTANT)\n";
        auto r4 = DeutschJozsa::solve(dj_balanced_oracle(1), 1);
        std::cout << "  balanced(1): " << (r4.type == DeutschJozsa::Result::CONSTANT ? "CONSTANT" : "BALANCED")
                  << " (expect BALANCED)\n";
    }

    // =========================================================================
    std::cout << "\n=== BernsteinVazirani ===\n";
    {
        for (const auto& s : std::vector<std::string>{"101","1100","000","111","1"}) {
            int n = s.size();
            auto r = BernsteinVazirani::solve(bv_oracle(s), n);
            std::cout << "  secret=" << s << " -> got='" << r.secret << "' (len=" << r.secret.size() << ")\n";
        }
    }

    // =========================================================================
    std::cout << "\n=== RecursiveBernsteinVazirani ===\n";
    {
        {
            std::string s = "1011";
            auto r = RecursiveBernsteinVazirani::solve({bv_oracle(s)}, 4);
            std::cout << "  depth1 s=1011 -> secrets[0]='" << r.secrets[0] << "'\n";
        }
        {
            std::string s0="101", s1="010";
            auto r = RecursiveBernsteinVazirani::solve({bv_oracle(s0),bv_oracle(s1)}, 3);
            std::cout << "  depth2 s0=101,s1=010 -> ['" << r.secrets[0] << "','" << r.secrets[1] << "']\n";
        }
        {
            std::string s0="110",s1="001",s2="111";
            auto r = RecursiveBernsteinVazirani::solve({bv_oracle(s0),bv_oracle(s1),bv_oracle(s2)}, 3, 1, 42);
            std::cout << "  depth3 -> ['" << r.secrets[0] << "','" << r.secrets[1] << "','" << r.secrets[2] << "']\n";
        }
        {
            std::string s0="000",s1="000";
            auto r = RecursiveBernsteinVazirani::solve({bv_oracle(s0),bv_oracle(s1)}, 3);
            std::cout << "  depth2 zeros -> ['" << r.secrets[0] << "','" << r.secrets[1] << "']\n";
        }
        {
            std::string s0="110",s1="011";
            auto r = RecursiveBernsteinVazirani::solve({bv_oracle(s0),bv_oracle(s1)}, 3, 2, 0);
            std::cout << "  depth2 shots=2 -> ['" << r.secrets[0] << "','" << r.secrets[1] << "']\n";
        }
    }

    // =========================================================================
    std::cout << "\n=== ProbabilisticBernsteinVazirani ===\n";
    {
        {
            std::string s="1100";
            auto r = ProbabilisticBernsteinVazirani::solve({bv_oracle(s)}, 4, {}, 20, 0);
            std::cout << "  single-key 1100: discovered=" << r.discovered_keys.size()
                      << " keys=[";
            for (auto& k : r.discovered_keys) std::cout << k << " ";
            std::cout << "]\n";
        }
        {
            std::string s0="101",s1="010";
            auto r = ProbabilisticBernsteinVazirani::solve({bv_oracle(s0),bv_oracle(s1)}, 3, {}, 40, 7);
            std::cout << "  two-key 101,010: discovered=" << r.discovered_keys.size()
                      << " keys=[";
            for (auto& k : r.discovered_keys) std::cout << k << " ";
            std::cout << "]\n";
        }
        {
            std::string s0="100",s1="010",s2="001";
            auto r = ProbabilisticBernsteinVazirani::solve({bv_oracle(s0),bv_oracle(s1),bv_oracle(s2)}, 3, {}, 60, 42);
            std::cout << "  three-key: discovered=" << r.discovered_keys.size()
                      << " keys=[";
            for (auto& k : r.discovered_keys) std::cout << k << " ";
            std::cout << "]\n";
        }
        {
            std::string s0="110",s1="001";
            auto r = ProbabilisticBernsteinVazirani::solve({bv_oracle(s0),bv_oracle(s1)}, 3, {9.0,1.0}, 200, 99);
            std::cout << "  skewed 9:1 -> key0_count=" << r.key_counts.count(s0)
                      << " key1_count=" << r.key_counts.count(s1) << "\n";
            if (r.key_counts.count(s0) && r.key_counts.count(s1))
                std::cout << "    counts: " << s0 << "=" << r.key_counts.at(s0)
                          << " " << s1 << "=" << r.key_counts.at(s1) << "\n";
            // Print all discovered
            for (auto& [k,c] : r.key_counts) std::cout << "    key='" << k << "' count=" << c << "\n";
        }
        {
            std::string s="11";
            auto r = ProbabilisticBernsteinVazirani::solve({bv_oracle(s)}, 2, {}, 17, 0);
            std::cout << "  shots_used: " << r.shots_used << " (expect 17)\n";
        }
        {
            std::string s0="001",s1="110";
            auto r = ProbabilisticBernsteinVazirani::solve({bv_oracle(s1),bv_oracle(s0)}, 3, {}, 60, 5);
            std::cout << "  sorted keys: [";
            for (auto& k : r.discovered_keys) std::cout << k << " ";
            std::cout << "]\n";
        }
        {
            std::string s="101";
            auto r = ProbabilisticBernsteinVazirani::solve({bv_oracle(s),bv_oracle(s)}, 3, {}, 30, 0);
            std::cout << "  duplicate: discovered=" << r.discovered_keys.size()
                      << " key='" << (r.discovered_keys.empty()?"":r.discovered_keys[0])
                      << "' count=" << (r.key_counts.count(s)?r.key_counts.at(s):0) << "\n";
        }
    }

    // =========================================================================
    std::cout << "\n=== SV_MeasureReset ===\n";
    {
        {
            QuantumCircuit qc(1,1); qc.h(0).measure(0,0);
            StatevectorSimulator sim;
            auto res = sim.run(qc, 1000, 42);
            int c0 = res.counts.count("0") ? res.counts.at("0") : 0;
            int c1 = res.counts.count("1") ? res.counts.at("1") : 0;
            std::cout << "  H+measure 1000 shots: '0'=" << c0 << " '1'=" << c1
                      << " (both should be >400)\n";
            std::cout << "  total keys in counts: " << res.counts.size() << "\n";
            for (auto& [k,v] : res.counts) std::cout << "    '" << k << "'=" << v << "\n";
        }
    }

    // =========================================================================
    std::cout << "\n=== CliffordPhase (Y expectation) ===\n";
    {
        {
            StabilizerState st(1); st.apply_h(0); st.apply_s(0);
            std::cout << "  H+S: expect_Y=" << st.expectation_pauli("Y") << " (expect 1)\n";
        }
        {
            StabilizerState st(1); st.apply_h(0); st.apply_s(0); st.apply_z(0);
            std::cout << "  H+S+Z: expect_Y=" << st.expectation_pauli("Y") << " (expect -1)\n";
        }
    }

    // =========================================================================
    std::cout << "\n=== CliffordExpectation (YY, YYX) ===\n";
    {
        {
            StabilizerState st(2); st.apply_h(0); st.apply_cx(0,1);
            std::cout << "  Bell: expect_YY=" << st.expectation_pauli("YY") << " (expect -1)\n";
            std::cout << "  Bell: expect_XX=" << st.expectation_pauli("XX") << " (expect 1)\n";
        }
        {
            StabilizerState st(3); st.apply_h(0); st.apply_cx(0,1); st.apply_cx(0,2);
            std::cout << "  GHZ: expect_YYX=" << st.expectation_pauli("YYX") << " (expect -1)\n";
        }
    }

    // =========================================================================
    std::cout << "\n=== DM_ExpectationSparse.MatchesSVForPureState ===\n";
    {
        Statevector sv(2);
        gates::apply_h(sv, 0);
        gates::apply_cx(sv, 0, 1);
        DensityMatrix rho = DensityMatrix::from_statevector(sv);

        // Print per-term to isolate which one is wrong
        for (const auto& label : std::vector<std::string>{"XX","YY","ZZ","XI"}) {
            double coeff = (label == "XI") ? 0.5 : 1.0;
            SparsePauliOp single = SparsePauliOp::from_list({{label, Complex128(coeff,0.0)}});
            double sv_t = single.expectation_value(sv);
            double dm_t = rho.expectation_value_sparse(single);
            std::cout << "  " << label << " coeff=" << coeff
                      << " sv=" << sv_t << " dm=" << dm_t << "\n";
        }

        SparsePauliOp op = SparsePauliOp::from_list({
            {"XX", Complex128(1.0,0.0)},
            {"YY", Complex128(1.0,0.0)},
            {"ZZ", Complex128(1.0,0.0)},
            {"XI", Complex128(0.5,0.0)}
        });
        double sv_exp = op.expectation_value(sv);
        double dm_exp = rho.expectation_value_sparse(op);
        std::cout << "  total: sv_exp=" << sv_exp << " dm_exp=" << dm_exp
                  << " (true answer=1.0)\n";
    }

    // =========================================================================
    std::cout << "\n=== DM_QubitOrdering.CZ_Symmetric ===\n";
    {
        QuantumCircuit qc(3);
        qc.h(0).h(2); qc.cz(2,0); qc.h(0).h(2); qc.measure_all();
        StatevectorSimulator sv_sim;
        auto sv_res = sv_sim.run(qc, 256, 42);
        NoiseModel ideal;
        DensityMatrixSimulator dm_sim;
        auto dm_res = dm_sim.run(qc, ideal, 256, 42);
        std::string sv_top = sv_res.counts.empty() ? "?" : sv_res.counts.begin()->first;
        std::string dm_top = dm_res.counts.empty() ? "?" : dm_res.counts.begin()->first;
        std::cout << "  SV dominant='" << sv_top << "' DM dominant='" << dm_top << "'\n";
        std::cout << "  SV counts: ";
        for (auto& [k,v] : sv_res.counts) std::cout << "'" << k << "'=" << v << " ";
        std::cout << "\n  DM counts: ";
        for (auto& [k,v] : dm_res.counts) std::cout << "'" << k << "'=" << v << " ";
        std::cout << "\n";
    }

    // =========================================================================
    std::cout << "\n=== MPSSim ===\n";
    {
        {
            QuantumCircuit qc(1); qc.h(0); qc.measure_all();
            MPSSimulator sim;
            auto res = sim.run(qc, 16, 1000, 42);
            int c0 = res.counts.count("0") ? res.counts.at("0") : 0;
            int c1 = res.counts.count("1") ? res.counts.at("1") : 0;
            std::cout << "  H 1-qubit 1000 shots: '0'=" << c0 << " '1'=" << c1 << "\n";
            for (auto& [k,v] : res.counts) std::cout << "    '" << k << "'=" << v << "\n";
        }
        {
            QuantumCircuit qc(2); qc.h(0).cx(0,1); qc.measure_all();
            MPSSimulator sim;
            auto res = sim.run(qc, 16, 1000, 42);
            int c00 = res.counts.count("00") ? res.counts.at("00") : 0;
            int c11 = res.counts.count("11") ? res.counts.at("11") : 0;
            std::cout << "  Bell 2-qubit 1000 shots: '00'=" << c00 << " '11'=" << c11 << "\n";
            for (auto& [k,v] : res.counts) std::cout << "    '" << k << "'=" << v << "\n";
        }
        {
            // to_statevector product state H(0)+X(1)
            QuantumCircuit qc(2); qc.h(0); qc.x(1);
            StatevectorSimulator sv_sim;
            auto sv_res = sv_sim.run(qc, 0, 42);
            MPSSimulator mps_sim;
            auto mps_res = mps_sim.run(qc, 16, 0, 42);
            Statevector mps_sv = mps_res.final_state.to_statevector();
            double F = fidelity(sv_res.final_state, mps_sv);
            std::cout << "  to_statevector H(0)+X(1): fidelity=" << F << " (expect ~1.0)\n";
        }
        {
            // to_statevector 4-qubit
            QuantumCircuit qc(4); qc.h(0); qc.cx(0,1); qc.x(2); qc.h(3);
            StatevectorSimulator sv_sim;
            auto sv_res = sv_sim.run(qc, 0, 42);
            MPSSimulator mps_sim;
            auto mps_res = mps_sim.run(qc, 16, 0, 42);
            Statevector mps_sv = mps_res.final_state.to_statevector();
            double F = fidelity(sv_res.final_state, mps_sv);
            std::cout << "  to_statevector 4-qubit: fidelity=" << F << " (expect ~1.0)\n";
        }
    }

    // =========================================================================
    std::cout << "\n=== QuantumInfo_PartialTrace ===\n";
    {
        Statevector sv(3);
        gates::apply_h(sv, 0);
        gates::apply_cx(sv, 0, 1);
        gates::apply_cx(sv, 0, 2);
        DensityMatrix rho = DensityMatrix::from_statevector(sv);
        DensityMatrix rho_sub = partial_trace(rho, {2});
        std::cout << "  partial_trace(rho_3qubit, {2}): n_qubits=" << rho_sub.n_qubits
                  << " (expect 2) trace=" << rho_sub.trace() << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
