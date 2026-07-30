"""Tests for the complete annular four-high c1 tail ledger."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "work/ns_collision/scripts"
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_full_c1_tail_ledger_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_rho_zero_full_c1_tail_ledger_audit import (  # noqa: E402
    _atomic_constant_certificate,
    _coefficient_expansion_certificate,
    _finite_replay,
    _leading_exclusion_certificate,
    _load_prerequisites,
    _operator_certificate,
    _packet_certificate,
    _tail_ledger,
    _vertex_stencil_certificate,
)


class AnnularFullC1TailLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        _, cls.payloads = _load_prerequisites()
        cls.branch = next(
            payload
            for path, payload in cls.payloads.items()
            if "second_jet_branch" in path
        )
        cls.fixed = next(
            payload
            for path, payload in cls.payloads.items()
            if "fixed_output" in path
        )

    def test_amplitude_one_expansion_is_exact(self) -> None:
        certificate = _coefficient_expansion_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        blocks = certificate["computed_blocks"]
        self.assertEqual(
            blocks["6S[u,E,E;Phi]"]["S[H,V,W;Phi]"],
            -24,
        )
        self.assertEqual(
            blocks["6S[u,u,B(u,E);Phi]"]["S[G1,H,H;Phi]"],
            6,
        )
        self.assertEqual(
            blocks["S[u,u,u;lambda2]"]["S[H,H,U;L0]"],
            -3,
        )

    def test_packet_uses_only_one_boundary_safe_difference(self) -> None:
        certificate = _packet_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertLess(
            certificate["derived_first_difference_constant"],
            4.0,
        )
        self.assertEqual(
            certificate["zero_extension_regularities_used"],
            ["bounded", "Lipschitz"],
        )
        self.assertIn(
            "C6",
            certificate["zero_extension_regularities_not_used"],
        )

    def test_every_degree_three_vertex_monomial_has_a_difference(self) -> None:
        certificate = _vertex_stencil_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(certificate["multiindex_count"], 20)
        self.assertEqual(
            certificate["maximum_residual_stencil_l1"],
            "1/2",
        )
        self.assertTrue(
            all(
                row["factored_axis"] is not None
                for row in certificate["multiindex_rows"]
            )
        )

    def test_full_euler_symbol_removes_the_shell_singularity(self) -> None:
        certificate = _operator_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(certificate["kernel_bound_constant"], 400)
        self.assertEqual(
            certificate["kernel_first_difference_constant"],
            280,
        )
        self.assertEqual(certificate["q_monomial_count_bound"], 216)
        self.assertIn(
            "No pressure-shell logarithm is needed",
            certificate["bounded_and_dyadic_output_conclusion"],
        )

    def test_termwise_constant_closes_all_fourteen_atomic_rows(self) -> None:
        operator = _operator_certificate()
        atomic = _atomic_constant_certificate(operator)
        ledger = _tail_ledger(atomic)
        self.assertTrue(atomic["all_checks_pass"])
        self.assertTrue(ledger["all_checks_pass"])
        self.assertEqual(
            atomic["per_atomic_contraction_constant"],
            375_840,
        )
        self.assertEqual(ledger["row_count"], 14)
        self.assertEqual(ledger["absolute_atomic_coefficient_mass"], 94)
        self.assertEqual(ledger["full_tail_constant"], 35_328_960)
        self.assertTrue(
            all(
                row["outer_pressure_output_held_fixed"]
                and row["one_difference_eligible"]
                for row in ledger["rows"]
            )
        )

    def test_only_the_two_continuum_terms_fail_the_shift_test(self) -> None:
        certificate = _leading_exclusion_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(
            [row["expression"] for row in certificate["rows"]],
            [
                "-2T(V,V,U;Phi)",
                "-4T(G,H,U;Phi)",
            ],
        )

    def test_tail_decomposition_replays_all_stored_carriers(self) -> None:
        replay = _finite_replay(self.branch, self.fixed)
        self.assertTrue(replay["all_checks_pass"])
        self.assertEqual(
            replay["sizes"],
            [5, 7, 9, 13, 17, 21, 25, 29],
        )
        self.assertLess(replay["maximum_replay_residual"], 1.0e-10)

    def test_result_closes_only_the_tail_gate(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            "annular_full_c1_over_N7_convergence_certified_sign_pending",
        )
        flags = self.result["certification_flags"]
        self.assertTrue(flags["full_c1_remainder_ledger_complete"])
        self.assertTrue(flags["full_c1_over_N7_convergence_proved"])
        self.assertFalse(flags["continuum_limit_nonzero_certified"])
        self.assertFalse(flags["continuum_limit_negative_certified"])
        self.assertFalse(flags["four_high_N9_coefficient_certified"])
        self.assertFalse(flags["uniform_second_jet_Taylor_bound_proved"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])
        self.assertEqual(
            self.result["full_limit_certificate"]["combined_constant"],
            35_578_960,
        )


if __name__ == "__main__":
    unittest.main()
