"""Focused tests for the annular first-jet remainder theorem."""

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
    "annular_rho_zero_first_jet_remainder_gate_audit_v1.json"
)
PREDECESSOR = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_first_jet_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_rho_zero_first_jet_remainder_gate_audit import (  # noqa: E402
    _bound_ledger_certificate,
    _compatible_stencil_certificate,
    _support_incidence_certificate,
    _viscous_fisher_row,
)


class AnnularRhoZeroFirstJetRemainderGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.predecessor = json.loads(
            PREDECESSOR.read_text(encoding="utf-8")
        )
        cls.stencil = _compatible_stencil_certificate()
        cls.support = _support_incidence_certificate()
        cls.ledger = _bound_ledger_certificate()

    def test_compatible_stencil_orders_are_exact(self) -> None:
        self.assertTrue(self.stencil["all_checks_pass"])
        self.assertEqual(
            self.stencil["one_dimensional_vanishing_orders"],
            {
                "value": 2,
                "first_derivative": 1,
                "second_derivative": 0,
            },
        )
        self.assertEqual(
            self.stencil["tensor_difference_orders"]["Phi"],
            6,
        )
        self.assertEqual(
            self.stencil["tensor_difference_orders"]["gradient_Phi"],
            5,
        )
        self.assertEqual(
            self.stencil["tensor_difference_orders"]["Laplacian_Phi"],
            4,
        )

    def test_odd_high_incidence_channels_are_excluded(self) -> None:
        self.assertTrue(self.support["all_checks_pass"])
        forbidden = self.support["forbidden_integrated_channels"]
        self.assertTrue(forbidden["one_high_leg"])
        self.assertTrue(forbidden["three_high_legs"])
        self.assertEqual(self.support["allowed_high_leg_parity"], "even")

    def test_dealiased_amplitude_parity_replay(self) -> None:
        parity = self.result["channel_parity_replay"]
        self.assertTrue(parity["all_checks_pass"])
        self.assertLess(parity["maximum_parity_residual"], 6.0e-15)
        self.assertEqual(len(parity["residuals"]), 9)

    def test_viscous_fisher_mixed_difference_identity_recomputes(
        self,
    ) -> None:
        predecessor_rows = {
            row["size"]: row for row in self.predecessor["carrier_rows"]
        }
        row = _viscous_fisher_row(25, predecessor_rows)
        stored = self.result["viscous_weighted_Fisher_theorem"][
            "finite_rows"
        ][0]
        self.assertTrue(row["all_checks_pass"])
        self.assertAlmostEqual(
            row["mixed_difference_heat_pair"],
            stored["mixed_difference_heat_pair"],
            places=14,
        )
        self.assertLess(row["FFT_replay_residual"], 3.0e-14)

    def test_viscous_fisher_rows_have_the_certified_scaling(self) -> None:
        theorem = self.result["viscous_weighted_Fisher_theorem"]
        rows = theorem["finite_rows"]
        self.assertTrue(theorem["all_checks_pass"])
        self.assertEqual(
            [row["size"] for row in rows],
            [25, 29, 33, 37, 41, 49, 65],
        )
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertLess(theorem["maximum_FFT_replay_residual"], 1.0e-13)
        self.assertTrue(
            all(20.0 < row["size_times_heat_pair"] < 25.0 for row in rows)
        )
        self.assertEqual(theorem["total_bound"], "R_F,nu(N)=O(N^3)=o(N^5)")

    def test_branch_power_ledger_is_computed_and_subcritical(self) -> None:
        self.assertTrue(self.ledger["all_checks_pass"])
        for branch in self.ledger["branch_rows"]:
            expected = (
                branch["fixed_scale_N_power"]
                + branch["low_amplitude_power"]
                + branch["coefficient_scale_power"]
            )
            self.assertEqual(
                branch["optimized_power_upper_bound"],
                expected,
            )
            self.assertLess(branch["optimized_power_upper_bound"], 5)
        self.assertEqual(
            self.ledger["maximum_optimized_power_upper_bound"],
            4,
        )
        self.assertEqual(
            self.ledger["total_remainder_bound"],
            "R_N=O(N^4)=o(N^5)",
        )

    def test_dangerous_two_high_two_low_branch_is_replayed(self) -> None:
        rows = self.result["two_high_two_low_finite_rows"]
        self.assertEqual([row["size"] for row in rows], [9, 13, 17, 21, 25])
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertTrue(
            all(
                abs(row["Euler_pressure_a2_coefficient_over_N"]) < 0.006
                for row in rows
            )
        )
        self.assertTrue(
            all(
                abs(
                    row[
                        "weight_advection_pressure_a2_coefficient_over_N"
                    ]
                )
                < 0.001
                for row in rows
            )
        )

    def test_total_first_jet_limit_is_negative(self) -> None:
        certificate = self.result["total_first_jet_limit_certificate"]
        predecessor = self.predecessor[
            "asymptotic_viscous_pressure_certificate"
        ]
        self.assertTrue(certificate["all_checks_pass"])
        self.assertAlmostEqual(
            certificate["total_first_jet_over_N5_limit"],
            predecessor["viscous_pressure_first_jet_over_N5_limit"],
            places=20,
        )
        self.assertLess(
            certificate["total_first_jet_over_N5_limit"],
            0.0,
        )
        self.assertEqual(
            certificate["finite_negative_sizes"],
            [25, 29, 33, 37, 41],
        )

    def test_scope_remains_fail_closed(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            "annular_rho_zero_total_first_jet_N5_limit_certified",
        )
        flags = self.result["certification_flags"]
        self.assertTrue(flags["total_first_jet_N5_limit_certified"])
        self.assertTrue(
            flags["total_first_jet_eventually_negative_proved"]
        )
        self.assertFalse(flags["required_N2_amplification_excluded"])
        self.assertFalse(flags["finite_parabolic_window_controlled"])
        self.assertTrue(flags["second_time_jet_needed"])
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
