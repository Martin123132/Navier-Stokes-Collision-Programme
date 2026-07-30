"""Focused tests for the annular inviscid second-jet branch audit."""

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
    "annular_rho_zero_inviscid_second_jet_branch_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_rho_zero_inviscid_second_jet_branch_audit import (  # noqa: E402
    _branch_row,
    _branch_support_certificate,
    _low_shear_certificate,
    _symbolic_compact_identity_certificate,
)


class AnnularRhoZeroInviscidSecondJetBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_compact_chain_rule_identity_is_exact(self) -> None:
        certificate = _symbolic_compact_identity_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(certificate["chain_rule_residual"], "0")
        self.assertIn(
            "6S(u,E,E;lambda)", certificate["compact_identity"]
        )
        self.assertEqual(
            certificate["annular_amplitude_polynomial"],
            "a**3*c_3 + a*c_1",
        )

    def test_low_shear_is_stationary(self) -> None:
        certificate = _low_shear_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertAlmostEqual(
            certificate["wave_dot_polarization"], 0.0, places=15
        )
        self.assertEqual(certificate["Euler_field"], "B(U,U)=0")

    def test_branch_projection_reduces_padding_to_eightfold(self) -> None:
        certificate = _branch_support_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(certificate["full_second_jet_dealias_factor"], 10)
        self.assertEqual(certificate["implemented_joint_factor"], 8)
        self.assertEqual(
            certificate["inviscid_pressure_amplitude_branches"]["a1"][
                "high_velocity_legs"
            ],
            4,
        )

    def test_small_branch_row_recomputes(self) -> None:
        stored = self.result["small_carrier_validation"]
        replay = _branch_row(3, dealias_factor=8)
        self.assertTrue(replay["all_checks_pass"])
        self.assertAlmostEqual(
            replay["a1_coefficient"],
            stored["a1_coefficient"],
            places=13,
        )
        self.assertAlmostEqual(
            replay["a3_coefficient"],
            stored["a3_coefficient"],
            places=13,
        )
        self.assertLess(
            replay["maximum_forbidden_amplitude_parity_residual"],
            1.0e-12,
        )

    def test_eightfold_and_tenfold_padding_agree(self) -> None:
        replay = self.result["padding_replay"]
        self.assertTrue(replay["all_checks_pass"])
        self.assertLess(replay["maximum_residual"], 1.0e-12)

    def test_compact_polynomial_replays_twenty_channels(self) -> None:
        rows = self.result["predecessor_twenty_channel_replays"]
        self.assertEqual([row["size"] for row in rows], [3, 5])
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertLess(
            max(row["absolute_residual"] for row in rows),
            1.0e-12,
        )

    def test_carrier_rows_have_only_a1_and_a3(self) -> None:
        rows = self.result["carrier_rows"]
        self.assertEqual(
            [row["size"] for row in rows],
            [5, 7, 9, 13, 17, 21, 25, 29],
        )
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertTrue(all(row["a1_coefficient"] < 0.0 for row in rows))
        self.assertTrue(all(row["a3_coefficient"] > 0.0 for row in rows))
        self.assertLess(
            max(
                row["maximum_forbidden_amplitude_parity_residual"]
                for row in rows
            ),
            1.0e-10,
        )

    def test_candidate_leading_signal_is_at_bounded_outputs(self) -> None:
        diagnostics = self.result["finite_pressure_output_diagnostics"]
        largest = self.result["carrier_rows"][-1]
        self.assertTrue(diagnostics["all_checks_pass"])
        self.assertEqual(diagnostics["bounded_output_definition"], "|q|<4")
        self.assertLess(
            largest["combined_dominant_bounded_output_sum"], 0.0
        )
        self.assertLess(
            abs(largest["combined_dominant_outside_bounded_output"]),
            0.1,
        )
        self.assertGreater(
            diagnostics["largest_bounded_fraction_of_dominant"],
            0.999,
        )
        self.assertLess(
            largest["maximum_shell_replay_residual"], 1.0e-8
        )

    def test_route_decision_supersedes_n7_only_triage(self) -> None:
        route = self.result["route_decision"]
        self.assertTrue(route["all_checks_pass"])
        self.assertIn(
            "candidate N9",
            route["candidate_leading_normalizations"][
                "four_high_one_low"
            ],
        )
        self.assertFalse(route["four_high_N9_limit_certified"])
        self.assertFalse(
            route["full_inviscid_pressure_N9_limit_certified"]
        )
        self.assertFalse(route["large_full_second_jet_FFT_authorized"])

    def test_scope_remains_fail_closed(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            "annular_inviscid_second_jet_pressure_branches_isolated",
        )
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags["combined_inviscid_pressure_identity_proved"]
        )
        self.assertTrue(
            flags["amplitude_polynomial_reduced_to_a1_a3"]
        )
        self.assertFalse(flags["four_high_N9_limit_certified"])
        self.assertFalse(
            flags["full_inviscid_pressure_N9_limit_certified"]
        )
        self.assertFalse(
            flags["full_second_jet_N7_coefficient_certified"]
        )
        self.assertFalse(flags["uniform_second_jet_Taylor_bound_proved"])
        self.assertFalse(flags["finite_parabolic_window_controlled"])
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
