"""Tests for the annular fixed-output continuum reduction."""

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
    "annular_rho_zero_fixed_output_continuum_gate_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_rho_zero_fixed_output_continuum_gate_audit import (  # noqa: E402
    _active_output_stencil,
    _continuum_certificate,
    _finite_output_diagnostics,
    _permutation_support_certificate,
    _remainder_certificate,
)


class AnnularFixedOutputContinuumGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        predecessor_path = (
            ROOT
            / cls.result["prerequisite"]["path"]
        )
        cls.predecessor = json.loads(
            predecessor_path.read_text(encoding="utf-8")
        )

    def test_active_stencil_is_exact(self) -> None:
        certificate = _active_output_stencil()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(certificate["active_output_count"], 36)
        self.assertEqual(certificate["maximum_active_radius_squared"], 6)
        self.assertEqual(certificate["sum_absolute_alpha_q"], "3/2")
        self.assertEqual(
            certificate["sqrt2_times_projector_sum_matrix"],
            [["0", "0", "0"], ["0", "-1/20", "0"], ["0", "0", "1/20"]],
        )
        self.assertEqual(
            certificate["projector_sum_matrix"],
            "Q=(sqrt(2)/40) diag(0,-1,1)",
        )

    def test_low_test_coefficient_formula_has_expected_pair(self) -> None:
        certificate = _active_output_stencil()
        rows = {
            tuple(row["wave"]): row
            for row in certificate["active_output_rows"]
        }
        self.assertEqual(
            rows[(0, 0, 1)][
                "alpha_q_where_A_q_equals_alpha_q_over_sqrt2"
            ],
            "-1/16",
        )
        self.assertEqual(
            rows[(0, 0, -1)][
                "alpha_q_where_A_q_equals_alpha_q_over_sqrt2"
            ],
            "-1/16",
        )
        self.assertEqual(rows[(0, 0, 1)]["parity"], -1)
        self.assertEqual(rows[(0, 0, -1)]["parity"], -1)

    def test_only_two_permutations_can_saturate_N7(self) -> None:
        certificate = _permutation_support_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(
            certificate["N7_saturating_permutations"],
            [
                "-2 T(V,V,U;Phi)",
                "-4 T(G,H,U;Phi)",
            ],
        )
        bounded_zero = [
            row
            for row in certificate["rows"]
            if row["active_pressure_outputs"]
            == "none for bounded q when N>=5"
        ]
        self.assertEqual(len(bounded_zero), 2)

    def test_continuum_formula_has_both_cancelling_pieces(self) -> None:
        certificate = _continuum_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        limit = certificate["combined_limit"]
        self.assertIn("v_z^2-v_y^2", limit["formula"])
        self.assertIn("g_y a_y-g_z a_z", limit["formula"])
        self.assertIn(
            "N^2",
            certificate["scaled_discrete_fields"]["BHH"],
        )
        self.assertIn(
            "N^5",
            certificate["scaled_discrete_fields"]["B_H_BHH"],
        )

    def test_remainder_is_quantitative_and_boundary_safe(self) -> None:
        certificate = _remainder_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(
            certificate["sampled_profile_bound"]["bound"],
            "epsilon_N <= 64/N for odd N>=5",
        )
        self.assertIn(
            "250000",
            certificate["conditional_full_remainder"],
        )
        self.assertIn(
            "log(2+N)",
            certificate["conditional_full_remainder"],
        )
        self.assertFalse(
            certificate["remaining_channel_bound"]["certified"]
        )
        self.assertFalse(certificate["full_c1_convergence_proved"])
        explanation = certificate["remaining_channel_bound"][
            "why_finite"
        ]
        self.assertIn("Lipschitz", explanation)
        self.assertIn("No C^2 or higher", explanation)

    def test_finite_mode_projection_replays_predecessor(self) -> None:
        stencil = _active_output_stencil()
        replay = _finite_output_diagnostics(
            self.predecessor,
            stencil,
        )
        stored = self.result["finite_output_diagnostics"]
        self.assertTrue(replay["all_checks_pass"])
        self.assertEqual(
            [row["size"] for row in replay["rows"]],
            [5, 7, 9, 13, 17, 21, 25, 29],
        )
        self.assertAlmostEqual(
            replay["largest_carrier"]["active_combined"],
            stored["largest_carrier"]["active_combined"],
            places=10,
        )
        self.assertLess(
            replay["recent_inverse_N_fits"]["combined_quadratic"][
                "candidate_limit"
            ],
            0.0,
        )
        self.assertLess(
            replay["recent_inverse_N_fits"]["combined_cubic"][
                "candidate_limit"
            ],
            0.0,
        )

    def test_scope_is_fail_closed_at_the_sign_gate(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            "annular_four_high_leading_continuum_reduced_tail_sign_pending",
        )
        flags = self.result["certification_flags"]
        self.assertTrue(flags["continuum_limit_formula_proved"])
        self.assertTrue(
            flags["dominant_fixed_output_over_N7_convergence_proved"]
        )
        self.assertFalse(flags["full_c1_over_N7_convergence_proved"])
        self.assertFalse(flags["full_c1_remainder_ledger_complete"])
        self.assertFalse(flags["continuum_limit_nonzero_certified"])
        self.assertFalse(flags["continuum_limit_negative_certified"])
        self.assertFalse(flags["four_high_N9_coefficient_certified"])
        self.assertFalse(flags["finite_parabolic_window_controlled"])
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
