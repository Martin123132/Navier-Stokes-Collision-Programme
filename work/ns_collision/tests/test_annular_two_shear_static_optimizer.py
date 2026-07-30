"""Tests for the annular two-shear static optimizer route guard."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "work/ns_collision/scripts"
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_two_shear_static_optimizer_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_two_shear_static_optimizer_audit import (  # noqa: E402
    _symbolic_low_field_certificate,
)


class AnnularTwoShearStaticOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_exact_symbolic_low_field_coefficients(self) -> None:
        symbolic = _symbolic_low_field_certificate()
        plus = symbolic["plus_vertex_exact"]
        self.assertTrue(symbolic["all_symbolic_checks_pass"])
        self.assertEqual(plus["weighted_Fisher_mass"], "17/16")
        self.assertEqual(plus["kinetic_flux_load"], "-sqrt(2)/16")
        self.assertEqual(plus["pressure_flux_load"], "-sqrt(2)/48")
        self.assertEqual(plus["complete_flux_load"], "-sqrt(2)/12")
        self.assertEqual(symbolic["combined_L2_mass"], "4")

    def test_full_field_polynomials_have_no_omitted_terms(self) -> None:
        replay = self.audit["full_field_support_replay"]
        self.assertTrue(replay["all_support_replay_checks_pass"])
        self.assertLess(
            replay["maximum_complete_polynomial_residual"],
            3.0e-12,
        )
        self.assertLess(
            replay["maximum_pressure_polynomial_residual"],
            3.0e-12,
        )
        self.assertLess(
            replay["maximum_Fisher_polynomial_residual"],
            3.0e-12,
        )
        self.assertAlmostEqual(
            replay["unit_low_weighted_Fisher"],
            17.0 / 16.0,
            delta=2.0e-15,
        )
        self.assertAlmostEqual(
            replay["unit_low_complete_load"],
            -math.sqrt(2.0) / 12.0,
            delta=2.0e-15,
        )
        self.assertAlmostEqual(
            replay["unit_low_pressure_load"],
            -math.sqrt(2.0) / 48.0,
            delta=2.0e-15,
        )

    def test_finite_rows_retain_hhl_sign_and_fisher_scaling(self) -> None:
        rows = self.audit["finite_annular_rows"]
        self.assertEqual(
            [row["size"] for row in rows],
            [3, 5, 9, 13, 17, 25, 33, 49],
        )
        for row in rows:
            self.assertTrue(row["all_finite_checks_pass"])
            self.assertLess(row["complete_HHL_load_over_N"], 0.0)
            self.assertLess(row["pressure_HHL_load_over_N"], 0.0)
            self.assertGreater(row["high_plus_Fisher"], 0.0)
            self.assertLess(
                row["maximum_divergence_residual"],
                2.0e-12,
            )
        self.assertAlmostEqual(
            rows[-1]["complete_HHL_load_over_N"],
            -0.0015147752912734163,
            delta=2.0e-16,
        )
        self.assertAlmostEqual(
            rows[-1]["pressure_HHL_load_over_N"],
            -0.001514775180289843,
            delta=2.0e-16,
        )

    def test_old_finite_optimizer_and_restart_scale_are_rejected(self) -> None:
        optimizer = self.audit["optimizer_and_scaling_certificate"]
        finite = optimizer["finite_N_joint_optimization"]
        restart = optimizer["restart_scaling_decision"]
        self.assertEqual(finite["joint_supremum"], "+infinity")
        self.assertFalse(finite["finite_stationary_optimizer_exists"])
        self.assertFalse(
            restart["old_finite_optimizer_ports_unchanged"]
        )
        self.assertFalse(
            restart["old_a_and_t_Theta_N_scaling_ports_unchanged"]
        )
        self.assertFalse(
            restart["old_Omega_N5_average_generator_gate_ports_unchanged"]
        )
        old_ray = optimizer["carrier_power_table"][
            "old_amplitude_alpha_one"
        ]
        self.assertEqual(old_ray["optimized_t"], "Theta(N^(3/2))")
        self.assertEqual(
            old_ray["optimized_objective"],
            "Theta(N^(9/2))",
        )

    def test_finite_amplitude_N_replay_crosses_both_objectives(self) -> None:
        summary = self.audit["finite_amplitude_equals_N_summary"]
        self.assertEqual(summary["first_complete_positive_size"], 9)
        self.assertEqual(summary["first_pressure_positive_size"], 49)
        self.assertIn(49, summary["complete_positive_sizes"])
        self.assertIn(49, summary["pressure_positive_sizes"])

    def test_scope_does_not_overclaim(self) -> None:
        flags = self.audit["certification_flags"]
        self.assertTrue(self.audit["all_positive_checks_pass"])
        self.assertTrue(flags["joint_static_supremum_is_infinite"])
        self.assertFalse(flags["finite_static_optimizer_exists"])
        self.assertFalse(flags["phase_cancellation_gate_proved"])
        self.assertFalse(flags["complete_finite_first_jet_ported"])
        self.assertFalse(flags["complete_finite_second_jet_ported"])
        self.assertFalse(flags["critical_L3_control_proved"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
