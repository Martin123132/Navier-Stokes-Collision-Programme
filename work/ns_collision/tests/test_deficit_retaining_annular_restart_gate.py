"""Focused tests for the deficit-retaining annular restart gate."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "work/ns_collision/scripts"
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "deficit_retaining_annular_restart_gate_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from deficit_retaining_annular_restart_gate_audit import (  # noqa: E402
    _exact_partition_norms,
    _pressure_ray_row,
    _reset_tax_row,
    _symbolic_deficit_certificate,
)


class DeficitRetainingAnnularRestartGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.symbolic = _symbolic_deficit_certificate()
        cls.partition = _exact_partition_norms()
        cls.row25 = _pressure_ray_row(25)
        cls.tax25 = _reset_tax_row(cls.row25, 0.1)

    def test_legendre_deficit_factorization_is_exact(self) -> None:
        self.assertTrue(self.symbolic["all_checks_pass"])
        self.assertEqual(
            self.symbolic["factorization_symbolic_residual"], "0"
        )
        self.assertEqual(
            self.symbolic["endpoint_algebra_symbolic_residual"], "0"
        )
        self.assertIn(
            "J_0[lambda_T]-Delta_s(lambda_s)",
            self.symbolic[
                "exact_deficit_retaining_restart_identity"
            ],
        )

    def test_partition_norms_and_weight_fisher_are_exact(self) -> None:
        self.assertTrue(self.partition["all_checks_pass"])
        self.assertEqual(
            self.partition["Phi_plus_plus_plus_cube_mean"], "125/4096"
        )
        self.assertEqual(
            self.partition["Phi_plus_plus_plus_L3_norm"], "5/16"
        )
        self.assertEqual(
            self.partition["Phi_weight_Fisher_mean"], "75/4096"
        )
        self.assertEqual(
            self.partition["unit_low_plane_wave_L2_squared"], "2"
        )

    def test_pressure_only_N25_escape_recomputes(self) -> None:
        stored = next(
            row
            for row in self.result["pressure_only_annular_rows"]
            if row["size"] == 25
        )
        self.assertTrue(self.row25["all_checks_pass"])
        self.assertTrue(
            self.row25["ray_optimization"]["positive_escape"]
        )
        self.assertAlmostEqual(
            self.row25["rho_zero_pressure_HHL_load"],
            stored["rho_zero_pressure_HHL_load"],
            places=14,
        )
        self.assertAlmostEqual(
            self.row25["ray_optimization"]["optimized_objective"],
            stored["ray_optimization"]["optimized_objective"],
            places=14,
        )

    def test_rho_zero_correction_removes_kinetic_load(self) -> None:
        self.assertNotEqual(self.row25["kinetic_HHL_load"], 0.0)
        self.assertAlmostEqual(
            self.row25["rho_zero_pressure_HHL_load"],
            self.row25["high_high_pressure_HHL_load"]
            + self.row25["cross_pressure_HHL_load"],
            places=15,
        )
        self.assertAlmostEqual(
            self.row25["pressure_vs_complete_residual"],
            -self.row25["kinetic_HHL_load"],
            places=15,
        )

    def test_reset_tax_dominates_frozen_parabolic_replay(self) -> None:
        self.assertTrue(self.tax25["all_checks_pass"])
        self.assertGreater(
            self.tax25["full_L2_reset_deficit_lower_bound"], 0.5
        )
        self.assertLess(
            self.tax25["frozen_generator_fraction_of_full_tax"], 1e-6
        )
        self.assertGreater(
            self.tax25[
                "required_average_amplification_over_initial_generator"
            ],
            1e6,
        )

    def test_asymptotic_time_ratio_is_five_over_288(self) -> None:
        asymptotic = self.result["asymptotic_reset_tax_certificate"]
        self.assertTrue(asymptotic["all_checks_pass"])
        self.assertAlmostEqual(
            asymptotic[
                "reset_tax_to_three_static_generator_time_limit"
            ],
            5.0 / 288.0,
            places=17,
        )
        self.assertAlmostEqual(
            asymptotic["required_amplification_coefficient"],
            5.0 / 28.8,
            places=16,
        )

    def test_finite_rows_preserve_pressure_limit(self) -> None:
        rows = self.result["pressure_only_annular_rows"]
        self.assertEqual([row["size"] for row in rows], [17, 25, 33, 65])
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertLess(
            rows[-1]["relative_pressure_vs_complete_difference"], 2e-8
        )
        self.assertLess(rows[-1]["pressure_load_over_size"], 0.0)

    def test_scope_remains_fail_closed(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            "annular_static_escape_reset_tax_certified",
        )
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags["exact_deficit_retaining_restart_identity_proved"]
        )
        self.assertTrue(
            flags[
                "parabolic_survival_requires_order_N2_amplification_proved"
            ]
        )
        self.assertFalse(
            flags["static_escape_is_direct_dynamic_counterexample"]
        )
        self.assertFalse(flags["required_nonlinear_amplification_excluded"])
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
