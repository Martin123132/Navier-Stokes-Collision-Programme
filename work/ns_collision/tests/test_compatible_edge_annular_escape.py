"""Focused tests for the compatible-edge annular escape."""

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
    "compatible_edge_annular_escape_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compatible_edge_annular_escape_audit import (  # noqa: E402
    _edge_penalty_certificate,
    _finite_row,
    _full_field_support_replay,
    _joint_ray_optimum,
)


class CompatibleEdgeAnnularEscapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.edge = _edge_penalty_certificate()
        cls.support = _full_field_support_replay()
        cls.row25 = _finite_row(25)

    def test_exact_delta_penalty_is_common(self) -> None:
        self.assertTrue(self.edge["all_checks_pass"])
        self.assertEqual(self.edge["common_exact_value"], "75/256")
        self.assertEqual(
            set(self.edge["delta_vertex_cubic_energies"].values()),
            {"75/256"},
        )

    def test_full_field_support_and_low_cost_replay(self) -> None:
        self.assertTrue(self.support["all_checks_pass"])
        self.assertLess(
            self.support["maximum_full_load_vs_linear_HHL_residual"],
            4e-12,
        )
        self.assertLess(
            self.support[
                "maximum_full_Fisher_vs_additive_formula_residual"
            ],
            4e-12,
        )
        self.assertEqual(
            set(self.support["low_only_Fisher_by_vertex"].values()),
            {0.4999999999999999},
        )

    def test_joint_ray_optimizer_has_both_stationary_points(self) -> None:
        value = _joint_ray_optimum(2.0, 0.25, 3.0, 0.5, 1.25)
        self.assertTrue(value["all_checks_pass"])
        self.assertTrue(value["positive_escape"])
        self.assertAlmostEqual(
            value["optimal_oriented_low_amplitude"],
            2.0 / (1.25 * 3.0),
            places=15,
        )
        self.assertLess(
            value["coefficient_scale_stationarity_residual"], 2e-14
        )
        self.assertAlmostEqual(
            value["optimized_objective"],
            value["replayed_optimized_objective"],
            places=14,
        )

    def test_N25_recomputes_positive_optimized_escape(self) -> None:
        stored = next(
            row
            for row in self.result["finite_annular_rows"]
            if row["size"] == 25
        )
        self.assertTrue(self.row25["all_checks_pass"])
        self.assertTrue(
            self.row25["ray_optimization"]["positive_escape"]
        )
        self.assertAlmostEqual(
            self.row25["signed_complete_HHL_load"],
            stored["signed_complete_HHL_load"],
            places=14,
        )
        self.assertAlmostEqual(
            self.row25["ray_optimization"]["optimized_objective"],
            stored["ray_optimization"]["optimized_objective"],
            places=14,
        )

    def test_stored_bounded_weight_crossing_is_positive(self) -> None:
        row = next(
            row
            for row in self.result["finite_annular_rows"]
            if row["size"] == 137
        )
        optimum = row["ray_optimization"]
        self.assertTrue(row["all_checks_pass"])
        self.assertTrue(optimum["positive_escape"])
        self.assertTrue(optimum["bounded_scale_one_escape"])
        self.assertGreater(
            optimum["bounded_coefficient_scale_one_objective"], 1e-3
        )

    def test_asymptotic_constants_and_fixed_ray_dichotomy(self) -> None:
        certificate = self.result["asymptotic_ray_certificate"]
        self.assertTrue(certificate["all_checks_pass"])
        self.assertLess(certificate["beta_plus_signed"], 0.0)
        limits = certificate["delta_plus_asymptotics"]
        self.assertGreater(
            limits["linear_margin_over_N_squared_limit"], 0.0
        )
        self.assertGreater(
            limits["optimized_objective_over_N_cubed_limit"], 0.0
        )
        expected = (
            32.0
            * math.sqrt(2.0)
            * certificate["beta_star"] ** 3
            / 45.0
        )
        self.assertAlmostEqual(
            limits["optimized_objective_over_N_cubed_limit"],
            expected,
            places=18,
        )
        dichotomy = certificate["fixed_ray_dichotomy"]
        self.assertIn("Theta(N^3)", dichotomy["suppressed_class"])
        self.assertIn("Theta(N^2)", dichotomy["escaping_class"])

    def test_finite_summary_records_both_escapes(self) -> None:
        summary = self.result["finite_escape_summary"]
        self.assertTrue(summary["finite_escape_checks_pass"])
        self.assertEqual(
            summary["first_audited_positive_optimized_size"], 25
        )
        self.assertEqual(
            summary["first_audited_positive_bounded_scale_one_size"], 137
        )
        self.assertLess(
            summary["largest_size_load_limit_relative_error"], 0.04
        )

    def test_scope_is_fail_closed(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            "complete_compatible_edge_escape_certified",
        )
        flags = self.result["certification_flags"]
        self.assertTrue(flags["complete_full_field_flux_included"])
        self.assertTrue(
            flags["exact_twelve_edge_cubic_penalty_included"]
        )
        self.assertTrue(
            flags["bounded_compatible_coefficient_escape_proved"]
        )
        self.assertFalse(
            flags["static_arbitrary_coefficient_coercivity_proved"]
        )
        self.assertFalse(flags["dynamic_adjoint_coefficient_escape_proved"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
