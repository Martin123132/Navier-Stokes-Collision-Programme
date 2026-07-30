"""Focused tests for the annular rho-zero first-jet gate."""

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
    "annular_rho_zero_first_jet_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_rho_zero_first_jet_audit import (  # noqa: E402
    _asymptotic_viscous_pressure_certificate,
    _first_jet_row,
    _heat_weighted_pressure_limit_certificate,
    _symbolic_first_variation_certificate,
)


class AnnularRhoZeroFirstJetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.symbolic = _symbolic_first_variation_certificate()
        cls.limit = _heat_weighted_pressure_limit_certificate()
        cls.asymptotic = _asymptotic_viscous_pressure_certificate(
            cls.limit,
            1.0,
        )
        cls.small = _first_jet_row(
            3,
            low_amplitude_override=0.7,
            coefficient_scale_override=0.9,
            finite_difference_epsilon=1.0e-6,
        )

    def test_directional_first_variations_are_exact(self) -> None:
        self.assertTrue(self.symbolic["all_checks_pass"])
        self.assertEqual(self.symbolic["velocity_symbolic_residual"], "0")
        self.assertEqual(self.symbolic["weight_symbolic_residual"], "0")
        self.assertIn(
            "p'[u;v]",
            self.symbolic["velocity_directional_derivative"],
        )

    def test_small_carrier_central_differences_validate_all_directions(
        self,
    ) -> None:
        finite = self.small["finite_difference_validation"]
        self.assertTrue(self.small["all_checks_pass"])
        self.assertIsNotNone(finite)
        self.assertEqual(
            set(finite["rows"]),
            {
                "velocity_Euler",
                "velocity_viscous",
                "weight_advection",
                "weight_antidiffusion",
            },
        )
        self.assertLess(finite["maximum_absolute_residual"], 4.0e-10)

    def test_independent_padding_replay_is_stable(self) -> None:
        padding = self.result["padding_replay"]
        self.assertTrue(padding["all_checks_pass"])
        self.assertNotEqual(
            padding["base_grid_shape"],
            padding["padded_grid_shape"],
        )
        self.assertLess(padding["maximum_component_residual"], 4.0e-15)

    def test_heat_weighted_continuum_limit_has_analytic_sign(self) -> None:
        self.assertTrue(self.limit["sign_and_margin_check"])
        self.assertLess(self.limit["static_pressure_load_limit"], 0.0)
        self.assertLess(
            self.limit["heat_weighted_pressure_load_limit"],
            -self.limit["analytic_absolute_lower_bound"],
        )
        self.assertAlmostEqual(
            self.limit["heat_weighted_pressure_load_limit"],
            -0.017493957024435965,
            places=16,
        )

    def test_finite_viscous_pressure_identity_replays(self) -> None:
        rows = self.result["carrier_rows"]
        self.assertEqual([row["size"] for row in rows], [25, 29, 33, 37, 41])
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertLess(
            max(row["viscous_pressure_replay_residual"] for row in rows),
            3.0e-13,
        )
        row25 = rows[0]
        self.assertAlmostEqual(
            row25["viscous_pressure_contribution"],
            row25["expected_viscous_pressure_contribution"],
            places=13,
        )
        self.assertLess(
            row25["heat_weighted_pressure_HHL_loads"]["combined"],
            0.0,
        )

    def test_complete_finite_first_jet_is_negative_and_pressure_led(
        self,
    ) -> None:
        rows = self.result["carrier_rows"]
        self.assertTrue(all(row["first_derivative"] < 0.0 for row in rows))
        self.assertTrue(
            all(
                row["viscous_pressure_contribution"] < 0.0
                for row in rows
            )
        )
        scaling = self.result["scaling_diagnostics"]
        self.assertLess(
            scaling["largest_carrier_absolute_remainder_fraction"],
            0.02,
        )
        self.assertLess(scaling["largest_total_over_N5"], 0.0)

    def test_viscous_pressure_N5_limit_is_certified(self) -> None:
        self.assertTrue(self.asymptotic["all_checks_pass"])
        self.assertAlmostEqual(
            self.asymptotic[
                "viscous_pressure_first_jet_over_N5_limit"
            ],
            -1.0442344590350905e-7,
            places=19,
        )
        self.assertLess(
            self.asymptotic[
                "viscous_pressure_first_jet_over_N5_limit"
            ],
            self.asymptotic["analytic_strict_negative_upper_bound"],
        )

    def test_scope_remains_fail_closed(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            "annular_rho_zero_viscous_pressure_N5_limit_certified",
        )
        flags = self.result["certification_flags"]
        self.assertTrue(flags["exact_first_variation_formula_proved"])
        self.assertTrue(
            flags[
                "asymptotic_viscous_pressure_N5_coefficient_certified"
            ]
        )
        self.assertFalse(
            flags["asymptotic_total_first_jet_N5_coefficient_certified"]
        )
        self.assertFalse(flags["required_N2_amplification_excluded"])
        self.assertTrue(flags["second_time_jet_needed"])
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
