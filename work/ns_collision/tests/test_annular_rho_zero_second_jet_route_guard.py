"""Focused tests for the annular rho-zero second-jet route guard."""

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
    "annular_rho_zero_second_jet_route_guard_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_rho_zero_second_jet_route_guard_audit import (  # noqa: E402
    _second_jet_row,
    _support_ledger_certificate,
    _symbolic_second_variation_certificate,
)


class AnnularRhoZeroSecondJetRouteGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_symbolic_second_variation_is_exact(self) -> None:
        certificate = _symbolic_second_variation_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(
            certificate["symbolic_residuals"],
            {
                "velocity_Hessian": "0",
                "mixed_Hessian": "0",
                "weight_Hessian": "0",
            },
        )
        self.assertIn("u_2", certificate["second_chain_rule"])
        self.assertIn("lambda_2", certificate["second_chain_rule"])

    def test_support_ledger_requires_tenfold_dealiasing(self) -> None:
        ledger = _support_ledger_certificate()
        self.assertTrue(ledger["all_checks_pass"])
        self.assertEqual(
            ledger["maximum_velocity_acceleration_support"], "3K"
        )
        self.assertEqual(
            ledger["maximum_second_jet_integrand_support"], "5K+O(L)"
        )
        self.assertEqual(ledger["implemented_dealias_factor"], 10)

    def test_small_carrier_chain_rule_recomputes(self) -> None:
        stored = self.result["small_carrier_validation"]
        replay = _second_jet_row(
            3,
            dealias_factor=10,
            low_amplitude_override=0.7,
            coefficient_scale_override=0.9,
        )
        self.assertTrue(replay["all_checks_pass"])
        self.assertAlmostEqual(
            replay["second_variation"]["direct_second_derivative"],
            stored["second_variation"]["direct_second_derivative"],
            places=10,
        )
        self.assertLess(
            replay["second_variation"]["decomposition_residual"],
            2.0e-10,
        )
        self.assertLess(replay["pure_heat_pressure_replay_residual"], 1.0e-10)

    def test_finite_difference_and_padding_replay(self) -> None:
        validation = self.result["small_carrier_validation"]
        finite = validation["finite_difference_validation"]
        padding = self.result["padding_replay"]
        self.assertLess(finite["relative_residual"], 3.0e-10)
        self.assertLess(finite["absolute_residual"], 2.0e-7)
        self.assertTrue(padding["all_checks_pass"])
        self.assertLess(padding["maximum_channel_residual"], 1.0e-10)
        self.assertLess(
            padding["total_second_derivative_residual"], 1.0e-10
        )

    def test_second_small_carrier_row_is_nondegenerate(self) -> None:
        row = self.result["fixed_amplitude_second_small_carrier_row"]
        self.assertTrue(row["all_checks_pass"])
        self.assertEqual(row["size"], 5)
        self.assertFalse(row["static_optimizer_used"])
        self.assertGreater(row["low_amplitude"], 0.0)
        self.assertGreater(row["coefficient_scale"], 0.0)
        self.assertNotEqual(
            row["second_variation"]["direct_second_derivative"], 0.0
        )
        self.assertGreater(
            row[
                "expected_pure_velocity_heat_pressure_second_derivative"
            ],
            0.0,
        )
        self.assertLess(row["pure_heat_pressure_replay_residual"], 1.0e-10)

    def test_second_heat_limit_replays_predecessor_constants(self) -> None:
        certificate = self.result[
            "second_heat_pressure_limit_certificate"
        ]
        self.assertTrue(certificate["all_checks_pass"])
        residuals = certificate["predecessor_replay_residuals"]
        self.assertLess(max(residuals.values()), 2.0e-16)
        limits = certificate["pressure_load_limits"]
        self.assertLess(limits["static_B0_over_N"], 0.0)
        self.assertLess(limits["first_heat_B1_over_N3"], 0.0)
        self.assertLess(limits["second_heat_B2_over_N5"], 0.0)

    def test_positive_pure_heat_N7_coefficient_recomputes(self) -> None:
        limit = self.result["second_heat_pressure_limit_certificate"]
        asymptotic = self.result[
            "pure_heat_pressure_asymptotic_certificate"
        ]
        b0 = limit["pressure_load_limits"]["static_B0_over_N"]
        b2 = limit["pressure_load_limits"]["second_heat_B2_over_N5"]
        ray = math.sqrt(8.0 / (3.0 * 75.0 / 256.0))
        expected = -(abs(b0) ** 2) * ray * b2
        self.assertTrue(asymptotic["all_checks_pass"])
        self.assertAlmostEqual(
            asymptotic[
                "pure_heat_pressure_second_jet_over_N7_limit"
            ],
            expected,
            places=18,
        )
        self.assertGreater(
            expected,
            asymptotic["analytic_strict_positive_lower_bound"],
        )
        self.assertAlmostEqual(
            asymptotic[
                "pure_heat_quadratic_slope_turnaround_scale_N2t"
            ],
            0.07875390524396945,
            places=14,
        )

    def test_finite_heat_rows_have_the_certified_sign(self) -> None:
        rows = self.result["finite_second_heat_load_rows"]
        limit = self.result["second_heat_pressure_limit_certificate"][
            "pressure_load_limits"
        ]["second_heat_B2_over_N5"]
        self.assertEqual(
            [row["size"] for row in rows], [25, 33, 49, 65]
        )
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertTrue(
            all(
                row["second_heat_pressure_load_over_N5"] < 0.0
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["pure_heat_pressure_second_jet_over_N7"] > 0.0
                for row in rows
            )
        )
        self.assertLess(
            abs(rows[-1]["second_heat_pressure_load_over_N5"] - limit),
            abs(rows[0]["second_heat_pressure_load_over_N5"] - limit),
        )

    def test_route_guard_keeps_two_N7_groups_open(self) -> None:
        guard = self.result["second_jet_power_route_guard"]
        self.assertTrue(guard["all_checks_pass"])
        self.assertEqual(
            guard["certified_N7_channel"],
            "double_velocity_heat_pressure",
        )
        self.assertEqual(
            guard["unresolved_possible_N7_channel_groups"],
            [
                "pure_nonlinear_velocity_pressure",
                "pure_weight_transport_and_mixed_pressure",
            ],
        )
        nonlinear = next(
            row
            for row in guard["rows"]
            if row["channel_group"]
            == "pure_nonlinear_velocity_pressure"
        )
        self.assertEqual(nonlinear["route_power"], 9)
        self.assertFalse(guard["full_N7_coefficient_certified"])
        self.assertFalse(guard["all_channels_above_N7_excluded"])
        self.assertFalse(guard["large_carrier_FFT_authorized"])

    def test_scope_remains_fail_closed(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            "annular_rho_zero_second_jet_route_guard_certified",
        )
        flags = self.result["certification_flags"]
        self.assertTrue(flags["exact_second_variation_formula_proved"])
        self.assertTrue(flags["Navier_Stokes_acceleration_retained"])
        self.assertTrue(flags["pressure_Hessian_retained"])
        self.assertTrue(flags["pure_heat_pressure_N7_limit_certified"])
        self.assertFalse(
            flags["full_second_jet_N7_coefficient_certified"]
        )
        self.assertFalse(
            flags["all_second_jet_channels_above_N7_excluded"]
        )
        self.assertFalse(flags["uniform_second_jet_Taylor_bound_proved"])
        self.assertFalse(flags["finite_parabolic_window_controlled"])
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
