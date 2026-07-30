"""Tests for the annular parallel-shear finite jet port."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "work/ns_collision/scripts"
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_finite_jet_port_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_parallel_shear_finite_jet_port_audit import (  # noqa: E402
    _structural_certificate,
)


class AnnularParallelShearFiniteJetPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_parallel_low_plane_is_stationary_and_heat_invariant(self) -> None:
        certificate = _structural_certificate()
        self.assertTrue(certificate["all_structural_checks_pass"])
        self.assertEqual(
            certificate["heat_identity"],
            "Delta U=-2U for every (a_yz,a_xy)",
        )
        self.assertIn("HHHL and HLLL", certificate["quartic_support_rule"])

    def test_finite_chain_rule_padding_and_differences_pass(self) -> None:
        small = self.audit["small_carrier_finite_jet_validation"]
        fixed = self.audit["fixed_amplitude_N5_jet_row"]
        self.assertLess(
            small["first_variation"]["decomposition_residual"], 2.0e-9
        )
        self.assertLess(
            small["second_variation"]["decomposition_residual"], 2.0e-8
        )
        self.assertLess(
            fixed["second_variation"]["decomposition_residual"], 2.0e-8
        )
        self.assertLess(
            small["finite_difference_validation"]["first"][
                "relative_residual"
            ],
            3.0e-8,
        )
        self.assertLess(
            small["finite_difference_validation"]["second"][
                "relative_residual"
            ],
            3.0e-8,
        )
        self.assertTrue(self.audit["padding_replay"][
            "all_padding_checks_pass"
        ])
        self.assertTrue(
            self.audit["weight_scale_homogeneity_replay"][
                "all_weight_homogeneity_checks_pass"
            ]
        )

    def test_quartic_mixed_channels_are_explicit(self) -> None:
        projection = self.audit[
            "two_low_amplitude_polynomial_projection"
        ]
        self.assertTrue(projection["all_projection_checks_pass"])
        rows = {
            row["branch"]: row
            for row in projection[
                "quartic_first_inviscid_pressure_enumeration"
            ]
        }
        self.assertLess(
            rows["HHHL"]["maximum_absolute_coefficient"], 2.0e-13
        )
        self.assertLess(
            rows["HLLL"]["maximum_absolute_coefficient"], 2.0e-13
        )
        self.assertLess(
            rows["LLLL"]["maximum_absolute_coefficient"], 2.0e-13
        )
        mixed = next(
            term
            for term in rows["HHLL"]["mixed_polarization_terms"]
            if term["yz_power"] == 1 and term["xy_power"] == 1
        )
        self.assertAlmostEqual(
            mixed["coefficient"],
            0.0027527481207719195,
            delta=2.0e-13,
        )

    def test_quintic_branch_reduction_and_finite_signs(self) -> None:
        projection = self.audit[
            "two_low_amplitude_polynomial_projection"
        ]
        rows = {
            row["branch"]: row
            for row in projection[
                "quintic_second_inviscid_pressure_enumeration"
            ]
        }
        for label in ("HHHHH", "HHHLL", "HLLLL", "LLLLL"):
            self.assertLess(
                rows[label]["maximum_absolute_coefficient"], 3.0e-12
            )
        self.assertAlmostEqual(
            projection["finite_c1_equal_amplitude_coefficient"],
            -0.19198263174267377,
            delta=3.0e-12,
        )
        self.assertAlmostEqual(
            projection["finite_c3_equal_amplitude_coefficient"],
            0.051827377779792495,
            delta=3.0e-12,
        )

    def test_heat_loads_and_continuum_constants(self) -> None:
        constants = self.audit["continuum_heat_constants"]["beta"]
        self.assertAlmostEqual(
            constants["0"], 0.0011545988274271606, delta=2.0e-18
        )
        self.assertAlmostEqual(
            constants["1"], 0.014358876443453603, delta=2.0e-17
        )
        self.assertAlmostEqual(
            constants["2"], 0.18231326692566646, delta=2.0e-16
        )
        rows = self.audit["finite_heat_weighted_HHL_rows"]
        self.assertEqual(
            [row["size"] for row in rows], [5, 9, 17, 25, 33, 49]
        )
        for row in rows:
            self.assertTrue(row["all_heat_load_checks_pass"])
            for power in ("0", "1", "2"):
                self.assertLess(row["powers"][power]["combined"], 0.0)
        self.assertLess(
            abs(
                rows[-1]["powers"]["1"]["normalized_combined"]
                + constants["1"]
            ),
            abs(
                rows[0]["powers"]["1"]["normalized_combined"]
                + constants["1"]
            ),
        )

    def test_scope_distinguishes_inviscid_and_complete_second_jet(self) -> None:
        flags = self.audit["certification_flags"]
        ledger = self.audit["carrier_power_ledger"]
        self.assertTrue(self.audit["all_positive_checks_pass"])
        self.assertTrue(
            flags["parallel_complete_finite_first_jet_ported"]
        )
        self.assertTrue(
            flags["parallel_complete_finite_second_jet_ported"]
        )
        self.assertTrue(
            flags["parallel_first_total_N5_limit_negative"]
        )
        self.assertTrue(
            flags["parallel_second_inviscid_pressure_N9_limit_negative"]
        )
        self.assertFalse(
            flags["parallel_complete_second_N9_limit_certified"]
        )
        self.assertTrue(
            ledger["second_inviscid_pressure_N9_limit_certified"]
        )
        self.assertFalse(ledger["total_second_N9_limit_certified"])
        self.assertFalse(flags["uniform_second_jet_Taylor_remainder_proved"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
