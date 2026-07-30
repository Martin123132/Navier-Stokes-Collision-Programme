"""Tests for the parallel-shear Euler-transport Fisher exclusion."""

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
    "annular_parallel_shear_euler_transport_fisher_exclusion_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_parallel_shear_euler_transport_fisher_exclusion_audit import (  # noqa: E402
    _material_identity_certificate,
    _power_exclusion_certificate,
    _vertex_difference_certificate,
)


class ParallelShearEulerTransportFisherExclusionTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_exact_chain_rule_contains_all_five_rows(self) -> None:
        certificate = _material_identity_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(len(certificate["channel_rows"]), 5)
        self.assertEqual(certificate["velocity_degree"], 4)
        self.assertIn(
            "D_t^2 |grad u|^2",
            certificate["combined_material_identity"],
        )
        companion = certificate["weight_self_companion"]
        self.assertEqual(companion["channel_subterm_count"], 7)
        self.assertEqual(companion["velocity_degree"], 2)
        self.assertEqual(companion["weight_degree"], 3)
        self.assertIn(
            "D_t^2 |grad lambda|^2",
            companion["combined_material_identity"],
        )

    def test_degree_four_vertex_leaves_two_differences(self) -> None:
        certificate = _vertex_difference_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(certificate["multiindex_count"], 35)
        self.assertEqual(
            certificate["minimum_compatible_difference_order"], 2
        )
        self.assertEqual(
            certificate["one_dimensional_order_by_coordinate_power"],
            {"0": 2, "1": 1, "2": 0, "3": 1, "4": 0},
        )

    def test_zero_extended_profile_norms_pass(self) -> None:
        certificate = self.audit["packet_difference_norm_certificate"]
        self.assertTrue(certificate["all_checks_pass"])
        maxima = certificate["maxima"]
        self.assertLess(maxima["scaled_l_infinity"], 0.5)
        self.assertLess(maxima["scaled_l1"], 0.2)
        self.assertLess(
            maxima["maximum_scaled_first_difference_l1"], 0.5
        )
        self.assertLess(
            maxima["maximum_pure_second_difference_l1"], 2.4
        )
        self.assertLess(
            maxima["maximum_mixed_second_difference_l1"], 1.3
        )

    def test_five_channel_polynomial_reconstructs(self) -> None:
        replay = self.audit["finite_channel_projection_replay"]
        self.assertTrue(replay["all_checks_pass"])
        self.assertEqual(replay["selected_channel_count"], 5)
        self.assertLess(
            replay["maximum_combined_coefficient_residual"], 1.0e-15
        )
        rows = {
            (row["yz_power"], row["xy_power"]): row
            for row in replay["combined_N5_polynomial"]
        }
        self.assertAlmostEqual(
            rows[(0, 0)]["combined_coefficient"],
            -0.05166846710339710,
            delta=2.0e-15,
        )
        self.assertAlmostEqual(
            rows[(1, 1)]["combined_coefficient"],
            -0.021425969888489327,
            delta=2.0e-15,
        )
        weight_self = self.audit[
            "finite_weight_self_projection_replay"
        ]
        self.assertTrue(weight_self["all_checks_pass"])
        self.assertEqual(weight_self["selected_subterm_count"], 7)
        self.assertEqual(
            weight_self["maximum_combined_coefficient_residual"], 0.0
        )

    def test_power_ledger_closes_only_this_block(self) -> None:
        certificate = _power_exclusion_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(
            certificate["internal_output_shell"]["dyadic_sum_bound"],
            "O(N^7)",
        )
        self.assertTrue(
            all(
                row["strictly_below_N9"]
                for row in certificate["branches"]
            )
        )
        self.assertTrue(
            all(
                row["strictly_below_N9"]
                for row in certificate["weight_self_branches"]
            )
        )
        flags = self.audit["certification_flags"]
        self.assertTrue(
            flags["Euler_transport_weighted_Fisher_o_N9_proved"]
        )
        self.assertTrue(
            flags["Euler_transport_weight_self_o_N9_proved"]
        )
        self.assertTrue(
            flags[
                "all_pure_EA_viscosity_bearing_Fisher_rows_o_N9_proved"
            ]
        )
        self.assertFalse(
            flags["all_viscosity_bearing_second_rows_o_N9_proved"]
        )
        self.assertFalse(flags["complete_second_N9_limit_certified"])

    def test_production_record_passes_and_remains_fail_closed(self) -> None:
        self.assertTrue(self.audit["all_positive_checks_pass"])
        self.assertEqual(self.audit["status"], "passed")
        self.assertIn("O(N^8)=o(N^9)", self.audit["theorem"])
        self.assertIn(
            "v or d",
            self.audit["remaining_gate"].lower(),
        )
        self.assertFalse(
            self.audit["certification_flags"][
                "Navier_Stokes_global_regularity_proved"
            ]
        )


if __name__ == "__main__":
    unittest.main()
