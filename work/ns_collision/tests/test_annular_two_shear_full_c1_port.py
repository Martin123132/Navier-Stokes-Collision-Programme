"""Tests for the two-shear full c1 tail and limit port."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_two_shear_full_c1_port_audit_v1.json"
)


class AnnularTwoShearFullC1PortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_modified_multiplier_retains_single_difference_bound(self) -> None:
        certificate = self.audit[
            "modified_packet_multiplier_certificate"
        ]
        self.assertTrue(certificate["all_multiplier_checks_pass"])
        self.assertLess(
            certificate[
                "maximum_vector_coordinate_derivative_constant"
            ],
            6.0,
        )
        self.assertLess(
            certificate["derived_first_difference_constant"],
            4.0,
        )
        approximation = certificate[
            "continuum_profile_approximation"
        ]
        self.assertLess(
            approximation["derived_L1_plus_L2_error_constant"],
            256.0,
        )
        self.assertEqual(
            approximation["certified_bound"],
            "epsilon_N=||b_N-b||_1+||b_N-b||_2<=256/N",
        )
        replay = self.audit["finite_first_difference_replay"]
        self.assertTrue(replay["all_replay_checks_pass"])
        self.assertTrue(
            replay["replay_is_not_source_of_analytic_bound"]
        )
        self.assertEqual(
            [row["size"] for row in replay["rows"]],
            [5, 9, 17, 33],
        )
        self.assertLess(
            max(
                row["maximum_scaled_first_difference"]
                for row in replay["rows"]
            ),
            1.6,
        )

    def test_fourteen_tail_profiles_port_linearly(self) -> None:
        port = self.audit["two_shear_tail_port_certificate"]
        self.assertTrue(port["all_tail_port_checks_pass"])
        checks = port["checks"]
        self.assertTrue(checks["fourteen_rows_retained"])
        self.assertTrue(
            checks["every_structural_profile_is_linear_in_low_field"]
        )
        self.assertEqual(port["old_low_fourier_l1"], 2)
        self.assertEqual(port["new_low_fourier_l1"], 4)
        self.assertEqual(port["new_per_atomic_constant"], 751_680)
        self.assertEqual(port["atomic_coefficient_mass"], 94)
        self.assertEqual(port["new_tail_constant"], 70_657_920)

    def test_fixed_output_and_full_limit_close(self) -> None:
        fixed = self.audit["fixed_output_convergence_port"]
        limit = self.audit["full_limit_certificate"]
        self.assertTrue(fixed["all_fixed_output_port_checks_pass"])
        self.assertEqual(fixed["conclusion"], "D_*,N/N^7 -> L_*")
        self.assertTrue(limit["all_full_limit_checks_pass"])
        self.assertEqual(
            limit["conclusion"],
            "c1_*,N/N^7 -> L_*<0",
        )
        self.assertEqual(
            limit["tail_inequality"],
            "|c1_*,N-D_*,N|<=70657920*N^6 for odd N>=5",
        )

    def test_certification_scope_remains_fail_closed(self) -> None:
        flags = self.audit["certification"]
        self.assertTrue(self.audit["all_port_checks_pass"])
        self.assertTrue(
            flags["modified_full_c1_over_N7_convergence_proved"]
        )
        self.assertTrue(
            flags["modified_full_c1_limit_negative_certified"]
        )
        self.assertFalse(flags["original_single_shear_L_EE_sign_certified"])
        self.assertFalse(flags["modified_static_optimizer_ported"])
        self.assertFalse(flags["modified_complete_second_jet_ported"])
        self.assertFalse(flags["modified_parabolic_window_closed"])
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
