"""Focused tests for the joint primitive HHL incidence-Schur gate."""

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
    "joint_primitive_hhl_incidence_schur_gate_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from joint_primitive_hhl_incidence_schur_gate_audit import (  # noqa: E402
    WINDOWS,
    audit,
)


class JointPrimitiveHHLIncidenceSchurGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.smallest = audit(windows=WINDOWS[:1])["window_rows"][0]

    def test_smallest_window_recomputes_and_reconstructs(self) -> None:
        row = self.smallest
        self.assertTrue(row["all_checks_pass"])
        self.assertEqual(row["complex_high_variable_dimension"], 8)
        self.assertEqual(row["real_low_coordinate_count"], 52)
        self.assertEqual(row["same_sign_high_alias_count"], 0)
        replay = row["direct_reconstruction"]
        self.assertLess(
            replay["maximum_Fisher_matrix_residual"], 3.0e-11
        )
        self.assertLess(
            replay["maximum_load_matrix_residual"], 3.0e-11
        )

    def test_all_eight_joint_windows_pass(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            "finite_window_joint_pressure_growth_witnesses_validated",
        )
        rows = self.result["window_rows"]
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [row["complex_high_variable_dimension"] for row in rows],
            [8, 16, 24, 40, 72, 120, 108, 144],
        )
        self.assertTrue(all(row["all_checks_pass"] for row in rows))

    def test_every_joint_witness_has_unit_cost_and_direct_replay(
        self,
    ) -> None:
        for row in self.result["window_rows"]:
            witness = row["normalized_spectra"]["joint_l2_witness"]
            self.assertAlmostEqual(witness["Fisher_energy"], 1.0, places=10)
            self.assertAlmostEqual(
                witness["low_coordinate_l2_norm"], 1.0, places=10
            )
            self.assertLess(
                witness["matrix_vs_direct_complete_load_residual"],
                3.0e-10,
            )
            self.assertLess(
                witness["component_vs_direct_flux_residual"],
                3.0e-10,
            )

    def test_axial_saturates_while_transverse_witnesses_grow(
        self,
    ) -> None:
        comparisons = self.result["growth_summary"][
            "directional_comparisons"
        ]
        self.assertLess(
            comparisons["axial_length_8_over_4_joint_lower_ratio"],
            1.3,
        )
        self.assertGreater(
            comparisons["strip_width_5_over_3_joint_lower_ratio"],
            1.9,
        )
        self.assertGreater(
            comparisons["slab_width_5_over_3_joint_lower_ratio"],
            2.0,
        )
        slab = comparisons["slab_length_rows"]
        self.assertEqual([row["length"] for row in slab], [4, 6, 8])
        self.assertTrue(
            slab[0]["joint_lower"]
            < slab[1]["joint_lower"]
            < slab[2]["joint_lower"]
        )
        self.assertGreater(slab[-1]["joint_lower"], 0.8)

    def test_growth_witnesses_are_high_high_pressure_dominated(
        self,
    ) -> None:
        fractions = self.result["growth_summary"][
            "directional_comparisons"
        ]["pressure_high_high_fraction_by_window"]
        self.assertEqual(fractions["axial_4"], 0.0)
        self.assertEqual(fractions["axial_8"], 0.0)
        self.assertGreater(fractions["strip_4x3"], 0.9)
        for label in (
            "strip_4x5",
            "slab_4x3x3",
            "slab_4x5x3",
            "slab_6x3x3",
            "slab_8x3x3",
        ):
            self.assertGreater(fractions[label], 0.96)

    def test_scope_remains_fail_closed(self) -> None:
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags[
                "finite_window_pressure_growth_witnesses_validated"
            ]
        )
        self.assertFalse(flags["analytic_unbounded_pressure_family_proved"])
        self.assertFalse(flags["window_uniform_joint_Schur_bound_proved"])
        self.assertFalse(flags["all_cross_shell_HHL_absorbed"])
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
