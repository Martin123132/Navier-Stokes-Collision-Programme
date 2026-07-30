from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from annular_vertex_commutator_gate_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_vertex_commutator_gate_audit_v1.json"
)


class AnnularVertexCommutatorGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_sharp_residue_chain_toggle_theorem(self) -> None:
        theorem = self.result["residue_chain_toggle_theorem"]
        self.assertTrue(theorem["all_checks_pass"])
        self.assertEqual(len(theorem["rows"]), 7)
        for row in theorem["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertEqual(
                row["maximum_chain_length"],
                row["expected_maximum_chain_length"],
            )
            self.assertAlmostEqual(
                row["numerical_sum_over_difference_norm"],
                row["sharp_toggle_constant"],
                places=11,
            )
            self.assertAlmostEqual(
                row["numerical_difference_over_sum_norm"],
                row["sharp_toggle_constant"],
                places=11,
            )
            self.assertLess(
                row["sharp_toggle_constant"],
                row["cotangent_upper_bound"],
            )

    def test_tensor_hamming_collapse(self) -> None:
        collapse = self.result["tensor_Hamming_collapse"]
        self.assertTrue(collapse["all_checks_pass"])
        constant = collapse["example_toggle_constant"]
        for distance in range(4):
            self.assertAlmostEqual(
                collapse["extremizing_tensor_ratios"][str(distance)],
                constant**distance,
                places=11,
            )
        self.assertIn(
            "Hamming(v,w)",
            collapse["three_dimensional_inequality"],
        )

    def test_annular_pressure_commutator_theorem(self) -> None:
        theorem = self.result["annular_pressure_commutator_theorem"]
        self.assertTrue(theorem["all_checks_pass"])
        self.assertAlmostEqual(
            theorem["Lambda_equals_2_theta_upper"],
            6.0 / 3.141592653589793,
            places=14,
        )
        self.assertLess(theorem["Lambda_equals_2_theta_upper"], 2.0)
        self.assertEqual(theorem["validity_threshold"], "K>sqrt(3)m")
        self.assertIn(
            "E_v/sqrt(K(K-sqrt(3)m))",
            theorem["single_vertex_pressure_load_bound"],
        )
        self.assertIn("low-output", theorem["scope"])

    def test_shellized_counterexample_stress(self) -> None:
        stress = self.result["shellized_counterexample_stress"]
        self.assertTrue(stress["all_checks_pass"])
        self.assertEqual(
            [row["order"] for row in stress["rows"]],
            [3, 4, 5, 6, 7],
        )
        ratios = []
        for row in stress["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertLess(row["radial_shell_ratio"], 2.0)
            self.assertLess(
                row["maximum_product_coordinate_mode_upper"],
                row["nyquist"],
            )
            self.assertLess(
                row["maximum_relative_divergence_residual"],
                1.0e-10,
            )
            ratios.append(row["diagonal_commutator_ratio"])
        self.assertTrue(
            all(
                first > second
                for first, second in zip(ratios, ratios[1:])
            )
        )
        self.assertLess(stress["ratio_variation_factor"], 1.2)

    def test_scope_flags_and_next_gate(self) -> None:
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags["sharp_residue_chain_toggle_inequality_proved"]
        )
        self.assertTrue(
            flags["annular_Hamming_leakage_collapse_proved"]
        )
        self.assertTrue(
            flags[
                "single_vertex_annular_high_output_pressure_bound_proved"
            ]
        )
        self.assertFalse(flags["sharp_high_output_cutoff_supported"])
        self.assertFalse(flags["low_output_high_high_beat_controlled"])
        self.assertFalse(flags["cross_shell_paraproduct_summation_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])
        self.assertIn(
            "low pressure output",
            self.result["next_theorem_target"],
        )

    def test_stored_result_replays_exactly(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored, self.result)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "annular_diagonal_pressure_commutator_certified",
        )


if __name__ == "__main__":
    unittest.main()
