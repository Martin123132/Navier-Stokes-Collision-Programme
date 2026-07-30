from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compatible_eight_cell_cubic_graph_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "compatible_eight_cell_cubic_graph_audit_v1.json"
)


class CompatibleEightCellCubicGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_projection_translation_and_projective_reduction(self) -> None:
        graph = self.result["exact_projection_and_cubic_energy"]
        self.assertTrue(graph["all_checks_pass"])
        self.assertEqual(graph["translation_residual"], "0")
        self.assertEqual(graph["constant_vector_energy"], "0")
        self.assertEqual(graph["projective_symbolic_residual"], "0")
        self.assertIn("sum_v b_v=0", graph["load_conservation"])
        flags = self.result["certification_flags"]
        self.assertTrue(flags["eight_cell_pressure_load_projection_derived"])
        self.assertTrue(
            flags["eight_cell_projective_cubic_reduction_derived"]
        )

    def test_exact_nonconvexity_witness(self) -> None:
        witness = self.result["nonconvexity_witness"]
        self.assertTrue(witness["all_checks_pass"])
        self.assertEqual(witness["exact_convexity_violation"], "39/128")
        self.assertEqual(witness["midpoint_energy"], "95/4")
        self.assertTrue(
            self.result["certification_flags"][
                "eight_cell_cubic_energy_nonconvexity_proved"
            ]
        )

    def test_vertex_saturator_and_scope(self) -> None:
        saturator = self.result["abstract_vertex_saturator"]
        self.assertTrue(saturator["all_checks_pass"])
        self.assertEqual(
            saturator[
                "normalized_cubic_energy_c_equals_t_equals_one"
            ],
            "75/256",
        )
        self.assertEqual(saturator["normalized_pressure_load"], "225/256")
        self.assertEqual(saturator["normalized_objective"], "75/128")
        self.assertEqual(
            saturator["normalized_objective"],
            saturator["normalized_directionwise_envelope"],
        )
        self.assertEqual(
            saturator["normalized_load_by_Hamming_distance"],
            {
                "0": "225/256",
                "1": "-45/256",
                "2": "-27/256",
                "3": "-9/256",
            },
        )
        self.assertTrue(
            all(
                row["load_sum"] == "0"
                for row in saturator["vertex_rows"]
            )
        )
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags["compatibility_only_uniform_strict_gain_falsified"]
        )
        self.assertFalse(flags["abstract_vertex_saturator_PDE_realized"])

    def test_taylor_green_projection_and_nonpromotion(self) -> None:
        taylor_green = self.result["taylor_green_projection"]
        self.assertTrue(taylor_green["all_checks_pass"])
        self.assertLess(
            taylor_green["maximum_numerical_load"],
            1.0e-17,
        )
        self.assertEqual(
            taylor_green["normalized_global_graph_supremum"],
            0.0,
        )
        self.assertGreater(
            taylor_green["normalized_directionwise_envelope"],
            0.0,
        )
        flags = self.result["certification_flags"]
        self.assertFalse(
            flags["Navier_Stokes_pressure_load_cone_characterized"]
        )
        self.assertFalse(flags["critical_signed_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored["kind"], self.result["kind"])
        self.assertEqual(stored["status"], self.result["status"])
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertIn(
            "Fourier-triad map",
            stored["next_theorem_target"],
        )


if __name__ == "__main__":
    unittest.main()
