from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cubic_zero_face_edge_envelope_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "cubic_zero_face_edge_envelope_audit_v1.json"
)


class CubicZeroFaceEdgeEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_sharp_zero_face_supremum(self) -> None:
        sharp = self.result["sharp_cubic_edge_envelope"]
        self.assertTrue(sharp["all_checks_pass"])
        self.assertEqual(sharp["symbolic_optimum_residual"], "0")
        self.assertIn(
            "one of A,B is exactly zero",
            sharp["optimizer_geometry"],
        )
        self.assertLess(sharp["maximum_random_residual"], 1.0e-10)
        flags = self.result["certification_flags"]
        self.assertTrue(flags["sharp_zero_face_edge_supremum_derived"])
        self.assertTrue(
            flags["zero_face_reciprocal_singularity_removed"]
        )

    def test_conditional_pressure_L32_reduction(self) -> None:
        pressure = self.result["conditional_pressure_L32_reduction"]
        self.assertTrue(pressure["all_checks_pass"])
        self.assertEqual(
            pressure["partition_derivative_cube_mean"],
            "m**3/(6*pi)",
        )
        self.assertIn(
            "U^2/nu",
            pressure["pressure_L32_reduction"],
        )
        self.assertTrue(
            self.result["certification_flags"][
                "edge_remainder_reduced_to_pressure_L32"
            ]
        )

    def test_scale_homogeneity(self) -> None:
        scaling = self.result["scale_homogeneity"]
        self.assertTrue(scaling["all_checks_pass"])
        self.assertEqual(scaling["symbolic_residual"], "0")
        self.assertEqual(
            scaling["local_Reynolds_power"],
            "(a/(nu m))^(3/2)",
        )
        self.assertFalse(
            scaling["fixed_frequency_uniform_absorption_possible"]
        )

    def test_taylor_green_stress_and_nonpromotion(self) -> None:
        stress = self.result["taylor_green_edge_stress"]
        self.assertTrue(stress["all_checks_pass"])
        self.assertGreater(stress["summed_sharp_envelope"], 0.0)
        self.assertLess(
            stress["x_direction"]["maximum_pointwise_residual"],
            2.0e-16,
        )
        flags = self.result["certification_flags"]
        self.assertFalse(
            flags["globally_compatible_partition_supremum_evaluated"]
        )
        self.assertFalse(flags["pressure_L32_remainder_absorbed"])
        self.assertFalse(flags["critical_signed_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored["kind"], self.result["kind"])
        self.assertEqual(stored["status"], self.result["status"])
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertIn(
            "globally compatible",
            stored["next_theorem_target"],
        )


if __name__ == "__main__":
    unittest.main()
