from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fourier_pressure_load_surjectivity_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "fourier_pressure_load_surjectivity_audit_v1.json"
)


class FourierPressureLoadSurjectivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_exact_pressure_and_walsh_map(self) -> None:
        identity = self.result["exact_Fourier_and_Walsh_map"]
        self.assertTrue(identity["all_checks_pass"])
        self.assertEqual(
            identity["target_loads"],
            identity["expected_Hamming_loads"],
        )
        self.assertEqual(
            identity["target_loads"],
            [
                "-9/256",
                "-27/256",
                "-27/256",
                "-45/256",
                "-27/256",
                "-45/256",
                "-45/256",
                "225/256",
            ],
        )
        self.assertIn("sum_v b_v=0", identity["load_conservation"])

    def test_lacunary_support_certificate(self) -> None:
        construction = self.result[
            "lacunary_surjectivity_construction"
        ]
        support = construction["support_certificate"]
        self.assertTrue(support["all_checks_pass"])
        self.assertEqual(support["signed_mode_count"], 42)
        self.assertEqual(support["unordered_low_triple_count"], 14)
        self.assertEqual(support["invalid_low_triple_count"], 0)
        self.assertEqual(
            construction["block_scales"],
            [8, 32, 128, 512, 2048, 8192, 32768],
        )
        self.assertTrue(
            all(
                row["exact_unit_coupling"] != "0"
                and row["divergence_residuals"] == [0, 0, 0]
                for row in construction["block_rows"]
            )
        )

    def test_surjective_load_realization_and_scope(self) -> None:
        construction = self.result[
            "lacunary_surjectivity_construction"
        ]
        self.assertTrue(construction["all_checks_pass"])
        self.assertLess(
            construction["maximum_relative_divergence_residual"],
            1.0e-14,
        )
        self.assertLess(
            construction["maximum_target_transport_mode_residual"],
            1.0e-11,
        )
        self.assertLess(
            construction["maximum_load_residual"],
            1.0e-11,
        )
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags[
                "instantaneous_zero_sum_load_space_surjectivity_proved"
            ]
        )
        self.assertTrue(
            flags["vertex_saturating_Hamming_load_ray_PDE_realized"]
        )
        self.assertFalse(
            flags["abstract_pointwise_edge_saturator_PDE_realized"]
        )

    def test_taylor_green_seed81_and_nonpromotion(self) -> None:
        taylor_green = self.result["taylor_green_sparse_check"]
        seed81 = self.result["seed81_sparse_benchmark"]
        self.assertTrue(taylor_green["all_checks_pass"])
        self.assertEqual(
            taylor_green["maximum_stencil_transport_mode"],
            0.0,
        )
        self.assertTrue(seed81["all_checks_pass"])
        self.assertLess(abs(seed81["load_sum"]), 1.0e-12)
        self.assertLess(abs(seed81["pressure_residual"]), 1.0e-11)
        flags = self.result["certification_flags"]
        self.assertFalse(flags["uniform_quantitative_load_bound_proved"])
        self.assertFalse(flags["critical_signed_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored["kind"], self.result["kind"])
        self.assertEqual(stored["status"], self.result["status"])
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertIn(
            "least velocity energy",
            stored["next_theorem_target"],
        )


if __name__ == "__main__":
    unittest.main()
