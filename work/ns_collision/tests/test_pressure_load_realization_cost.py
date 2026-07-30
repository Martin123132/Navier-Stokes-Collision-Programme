from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pressure_load_realization_cost_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "pressure_load_realization_cost_audit_v1.json"
)


class PressureLoadRealizationCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_exact_amplitude_and_partition_scaling(self) -> None:
        scaling = self.result["exact_scaling"]
        self.assertTrue(scaling["all_checks_pass"])
        self.assertEqual(
            scaling["load_scaling"],
            "b_m[u_(a,m)]=a^3 m b_1[u]",
        )
        self.assertEqual(
            scaling["amplitude_homogeneity"]["L3_cubed_for_tau_b"],
            "tau",
        )
        self.assertIn("m||u||_3^3/|b|", scaling["interpretation"])

    def test_single_block_exact_cost_minima(self) -> None:
        block = self.result["single_block_optimization"]
        self.assertTrue(block["all_checks_pass"])
        self.assertEqual(block["H1_product_residual"], "0")
        self.assertEqual(block["L2_optimizer"], "x=y=z=P^(1/3)")
        self.assertIn("6*P**(2/3)", block["exact_L2_minimum"])
        self.assertIn("36P", block["critical_L3_upper_bound"])

    def test_uniform_cubic_and_quadratic_support(self) -> None:
        limits = self.result["normalized_coupling_limits"]
        support = self.result["uniform_lacunary_support"]
        self.assertTrue(limits["all_checks_pass"])
        self.assertEqual(len(limits["rows"]), 7)
        self.assertGreater(
            limits["minimum_limit_normalized_coupling"],
            0.0,
        )
        self.assertTrue(support["all_checks_pass"])
        self.assertEqual(support["leading_signed_mode_count"], 42)
        self.assertEqual(support["leading_zero_triple_count"], 14)
        self.assertEqual(support["invalid_leading_zero_triple_count"], 0)
        self.assertEqual(support["leading_zero_pair_count"], 21)
        self.assertEqual(support["invalid_leading_zero_pair_count"], 0)

    def test_high_carrier_family_and_fisher_visibility(self) -> None:
        family = self.result["high_carrier_realization_family"]
        self.assertTrue(family["all_checks_pass"])
        self.assertLess(family["bounded_L2_ratio_over_sample"], 1.2)
        self.assertLess(
            family["bounded_L3_upper_ratio_over_sample"],
            1.2,
        )
        self.assertLess(
            family["H1_over_M2_ratio_over_sample"],
            1.2,
        )
        for row in family["direct_sparse_rows"]:
            self.assertEqual(row["velocity_mode_count"], 42)
            self.assertLess(row["maximum_load_residual"], 1.0e-10)
            self.assertLess(
                row["maximum_relative_divergence_residual"],
                1.0e-14,
            )
            self.assertTrue(
                row["quadratic_stencil_silence"]["all_checks_pass"]
            )
            self.assertAlmostEqual(
                row["vertex_weighted_H1_squared"],
                row["costs"]["H1_squared"] / 8.0,
            )

    def test_seed81_benchmark_and_nonpromotion(self) -> None:
        seed81 = self.result["seed81_cost_benchmark"]
        self.assertTrue(seed81["all_checks_pass"])
        self.assertAlmostEqual(seed81["velocity_L2_squared"], 100.0)
        self.assertGreater(
            seed81["sampled_velocity_L3_cubed"],
            seed81["velocity_L2_squared"] ** 1.5,
        )
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags["explicit_block_H1_cost_grows_quadratically"]
        )
        self.assertFalse(flags["global_H1_least_cost_coercivity_proved"])
        self.assertFalse(flags["critical_signed_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored, self.result)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertIn(
            "high-carrier absorption threshold",
            stored["next_theorem_target"],
        )


if __name__ == "__main__":
    unittest.main()
