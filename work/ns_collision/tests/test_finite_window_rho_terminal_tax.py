from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from finite_window_rho_terminal_tax_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "finite_window_rho_terminal_tax_audit_v1.json"
)


class FiniteWindowRhoTerminalTaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_exact_terminal_tax_identity(self) -> None:
        theorem = self.result["theorem"]
        self.assertTrue(theorem["all_checks_pass"])
        self.assertEqual(theorem["difference_symbolic_residual"], "0")
        self.assertIn(
            "J_rho-J_0=(3/2)",
            theorem["finite_window_terminal_tax_identity"],
        )
        flags = self.result["certification_flags"]
        self.assertTrue(flags["terminal_tax_identity_derived"])
        self.assertTrue(
            flags["rho_zero_globally_minimizes_fixed_weight_generator"]
        )
        self.assertTrue(
            flags["rho_zero_globally_minimizes_generator_supremum"]
        )

    def test_weighted_chaos_tax_is_monotone(self) -> None:
        stress = self.result["weighted_chaos_stress"]
        self.assertTrue(stress["all_checks_pass"])
        self.assertGreater(stress["terminal_weight_minimum"], 0.0)
        self.assertTrue(stress["tax_monotone_on_sampled_grid"])
        self.assertTrue(
            stress["tax_strictly_positive_for_sampled_positive_rho"]
        )
        self.assertLess(stress["rho_one_tax_residual"], 1.0e-13)
        self.assertLess(
            stress[
                "maximum_Gauss_Hermite_correlation_or_derivative_residual"
            ],
            3.0e-13,
        )

    def test_formal_crossover_is_not_a_sign_target(self) -> None:
        taylor = self.result["taylor_crossover_reinterpretation"]
        self.assertTrue(taylor["all_checks_pass"])
        self.assertFalse(taylor["formal_crossover_is_a_search_target"])
        self.assertFalse(
            taylor["finite_window_seed81_solver_needed_to_decide_net_sign"]
        )
        lower, upper = taylor["formal_integrated_crossover_range"]
        self.assertAlmostEqual(lower, 0.0756122707598066, places=12)
        self.assertAlmostEqual(upper, 0.0756122707598066, places=12)

    def test_scope_and_nonpromotion(self) -> None:
        flags = self.result["certification_flags"]
        self.assertFalse(
            flags[
                "positive_rho_finite_window_advantage_in_this_dual_class"
            ]
        )
        self.assertFalse(
            flags["random_or_path_adapted_weight_route_excluded"]
        )
        self.assertFalse(flags["signed_or_multi_replica_route_excluded"])
        self.assertFalse(
            flags["intrinsic_scale_pressure_tail_bound_proved"]
        )
        self.assertFalse(flags["critical_signed_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored["kind"], self.result["kind"])
        self.assertEqual(stored["status"], self.result["status"])
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertTrue(
            stored["certification_flags"][
                "terminal_tax_nonnegative_for_positive_rho"
            ]
        )


if __name__ == "__main__":
    unittest.main()
