from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pressure_hamming_commutator_gate_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "pressure_hamming_commutator_gate_audit_v1.json"
)


class PressureHammingCommutatorGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_exact_vertex_walsh_identity(self) -> None:
        calculus = self.result["exact_vertex_Walsh_calculus"]
        self.assertTrue(calculus["all_checks_pass"])
        self.assertTrue(calculus["distance_two_terms_are_genuine"])
        self.assertTrue(calculus["distance_three_term_is_genuine"])
        self.assertTrue(
            calculus["nonzero_masks_by_hamming_order"]["2"]
        )
        self.assertTrue(
            calculus["nonzero_masks_by_hamming_order"]["3"]
        )
        self.assertLess(
            calculus["phase_corrected_Fourier_identity_residual"],
            1.0e-15,
        )
        self.assertIn("psi_(v xor S)", calculus["mixed_derivative_toggle"])

    def test_hamming_leakage_matrix(self) -> None:
        leakage = self.result["Hamming_leakage_bound"]
        self.assertTrue(leakage["all_checks_pass"])
        self.assertAlmostEqual(
            leakage["smooth_prototype_matrix_norm"],
            leakage["smooth_prototype_exact_row_sum"],
            places=12,
        )
        self.assertAlmostEqual(
            leakage["bounded_multiplier_matrix_norm"],
            8.0,
            places=12,
        )
        self.assertEqual(
            sorted(leakage["higher_strata"]),
            [
                "distance_0",
                "distance_1",
                "distance_2",
                "distance_3",
            ],
        )

    def test_two_scale_divergence_free_no_go(self) -> None:
        theorem = self.result["two_scale_counterexample_theorem"]
        self.assertTrue(theorem["all_checks_pass"])
        self.assertEqual(
            theorem["fixed_minimum_velocity_mode"],
            "sqrt(3)*(2+1)=3*sqrt(3)",
        )
        self.assertEqual(
            theorem["proved_asymptotic_bounds"][
                "diagonal_ratio_lower"
            ],
            "ratio>=c N^(1/2)",
        )
        self.assertIn(
            "div u_N=0",
            theorem["exact_divergence"],
        )
        self.assertIn(
            "no bandwidth-independent C",
            theorem["falsified_statement"],
        )

    def test_finite_fourier_counterexample_pilot(self) -> None:
        pilot = self.result["finite_Fourier_counterexample_pilot"]
        self.assertTrue(pilot["all_checks_pass"])
        self.assertEqual(
            [row["order"] for row in pilot["rows"]],
            [4, 6, 8, 10, 12, 14],
        )
        ratios = []
        scaled_pressure = []
        for row in pilot["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertAlmostEqual(
                row["minimum_velocity_mode"],
                row["expected_minimum_velocity_mode"],
                places=10,
            )
            self.assertLess(
                row["maximum_product_coordinate_mode_upper"],
                row["nyquist"],
            )
            self.assertLess(
                row["maximum_relative_divergence_residual"],
                1.0e-10,
            )
            ratios.append(row["diagonal_commutator_ratio"])
            scaled_pressure.append(row["N_cubed_weighted_pressure"])
        self.assertTrue(
            all(
                first < second
                for first, second in zip(ratios, ratios[1:])
            )
        )
        self.assertTrue(
            all(
                first < second
                for first, second in zip(
                    scaled_pressure,
                    scaled_pressure[1:],
                )
            )
        )
        self.assertGreater(
            pilot["observed_ratio_growth_factor"],
            50.0,
        )

    def test_scope_and_route_decision(self) -> None:
        flags = self.result["certification_flags"]
        self.assertTrue(flags["exact_eight_shift_Walsh_identity_proved"])
        self.assertTrue(
            flags["distance_two_and_three_pressure_leakage_genuine"]
        )
        self.assertTrue(
            flags["coupled_eight_cell_multiplier_bound_proved"]
        )
        self.assertTrue(
            flags[
                "lower_carrier_only_diagonal_commutator_bound_falsified"
            ]
        )
        self.assertFalse(
            flags[
                "intrinsic_amplitude_condition_repairs_diagonal_bound"
            ]
        )
        self.assertFalse(
            flags["annular_shell_diagonal_commutator_bound_proved"]
        )
        self.assertFalse(flags["critical_signed_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])
        self.assertIn("dyadic annular", self.result["route_decision"])

    def test_stored_result_replays_exactly(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored, self.result)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertIn(
            "annulus",
            stored["next_theorem_target"],
        )


if __name__ == "__main__":
    unittest.main()
