from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from high_carrier_weighted_fisher_gate_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "high_carrier_weighted_fisher_gate_audit_v1.json"
)


class HighCarrierWeightedFisherGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_unweighted_highpass_coercivity_theorem(self) -> None:
        theorem = self.result["unweighted_highpass_coercivity"]
        self.assertTrue(theorem["all_checks_pass"])
        self.assertEqual(
            theorem["carrier_exponent_for_fixed_load"],
            1,
        )
        self.assertIn(
            "/(2*K**(3/2))",
            theorem["load_upper_bound"],
        )
        self.assertIn(
            "sqrt(3)m/2",
            theorem["vertex_partition_gradient_bound"],
        )
        bridge = self.result["square_factor_highpass_bridge"]
        self.assertTrue(bridge["all_checks_pass"])
        self.assertEqual(
            bridge["validity_threshold"],
            "K>sqrt(3)m",
        )
        self.assertEqual(
            bridge["weighted_mass_upper_bound"],
            "E_v/(K*(K - sqrt(3)*m))",
        )
        self.assertEqual(
            bridge["gradient_factor_at_K_equals_2sqrt3m"],
            "1 + 3*sqrt(2)/4",
        )

    def test_exact_zero_face_uncertainty_packet(self) -> None:
        packet = self.result["zero_face_uncertainty_packet"]
        self.assertTrue(packet["all_checks_pass"])
        self.assertEqual(len(packet["rows"]), 6)
        self.assertLess(packet["maximum_weighted_Dirichlet"], 6.0)
        self.assertGreater(packet["ratio_drop"], 800.0)
        for row in packet["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertAlmostEqual(
                row["weighted_Dirichlet"],
                row["difference_form_Dirichlet"],
                places=9,
            )
            self.assertGreaterEqual(
                row["unweighted_Dirichlet"],
                row["minimum_mode"] ** 2,
            )

    def test_zero_face_concentration_scaling(self) -> None:
        scaling = self.result["zero_face_concentration_scaling"]
        self.assertTrue(scaling["all_checks_pass"])
        self.assertEqual(
            [
                row["fixed_load_weighted_delta_exponent"]
                for row in scaling["rows"]
            ],
            ["-1/3", "0", "1/3"],
        )
        self.assertTrue(
            all(
                row["pressure_to_weighted_Fisher_ratio"]
                == "A*delta/nu"
                for row in scaling["rows"]
            )
        )
        self.assertIn("A/(nu K)", scaling["intrinsic_ratio"])

    def test_pressure_active_finite_fourier_pilot(self) -> None:
        pilot = self.result["PDE_zero_face_packet_pilot"]
        self.assertTrue(pilot["all_checks_pass"])
        self.assertEqual(
            [row["order"] for row in pilot["rows"]],
            [2, 3, 4, 5],
        )
        weighted = []
        unweighted = []
        intrinsic = []
        for row in pilot["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertLess(
                row["maximum_relative_divergence_residual"],
                1.0e-10,
            )
            self.assertGreaterEqual(
                row["minimum_mode_over_order"],
                3.0,
            )
            self.assertAlmostEqual(
                abs(row["normalized_pressure_load"]),
                1.0,
                places=10,
            )
            weighted.append(row["normalized_weighted_Fisher"])
            unweighted.append(row["normalized_unweighted_Fisher"])
            intrinsic.append(row["intrinsic_Reynolds_proxy"])
        self.assertTrue(
            all(
                first > second
                for first, second in zip(weighted, weighted[1:])
            )
        )
        self.assertTrue(
            all(
                first < second
                for first, second in zip(unweighted, unweighted[1:])
            )
        )
        self.assertTrue(
            all(
                first < second
                for first, second in zip(intrinsic, intrinsic[1:])
            )
        )

    def test_scope_and_nonpromotion(self) -> None:
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags[
                "pure_highpass_unweighted_H1_least_cost_linear_coercivity_proved"
            ]
        )
        self.assertTrue(
            flags["vertex_square_factor_highpass_mass_bridge_proved"]
        )
        self.assertTrue(
            flags[
                "vertex_zero_face_gradient_controlled_by_weighted_Fisher"
            ]
        )
        self.assertTrue(
            flags["zero_face_uncertainty_mechanism_exactly_realized"]
        )
        self.assertTrue(
            flags["PDE_pressure_packet_mechanism_numerically_realized"]
        )
        self.assertFalse(
            flags["global_quadratic_carrier_coercivity_proved"]
        )
        self.assertFalse(
            flags["PDE_pressure_packet_asymptotic_counterexample_proved"]
        )
        self.assertFalse(
            flags["general_intrinsic_high_carrier_absorption_proved"]
        )
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result_replays_exactly(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored, self.result)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertIn(
            "commutator",
            stored["next_theorem_target"],
        )


if __name__ == "__main__":
    unittest.main()
