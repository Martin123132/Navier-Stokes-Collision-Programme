from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adjoint_replica_pressure_edge_gate_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "adjoint_replica_pressure_edge_gate_audit_v1.json"
)


class AdjointReplicaPressureEdgeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_backward_restart_dual_and_nonpromotion(self) -> None:
        dual = self.result["backward_restart_dual"]
        self.assertTrue(dual["all_checks_pass"])
        self.assertEqual(
            dual["general_generator_symbolic_residual"],
            "0",
        )
        self.assertEqual(dual["rho_zero_symbolic_residual"], "0")
        self.assertIn(
            "lambda|grad lambda|^2",
            dual["rho_zero_reduction"],
        )

        flags = self.result["certification_flags"]
        self.assertTrue(
            flags["backward_adjoint_restart_dual_inequality_derived"]
        )
        self.assertTrue(
            flags["backward_terminal_penalty_contraction_proved"]
        )
        self.assertFalse(
            flags["universal_restart_dual_flux_is_nonpositive"]
        )
        self.assertFalse(flags["critical_signed_replica_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_exact_shear_and_abc_stresses(self) -> None:
        shear = self.result["exact_periodic_shear"]
        self.assertTrue(shear["all_checks_pass"])
        self.assertEqual(shear["backward_PDE_symbolic_residual"], "0")
        self.assertEqual(
            shear["terminal_condition_symbolic_residual"],
            "0",
        )
        self.assertLess(shear["identity_residual"], 2.0e-13)
        self.assertLess(
            shear["terminal_physical_L3_cubed"],
            shear["initial_physical_L3_cubed"],
        )

        abc = self.result["periodic_ABC"]
        self.assertTrue(abc["all_checks_pass"])
        self.assertLess(abs(abc["pressure_work"]), 2.0e-14)
        self.assertLess(
            abc["Delta_u_equals_minus_u_balance_residual"],
            4.0e-6,
        )

    def test_reset_rho_ordering(self) -> None:
        ordering = self.result["reset_rho_ordering"]
        self.assertTrue(ordering["all_checks_pass"])
        self.assertTrue(ordering["rho_zero_is_instantaneously_optimal"])
        values = [
            row["dual_generator_excess_over_rho_zero"]
            for row in ordering["rows"]
        ]
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values, sorted(values))

    def test_resolved_high_amplitude_sign_falsifier(self) -> None:
        stress = self.result["high_amplitude_pressure_sign_falsifier"]
        self.assertTrue(stress["all_checks_pass"])
        lower, upper = stress["rho_zero_sign_change_amplitude_range"]
        self.assertGreater(lower, 168.16)
        self.assertLess(upper, 168.18)
        for row in stress["rows"]:
            scaled = row["scaled_rho_zero_rows"]
            self.assertLess(
                scaled[0]["rho_zero_rate_over_3_scale_cubed"],
                0.0,
            )
            self.assertGreater(
                scaled[1]["rho_zero_rate_over_3_scale_cubed"],
                0.0,
            )

    def test_partition_edge_gate_and_stored_result(self) -> None:
        partition = self.result["partition_pressure_edge_gate"]
        self.assertTrue(partition["all_checks_pass"])
        self.assertLess(
            partition["pressure_gradient_reconstruction_residual"],
            2.0e-12,
        )
        direct = partition["direct_weighted_pressure_flux"]
        self.assertAlmostEqual(
            direct,
            partition["edge_weighted_pressure_flux"],
            places=12,
        )
        self.assertAlmostEqual(
            partition["direct_weight_fisher"],
            partition["conditional_weight_fisher"],
            places=12,
        )
        self.assertGreater(
            partition["direct_weighted_pressure_flux"],
            1.0,
        )
        self.assertLess(
            partition["smooth_partition_sign_change_amplitude"],
            700.0,
        )
        self.assertGreater(
            partition[
                "smooth_partition_scaled_rate_over_3_scale_cubed_at_700"
            ],
            0.0,
        )
        self.assertGreater(
            partition["edge_young_upper_at_nu_one"],
            partition["exact_dual_flux_at_nu_one"],
        )

        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored["kind"], self.result["kind"])
        self.assertEqual(stored["status"], self.result["status"])
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertFalse(
            stored["certification_flags"][
                "edge_Young_remainder_absorbed_by_replica_dissipation"
            ]
        )


if __name__ == "__main__":
    unittest.main()
