from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from signed_projected_replica_generator_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "signed_projected_replica_generator_audit_v1.json"
)


class SignedProjectedReplicaGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_projected_two_replica_generator(self) -> None:
        projected = self.result["projected_two_replica_generator"]
        self.assertTrue(projected["all_checks_pass"])
        self.assertEqual(projected["stretch_symbolic_residual"], "0")
        self.assertEqual(projected["diffusion_symbolic_residual"], "0")
        self.assertIn("2nu(1-rho)", projected["two_replica_local_balance"])

        flags = self.result["certification_flags"]
        self.assertTrue(flags["projected_Weber_SPDE_derived"])
        self.assertTrue(flags["rho_two_replica_local_balance_derived"])
        self.assertTrue(
            flags["independent_endpoint_recovers_energy_equality"]
        )
        self.assertFalse(flags["signed_replica_L3_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_rho_endpoints_and_indefinite_strain(self) -> None:
        stresses = self.result["rho_reset_stresses"]
        self.assertTrue(stresses["all_checks_pass"])
        for name in ("periodic_shear", "abc_flow"):
            flow = stresses[name]
            self.assertAlmostEqual(
                flow["rows"][0]["cross_energy_derivative_at_reset"],
                flow["physical_energy_derivative"],
                places=14,
            )
            self.assertEqual(
                flow["rows"][-1]["cross_energy_derivative_at_reset"],
                0.0,
            )
        burgers = stresses["burgers_strain_sign_stress"]
        self.assertLess(
            burgers["V1_equals_V2_equals_e3_rate_over_a"],
            0.0,
        )
        self.assertGreater(
            burgers["V1_equals_V2_equals_e1_rate_over_a"],
            0.0,
        )

    def test_gaussian_correlation_homotopy(self) -> None:
        homotopy = self.result["Gaussian_chaos_homotopy"]
        self.assertTrue(homotopy["all_checks_pass"])
        self.assertTrue(homotopy["monotone_on_zero_one"])
        self.assertTrue(
            homotopy["gradient_cross_pairing_nonnegative"]
        )
        self.assertIn("|grad u|^2", homotopy["gradient_lower_bound"])
        expected_energies = [1.25, 1.0625, 1.625, 0.3]
        for actual, expected in zip(
            homotopy["chaos_energies"],
            expected_energies,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=14)
        for row in homotopy["rows"]:
            self.assertLess(row["correlation_residual"], 3.0e-13)
            self.assertLess(row["derivative_residual"], 3.0e-13)
            self.assertGreaterEqual(
                row["chaos_polynomial_derivative"],
                0.0,
            )
        self.assertTrue(
            self.result["certification_flags"][
                "positive_rho_cross_gradient_dissipation_proved"
            ]
        )

    def test_weighted_critical_pressure_gate(self) -> None:
        weighted = self.result["weighted_critical_formulation"]
        self.assertTrue(weighted["all_checks_pass"])
        self.assertEqual(weighted["dual_gap_symbolic_residual"], "0")
        self.assertEqual(weighted["optimizer_symbolic_residual"], "0")

        pressure = self.result["adversarial_pressure_stress"]
        self.assertTrue(pressure["all_checks_pass"])
        self.assertGreater(
            pressure["critical_pressure_work_range"][0],
            40.5,
        )
        self.assertLess(
            pressure["critical_pressure_relative_spread"],
            1.0e-5,
        )
        self.assertLess(
            pressure["maximum_absolute_critical_convective_work"],
            3.1e-4,
        )
        for row in pressure["rows"]:
            self.assertAlmostEqual(row["velocity_rms"], 10.0, places=11)
            self.assertGreater(row["weights"][0]["pressure_work"], 40.2)
            self.assertGreater(row["weights"][-1]["pressure_work"], 40.5)

    def test_three_replica_generator_and_stored_result(self) -> None:
        triple = self.result["three_replica_tensor_generator"]
        self.assertTrue(triple["all_checks_pass"])
        self.assertTrue(
            triple["sample_correlation_matrix_is_positive_definite"]
        )
        self.assertIn("|u|^3", triple["critical_contraction"])

        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored["kind"], self.result["kind"])
        self.assertEqual(stored["status"], self.result["status"])
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertFalse(
            stored["certification_flags"][
                "weighted_pressure_flux_bound_proved"
            ]
        )
        self.assertFalse(
            stored["certification_flags"][
                "signed_replica_L3_bound_proved"
            ]
        )


if __name__ == "__main__":
    unittest.main()
