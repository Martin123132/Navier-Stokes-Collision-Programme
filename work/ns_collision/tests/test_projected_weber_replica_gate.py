from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from projected_weber_replica_gate_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "projected_weber_replica_gate_audit_v1.json"
)


class ProjectedWeberReplicaGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_joint_generator_and_nonpromotion_flags(self) -> None:
        result = self.result
        self.assertTrue(result["all_positive_checks_pass"])
        flags = result["certification_flags"]
        self.assertTrue(
            flags[
                "joint_common_path_tangent_covector_generator_derived"
            ]
        )
        self.assertFalse(
            flags["bare_directional_cubic_has_direct_collision_diffusion"]
        )
        self.assertFalse(
            flags["single_superharmonic_collision_weight_closes_pointwise"]
        )
        self.assertFalse(
            flags["signed_projected_replica_closure_bound_proved"]
        )
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_tensor_proxy_and_affine_sign_obstruction(self) -> None:
        tensor = self.result["tensor_proxy"]
        self.assertTrue(tensor["all_checks_pass"])
        for row in tensor["rows"]:
            self.assertLess(row["trace_derivative_residual"], 2.0e-10)
            self.assertLess(
                row["proxy_derivative_relative_residual"],
                2.0e-12,
            )
        affine = self.result["affine_generator_stress"]
        self.assertGreaterEqual(
            affine[
                "minimum_rate_over_0_le_q_le_1_with_expanding_tangent"
            ],
            0.5 - 2.0e-12,
        )
        self.assertGreater(
            affine["contracting_tangent_direction"][
                "far_field_generator_rate_over_a"
            ],
            0.0,
        )

    def test_exact_shear_exposes_both_projection_losses(self) -> None:
        shear = self.result["exact_periodic_shear"]
        self.assertTrue(shear["symbolic"]["all_checks_pass"])
        inflation = shear["unprojected_magnetization_inflation"]
        self.assertTrue(inflation["all_checks_pass"])
        self.assertGreater(
            inflation["rows"][-1]["unprojected_inflation_ratio"],
            100.0,
        )
        harmonic = shear["projected_common_path_harmonic_variance"]
        self.assertTrue(harmonic["all_checks_pass"])
        self.assertGreater(harmonic["closed_form_variance"], 0.0)
        self.assertLess(harmonic["relative_residual"], 2.0e-12)

    def test_signed_replica_target_and_stored_result(self) -> None:
        identities = self.result["signed_replica_identities"]
        self.assertIn("v_1 dot v_2", identities["two_replica"])
        self.assertIn("product_", identities["three_replica"])
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored["kind"], self.result["kind"])
        self.assertEqual(stored["status"], self.result["status"])
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertFalse(
            stored["certification_flags"][
                "signed_projected_replica_closure_bound_proved"
            ]
        )


if __name__ == "__main__":
    unittest.main()
