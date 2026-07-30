from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "parabolic_gramian_continuation_audit.py"
RESULT = ROOT / "results" / "parabolic_gramian_continuation_audit_v1.json"
SPEC = importlib.util.spec_from_file_location(
    "parabolic_gramian_continuation_audit",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParabolicGramianContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = MODULE.audit()

    def test_continuation_hierarchy_and_nonpromotion(self) -> None:
        result = self.result
        self.assertTrue(result["all_positive_checks_pass"])
        self.assertEqual(
            result["status"],
            (
                "critical_continuation_hierarchy_proved_"
                "unconditional_Gramian_moment_bound_open"
            ),
        )
        flags = result["certification_flags"]
        self.assertTrue(
            flags[
                "exact_directional_cubic_moment_is_sufficient_for_L3_control"
            ]
        )
        self.assertTrue(
            flags[
                "tensor_spectral_cubic_moment_is_sufficient_for_L3_control"
            ]
        )
        self.assertTrue(
            flags["scalar_radial_cubic_moment_is_sufficient_for_L3_control"]
        )
        self.assertFalse(
            flags["scalar_radial_criterion_is_quantitatively_viable"]
        )
        self.assertFalse(flags["Leray_energy_bounds_tensor_spectral_moment"])
        self.assertFalse(flags["low_regularity_inverse_time_probe_justified"])
        self.assertFalse(flags["exceptional_set_upgrade_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_window_hierarchy_and_Burgers_falsifier(self) -> None:
        rows = self.result["Burgers_vortex_axis_stress"]["rows"]
        for row in rows:
            self.assertTrue(row["all_window_checks_pass"])
            self.assertTrue(row["all_integration_checks_pass"])
            self.assertLessEqual(
                row["exact_inverse_J_cubic"],
                row["tensor_inverse_cubic_bound"] * (1.0 + 1.0e-8),
            )
            self.assertLessEqual(
                row["tensor_inverse_cubic_bound"],
                row["radial_inverse_cubic_bound"] * (1.0 + 1.0e-8),
            )
        self.assertGreater(
            rows[-1]["radial_loss_over_tensor"],
            1.0e12,
        )

    def test_exact_periodic_shear_and_ABC(self) -> None:
        shear = self.result["periodic_finite_Fourier_shear"]
        self.assertTrue(
            shear["exact_solution_audit"][
                "all_exact_solution_checks_pass"
            ]
        )
        self.assertLess(shear["window"]["exact_J_residual"], 2.0e-9)
        self.assertTrue(shear["window"]["all_window_checks_pass"])

        abc = self.result["periodic_finite_Fourier_ABC"]
        self.assertTrue(
            abc["exact_solution_audit"]["all_exact_solution_checks_pass"]
        )
        self.assertTrue(abc["summary"]["all_trajectory_checks_pass"])
        self.assertTrue(
            abc["restart_cocycle"]["all_cocycle_checks_pass"]
        )
        for residual in abc["restart_cocycle"]["residuals"].values():
            self.assertLess(residual, 8.0e-9)

    def test_small_window_expansion_and_stored_result(self) -> None:
        expansion = self.result["small_window_strain_expansion"]
        self.assertTrue(expansion["all_small_window_checks_pass"])
        self.assertAlmostEqual(expansion["predicted_coefficient"], 4.0 / 3.0)

        stored = json.loads(RESULT.read_text(encoding="ascii"))
        self.assertEqual(stored["kind"], self.result["kind"])
        self.assertEqual(stored["schema_version"], 1)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertFalse(
            stored["certification_flags"][
                "Navier_Stokes_global_regularity_proved"
            ]
        )


if __name__ == "__main__":
    unittest.main()
