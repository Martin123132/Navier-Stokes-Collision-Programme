from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "correlated_replica_tangent_gramian_audit.py"
RESULT = ROOT / "results" / "correlated_replica_tangent_gramian_audit_v1.json"
SPEC = importlib.util.spec_from_file_location(
    "correlated_replica_tangent_gramian_audit",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CorrelatedReplicaTangentGramianTests(unittest.TestCase):
    def test_symbolic_correlation_and_tangent_identities(self) -> None:
        result = MODULE.audit(quadrature_order=64)
        symbolic = result["symbolic_correlated_replica_audit"]
        self.assertTrue(symbolic["all_symbolic_checks_pass"])
        self.assertEqual(
            symbolic["three_dimensional_log_squared_gap_source"],
            "4*nu*(1 - rho)",
        )
        for value in symbolic["checks"].values():
            self.assertTrue(value)

    def test_all_Gramian_cases_and_falsifiers(self) -> None:
        result = MODULE.audit(quadrature_order=64)
        self.assertTrue(result["all_positive_checks_pass"])
        for row in result["cases"].values():
            self.assertTrue(row["all_case_checks_pass"])
            for value in row["checks"].values():
                self.assertTrue(value)

        flags = result["certification_flags"]
        self.assertTrue(
            flags["conditional_forward_inverse_Gramian_congruence_proved"]
        )
        self.assertTrue(
            flags["conditional_cross_covariance_recovers_flow_Jacobian"]
        )
        self.assertTrue(flags["Minkowski_determinant_floor_proved"])
        self.assertTrue(flags["radial_trace_deformation_bound_proved"])
        self.assertFalse(flags["radial_noncollision_alone_controls_deformation"])
        self.assertFalse(
            flags["Leray_energy_controls_critical_forward_inverse_traces"]
        )
        self.assertFalse(flags["low_regularity_inverse_time_probe_justified"])
        self.assertFalse(flags["critical_L3_continuation_bridge_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

        stress = result["stress_tests"]
        self.assertTrue(
            stress["one_sided_noncollision"][
                "noncollision_covariance_remains_positive"
            ]
        )
        self.assertGreater(
            stress["one_sided_noncollision"]["maximum_deformation_norm"],
            1000.0,
        )
        self.assertGreater(
            stress["simple_shear_nonnormality"][
                "trace_bound_over_actual_squared"
            ],
            100.0,
        )
        self.assertTrue(
            stress["rigid_rotation_orientation"][
                "cross_covariance_retains_rotation"
            ]
        )
        self.assertGreater(
            stress["noncommuting_generator_commutator_norm"],
            0.5,
        )

    def test_closed_form_planar_strain_and_shear_values(self) -> None:
        result = MODULE.audit(quadrature_order=64)
        nu = 0.7
        terminal_time = 1.0

        planar = result["cases"]["planar_strain"]
        strength = 8.0
        expected_expanding = (
            2.0 * nu * math.expm1(2.0 * strength * terminal_time) / strength
        )
        expected_contracting = (
            2.0
            * nu
            * (-math.expm1(-2.0 * strength * terminal_time))
            / strength
        )
        expected_neutral = 4.0 * nu * terminal_time
        self.assertAlmostEqual(
            planar["forward_Gramian"][0][0],
            expected_expanding,
            places=7,
        )
        self.assertAlmostEqual(
            planar["forward_Gramian"][1][1],
            expected_contracting,
            places=12,
        )
        self.assertAlmostEqual(
            planar["forward_Gramian"][2][2],
            expected_neutral,
            places=12,
        )
        self.assertAlmostEqual(
            planar["inverse_time_Gramian"][0][0],
            expected_contracting,
            places=12,
        )
        self.assertAlmostEqual(
            planar["inverse_time_Gramian"][1][1],
            expected_expanding,
            places=7,
        )

        shear = result["cases"]["simple_shear"]
        shear_strength = 12.0
        expected_normalized_trace = (
            3.0 + (shear_strength * terminal_time) ** 2 / 3.0
        )
        self.assertAlmostEqual(
            shear["normalized_forward_radial_variance"],
            expected_normalized_trace,
            places=11,
        )
        self.assertAlmostEqual(
            shear["normalized_inverse_radial_variance"],
            expected_normalized_trace,
            places=11,
        )

    def test_parabolic_scaling_and_stored_result(self) -> None:
        result = MODULE.audit(quadrature_order=64)
        scaling = result["parabolic_scaling_audit"]
        self.assertTrue(all(scaling["checks"].values()))
        self.assertEqual(
            result["status"],
            (
                "smooth_flow_correlation_to_deformation_identity_proved_"
                "critical_trace_estimate_open"
            ),
        )

        stored = json.loads(RESULT.read_text(encoding="ascii"))
        self.assertEqual(stored["kind"], result["kind"])
        self.assertEqual(stored["schema_version"], 1)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertFalse(
            stored["certification_flags"][
                "Navier_Stokes_global_regularity_proved"
            ]
        )


if __name__ == "__main__":
    unittest.main()
