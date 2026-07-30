from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cross_shell_modulated_wave_gate_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "cross_shell_modulated_wave_gate_audit_v1.json"
)


class CrossShellModulatedWaveGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_exact_modulated_wave_geometry(self) -> None:
        theorem = self.result["analytic_no_go"]
        self.assertTrue(theorem["all_checks_pass"])
        self.assertIn("q=a_H-b_H=(1,1,0)", theorem["low_wave_and_stencil"])
        self.assertIn(
            "(-1/9,-1/9,4/9)",
            theorem["low_velocity"],
        )
        self.assertIn(
            "2e_1 tensor e_1",
            theorem["low_Reynolds_stress_limit"],
        )
        self.assertIn(
            "->-1",
            theorem["exact_low_pressure_coefficient"],
        )
        exact = theorem["exact_rational_checks"]
        self.assertTrue(exact["all_checks_pass"])
        self.assertEqual(exact["q_plus_k_plus_r"], [0, 0, 0])
        self.assertEqual(exact["k_dot_C"], "0")
        self.assertEqual(exact["pressure_load_limit"], "1/144")
        self.assertEqual(
            exact["anisotropic_flux_load_limit"],
            "1/144",
        )

    def test_pressure_coefficient_and_shell_replay(self) -> None:
        replay = self.result["finite_mode_asymptotic_replay"]
        self.assertTrue(replay["all_checks_pass"])
        self.assertEqual(
            replay["carriers"],
            [8, 16, 32, 64, 128, 256, 512, 1024],
        )
        for row in replay["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertLess(row["high_shell_ratio"], 1.02)
            self.assertAlmostEqual(
                row["low_pressure_coefficient_at_q"],
                -row["exact_q_dot_first_polarization"],
                places=13,
            )
            self.assertLess(
                row["maximum_divergence_residual"],
                1.0e-12,
            )

    def test_complete_flux_matches_direct_cubic_polynomial(self) -> None:
        rows = self.result["finite_mode_asymptotic_replay"]["rows"]
        for row in rows:
            self.assertAlmostEqual(
                row["combined_HHL_load"],
                row["direct_polynomial_linear_load"],
                places=13,
            )
            self.assertLess(
                row["component_vs_direct_flux_residual"],
                1.0e-12,
            )
            self.assertLess(
                row["maximum_imaginary_load_residual"],
                1.0e-12,
            )

    def test_nonzero_pressure_and_complete_flux_limits(self) -> None:
        replay = self.result["finite_mode_asymptotic_replay"]
        limit = 1.0 / 144.0
        self.assertAlmostEqual(
            replay["analytic_pressure_load_limit"],
            limit,
            places=15,
        )
        self.assertAlmostEqual(
            replay["analytic_combined_HHL_load_limit"],
            limit,
            places=15,
        )
        self.assertGreater(replay["pressure_last_over_limit"], 0.999)
        self.assertGreater(replay["combined_last_over_limit"], 0.999)
        self.assertLess(
            replay["maximum_H_times_kinetic_load"],
            0.02,
        )
        self.assertLess(
            replay["maximum_H_squared_times_cross_pressure_load"],
            0.25,
        )

    def test_scope_flags_and_route_decision(self) -> None:
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags["pressure_only_cross_shell_H_decay_falsified"]
        )
        self.assertTrue(
            flags["complete_signed_HHL_flux_H_decay_falsified"]
        )
        self.assertTrue(
            flags["anisotropic_Reynolds_stress_survives_in_flux"]
        )
        self.assertTrue(flags["self_shell_pressure_closure_preserved"])
        self.assertFalse(flags["dyadic_amplitude_summation_proved"])
        self.assertFalse(flags["inter_shell_telescoping_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])
        self.assertIn(
            "dyadic interaction atlas",
            self.result["route_decision"],
        )

    def test_stored_result_replays_exactly(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored, self.result)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "cross_shell_carrier_decay_falsified",
        )


if __name__ == "__main__":
    unittest.main()
