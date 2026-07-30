from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scale_uniform_low_output_tail_gate_audit import (
    PRIOR_RESULT_SHA256,
    _dyadic_tail_summation_audit,
    _endpoint_pulse_audit,
    _lattice_shell_audit,
    _max_norm_shell_count,
    _pulse_response_l2_squared,
)


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "scale_uniform_low_output_tail_gate_audit_v1.json"
)


class ScaleUniformLowOutputTailGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stored = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_exact_lattice_shell_count(self) -> None:
        audit = _lattice_shell_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertEqual(audit["maximum_formula_residual"], 0)
        self.assertLessEqual(audit["maximum_cubic_ratio"], 56.0)
        for scale in (1, 2, 4, 8, 16):
            self.assertEqual(
                _max_norm_shell_count(scale),
                56 * scale**3 - 36 * scale**2 + 6 * scale,
            )

    def test_dyadic_H_minus_s_threshold(self) -> None:
        audit = _dyadic_tail_summation_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertTrue(
            audit["monotone_convergence_for_every_s_gt_1"]
        )
        self.assertTrue(audit["H_minus_1_infinite_series_diverges"])
        self.assertEqual(
            audit["H_minus_1_high_output_increment_per_shell"],
            1.0,
        )

    def test_exact_endpoint_pulse_identity(self) -> None:
        for carrier in (32, 64, 128, 1024, 8192):
            actual, expected = _pulse_response_l2_squared(carrier)
            self.assertAlmostEqual(actual, expected, places=14)
        audit = _endpoint_pulse_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertTrue(audit["endpoint_residual_strictly_decreases"])
        self.assertGreater(
            audit["rows"][-1]["H_minus_1_block_square"],
            1.25,
        )

    def test_subcritical_tail_decays_but_endpoint_does_not(self) -> None:
        audit = _endpoint_pulse_audit()
        first = audit["rows"][0]
        last = audit["rows"][-1]
        self.assertGreater(last["H_minus_1_block_square"], 1.25)
        for epsilon in ("0.1", "0.25", "0.5"):
            self.assertLess(
                last["subcritical_rows"][epsilon][
                    "H_minus_1_plus_epsilon_square"
                ],
                first["subcritical_rows"][epsilon][
                    "H_minus_1_plus_epsilon_square"
                ],
            )

    def test_stored_status_prerequisite_and_scope(self) -> None:
        stored = self.stored
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "scale_uniform_H_minus_one_plus_epsilon_"
            "Galerkin_stress_tail_certified",
        )
        self.assertEqual(
            stored["prerequisites"][
                "smooth_galerkin_shell_response_gate_sha256"
            ],
            PRIOR_RESULT_SHA256,
        )
        flags = stored["certification_flags"]
        self.assertTrue(
            flags["H_minus_s_tail_vanishes_for_every_s_gt_1"]
        )
        self.assertTrue(flags["fixed_mode_Galerkin_compactness_derived"])
        self.assertTrue(
            flags["H_minus_one_plus_epsilon_Galerkin_passage_proved"]
        )
        self.assertTrue(
            flags["H_minus_1_not_derivable_from_scalar_envelope_alone"]
        )
        self.assertFalse(flags["H_minus_1_endpoint_proved"])
        self.assertFalse(
            flags["H_minus_1_endpoint_falsified_for_actual_Navier_Stokes"]
        )
        self.assertFalse(
            flags["complete_suitable_weak_solution_passage_proved"]
        )
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
