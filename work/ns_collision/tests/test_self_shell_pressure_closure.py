from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from self_shell_pressure_closure_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "self_shell_pressure_closure_audit_v1.json"
)


class SelfShellPressureClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_exact_far_low_support_exclusion(self) -> None:
        support = self.result["support_exclusion"]
        self.assertTrue(support["all_checks_pass"])
        self.assertEqual(len(support["rows"]), 4)
        for row in support["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertGreater(row["strict_triangle_margin"], 0.0)
            self.assertEqual(
                row["admissible_support_resonance_count"],
                0,
            )
        self.assertIn("q+k+r=0", support["proof"])

    def test_gap_dependent_full_self_shell_theorem(self) -> None:
        theorem = self.result["full_self_shell_theorem"]
        self.assertTrue(theorem["all_checks_pass"])
        self.assertIn("K>sqrt(3)m", theorem["assumptions"])
        self.assertIn(
            "(K-R)/2",
            theorem["exact_low_output_identity"],
        )
        self.assertIn(
            "delta^(-3)",
            theorem["annular_constant"],
        )
        for row in theorem["adaptive_gap_replay"]:
            self.assertTrue(row["strict_support_exclusion_holds"])
            self.assertLess(
                row["cutoff_plus_stencil_over_carrier"],
                1.0,
            )

    def test_valid_adversarial_shells(self) -> None:
        stress = self.result["adversarial_sparse_shell_stress"]
        self.assertTrue(stress["all_checks_pass"])
        rows = stress["valid_shell_rows"]
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertTrue(row["all_checks_pass"])
            self.assertGreater(row["smooth_low_pressure_L2"], 1.0e-8)
            self.assertGreater(
                row["maximum_full_pressure_load"],
                1.0e-8,
            )
            self.assertEqual(
                row["maximum_smooth_low_pressure_load"],
                0.0,
            )
            self.assertEqual(
                row["maximum_full_minus_smooth_high_load"],
                0.0,
            )
            self.assertEqual(
                row["maximum_low_load_resonance_count"],
                0,
            )
            self.assertLess(
                row["maximum_divergence_residual"],
                1.0e-12,
            )

    def test_fixed_split_threshold_probe_is_sensitive(self) -> None:
        support_probe = self.result["support_exclusion"][
            "below_threshold_probe"
        ]
        field_probe = self.result["adversarial_sparse_shell_stress"][
            "below_threshold_nonzero_channel"
        ]
        self.assertFalse(
            support_probe["condition_K_gt_2sqrt3m_holds"]
        )
        self.assertGreater(support_probe["support_resonance_count"], 0)
        self.assertTrue(field_probe["all_checks_pass"])
        self.assertFalse(
            field_probe["condition_K_gt_2sqrt3m_holds"]
        )
        self.assertGreater(
            field_probe["maximum_smooth_low_pressure_load"],
            1.0e-8,
        )
        self.assertGreater(
            field_probe["maximum_low_load_resonance_count"],
            0,
        )

    def test_scope_flags_and_next_channel(self) -> None:
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags["exact_far_low_pressure_load_orthogonality_proved"]
        )
        self.assertTrue(
            flags["full_self_shell_pressure_load_bound_proved"]
        )
        self.assertTrue(
            flags["gap_dependent_closure_for_K_gt_sqrt3m_proved"]
        )
        self.assertTrue(
            flags["uniform_fixed_split_for_K_ge_2sqrt3m_proved"]
        )
        self.assertFalse(
            flags["K_ge_2sqrt3m_uniform_threshold_proved_sharp"]
        )
        self.assertFalse(
            flags["cross_shell_high_high_to_low_controlled"]
        )
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])
        self.assertIn(
            "modulated-wave",
            self.result["route_decision"],
        )

    def test_stored_result_replays_exactly(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored, self.result)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "full_self_shell_pressure_closure_certified",
        )


if __name__ == "__main__":
    unittest.main()
