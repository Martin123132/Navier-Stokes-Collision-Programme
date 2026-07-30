from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from joint_scale_cell_viscous_occupation_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "joint_scale_cell_viscous_occupation_audit_v1.json"
)


class JointScaleCellViscousOccupationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_baseline_cumulative_stress_bound(self) -> None:
        baseline = self.result["baseline_cumulative_stress_bound"]
        flags = self.result["certification_flags"]
        self.assertTrue(baseline["all_checks_pass"])
        self.assertIn("L^(3/2)", baseline["estimate"])
        self.assertIn("ell1", baseline["scope"])
        self.assertTrue(
            flags["baseline_cumulative_stress_ell1_bound_proved"]
        )

    def test_common_low_mode_stress_no_go(self) -> None:
        channel = self.result["pointwise_Fourier_Walsh_channel"]
        self.assertTrue(channel["all_checks_pass"])
        self.assertEqual(channel["common_low_Fourier_mode"], [1, 1, 0])
        self.assertEqual(channel["common_cell_Walsh_mask"], 7)
        for row in channel["prefix_rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertLess(
                row["stress_cross_shell_residual"],
                1.0e-12,
            )
            self.assertGreaterEqual(
                row["stress_sum_over_square_function"] + 1.0e-12,
                row["analytic_stress_ratio_lower"],
            )
        self.assertGreater(
            channel["prefix_rows"][-1][
                "stress_sum_over_square_function"
            ],
            2.0,
        )

    def test_common_top_walsh_flux_no_go_and_scope(self) -> None:
        channel = self.result["pointwise_Fourier_Walsh_channel"]
        flags = self.result["certification_flags"]
        for row in channel["individual_channels"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertGreater(row["all_cosine_vertex_load"], 0.0)
            self.assertLess(
                row["maximum_off_top_Walsh_coefficient"],
                1.0e-13,
            )
        self.assertGreater(
            channel["prefix_rows"][-1][
                "flux_sum_over_square_function"
            ],
            2.0,
        )
        self.assertTrue(
            flags[
                "pointwise_high_shell_ell2_orthogonality_gain_falsified"
            ]
        )
        self.assertFalse(
            flags["all_pointwise_Carleson_estimates_falsified"]
        )

    def test_stokes_damping_and_viscous_occupation(self) -> None:
        occupation = self.result["viscous_occupation_bound"]
        flags = self.result["certification_flags"]
        self.assertTrue(occupation["all_checks_pass"])
        self.assertEqual(occupation["beta_in_mu_squared"], 5.5)
        self.assertGreater(
            occupation["minimum_lacunarity_ratio"],
            1.9,
        )
        self.assertLess(
            occupation["maximum_exact_Stokes_damping_residual"],
            1.0e-12,
        )
        self.assertLessEqual(
            occupation["exact_L2_time_norm_squared"],
            occupation["Schur_upper"],
        )
        self.assertLessEqual(
            occupation["half_peak_occupation"],
            occupation["Chebyshev_occupation_upper"],
        )
        self.assertTrue(
            flags["linear_Stokes_HHL_viscous_occupation_bound_proved"]
        )
        self.assertFalse(
            flags[
                "Navier_Stokes_time_integrated_compensation_proved"
            ]
        )

    def test_conditional_forced_relaxation_gate(self) -> None:
        forced = self.result["forced_relaxation_bound"]
        flags = self.result["certification_flags"]
        self.assertTrue(forced["all_checks_pass"])
        self.assertLessEqual(
            forced["exact_replay_response_L2_norm"],
            forced["replay_upper"],
        )
        self.assertTrue(
            flags["conditional_forced_relaxation_bound_proved"]
        )
        self.assertFalse(
            flags[
                "Navier_Stokes_nonlinear_regeneration_bound_proved"
            ]
        )
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result_replays_exactly(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored, self.result)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "pointwise_joint_channel_no_go_Stokes_occupation_certified",
        )


if __name__ == "__main__":
    unittest.main()
