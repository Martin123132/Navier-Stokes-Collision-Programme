"""Tests for the annular common-polarization phase repair."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "work/ns_collision/scripts"
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_phase_repair_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_parallel_shear_phase_repair_audit import (  # noqa: E402
    _parallel_shear_certificate,
)


class AnnularParallelShearPhaseRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_scalar_phase_family_has_strict_quadrant_no_go(self) -> None:
        phase = self.audit["scalar_phase_family"]
        self.assertTrue(phase["all_phase_checks_pass"])
        self.assertEqual(
            phase["common_interaction_polynomial"],
            "p1**2*p2 + p1*p2**2 + p1*q2**2 + p2*q1**2",
        )
        self.assertIn("p1>0 and p2>0", phase["strict_square_quadrant"])

    def test_polarization_factorization_has_common_zero_branch(self) -> None:
        polarization = self.audit["exact_square_polarization_family"]
        self.assertTrue(
            polarization["all_polarization_checks_pass"]
        )
        self.assertEqual(
            polarization["pressure_self_flux_zero_cosine"],
            "-(a - b)*(b + d)*(a - 2*b - d)/24",
        )
        self.assertEqual(
            polarization["complete_self_flux_zero_cosine"],
            "(a + 2*b)*(2*b - d)*(a - 2*b - d)/24",
        )
        self.assertEqual(
            polarization["common_zero_branch"],
            "a=d+2b",
        )
        diagonal = self.audit["diagonal_cosine_no_go"]
        self.assertTrue(
            diagonal["all_diagonal_cosine_checks_pass"]
        )
        self.assertIn("2*b**2", diagonal["pressure_self_flux"])

    def test_parallel_low_field_is_exact_stationary_shear(self) -> None:
        parallel = _parallel_shear_certificate()
        self.assertTrue(parallel["all_parallel_shear_checks_pass"])
        self.assertEqual(parallel["weighted_Fisher"], "9/8")
        self.assertEqual(parallel["L2_mass"], "4")
        self.assertEqual(parallel["kinetic_self_flux_load"], "0")
        self.assertEqual(parallel["pressure_self_flux_load"], "0")
        self.assertEqual(parallel["complete_self_flux_load"], "0")

    def test_stencil_symmetry_retains_strict_square(self) -> None:
        stencil = self.audit["stencil_and_curvature_symmetry"]
        self.assertTrue(stencil["all_stencil_symmetry_checks_pass"])
        self.assertEqual(
            stencil["reflection_average_of_generic_symmetric_tensor"],
            [["Cxx", "0", "0"], ["0", "Cyy", "0"], ["0", "0", "Czz"]],
        )
        reduction = stencil["four_high_reduction"]
        self.assertEqual(
            reduction["reduced_formula"],
            "-sqrt(3)*Cyy/10",
        )
        self.assertIn("<0", reduction["strict_result"])
        for row in self.audit["curvature_matrix_replays"]:
            self.assertTrue(row["all_curvature_replay_checks_pass"])
            self.assertLess(row["maximum_off_diagonal"], 2.0e-20)
            self.assertLess(row["trace_residual"], 3.0e-20)

    def test_complete_full_field_replay_is_linear_in_low_amplitude(self) -> None:
        replay = self.audit["full_field_support_replay"]
        self.assertTrue(replay["all_support_replay_checks_pass"])
        self.assertAlmostEqual(
            replay["unit_low_weighted_Fisher"],
            9.0 / 8.0,
            delta=2.0e-15,
        )
        self.assertLess(
            abs(replay["unit_low_complete_load"]),
            2.0e-15,
        )
        self.assertLess(
            abs(replay["unit_low_pressure_load"]),
            2.0e-15,
        )
        self.assertLess(
            replay["maximum_complete_linear_residual"],
            2.0e-12,
        )
        self.assertLess(
            replay["maximum_pressure_linear_residual"],
            2.0e-12,
        )
        self.assertLess(
            replay["maximum_Fisher_quadratic_residual"],
            2.0e-12,
        )

    def test_finite_HHL_rows_and_optimizer_repair(self) -> None:
        rows = self.audit["finite_annular_rows"]
        self.assertEqual(
            [row["size"] for row in rows],
            [3, 5, 9, 13, 17, 25, 33, 49],
        )
        for row in rows:
            self.assertTrue(row["all_finite_checks_pass"])
            self.assertLess(row["complete_HHL_load_over_N"], 0.0)
            self.assertLess(row["pressure_HHL_load_over_N"], 0.0)
        summary = self.audit["finite_escape_summary"]
        self.assertEqual(summary["first_complete_positive_size"], 25)
        self.assertEqual(summary["first_pressure_positive_size"], 25)
        self.assertAlmostEqual(
            rows[-1]["pressure_HHL_load_over_N"],
            -0.001236808755580836,
            delta=2.0e-16,
        )

    def test_optimizer_reset_and_tail_constants(self) -> None:
        optimizer = self.audit["optimizer_and_restart_certificate"]
        self.assertTrue(optimizer["all_optimizer_restart_checks_pass"])
        self.assertAlmostEqual(
            optimizer["parallel_continuum_HHL_limit"],
            -0.001154598827427155,
            delta=2.0e-18,
        )
        reset = optimizer["reset_deficit_port"]
        self.assertEqual(reset["ratio_formula"], "5/(36nu)")
        self.assertAlmostEqual(
            reset["deficit_over_three_static_generator_liminf"],
            5.0 / 36.0,
            delta=2.0e-15,
        )
        tail = self.audit["parallel_full_c1_tail_port"]
        self.assertTrue(tail["all_tail_port_checks_pass"])
        self.assertIn("70,657,920", tail["tail_bound"])
        self.assertIn("-(", tail["complete_limit"])

    def test_scope_is_fail_closed_beyond_repair(self) -> None:
        flags = self.audit["certification_flags"]
        self.assertTrue(self.audit["all_positive_checks_pass"])
        self.assertTrue(
            flags[
                "parallel_shear_pressure_and_energy_flux_"
                "divergence_zero_pointwise"
            ]
        )
        self.assertTrue(flags["parallel_full_c1_limit_negative"])
        self.assertTrue(
            flags["parallel_finite_static_optimizer_restored"]
        )
        self.assertTrue(flags["parallel_reset_N5_gate_restored"])
        self.assertFalse(
            flags["parallel_complete_finite_first_jet_ported"]
        )
        self.assertFalse(
            flags["parallel_complete_finite_second_jet_ported"]
        )
        self.assertFalse(flags["critical_L3_control_proved"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])
        self.assertTrue(math.isfinite(
            self.audit["optimizer_and_restart_certificate"][
                "asymptotic_optimizer"
            ]["g_N_over_N_cubed_limit"]
        ))


if __name__ == "__main__":
    unittest.main()
