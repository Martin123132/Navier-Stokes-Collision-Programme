"""Tests for the annular two-shear exact-square route guard."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "work/ns_collision/scripts"
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_two_shear_square_gate_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_two_shear_square_gate_audit import (  # noqa: E402
    ALPHA,
    _dominant_profile_samples,
    _quadrature_row,
    _stencil_audit,
)


class AnnularTwoShearSquareGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_exact_low_stencil_and_static_strain_matrices(self) -> None:
        stencil = _stencil_audit()
        self.assertTrue(stencil["all_exact_stencil_checks_pass"])
        self.assertEqual(stencil["combined_active_output_count"], 58)
        self.assertEqual(
            stencil[
                "combined_active_output_l1_before_dividing_by_sqrt2"
            ],
            "3",
        )
        self.assertEqual(stencil["combined_maximum_radius_squared"], 6)
        self.assertEqual(
            stencil["combined_sqrt2_times_projector_matrix"],
            [["1/20", "0", "0"], ["0", "-1/10", "0"], ["0", "0", "1/20"]],
        )
        self.assertEqual(
            stencil["combined_static_strain_after_removing_sqrt2"],
            [["-1", "0", "0"], ["0", "2", "0"], ["0", "0", "-1"]],
        )
        self.assertTrue(
            stencil["checks"][
                "fixed_output_matrix_is_negative_static_strain_over_20"
            ]
        )

    def test_modified_profile_is_admissible_by_itself(self) -> None:
        x, y, z, profile = _dominant_profile_samples(8)
        divergence = (
            x[:, None, None] * profile[0]
            + y[None, :, None] * profile[1]
            + z[None, None, :] * profile[2]
        )
        self.assertLess(float(np.max(np.abs(divergence))), 1.0e-14)
        self.assertEqual(float(np.max(np.abs(profile[1]))), 0.0)
        for axis in range(1, 4):
            self.assertEqual(
                float(np.max(np.abs(np.take(profile, 0, axis=axis)))),
                0.0,
            )
            self.assertEqual(
                float(np.max(np.abs(np.take(profile, -1, axis=axis)))),
                0.0,
            )

    def test_production_rows_replay_exact_negative_square(self) -> None:
        rows = self.audit["rows"]
        self.assertEqual([row["size"] for row in rows], [8, 16, 32])
        expected = [
            -2.7254181695245514e-7,
            -2.982757380401352e-7,
            -3.050326849524017e-7,
        ]
        for row, value in zip(rows, expected):
            velocity_y_energy = row["velocity_energy_components"][1]
            profile_energy = sum(
                row["positive_packet_profile_energy_components"]
            )
            self.assertAlmostEqual(
                row["combined_fixed_output_functional"],
                value,
                delta=2.0e-20,
            )
            self.assertAlmostEqual(
                row["combined_fixed_output_functional"],
                -3.0 * ALPHA * velocity_y_energy,
                delta=2.0e-20,
            )
            self.assertAlmostEqual(
                row["static_combined"],
                -ALPHA * profile_energy,
                delta=2.0e-18,
            )
            self.assertLess(
                abs(sum(row["component_energy_curvatures"])),
                2.0e-18,
            )
            self.assertTrue(row["all_numerical_checks_pass"])

    def test_covariance_replays_strict_nonvanishing_direction(self) -> None:
        for row in self.audit["rows"]:
            covariance = np.asarray(row["covariance_matrix"])
            self.assertEqual(covariance[1, 1], 0.0)
            self.assertGreater(covariance[2, 2], 0.039)
            self.assertLess(
                float(
                    np.max(
                        np.abs(
                            covariance
                            - np.diag(np.diag(covariance))
                        )
                    )
                ),
                1.0e-18,
            )
            replay = row["near_zero_nonvanishing_replay"]
            self.assertTrue(replay["quadrature_y_component_negative"])
            self.assertTrue(replay["leading_y_component_negative"])

    def test_odd_finite_packet_static_replay_uses_positive_domain(self) -> None:
        replay = self.audit["finite_static_packet_replay"]
        rows = replay["rows"]
        self.assertEqual(
            [row["size"] for row in rows],
            [5, 9, 13, 17, 25],
        )
        expected = [
            -0.002640785765643764,
            -0.002025849181034617,
            -0.0018206510133647194,
            -0.0017182972937519313,
            -0.0016162734760602766,
        ]
        errors = []
        for row, value in zip(rows, expected):
            self.assertAlmostEqual(
                row["combined_pressure_hhl_load_over_N"],
                value,
                delta=2.0e-16,
            )
            self.assertTrue(row["all_finite_static_checks_pass"])
            self.assertLess(row["maximum_divergence_residual"], 1.0e-12)
            errors.append(abs(row["normalized_error_from_reference"]))
        self.assertTrue(
            all(
                errors[index + 1] < errors[index]
                for index in range(len(errors) - 1)
            )
        )
        self.assertAlmostEqual(
            replay["positive_packet_continuum_reference_gauss80"],
            -0.0014140889924061505,
            delta=2.0e-17,
        )
        self.assertTrue(
            replay["continuum_reference_is_sign_replay_not_error_bound"]
        )

    def test_certification_scope_stays_exact_and_limited(self) -> None:
        certification = self.audit["certification"]
        self.assertTrue(self.audit["all_route_guard_checks_pass"])
        self.assertTrue(
            certification["modified_four_high_continuum_sign_analytic"]
        )
        self.assertTrue(certification["strict_nonzero_analytic"])
        self.assertTrue(
            certification["fft_rows_are_replays_not_sign_evidence"]
        )
        self.assertTrue(
            certification["modified_finite_static_packet_defined"]
        )
        self.assertTrue(
            certification["modified_finite_static_hhl_replayed"]
        )
        self.assertFalse(
            certification[
                "modified_static_limit_quantitative_remainder_ported"
            ]
        )
        self.assertFalse(
            certification["modified_finite_c1_tail_ledger_ported"]
        )
        self.assertFalse(
            certification["modified_first_and_second_jet_optimizer_ported"]
        )
        self.assertFalse(certification["navier_stokes_clay_problem_solved"])
        pilot = _quadrature_row(4)
        self.assertTrue(pilot["all_numerical_checks_pass"])
        self.assertLess(
            pilot["combined_fixed_output_functional"],
            0.0,
        )
        self.assertTrue(math.isfinite(pilot["functional_replay_error"]))


if __name__ == "__main__":
    unittest.main()
