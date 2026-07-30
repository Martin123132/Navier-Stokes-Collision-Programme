"""Tests for the annular continuum convolution quadrature."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "work/ns_collision/scripts"
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_rho_zero_continuum_convolution_quadrature_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_rho_zero_continuum_convolution_quadrature import (  # noqa: E402
    _grid_shape,
    _quadrature_row,
)


class AnnularContinuumConvolutionQuadratureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_fourfold_grid_is_minimal_for_quartic_mean(self) -> None:
        self.assertEqual(_grid_shape(9), (105, 18, 18))
        certificate = self.result["dealias_certificate"]
        self.assertEqual(certificate["factor"], 4)
        self.assertIn("quartic means", certificate["reason"])

    def test_small_row_recomputes_with_energy_trace(self) -> None:
        row = _quadrature_row(5)
        self.assertTrue(row["all_checks_pass"])
        self.assertGreater(
            row["first_form_continuum_quadrature"],
            0.0,
        )
        self.assertLess(
            row["second_form_continuum_quadrature"],
            0.0,
        )
        self.assertLess(
            row["combined_continuum_quadrature"],
            0.0,
        )
        self.assertLess(row["energy_trace_relative_residual"], 1.0e-12)
        self.assertLess(row["maximum_divergence_residual"], 1.0e-11)

    def test_production_rows_extend_to_sixty_five(self) -> None:
        rows = self.result["rows"]
        self.assertEqual(
            [row["size"] for row in rows],
            list(range(9, 66, 4)),
        )
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertTrue(
            all(
                row["first_form_continuum_quadrature"] > 0.0
                and row["second_form_continuum_quadrature"] < 0.0
                and row["combined_continuum_quadrature"] < 0.0
                for row in rows
            )
        )
        self.assertLess(
            max(row["energy_trace_relative_residual"] for row in rows),
            3.0e-15,
        )

    def test_tail_fits_agree_on_negative_candidate(self) -> None:
        fits = self.result["tail_inverse_N_fits"]["fits"]
        combined = {
            row["degree_in_inverse_N"]: row
            for row in fits["combined_continuum_quadrature"]
        }
        first = {
            row["degree_in_inverse_N"]: row
            for row in fits["first_form_continuum_quadrature"]
        }
        second = {
            row["degree_in_inverse_N"]: row
            for row in fits["second_form_continuum_quadrature"]
        }
        self.assertLess(-3.00e-7, combined[4]["candidate_limit"])
        self.assertLess(combined[4]["candidate_limit"], -2.99e-7)
        self.assertLess(
            combined[4]["maximum_replay_residual"],
            1.0e-15,
        )
        self.assertGreater(first[4]["candidate_limit"], 1.72e-7)
        self.assertLess(first[4]["candidate_limit"], 1.73e-7)
        self.assertLess(second[4]["candidate_limit"], -4.71e-7)
        self.assertGreater(second[4]["candidate_limit"], -4.72e-7)

    def test_original_fixed_output_data_cross_replay(self) -> None:
        replay = self.result["fixed_output_cross_replay"]
        self.assertTrue(replay["all_checks_pass"])
        self.assertEqual(replay["largest_common_size"], 29)
        self.assertLess(
            replay["largest_common_absolute_difference"],
            4.0e-9,
        )

    def test_numerical_sign_does_not_impersonate_interval_proof(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        route = self.result["route_decision"]
        self.assertTrue(route["continuum_sign_numerically_stable"])
        self.assertFalse(route["continuum_sign_interval_certified"])
        self.assertFalse(route["nonzero_N9_coefficient_certified"])
        flags = self.result["certification_flags"]
        self.assertFalse(flags["four_high_N9_coefficient_certified"])
        self.assertFalse(flags["finite_parabolic_window_controlled"])
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
