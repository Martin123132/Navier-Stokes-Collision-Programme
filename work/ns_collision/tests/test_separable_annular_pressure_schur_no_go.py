"""Focused tests for the separable annular pressure-Schur no-go."""

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
    "separable_annular_pressure_schur_no_go_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from separable_annular_pressure_schur_no_go_audit import (  # noqa: E402
    _annular_row,
    _dictionary_replay,
    _exact_algebra_certificates,
)


class SeparableAnnularPressureSchurNoGoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.exact = _exact_algebra_certificates()
        cls.small = _annular_row(3)
        cls.replay = _dictionary_replay()

    def test_exact_pressure_and_kinetic_matrices(self) -> None:
        self.assertTrue(self.exact["all_checks_pass"])
        self.assertEqual(
            self.exact[
                "pressure_limit_matrix_sqrt2_coefficients"
            ],
            [["0", "0", "0"], ["0", "1/20", "0"], ["0", "0", "-1/20"]],
        )
        self.assertTrue(
            self.exact["kinetic_leading_matrix_exactly_zero"]
        )
        self.assertTrue(
            self.exact[
                "one_dimensional_difference_identity_exact"
            ]
        )

    def test_small_family_recomputes(self) -> None:
        self.assertTrue(self.small["all_checks_pass"])
        stored = self.result["annular_family_rows"][0]
        self.assertEqual(stored["size"], 3)
        self.assertAlmostEqual(
            self.small["Fisher_energy_mixed_difference"],
            stored["Fisher_energy_mixed_difference"],
            places=14,
        )
        self.assertAlmostEqual(
            self.small["complete_HHL_load"],
            stored["complete_HHL_load"],
            places=14,
        )

    def test_dictionary_replay_is_independent_and_exact(self) -> None:
        self.assertTrue(self.replay["all_checks_pass"])
        self.assertLess(self.replay["Fisher_residual"], 2.0e-12)
        self.assertLess(
            self.replay["maximum_component_load_residual"], 2.0e-12
        )
        self.assertLess(
            self.replay["component_vs_direct_flux_residual"], 2.0e-11
        )

    def test_finite_rows_track_the_proved_asymptotics(self) -> None:
        rows = self.result["annular_family_rows"]
        self.assertEqual(
            [row["size"] for row in rows],
            [3, 5, 7, 9, 13, 17, 25, 33, 49, 65],
        )
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertGreater(
            rows[-1]["absolute_complete_load_over_Fisher"], 7000.0
        )
        self.assertGreater(rows[-1]["size_cubed_times_Fisher"], 3.0)
        self.assertLess(rows[-1]["size_cubed_times_Fisher"], 4.5)
        self.assertLess(
            abs(rows[-1]["kinetic_load_over_size"]), 1.0e-9
        )
        self.assertLess(
            abs(rows[-1]["pressure_cross_load_over_size"]), 1.0e-10
        )

    def test_pressure_limit_has_an_analytic_margin(self) -> None:
        exact = self.result["exact_algebra_certificates"]
        quadrature = self.result["continuum_pressure_quadrature"]
        self.assertEqual(
            exact["pressure_limit_absolute_lower_bound"],
            "51*sqrt(2)/438976",
        )
        self.assertTrue(quadrature["sign_and_margin_check"])
        self.assertLess(quadrature["pressure_load_limit"], 0.0)
        self.assertGreater(
            abs(quadrature["pressure_load_limit"]),
            quadrature["analytic_absolute_lower_bound"],
        )

    def test_fixed_transverse_control_saturates(self) -> None:
        controls = self.result["fixed_transverse_control_rows"]
        ratios = [
            row["absolute_complete_load_over_Fisher"]
            for row in controls
        ]
        self.assertTrue(
            all(
                ratios[index] > ratios[index + 1]
                for index in range(len(ratios) - 1)
            )
        )
        self.assertLess(ratios[-1], 0.3)

    def test_scope_is_fail_closed(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            (
                "analytic_separable_annular_complete_HHL_"
                "Schur_no_go_certified"
            ),
        )
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags[
                "uniform_joint_complete_HHL_Fisher_Schur_bound_falsified"
            ]
        )
        self.assertFalse(
            flags["isolated_primitive_chain_Hardy_theorem_falsified"]
        )
        self.assertFalse(flags["all_pressure_Fisher_methods_falsified"])
        self.assertFalse(flags["all_cross_shell_HHL_absorbed"])
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
