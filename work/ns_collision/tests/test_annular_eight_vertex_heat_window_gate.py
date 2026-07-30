"""Focused tests for the annular eight-vertex heat-window gate."""

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
    "annular_eight_vertex_heat_window_gate_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_eight_vertex_heat_window_gate_audit import (  # noqa: E402
    _dictionary_replay,
    _dynamic_row,
    _exact_incidence_certificate,
    _static_row,
)


class AnnularEightVertexHeatWindowGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.exact = _exact_incidence_certificate()
        cls.static_small = _static_row(3)
        cls.dynamic_small = _dynamic_row(3, 1.0, 0.1, 4)
        cls.replay = _dictionary_replay()

    def test_exact_incidence_has_six_surviving_characters(self) -> None:
        self.assertTrue(self.exact["all_checks_pass"])
        self.assertEqual(
            self.exact["nonzero_leading_Walsh_masks"],
            ["y", "xy", "z", "xz", "yz", "xyz"],
        )
        self.assertTrue(
            self.exact["pure_x_Walsh_matrix_exactly_zero"]
        )
        self.assertTrue(
            self.exact["equal_weight_pressure_cancellation_exact"]
        )
        self.assertTrue(
            self.exact[
                "every_vertex_kinetic_leading_matrix_exactly_zero"
            ]
        )

    def test_small_static_row_recomputes(self) -> None:
        self.assertTrue(self.static_small["all_checks_pass"])
        stored = self.result["static_family_rows"][0]
        self.assertAlmostEqual(
            self.static_small["plus_vertex_complete_load_over_size"],
            stored["plus_vertex_complete_load_over_size"],
            places=14,
        )
        self.assertAlmostEqual(
            self.static_small["plus_vertex_Fisher_times_size_cubed"],
            stored["plus_vertex_Fisher_times_size_cubed"],
            places=13,
        )

    def test_all_vertex_dictionary_replay_passes(self) -> None:
        self.assertTrue(self.replay["all_checks_pass"])
        self.assertEqual(self.replay["vertices_checked"], 8)
        self.assertEqual(
            self.replay["component_vertex_loads_checked"], 32
        )
        self.assertLess(
            self.replay[
                "maximum_dictionary_vs_Walsh_load_residual"
            ],
            3.0e-12,
        )
        self.assertLess(
            self.replay[
                "maximum_dictionary_vs_mixed_Fisher_residual"
            ],
            3.0e-12,
        )

    def test_fisher_partition_and_scaling_classes(self) -> None:
        rows = self.result["static_family_rows"]
        self.assertEqual(
            [row["size"] for row in rows], [3, 5, 9, 17, 33, 65]
        )
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        final = rows[-1]
        self.assertLess(final["global_Fisher_partition_residual"], 3e-10)
        exponents = {
            value["predicted_size_exponent"]
            for value in final["Fisher_scaling_diagnostics"].values()
        }
        self.assertEqual(exponents, {-3, -1, 1, 3})
        self.assertGreater(
            final["plus_vertex_absolute_load_over_Fisher"], 7000.0
        )

    def test_continuum_response_is_zero_sum_but_nonzero(self) -> None:
        continuum = self.result["continuum_response_certificate"]
        self.assertTrue(continuum["all_checks_pass"])
        self.assertLess(
            abs(continuum["equal_weight_static_limit_sum"]), 3e-14
        )
        self.assertGreater(continuum["static_vertex_l1_norm"], 0.03)
        self.assertEqual(
            continuum["nonzero_static_Walsh_characters"],
            ["y", "xy", "z", "xz", "yz", "xyz"],
        )

    def test_heat_window_recomputes_and_retains_N4_loss(self) -> None:
        self.assertTrue(self.dynamic_small["all_checks_pass"])
        stored = self.result["heat_window_rows"][0]
        self.assertAlmostEqual(
            self.dynamic_small[
                "plus_vertex_dynamic_ratio_over_size_to_fourth"
            ],
            stored["plus_vertex_dynamic_ratio_over_size_to_fourth"],
            places=10,
        )
        continuum = self.result["continuum_response_certificate"]
        self.assertLess(
            continuum["plus_vertex_heat_integrated_limit"], 0.0
        )
        self.assertGreater(
            abs(continuum["plus_vertex_heat_integrated_limit"]),
            continuum[
                "analytic_plus_heat_integral_absolute_lower_bound"
            ],
        )

    def test_equal_weight_cancellation_persists_at_each_heat_node(
        self,
    ) -> None:
        for row in self.result["heat_window_rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertLess(
                row[
                    "maximum_equal_weight_vertex_sum_during_quadrature"
                ],
                3.0e-13,
            )

    def test_scope_remains_fail_closed(self) -> None:
        self.assertTrue(self.result["all_positive_checks_pass"])
        self.assertEqual(
            self.result["status"],
            (
                "eight_vertex_cancellation_and_"
                "local_heat_persistence_certified"
            ),
        )
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags[
                "exact_equal_weight_eight_vertex_cancellation_proved"
            ]
        )
        self.assertTrue(
            flags["heat_viscosity_preserves_local_N4_obstruction_proved"]
        )
        self.assertTrue(flags["small_amplitude_NS_shadowing_transfer_proved"])
        self.assertFalse(
            flags["arbitrary_weighted_eight_vertex_flux_controlled"]
        )
        self.assertFalse(
            flags["large_amplitude_phase_compensation_excluded"]
        )
        self.assertFalse(flags["critical_L3_controlled"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
