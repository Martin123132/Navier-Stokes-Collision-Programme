"""Focused tests for the balanced-annular pressure-edge theorem."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from balanced_annular_pressure_edge_gate_audit import (  # noqa: E402
    audit,
)


class BalancedAnnularPressureEdgeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_prerequisites_are_pinned(self) -> None:
        self.assertTrue(
            self.result["prerequisites"]["all_checks_pass"]
        )

    def test_complete_vertex_bound_survives_sparse_fields(self) -> None:
        self.assertTrue(
            self.result["Taylor_Green_stress"]["all_checks_pass"]
        )
        self.assertTrue(
            self.result["seed81_stress"]["all_checks_pass"]
        )
        self.assertGreater(
            abs(
                self.result["seed81_stress"]["weighted_adversary"][
                    "pressure_load"
                ]
            ),
            1.0,
        )

    def test_compatible_extension_is_floor_free(self) -> None:
        theorem = self.result["theorem"]
        self.assertFalse(theorem["requires_positive_weight_floor"])
        self.assertTrue(
            self.result["seed81_stress"]["weighted_adversary"][
                "cubic_terminal_weight_Fisher_positive"
            ]
        )

    def test_coscaling_keeps_the_dimensionless_ratio(self) -> None:
        stress = self.result["coscaling_stress"]
        self.assertTrue(stress["all_checks_pass"])
        self.assertLess(stress["maximum_ratio_residual"], 1.0e-14)

    def test_scope_keeps_cross_shell_and_regularity_open(self) -> None:
        flags = self.result["certification_flags"]
        self.assertFalse(flags["full_multiband_pressure_edge_absorbed"])
        self.assertFalse(flags["cross_shell_HHL_pressure_absorbed"])
        self.assertFalse(flags["terminal_dual_supremum_controlled"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
