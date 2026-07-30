from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scale_adapted_edge_rho_expansion_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "scale_adapted_edge_rho_expansion_audit_v1.json"
)


class ScaleAdaptedEdgeRhoExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_scale_homogeneity_and_nonpromotion(self) -> None:
        scale = self.result["scale_homogeneity"]
        self.assertTrue(scale["all_checks_pass"])
        self.assertEqual(scale["exact_ratio_symbolic_residual"], "0")
        self.assertEqual(scale["edge_ratio_symbolic_residual"], "0")
        self.assertEqual(
            scale["local_Reynolds_number"],
            "Re_cell=a/(nu m)",
        )
        self.assertFalse(
            scale["fixed_scale_universal_absorption_possible"]
        )

        flags = self.result["certification_flags"]
        self.assertTrue(
            flags[
                "fixed_scale_universal_edge_absorption_falsified_by_scaling"
            ]
        )
        self.assertFalse(
            flags["scale_adapted_edge_remainder_absorbed"]
        )
        self.assertFalse(flags["critical_signed_replica_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_partition_frequency_identities(self) -> None:
        sweep = self.result["partition_frequency_sweep"]
        self.assertTrue(sweep["all_checks_pass"])
        self.assertEqual(
            sweep["positive_pressure_frequencies"],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(
            sweep["spectrally_silent_frequencies"],
            [7, 8, 9, 10, 11, 12],
        )
        for row in sweep["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertLess(row["representation_residual"], 2.0e-10)

    def test_frequency_adaptation_improves_resolved_edge_budget(self) -> None:
        rows = self.result["partition_frequency_sweep"]["rows"]
        first = rows[0]
        sixth = rows[5]
        self.assertLess(
            first["edge_absorption_amplitude_threshold"],
            0.5,
        )
        self.assertGreater(
            sixth["edge_absorption_amplitude_threshold"],
            5.0,
        )
        self.assertLess(
            sixth["scale_adapted_young_remainder"],
            first["scale_adapted_young_remainder"],
        )

    def test_short_time_rho_expansion(self) -> None:
        expansion = self.result["short_time_rho_expansion"]
        self.assertTrue(expansion["all_checks_pass"])
        lower, upper = expansion["leading_reset_loss_range"]
        self.assertGreater(lower, 2361.35)
        self.assertLess(upper, 2361.36)
        coefficient_lower, coefficient_upper = expansion[
            "first_time_coefficient_range"
        ]
        self.assertLess(coefficient_upper, -62459.6)
        self.assertGreater(coefficient_lower, -62459.7)
        for row in expansion["rows"]:
            self.assertLess(
                row["replica_pressure_gauge_residual"],
                2.0e-10,
            )
            self.assertAlmostEqual(
                row["formal_integrated_crossover_time"],
                0.0756122707598066,
                places=12,
            )

    def test_stored_result(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored["kind"], self.result["kind"])
        self.assertEqual(stored["status"], self.result["status"])
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertFalse(
            stored["certification_flags"][
                "finite_time_positive_rho_advantage_proved"
            ]
        )
        self.assertFalse(
            stored["certification_flags"][
                "Taylor_crossover_is_a_sign_certificate"
            ]
        )


if __name__ == "__main__":
    unittest.main()
