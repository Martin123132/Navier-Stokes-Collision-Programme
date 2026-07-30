from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dyadic_three_shell_atlas_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "dyadic_three_shell_atlas_audit_v1.json"
)


class DyadicThreeShellAtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_largest_two_scale_support_atlas(self) -> None:
        atlas = self.result["support_atlas"]
        occupied = self.result["occupied_triple_support_stress"]
        self.assertTrue(atlas["all_checks_pass"])
        self.assertIn("M_2>=(M_1-R)/2", atlas["sorted_support_rule"])
        self.assertIn("HHH", atlas["atlas"])
        self.assertIn("HHL", atlas["atlas"])
        for row in atlas["rows"]:
            self.assertTrue(
                row[
                    "at_most_four_when_largest_at_least_twice_stencil"
                ]
            )
        self.assertTrue(occupied["all_checks_pass"])
        self.assertGreater(occupied["occupied_ordered_triple_count"], 0)
        self.assertLess(
            occupied["maximum_largest_over_second_ratio"],
            4.0,
        )

    def test_localized_shell_skew_and_kinetic_reconstruction(self) -> None:
        skew = self.result["localized_shell_skew_identity"]
        self.assertTrue(skew["all_checks_pass"])
        self.assertIn(
            "T_Phi(a;b,c)+T_Phi(a;c,b)",
            skew["localized_identity"],
        )
        self.assertLess(
            skew["maximum_localized_skew_residual"],
            1.0e-12,
        )
        self.assertLess(
            skew["maximum_HHL_kinetic_reconstruction_residual"],
            1.0e-12,
        )
        self.assertLess(
            skew["maximum_global_antisymmetry_residual"],
            1.0e-12,
        )

    def test_eight_vertex_top_walsh_channel(self) -> None:
        vertices = self.result["eight_vertex_flux_structure"]
        self.assertTrue(vertices["all_checks_pass"])
        self.assertGreater(vertices["all_cosine_vertex_load"], 1.0e-4)
        self.assertEqual(vertices["equal_weight_eight_vertex_sum"], 0.0)
        self.assertEqual(
            vertices["maximum_off_top_Walsh_coefficient"],
            0.0,
        )
        self.assertAlmostEqual(
            vertices["Walsh_coefficients"]["7"],
            vertices["all_cosine_vertex_load"],
            places=14,
        )
        self.assertAlmostEqual(
            vertices["selector_sum_over_L1"],
            0.5,
            places=14,
        )

    def test_multishell_coherent_accumulation(self) -> None:
        stress = self.result["multishell_coherence_stress"]
        self.assertTrue(stress["all_checks_pass"])
        self.assertEqual(
            stress["individual_carriers"],
            [16, 32, 64, 128, 256],
        )
        for row in stress["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertGreater(row["combined_vertex_load"], 0.0)
            self.assertLess(
                row["cross_shell_coherence_residual"],
                1.0e-12,
            )
            self.assertAlmostEqual(
                row["high_Fourier_L2_energy_proxy"],
                4.0 * row["high_shell_count"],
                places=13,
            )

    def test_amplitude_envelope_and_scope_flags(self) -> None:
        envelope = self.result["HHL_amplitude_envelope"]
        flags = self.result["certification_flags"]
        self.assertTrue(envelope["all_checks_pass"])
        self.assertIn(
            "sqrt(2m)",
            envelope["dyadic_sequence_bound"],
        )
        for row in envelope["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertLessEqual(
                row["HHL_amplitude_sum"],
                row["universal_sqrt2_upper"],
            )
            self.assertLess(
                row["cubic_scaling_residual"],
                1.0e-13,
            )
        self.assertTrue(
            flags["largest_two_velocity_scales_comparable_proved"]
        )
        self.assertTrue(flags["HHL_amplitude_square_sum_bound_proved"])
        self.assertTrue(
            flags["fixed_vertex_pure_shell_telescoping_falsified"]
        )
        self.assertFalse(
            flags[
                "large_data_viscous_absorption_from_Leray_energy_proved"
            ]
        )
        self.assertFalse(flags["joint_scale_cell_Carleson_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result_replays_exactly(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored, self.result)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "dyadic_atlas_certified_naive_telescoping_falsified",
        )


if __name__ == "__main__":
    unittest.main()
