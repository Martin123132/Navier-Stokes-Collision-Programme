"""Focused tests for the primitive HHL chain Hardy envelope."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from primitive_hhl_chain_hardy_envelope_audit import audit  # noqa: E402


class PrimitiveHHLChainHardyEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_complete_symbol_budget_and_constants_are_exact(self) -> None:
        budget = self.result["proof_budget"]
        self.assertTrue(budget["all_checks_pass"])
        self.assertEqual(
            budget["ordered_complete_HHL_term_bounds"][
                "total_per_ordered_high_pair"
            ],
            "9/2",
        )
        self.assertEqual(budget["velocity_mass_budget_constant"], 27.0)
        self.assertEqual(budget["axial_chain_constant"], 108.0)
        rows = budget["multi_coordinate_step_rows"]
        self.assertEqual(rows[0]["chain_constant_exact"], "27/2")
        self.assertEqual(rows[1]["chain_constant_exact"], "6")

    def test_resonant_partner_degree_is_exhaustively_six(self) -> None:
        resonance = self.result["resonance_count_certificate"]
        self.assertTrue(resonance["all_checks_pass"])
        self.assertEqual(resonance["searched_steps"], 26)
        self.assertEqual(resonance["searched_low_waves_per_step"], 26)
        self.assertEqual(
            resonance["maximum_resonant_low_sign_offset_pairs"], 6
        )
        self.assertEqual(
            resonance["maximum_for_disjoint_support"], 6
        )

    def test_Hardy_and_orthogonal_phase_spectra_pass(self) -> None:
        rows = self.result["Hardy_certificate"][
            "finite_spectral_rows"
        ]
        self.assertEqual(rows[-1]["chain_length"], 16384)
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertLess(
            max(row["raw_Hardy_generalized_eigenvalue"] for row in rows),
            4.0,
        )
        self.assertLess(
            max(
                row["orthogonal_sine_phase_generalized_eigenvalue"]
                for row in rows
            ),
            2.0 / 3.0,
        )

    def test_sparse_primitive_atlas_reconstructs_complete_flux(
        self,
    ) -> None:
        rows = self.result["primitive_sparse_atlas"]
        self.assertEqual(len(rows), 21)
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertLess(
            max(row["component_vs_direct_flux_residual"] for row in rows),
            3.0e-11,
        )
        self.assertLess(
            max(row["maximum_divergence_residual"] for row in rows),
            3.0e-11,
        )

    def test_every_atlas_ratio_obeys_its_step_constant(self) -> None:
        rows = self.result["primitive_sparse_atlas"]
        for row in rows:
            self.assertLessEqual(
                row["normalized_m_times_load_over_U_Fisher"],
                row["claimed_chain_constant"],
            )
        scales = {row["partition_scale_m"] for row in rows}
        self.assertEqual(scales, {1, 2, 4})
        self.assertTrue(
            self.result["co_scaling_stress"]["all_checks_pass"]
        )

    def test_adversaries_and_joint_block_scope_remain_fail_closed(
        self,
    ) -> None:
        replays = self.result["mandatory_adversary_replays"]
        self.assertTrue(
            replays["canonical_pressure_active_chain"][
                "all_positive_checks_pass"
            ]
        )
        self.assertTrue(replays["Taylor_Green"]["all_checks_pass"])
        self.assertTrue(replays["seed81"]["all_checks_pass"])
        self.assertTrue(
            replays["modulated_wave_HHL"][
                "all_positive_checks_pass"
            ]
        )
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags["uniform_isolated_primitive_chain_envelope_proved"]
        )
        self.assertFalse(
            flags["finite_low_wave_vertex_Schur_bound_proved"]
        )
        self.assertFalse(flags["all_cross_shell_HHL_absorbed"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
