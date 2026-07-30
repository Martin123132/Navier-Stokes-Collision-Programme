"""Focused tests for the pressure-active Fisher-null compatibility gate."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pressure_active_fisher_null_compatibility_gate_audit import (  # noqa: E402
    audit,
)


class PressureActiveFisherNullCompatibilityGateTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_exact_symbolic_coefficient_certificate_passes(self) -> None:
        symbolic = self.result["symbolic_certificate"]
        self.assertTrue(symbolic["all_checks_pass"])
        self.assertTrue(
            symbolic["derived_complete_matches_claimed"]
        )
        self.assertEqual(symbolic["shear_identity_residual"], "0")
        self.assertTrue(
            symbolic["all_numerator_coefficients_positive"]
        )

    def test_sparse_complete_HHL_reconstruction_passes(self) -> None:
        rows = self.result["sparse_symbol_replays"]
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(row["all_checks_pass"] for row in rows))
        self.assertLess(
            max(row["maximum_component_load_residual"] for row in rows),
            2.0e-12,
        )
        self.assertLess(
            max(row["direct_flux_coefficient_residual"] for row in rows),
            2.0e-12,
        )

    def test_null_rows_vanish_and_phase_tilts_activate_pressure(
        self,
    ) -> None:
        rows = self.result["sparse_symbol_replays"]
        null_rows = [
            row
            for row in rows
            if not row["expected_pressure_load_active"]
        ]
        active_rows = [
            row for row in rows if row["expected_pressure_load_active"]
        ]
        self.assertTrue(
            all(
                abs(row["sparse_loads"]["combined"]) < 2.0e-12
                for row in null_rows
            )
        )
        self.assertTrue(
            all(row["pressure_load_active"] for row in active_rows)
        )

    def test_complete_load_is_bounded_by_half_the_full_Fisher(
        self,
    ) -> None:
        rows = self.result["sparse_symbol_replays"]
        self.assertLessEqual(
            max(
                row["complete_load_over_weighted_Fisher"]
                for row in rows
            ),
            0.5,
        )
        spectra = self.result["generalized_spectral_replays"]
        self.assertTrue(all(row["all_checks_pass"] for row in spectra))
        self.assertLess(
            max(row["two_polarization_maximum"] for row in spectra),
            0.5,
        )

    def test_pressure_and_anisotropic_kinetic_cancel_exactly(
        self,
    ) -> None:
        theorem = self.result["exact_edge_theorem"]
        self.assertIn(
            "cancels",
            theorem["high_high_pressure_cancellation"],
        )
        self.assertEqual(theorem["global_bound"], "|B_HHL|<=E_lambda(h)/2")
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags[
                "canonical_two_polarization_complete_HHL_Fisher_bound_proved"
            ]
        )

    def test_adversaries_and_global_scope_remain_fail_closed(self) -> None:
        replays = self.result["mandatory_adversary_replays"]
        self.assertTrue(replays["Taylor_Green"]["all_checks_pass"])
        self.assertTrue(replays["seed81"]["all_checks_pass"])
        self.assertTrue(
            replays["modulated_wave_HHL"][
                "all_positive_checks_pass"
            ]
        )
        flags = self.result["certification_flags"]
        self.assertFalse(flags["arbitrary_residue_chain_bound_proved"])
        self.assertFalse(flags["all_cross_shell_HHL_absorbed"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
