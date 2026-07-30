from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from smooth_galerkin_shell_response_gate_audit import (
    FORCING_SQUARE_CONSTANT,
    HEAT_WEIGHTED_HHL_CONSTANT,
    _forcing_square_audit,
    _heat_weighted_hhl_audit,
    _pairwise_evolution_audit,
    _response_and_initial_audit,
)


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "smooth_galerkin_shell_response_gate_audit_v1.json"
)


class SmoothGalerkinShellResponseGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stored = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_exact_pairwise_rates_are_retained(self) -> None:
        audit = _pairwise_evolution_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertFalse(audit["single_shell_scalar_rate_used"])
        self.assertGreater(audit["distinct_pair_rate_count"], 1)
        self.assertLess(audit["maximum_relative_residual"], 1.0e-12)

    def test_heat_weighted_hhl_commutator_keeps_low_factor(self) -> None:
        audit = _heat_weighted_hhl_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertEqual(HEAT_WEIGHTED_HHL_CONSTANT, 80.0)
        self.assertLessEqual(audit["maximum_bound_ratio"], 1.0)
        self.assertLess(
            audit["maximum_rate_identity_residual"],
            1.0e-12,
        )
        self.assertLessEqual(
            audit["maximum_selector_lipschitz_ratio"],
            1.0 + 1.0e-12,
        )

    def test_complete_weighted_forcing_square(self) -> None:
        audit = _forcing_square_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertLess(
            audit["derived_combined_constant"],
            FORCING_SQUARE_CONSTANT,
        )
        self.assertEqual(audit["retained_integer_constant"], 104.0)
        for row in audit["rows"]:
            self.assertLessEqual(
                row["complete_weighted_forcing_square"],
                row["analytic_bound"] * (1.0 + 1.0e-12),
            )

    def test_forced_and_initial_tail_summation(self) -> None:
        forcing = _forcing_square_audit()
        response = _response_and_initial_audit(forcing)
        self.assertTrue(response["all_checks_pass"])
        for row in response["rows"]:
            self.assertLessEqual(
                row["duhamel_triangle_bound"],
                row["duhamel_dyadic_tail_bound"]
                * (1.0 + 1.0e-12),
            )
            self.assertLessEqual(
                row["initial_exact_Gram_norm"],
                row["initial_energy_tail_bound"]
                * (1.0 + 1.0e-12),
            )

    def test_stored_scope_and_status(self) -> None:
        stored = self.stored
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "exact_pair_rate_complete_dyadic_smooth_galerkin_"
            "stress_response_certified",
        )
        flags = stored["certification_flags"]
        self.assertTrue(flags["smooth_Galerkin_exact_pair_rates_retained"])
        self.assertTrue(flags["heat_weighted_HHL_commutator_proved"])
        self.assertTrue(flags["sharp_Galerkin_cutoff_leakage_paid"])
        self.assertTrue(
            flags["complete_HHH_HHL_weighted_forcing_square_proved"]
        )
        self.assertTrue(flags["finite_low_channel_high_stress_tail_vanishes"])
        self.assertFalse(flags["scale_uniform_spatial_localization_proved"])
        self.assertFalse(flags["suitable_weak_solution_passage_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
