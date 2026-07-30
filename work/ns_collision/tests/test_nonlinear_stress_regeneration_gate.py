from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nonlinear_stress_regeneration_gate_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "nonlinear_stress_regeneration_gate_audit_v1.json"
)


class NonlinearStressRegenerationGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_exact_projected_stress_evolution(self) -> None:
        evolution = self.result["projected_stress_evolution"]
        flags = self.result["certification_flags"]
        self.assertTrue(evolution["all_checks_pass"])
        self.assertEqual(evolution["test_output_wave"], [1, 1, 0])
        self.assertEqual(
            evolution["first_generated_pair_output"],
            [1, 1, 0],
        )
        self.assertEqual(
            evolution["second_generated_pair_output"],
            [1, 1, 0],
        )
        self.assertLess(
            evolution["paired_e11_formula_residual"],
            1.0e-13,
        )
        self.assertTrue(
            flags["exact_projected_stress_evolution_derived"]
        )

    def test_HHL_low_factor_commutator(self) -> None:
        hhl = self.result["HHL_sweeping_commutator"]
        flags = self.result["certification_flags"]
        self.assertTrue(hhl["all_checks_pass"])
        self.assertIn("18 L", hhl["theorem"])
        self.assertLess(
            hhl["maximum_random_forcing_over_low_scale"],
            18.0,
        )
        self.assertLess(
            hhl["maximum_divergence_residual"],
            1.0e-11,
        )
        for row in hhl["rows"]:
            self.assertTrue(row["all_checks_pass"])
            self.assertLess(row["special_paired_over_unpaired"], 0.2)
        self.assertTrue(
            flags["HHL_leading_carrier_terms_cancel_proved"]
        )
        self.assertTrue(
            flags["HHL_regeneration_low_factor_bound_proved"]
        )

    def test_coherent_HHL_pump_cancellation(self) -> None:
        evolution = self.result["projected_stress_evolution"]
        self.assertGreater(
            abs(evolution["first_unpaired_e11"]),
            1.0,
        )
        self.assertGreater(
            abs(evolution["second_unpaired_e11"]),
            1.0,
        )
        self.assertGreater(
            evolution["unpaired_to_paired_e11_ratio"],
            1.0e6,
        )

    def test_HHH_pressure_strain_obstruction(self) -> None:
        hhh = self.result["HHH_pressure_strain_obstruction"]
        flags = self.result["certification_flags"]
        self.assertTrue(hhh["all_checks_pass"])
        self.assertGreater(
            hhh["limiting_projected_Frobenius_norm"],
            0.1,
        )
        self.assertLess(hhh["limiting_projected_trace"], 1.0e-12)
        self.assertLess(hhh["limiting_raw_transport_norm"], 1.0e-12)
        self.assertLess(
            hhh["rows"][-1]["normalized_residual_from_limit"],
            0.02,
        )
        self.assertTrue(
            flags[
                "HHH_anisotropic_pressure_strain_carrier_witness_proved"
            ]
        )
        self.assertFalse(
            flags["all_regeneration_low_factor_bound_proved"]
        )

    def test_sparse_parabolic_pulse_summability_and_scope(self) -> None:
        pulses = self.result["sparse_parabolic_pulse_test"]
        flags = self.result["certification_flags"]
        self.assertTrue(pulses["all_checks_pass"])
        self.assertGreater(
            pulses["rows"][-1][
                "coherent_sum_over_shell_square_function"
            ],
            2.5,
        )
        self.assertLess(pulses["energy_increment_ratio_max"], 0.5)
        self.assertLess(
            pulses["enstrophy_increment_ratio_max"],
            0.5,
        )
        self.assertLess(pulses["forcing_increment_ratio_max"], 0.1)
        self.assertTrue(
            flags["sparse_parabolic_forcing_summability_proved"]
        )
        self.assertFalse(
            flags["dense_packet_multiplicity_control_proved"]
        )
        self.assertFalse(
            flags[
                "full_Navier_Stokes_regeneration_norm_from_Leray_proved"
            ]
        )
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result_replays_exactly(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored, self.result)
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            (
                "HHL_regeneration_commutator_certified_"
                "HHH_pressure_strain_obstruction_exhibited"
            ),
        )


if __name__ == "__main__":
    unittest.main()
