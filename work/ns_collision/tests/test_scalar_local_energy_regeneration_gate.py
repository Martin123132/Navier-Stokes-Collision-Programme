from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dense_annular_hhh_packet_gate_audit as dense
import nonlinear_stress_regeneration_gate_audit as regeneration
from scalar_local_energy_regeneration_gate_audit import (
    POSITIVE_QUARTET_COEFFICIENT_LIMIT,
    REAL_QUARTET_LOAD_LIMIT,
    _central_quartic_audit,
    _dense_quartic_row,
    _independent_quartic_reconstruction,
    _pressure_pair,
    _quartic_symbol_components,
    _sharp_negative_shell_norm,
    _trace_flux_identity_audit,
)


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "scalar_local_energy_regeneration_gate_audit_v2.json"
)


class ScalarLocalEnergyRegenerationGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stored = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_exact_scalar_trace_identity(self) -> None:
        audit = _trace_flux_identity_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertLess(audit["maximum_identity_residual"], 1.0e-9)
        self.assertLess(audit["zero_output_trace_residual"], 1.0e-10)
        self.assertEqual(
            _pressure_pair(
                (1, 0, 0),
                np.asarray((0.0, 1.0, 0.0)),
                (-1, 0, 0),
                np.asarray((0.0, 0.0, 1.0)),
            ),
            0.0j,
        )
        self.assertTrue(
            self.stored["certification_flags"][
                "ordinary_scalar_local_energy_trace_removes_H_five_halves"
            ]
        )

    def test_independent_quartic_reconstruction(self) -> None:
        audit = _independent_quartic_reconstruction()
        self.assertTrue(audit["all_checks_pass"])
        self.assertLess(audit["maximum_vector_residual"], 1.0e-9)
        self.assertTrue(
            self.stored["certification_flags"][
                "linearized_low_velocity_evolution_included"
            ]
        )

    def test_complete_center_symbol_survives(self) -> None:
        audit = _central_quartic_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertEqual(audit["exact_limit"], "3*sqrt(2)/16")
        self.assertAlmostEqual(
            audit["numeric_limit"],
            REAL_QUARTET_LOAD_LIMIT,
        )
        self.assertLess(audit["last_complete_relative_error"], 2.0e-5)
        self.assertEqual(
            audit["frequency_bookkeeping"]["quartic_flux_output_wave"],
            [-1, -1, -1],
        )
        self.assertEqual(
            audit["frequency_bookkeeping"]["paired_scalar_output_wave"],
            [0, 0, 0],
        )
        self.assertTrue(
            self.stored["certification_flags"][
                "complete_differentiated_HHL_H_five_halves_survives"
            ]
        )

    def test_tau_zero_full_symbol_equals_leading_stress(self) -> None:
        samples = (
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            (
                (0.8, -0.4, 0.3),
                (-0.2, 0.1, -0.5),
            ),
            (
                (-1.0, 0.5, 0.25),
                (0.4, -0.75, 0.5),
            ),
        )
        low_value = 1j * np.asarray((1.0, -1.0, 0.0))
        gradient = 1j * np.ones(3) / 64.0
        for first_offset, second_offset in samples:
            third_offset = -(
                np.asarray(first_offset)
                + np.asarray(second_offset)
            )
            self.assertLessEqual(
                float(np.max(np.abs(third_offset))),
                1.0,
            )
            normalized_offsets = (
                np.asarray(first_offset) / 4096.0,
                np.asarray(second_offset) / 4096.0,
                third_offset / 4096.0,
            )
            waves = [
                np.asarray(direction, dtype=float) + offset
                for direction, offset in zip(
                    dense.CENTER_DIRECTIONS,
                    normalized_offsets,
                )
            ]
            values = [
                phase * regeneration._project(base, wave)
                for base, wave, phase in zip(
                    dense.BASE_VECTORS,
                    waves,
                    dense.PHASES,
                )
            ]
            complete = _quartic_symbol_components(
                waves[0],
                values[0],
                waves[1],
                values[1],
                waves[2],
                values[2],
                low_wave=np.zeros(3),
                low_value=low_value,
            )["complete"]
            stress = regeneration._hhh_stress_forcing(
                waves[0],
                values[0],
                waves[1],
                values[1],
                waves[2],
                values[2],
            )
            complete_load = float(np.dot(complete, gradient).real)
            stress_load = float(
                np.dot(stress @ low_value, gradient).real
            )
            self.assertAlmostEqual(
                complete_load,
                stress_load,
                places=11,
            )

    def test_lightweight_frequency_isolated_dense_packet(self) -> None:
        row = _dense_quartic_row(1, 4096)
        stored = self.stored["dense_spaced_packet"]["rows"][0]
        self.assertTrue(row["all_checks_pass"])
        self.assertEqual(row, stored)
        self.assertEqual(row["real_high_mode_count"], 162)
        self.assertEqual(row["exact_coherent_triad_count"], 343)
        self.assertGreater(
            row["minimum_positive_quartet_load_over_carrier"],
            0.0,
        )
        self.assertLessEqual(
            row["maximum_quartic_decomposition_roundoff_units"],
            256.0,
        )
        self.assertAlmostEqual(
            row["central_positive_quartet_coefficient_reference"],
            POSITIVE_QUARTET_COEFFICIENT_LIMIT,
        )
        self.assertGreaterEqual(
            row["coherent_count_normalized_complete_coefficient"],
            self.stored["dense_spaced_packet"][
                "continuum_positive_coefficient_lower"
            ],
        )

    def test_sharp_negative_three_halves_norm(self) -> None:
        theorem = _sharp_negative_shell_norm()
        self.assertTrue(theorem["all_checks_pass"])
        self.assertEqual(theorem["sharp_weight_exponent"], "3/2")
        self.assertEqual(theorem["squared_norm_weight"], "H^(-3)")
        self.assertTrue(
            theorem["finite_sequence_replay"]["all_checks_pass"]
        )
        self.assertLessEqual(
            theorem["finite_sequence_replay"][
                "weighted_forcing_over_energy_squared_dissipation"
            ],
            1.0,
        )
        flags = self.stored["certification_flags"]
        self.assertTrue(
            flags["sharp_shell_negative_three_halves_forcing_norm_proved"]
        )
        self.assertTrue(flags["Leray_control_of_weighted_HHH_forcing_proved"])

    def test_stored_scope_and_route_decision(self) -> None:
        stored = self.stored
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(len(stored["dense_spaced_packet"]["rows"]), 3)
        self.assertEqual(
            stored["dense_spaced_packet"]["rows"][-1][
                "exact_coherent_triad_count"
            ],
            50653,
        )
        self.assertTrue(
            stored["dense_spaced_packet"][
                "all_coherent_quartets_have_positive_selected_load"
            ]
        )
        flags = stored["certification_flags"]
        self.assertTrue(
            flags["fixed_width_center_limit_claim_withdrawn"]
        )
        self.assertTrue(
            flags["continuous_offset_domain_uniform_positivity_proved"]
        )
        self.assertFalse(flags["full_nonlinear_shell_response_closed"])
        self.assertFalse(flags["suitable_weak_solution_passage_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
