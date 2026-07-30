from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dense_low_output_block_gate_audit import (
    DEFAULT_CARRIER_MULTIPLE,
    DEFAULT_OUTPUT,
    DEFAULT_OUTPUT_HALF_WIDTH,
    _bounded_triple_count_1d,
    _central_interval_self_audit,
    _direct_count_1d,
    _finite_replay,
    _fixed_channel_symbol_interval,
    _multiplicity_audit,
    _scaling_certificate,
)


class DenseLowOutputBlockGateTests(unittest.TestCase):
    def test_inclusion_exclusion_count_matches_direct_enumeration(
        self,
    ) -> None:
        for radius in range(1, 7):
            for output in range(-3 * radius, 3 * radius + 1):
                self.assertEqual(
                    _bounded_triple_count_1d(radius, output),
                    _direct_count_1d(radius, output),
                )

    def test_interior_count_is_exact(self) -> None:
        for radius in range(1, 9):
            for output in range(-radius, radius + 1):
                self.assertEqual(
                    _bounded_triple_count_1d(radius, output),
                    3 * radius**2 + 3 * radius + 1 - output**2,
                )
        audit = _multiplicity_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertEqual(
            audit["uniform_one_dimensional_leading_lower"],
            "11/4",
        )

    def test_directed_interval_contains_center_and_stays_positive(
        self,
    ) -> None:
        center = _central_interval_self_audit()
        self.assertTrue(center["all_checks_pass"])
        block = _fixed_channel_symbol_interval()
        self.assertTrue(block["all_checks_pass"])
        lower, upper = block["unit_channel_interval_per_carrier"]
        self.assertGreater(lower, 0.0)
        self.assertGreater(upper, lower)
        self.assertIn("contains the true lattice polytope", block[
            "relaxed_domain"
        ])

    def test_small_exhaustive_lattice_replay_is_inside_interval(
        self,
    ) -> None:
        block = _fixed_channel_symbol_interval()
        lower, upper = block["unit_channel_interval_per_carrier"]
        replay = _finite_replay(
            (1,),
            DEFAULT_CARRIER_MULTIPLE,
            DEFAULT_OUTPUT_HALF_WIDTH,
            lower,
            upper,
            maximum_exhaustive_radius=1,
            sample_limit=64,
        )
        self.assertTrue(replay["all_checks_pass"])
        row = replay["rows"][0]
        self.assertEqual(row["symbol_evaluation_mode"], "exhaustive")
        self.assertEqual(row["symbol_evaluations"], 343)
        self.assertGreater(
            row["certified_uniform_forcing_lower_over_H_5_2"],
            0.0,
        )

    def test_scaling_certificate_keeps_temporal_gate_open(self) -> None:
        block = _fixed_channel_symbol_interval()
        lower = block["unit_channel_interval_per_carrier"][0]
        scaling = _scaling_certificate(
            lower,
            DEFAULT_OUTPUT_HALF_WIDTH,
            DEFAULT_CARRIER_MULTIPLE,
        )
        self.assertTrue(scaling["all_checks_pass"])
        self.assertGreater(scaling["explicit_H_5_2_constant"], 0.0)
        self.assertIn(
            "No lower bound on a parabolic-time forcing pulse",
            scaling["temporal_limit"],
        )

    def test_stored_scope_and_negative_claims(self) -> None:
        stored = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "positive_volume_low_output_HHH_block_certified",
        )
        flags = stored["certification_flags"]
        self.assertTrue(
            flags["positive_volume_low_output_block_realized"]
        )
        self.assertTrue(
            flags["simultaneous_spatial_H_5_2_output_scaling_proved"]
        )
        self.assertFalse(flags["parabolic_time_persistence_proved"])
        self.assertFalse(
            flags["H_minus_1_endpoint_for_actual_Navier_Stokes_proved"]
        )
        self.assertFalse(
            flags[
                "H_minus_1_endpoint_for_actual_Navier_Stokes_falsified"
            ]
        )
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
