"""Tests for the parallel-shear heat-block exclusion."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "work/ns_collision/scripts"
RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_heat_block_exclusion_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_parallel_shear_heat_block_exclusion_audit import (  # noqa: E402
    _exhaustive_partition_certificate,
    _flow_block_identity_certificate,
    _one_heat_pressure_certificate,
    _remaining_power_ledger,
)


class ParallelShearHeatBlockExclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(RESULT.read_text(encoding="utf-8"))
        predecessor_path = (
            ROOT
            / "work/ns_collision/results/"
            "annular_parallel_shear_finite_jet_port_audit_v1.json"
        )
        cls.predecessor = json.loads(
            predecessor_path.read_text(encoding="utf-8")
        )

    def test_all_atomic_second_subterms_are_partitioned_once(self) -> None:
        certificate = _exhaustive_partition_certificate(self.predecessor)
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(certificate["second_subterm_count"], 69)
        self.assertEqual(certificate["derived_aggregate_row_count"], 2)
        self.assertEqual(certificate["channel_count"], 20)
        self.assertEqual(certificate["duplicate_key_count"], 0)
        self.assertEqual(certificate["unclassified_keys"], [])
        self.assertEqual(
            certificate["block_subterm_counts"],
            {"pure_EA": 21, "one_heat": 31, "two_heat": 17},
        )

    def test_group_counts_and_degrees_are_exact(self) -> None:
        records = self.audit["exhaustive_subterm_partition"][
            "group_records"
        ]
        expected = {
            ("pure_EA", "pressure"): (9, [5, 1]),
            ("pure_EA", "velocity_Fisher"): (5, [4, 1]),
            ("pure_EA", "weight_self"): (7, [2, 3]),
            ("one_heat", "pressure"): (14, [4, 1]),
            ("one_heat", "velocity_Fisher"): (8, [3, 1]),
            ("one_heat", "weight_self"): (9, [1, 3]),
            ("two_heat", "pressure"): (8, [3, 1]),
            ("two_heat", "velocity_Fisher"): (4, [2, 1]),
            ("two_heat", "weight_self"): (5, [0, 3]),
        }
        for row in records:
            key = (row["block"], row["category"])
            count, degree = expected[key]
            self.assertEqual(row["row_count"], count)
            self.assertEqual(row["observed_degrees"], [degree])
            self.assertTrue(row["all_checks_pass"])

    def test_flow_chain_rule_has_disjoint_6_9_5_channels(self) -> None:
        certificate = _flow_block_identity_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        partition = certificate["channel_partition"]
        self.assertEqual(len(partition["pure_EA"]), 6)
        self.assertEqual(len(partition["one_heat"]), 9)
        self.assertEqual(len(partition["two_heat"]), 5)
        self.assertIn("X(Yg)+Y(Xg)", certificate[
            "one_heat_second_block"
        ])

    def test_borderline_pressure_branch_retains_two_differences(self) -> None:
        vertex = self.audit["one_heat_pressure_HHHH_certificate"]
        replay = _one_heat_pressure_certificate(
            {
                "minimum_compatible_difference_order": vertex[
                    "minimum_remaining_compatible_differences"
                ]
            }
        )
        self.assertTrue(replay["all_checks_pass"])
        self.assertEqual(
            vertex["minimum_remaining_compatible_differences"], 2
        )
        branches = {row["branch"]: row for row in vertex["branches"]}
        self.assertEqual(branches["HHHH"]["optimized_power"], 8)
        self.assertEqual(branches["HHLL"]["optimized_power"], 8)
        self.assertIn("remain fixed", vertex["outer_projector_guard"])

    def test_every_other_heat_group_is_strictly_sub_N9(self) -> None:
        certificate = _remaining_power_ledger()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertTrue(certificate["all_rows_strictly_below_N9"])
        self.assertEqual(certificate["maximum_optimized_power"], 8)
        self.assertEqual(
            sorted(row["optimized_power"] for row in certificate["rows"]),
            [3, 4, 7, 8, 8],
        )

    def test_complete_N9_is_closed_but_time_window_is_not(self) -> None:
        self.assertTrue(self.audit["all_positive_checks_pass"])
        asymptotic = self.audit["complete_second_jet_asymptotic"]
        self.assertTrue(asymptotic["certified"])
        self.assertIn("strict negative", asymptotic["conclusion"])
        flags = self.audit["certification_flags"]
        self.assertTrue(flags["all_69_second_subterms_partitioned"])
        self.assertTrue(
            flags["all_viscosity_bearing_second_rows_o_N9_proved"]
        )
        self.assertTrue(flags["complete_second_N9_limit_certified"])
        self.assertFalse(flags["uniform_second_jet_Taylor_remainder_proved"])
        self.assertFalse(flags["parabolic_window_turnaround_proved"])
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
