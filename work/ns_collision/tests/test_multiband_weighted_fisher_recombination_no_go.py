"""Focused tests for the multiband weighted-Fisher no-go."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from multiband_weighted_fisher_recombination_no_go_audit import (  # noqa: E402
    audit,
)


class MultibandWeightedFisherRecombinationNoGoTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_exact_chain_ratio_diverges(self) -> None:
        rows = self.result["exact_chain_rows"]
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertTrue(row["all_checks_pass"])
            self.assertEqual(
                row["component_to_physical_ratio"],
                float(row["dyadic_block_count_J"]),
            )

    def test_components_are_divergence_free_dyadic_annuli(self) -> None:
        rows = self.result["dyadic_annular_support"]
        self.assertTrue(
            all(row["annular_ratio_below_two"] for row in rows)
        )
        self.assertTrue(all(row["pressure"] == "0" for row in rows))

    def test_neighbor_edges_carry_the_missing_cancellation(self) -> None:
        counterexample = self.result["counterexample"]
        self.assertEqual(
            counterexample["nearest_neighbor_correction_per_interface"],
            "-1/4",
        )
        flags = self.result["certification_flags"]
        self.assertFalse(
            flags["finite_overlap_degree_implies_coercivity"]
        )
        self.assertTrue(
            flags["signed_neighboring_Fisher_edges_must_be_retained"]
        )

    def test_floor_weight_fisher_and_coscaling_do_not_rescue(self) -> None:
        self.assertTrue(
            self.result["coscaling_stress"]["ratio_invariant"]
        )
        self.assertEqual(
            self.result["terminal_weight_Fisher_no_rescue"]["base_value"],
            "integral lambda|grad lambda|^2=1/16",
        )
        self.assertIn(
            "approaches zero",
            self.result["strict_floor_limit"]["conclusion"],
        )

    def test_adversaries_and_scope_remain_fail_closed(self) -> None:
        fields = self.result["mandatory_finite_fields"]
        self.assertTrue(fields["Taylor_Green"]["all_checks_pass"])
        self.assertTrue(fields["seed81"]["all_checks_pass"])
        flags = self.result["certification_flags"]
        self.assertFalse(
            flags["balanced_single_band_pressure_theorem_invalidated"]
        )
        self.assertFalse(
            flags["joint_signed_pressure_Fisher_block_bound_proved"]
        )
        self.assertFalse(
            flags["Navier_Stokes_global_regularity_proved"]
        )


if __name__ == "__main__":
    unittest.main()
