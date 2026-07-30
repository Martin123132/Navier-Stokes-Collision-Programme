"""Tests for the parallel-shear third-jet route guard."""

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
    "annular_parallel_shear_third_jet_route_guard_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_parallel_shear_third_jet_route_guard_audit import (  # noqa: E402
    _bounded_output_exception_families,
    _carrier_ledger,
    _stencil_route_certificate,
    _taylor_threshold_certificate,
    _third_flow_identity_certificate,
)


class ParallelShearThirdJetRouteGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_exact_four_block_third_flow_identity(self) -> None:
        certificate = _third_flow_identity_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(
            sorted(certificate["third_state_blocks"]),
            ["z30", "z31", "z32", "z33"],
        )
        self.assertEqual(
            certificate["binomial_multiplicities"], [1, 3, 3, 1]
        )
        self.assertIn(
            "3D2g[Y,z21]",
            certificate["third_scalar_blocks"]["heat_2"],
        )

    def test_carrier_ledger_has_only_six_dangerous_rows(self) -> None:
        ledger = _carrier_ledger()
        self.assertTrue(ledger["all_checks_pass"])
        self.assertEqual(ledger["row_count"], 28)
        self.assertEqual(ledger["automatic_row_count"], 22)
        self.assertEqual(ledger["dangerous_row_count"], 6)
        observed = {
            (
                row["sector"],
                row["heat_count"],
                row["high_leaf_count"],
                row["gains_required_for_O_N10"],
            )
            for row in ledger["dangerous_rows"]
        }
        self.assertEqual(
            observed,
            {
                ("pressure", 0, 4, 2),
                ("pressure", 0, 6, 4),
                ("pressure", 1, 4, 2),
                ("pressure", 2, 4, 2),
                ("velocity_Fisher", 0, 4, 2),
                ("velocity_Fisher", 1, 4, 2),
            },
        )

    def test_bounded_output_inventory_is_exact_and_sub_N12(self) -> None:
        exceptions = _bounded_output_exception_families()
        self.assertTrue(exceptions["all_checks_pass"])
        self.assertEqual(exceptions["family_count"], 13)
        self.assertEqual(
            exceptions["family_count_by_heat"], {"0": 6, "1": 7}
        )
        self.assertEqual(exceptions["N11_capable_family_count"], 5)
        self.assertEqual(
            exceptions["maximum_optimized_power_upper_bound"], 11
        )
        self.assertNotIn("2", exceptions["family_count_by_heat"])

    def test_route_guard_keeps_internal_shell_obligation_open(self) -> None:
        ledger = _carrier_ledger()
        exceptions = _bounded_output_exception_families()
        route = _stencil_route_certificate(ledger, exceptions)
        self.assertTrue(route["all_checks_pass"])
        self.assertEqual(route["parity_gauged_vertex_difference_budget"], 6)
        self.assertEqual(route["four_high_requirement"], 2)
        self.assertEqual(route["six_high_requirement"], 4)
        self.assertFalse(route["restart_time_O_N11_certified"])
        self.assertIn("still required", route["internal_shell_obligation"])

    def test_uniform_big_O_N11_is_the_sufficient_target(self) -> None:
        certificate = _taylor_threshold_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(
            certificate["sufficient_window_condition"],
            "0<T<=c2/(2 C3)",
        )
        self.assertIn(
            "O(N^11)",
            certificate["correction_to_previous_target"],
        )
        self.assertIn(
            "evolving coupled",
            certificate["restart_time_bound_is_not_enough"],
        )

    def test_finite_multilinear_replay_and_padding_pass(self) -> None:
        self.assertTrue(self.audit["all_route_guard_checks_pass"])
        finite = self.audit["finite_spectral_replay"]
        self.assertLess(
            max(
                finite[
                    "third_velocity_relative_divergence_residuals"
                ].values()
            ),
            1.0e-9,
        )
        self.assertLess(
            finite["finite_difference_replay"]["relative_residual"],
            2.0e-5,
        )
        self.assertLess(
            self.audit["padding_replay"]["relative_total_residual"],
            1.0e-9,
        )
        flags = self.audit["certification_flags"]
        self.assertTrue(flags["all_28_carrier_rows_partitioned"])
        self.assertFalse(flags["complete_restart_time_third_O_N11_proved"])
        self.assertFalse(
            flags["uniform_parabolic_window_third_O_N11_proved"]
        )
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
