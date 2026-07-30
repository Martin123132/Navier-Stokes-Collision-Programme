"""Tests for the restart-time third internal-shell lemma."""

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
    "annular_parallel_shear_third_internal_shell_lemma_audit_v1.json"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from annular_parallel_shear_third_internal_shell_lemma_audit import (  # noqa: E402
    _difference_allocation_certificate,
    _euler_shell_certificate,
    _explicit_constant_certificate,
    _functional_atoms,
    _optimizer_bound_certificate,
    _packet_difference_certificate,
    _power_closure_certificate,
    _state_tree_expansions,
    _topology_ledger,
    _tree_expansion_certificate,
)
from annular_parallel_shear_third_jet_route_guard_audit import (  # noqa: E402
    _bounded_output_exception_families,
    _carrier_ledger,
)


class ParallelShearThirdInternalShellLemmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expansions = _state_tree_expansions()
        cls.atoms = _functional_atoms(cls.expansions)
        cls.tree = _tree_expansion_certificate(
            cls.expansions, cls.atoms
        )
        cls.topology = _topology_ledger(cls.atoms)
        cls.allocation = _difference_allocation_certificate(
            cls.topology
        )
        cls.audit = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_exact_third_tree_mass_is_1412(self) -> None:
        self.assertTrue(self.tree["all_checks_pass"])
        self.assertEqual(
            self.tree["total_functional_absolute_coefficient_mass"],
            1412,
        )
        pressure = [
            row
            for row in self.tree["functional_rows"]
            if row["sector"] == "pressure"
        ]
        self.assertEqual(
            [row["atom_count"] for row in pressure],
            [20, 49, 44, 13],
        )
        self.assertEqual(
            [row["absolute_coefficient_mass"] for row in pressure],
            [120, 300, 244, 64],
        )

    def test_dangerous_topology_inventory_is_exhaustive(self) -> None:
        topology = self.topology
        self.assertTrue(topology["all_checks_pass"])
        self.assertEqual(
            topology["protected_pressure_assignment_count"], 579
        )
        self.assertEqual(
            topology["expanded_pressure_exception_count"], 30
        )
        self.assertEqual(
            topology["protected_velocity_Fisher_assignment_count"], 81
        )
        self.assertEqual(topology["maximum_protected_B_path_depth"], 3)
        self.assertEqual(topology["six_high_maximum_B_path_depth"], 2)
        self.assertEqual(
            topology[
                "protected_four_high_rows_with_fixed_bounded_B_node"
            ],
            4,
        )
        self.assertEqual(
            topology[
                "protected_four_high_maximum_fixed_bounded_B_nodes"
            ],
            1,
        )
        self.assertEqual(
            topology["six_high_fixed_bounded_B_node_mass"], 0
        )
        self.assertEqual(
            topology["post_resonance_topology_failures"], []
        )

    def test_packet_certificate_uses_boundary_safe_l1_bounds(self) -> None:
        packet = _packet_difference_certificate()
        self.assertTrue(packet["all_checks_pass"])
        rows = packet["tensor_multiindex_rows"]
        self.assertEqual(len(rows), 27)
        self.assertEqual(
            max(row["difference_order"] for row in rows), 6
        )
        self.assertTrue(
            all(
                row["L1_carrier_power"]
                == 2 - row["difference_order"]
                for row in rows
            )
        )
        self.assertIn("zero-extended", packet["conclusion"])

    def test_shell_gain_and_difference_allocation_are_separated(self) -> None:
        shell = _euler_shell_certificate()
        self.assertTrue(shell["all_checks_pass"])
        self.assertEqual(
            [
                row["minimum_gain_over_0_le_kappa_le_1"]
                for row in shell["rows"]
            ],
            [0, 1, 2, 3, 4, 4, 4],
        )
        allocation = self.allocation
        self.assertTrue(allocation["all_checks_pass"])
        self.assertEqual(
            [
                row["minimum_carrier_gain"]
                for row in allocation[
                    "strictly_nested_free_shell_depth_rows"
                ]
            ],
            [6, 4, 4, 4],
        )
        self.assertEqual(
            allocation["protected_four_high_minimum_gain"], 1
        )
        self.assertEqual(
            allocation["all_high_pressure_minimum_gain"], 4
        )
        self.assertFalse(
            allocation["protected_four_high_single_factor_route"][
                "dyadic_log_at_minimum"
            ]
        )

    def test_all_power_routes_close_at_restart_N11(self) -> None:
        closure = _power_closure_certificate(
            _carrier_ledger(),
            _bounded_output_exception_families(),
            self.allocation,
            self.topology,
        )
        self.assertTrue(closure["all_checks_pass"])
        self.assertEqual(closure["maximum_final_power"], 11)
        self.assertEqual(
            [
                row["final_power_upper_bound"]
                for row in closure["rows"]
            ],
            [10, 11, 11, 11],
        )
        self.assertEqual(
            closure["complete_restart_time_third_bound"],
            "g_N'''(0)=O_nu(N^11)",
        )

    def test_optimizer_and_explicit_constant_are_finite(self) -> None:
        optimizer = _optimizer_bound_certificate()
        self.assertTrue(optimizer["all_checks_pass"])
        self.assertEqual(optimizer["raw_amplitude_constant"], 57)
        self.assertEqual(optimizer["raw_weight_constant"], 183)
        explicit = _explicit_constant_certificate(self.tree)
        self.assertTrue(explicit["all_checks_pass"])
        self.assertEqual(explicit["C0_decimal_digits"], 422)
        self.assertEqual(
            explicit["C0_leading_16_digits"], "2746219328370245"
        )
        self.assertTrue(
            explicit["finite_shell_count_charged_per_factor"]
        )
        self.assertIn("N^11", explicit["explicit_bound"])

    def test_production_record_closes_restart_only(self) -> None:
        self.assertEqual(self.audit["status"], "passed")
        self.assertTrue(self.audit["all_positive_checks_pass"])
        flags = self.audit["certification_flags"]
        self.assertTrue(flags["all_dangerous_topologies_enumerated"])
        self.assertTrue(
            flags[
                "six_high_post_resonance_nested_shell_rank_certified"
            ]
        )
        self.assertTrue(
            flags["protected_four_high_O_N11_bound_certified"]
        )
        self.assertTrue(flags["complete_restart_time_third_O_N11_proved"])
        self.assertFalse(
            flags["uniform_parabolic_window_third_O_N11_proved"]
        )
        self.assertFalse(flags["finite_time_blowup_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
