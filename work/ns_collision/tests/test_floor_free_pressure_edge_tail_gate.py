from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from floor_free_pressure_edge_tail_gate_audit import (
    DEFAULT_OUTPUT,
    DIRECT_RESULT_SHA256,
    TAIL_CONSTANT,
    _fourier_duality_audit,
    _prerequisite_audit,
    _scale_family_audit,
    _weight_product_certificate,
)


class FloorFreePressureEdgeTailGateTests(unittest.TestCase):
    def test_direct_endpoint_prerequisite_is_pinned(self) -> None:
        prerequisite = _prerequisite_audit()
        self.assertTrue(prerequisite["all_checks_pass"])
        self.assertEqual(
            prerequisite["actual_sha256"],
            DIRECT_RESULT_SHA256,
        )
        self.assertEqual(prerequisite["tail_constant"], TAIL_CONSTANT)

    def test_discrete_sobolev_duality(self) -> None:
        duality = _fourier_duality_audit()
        self.assertTrue(duality["all_checks_pass"])
        self.assertLessEqual(duality["cauchy_ratio"], 1.0)

    def test_product_chain_is_floor_free(self) -> None:
        product = _weight_product_certificate()
        self.assertTrue(product["all_checks_pass"])
        self.assertIn("No lower bound", product["floor_free"])
        self.assertIn("grad^2 lambda", product["product_rule"])

    def test_scale_adapted_diagonal_tail_decays(self) -> None:
        scaling = _scale_family_audit()
        self.assertTrue(scaling["all_checks_pass"])
        self.assertTrue(scaling["monotone_decay"])
        self.assertEqual(scaling["maximum_identity_residual"], 0.0)
        self.assertEqual(
            scaling["rows"][-1]["far_carrier_cutoff_K"],
            128**5,
        )

    def test_stored_scope_keeps_near_carrier_gate_open(self) -> None:
        stored = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "floor_free_far_carrier_pressure_edge_tail_certified",
        )
        theorem = stored["theorem"]
        self.assertTrue(theorem["uniform_in_terminal_time"])
        self.assertFalse(theorem["requires_positive_weight_floor"])
        flags = stored["certification_flags"]
        self.assertTrue(
            flags["floor_free_far_carrier_pressure_edge_tail_vanishes"]
        )
        self.assertFalse(flags["near_carrier_signed_pressure_edge_absorbed"])
        self.assertFalse(flags["terminal_dual_supremum_controlled"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
