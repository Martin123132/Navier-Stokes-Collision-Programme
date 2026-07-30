from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from intrinsic_pressure_tail_gate_audit import audit


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "intrinsic_pressure_tail_gate_audit_v1.json"
)


class IntrinsicPressureTailGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_dyadic_tail_decomposition_and_bound(self) -> None:
        tail = self.result["dyadic_tail_decomposition"]
        self.assertTrue(tail["all_checks_pass"])
        self.assertIn("low-low term is exactly absent", tail[
            "sharp_support_identity"
        ])
        self.assertEqual(
            tail["conditional_absorption_threshold"],
            "m>=||u||_infinity^2/(nu lambda_*)",
        )
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags["dyadic_high_pressure_tail_identity_derived"]
        )
        self.assertTrue(
            flags["unweighted_L2_pressure_tail_bound_derived"]
        )
        self.assertTrue(
            flags["positive_floor_intrinsic_absorption_derived"]
        )

    def test_taylor_green_tail_scaling(self) -> None:
        scaling = self.result[
            "taylor_green_amplitude_frequency_gate"
        ]
        self.assertTrue(scaling["all_checks_pass"])
        self.assertEqual(scaling["divergence_residual"], "0")
        self.assertEqual(
            scaling["Euler_pressure_balance_residual"],
            ["0", "0"],
        )
        self.assertEqual(
            scaling["normalized_pressure_tail_flux"],
            "beta/32",
        )
        self.assertFalse(
            scaling["fixed_frequency_universal_absorption_possible"]
        )
        self.assertTrue(
            scaling["intrinsic_frequency_keeps_ratio_scale_invariant"]
        )

    def test_zero_face_weight_obstruction(self) -> None:
        gate = self.result["zero_face_weight_gate"]
        self.assertTrue(gate["all_checks_pass"])
        self.assertEqual(
            gate["weight_family"],
            "w_epsilon(x)=epsilon+sin(x/2)^2",
        )
        self.assertGreater(
            gate["rows"][-1][
                "explicit_Hilbert_norm_lower_bound"
            ],
            70.0,
        )
        self.assertFalse(
            self.result["certification_flags"][
                "uniform_arbitrary_weight_CZ_localization_available"
            ]
        )

    def test_scope_and_nonpromotion(self) -> None:
        flags = self.result["certification_flags"]
        self.assertTrue(
            flags[
                "fixed_frequency_tail_absorption_falsified_by_scaling"
            ]
        )
        self.assertFalse(
            flags["zero_face_full_terminal_supremum_preserved"]
        )
        self.assertFalse(
            flags["floor_free_signed_pressure_edge_bound_proved"]
        )
        self.assertFalse(
            flags["intrinsic_scale_pressure_tail_bound_proved"]
        )
        self.assertFalse(flags["critical_signed_bound_proved"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result(self) -> None:
        stored = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(stored["kind"], self.result["kind"])
        self.assertEqual(stored["status"], self.result["status"])
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertIn(
            "Carleson",
            stored["next_theorem_target"],
        )


if __name__ == "__main__":
    unittest.main()
