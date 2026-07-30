from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dense_annular_hhh_packet_gate_audit import (
    _central_witness_audit,
    _dense_packet_row,
    _parabolic_no_go,
    _sharp_scaling_theorem,
    _walsh_coupling_audit,
)


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "dense_annular_hhh_packet_gate_audit_v1.json"
)


class DenseAnnularHHHPacketGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stored = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_exact_rational_center_witness(self) -> None:
        center = _central_witness_audit()
        self.assertTrue(center["all_checks_pass"])
        self.assertEqual(center["exact_Frobenius_norm"], "12*sqrt(43)")
        self.assertLess(center["exact_matrix_residual"], 1.0e-13)
        self.assertLess(
            center["phase_rotated_real_matrix_residual"],
            1.0e-13,
        )
        self.assertGreater(center["central_channel_pairing"], 70.0)

    def test_lightweight_dense_lattice_replay(self) -> None:
        row = _dense_packet_row(1, 32)
        stored = self.stored["dense_annular_packet"]["rows"][0]
        self.assertTrue(row["all_checks_pass"])
        self.assertEqual(row, stored)
        self.assertEqual(row["real_field_mode_count"], 162)
        self.assertEqual(row["exact_coherent_triad_count"], 343)
        self.assertAlmostEqual(row["normalized_energy"], 1.0)
        self.assertGreater(
            row["minimum_unit_triad_channel_over_carrier"],
            60.0,
        )

    def test_sharp_five_halves_scaling_theorem(self) -> None:
        theorem = _sharp_scaling_theorem()
        flags = self.stored["certification_flags"]
        self.assertTrue(theorem["all_checks_pass"])
        self.assertEqual(theorem["sharp_forcing_exponent"], "5/2")
        self.assertEqual(
            theorem["parabolic_forcing_L2_cost_exponent"],
            "3",
        )
        self.assertEqual(
            theorem["parabolic_enstrophy_cost_exponent"],
            "0",
        )
        self.assertTrue(
            flags["sharp_H_five_halves_tensor_forcing_growth_proved"]
        )

    def test_fixed_top_walsh_coupling(self) -> None:
        walsh = _walsh_coupling_audit()
        self.assertTrue(walsh["all_checks_pass"])
        self.assertEqual(walsh["Walsh_character"], "chi_123(v)=v_1 v_2 v_3")
        self.assertEqual(walsh["exact_pairing_magnitude"], "1/sqrt(86)")
        self.assertLess(
            walsh["pairing_magnitude_residual"],
            1.0e-13,
        )

    def test_parabolic_no_go_scope(self) -> None:
        theorem = _sharp_scaling_theorem()
        no_go = _parabolic_no_go(theorem)
        flags = self.stored["certification_flags"]
        self.assertTrue(no_go["all_checks_pass"])
        self.assertTrue(
            flags[
                "raw_tensor_forcing_bound_from_Leray_inputs_alone_falsified"
            ]
        )
        self.assertFalse(
            flags["unforced_Navier_Stokes_dynamic_counterexample_proved"]
        )
        self.assertFalse(flags["trace_local_energy_channel_obstructed"])
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])

    def test_stored_result_scope_and_dense_rows(self) -> None:
        stored = self.stored
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            (
                "sharp_dense_HHH_Bernstein_loss_certified_"
                "Leray_input_only_forcing_bound_falsified"
            ),
        )
        rows = stored["dense_annular_packet"]["rows"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1]["real_field_mode_count"], 4374)
        self.assertEqual(
            rows[-1]["exact_coherent_triad_count"],
            226981,
        )
        self.assertGreater(
            rows[-1]["channel_over_full_tensor_norm"],
            0.99999,
        )
        self.assertTrue(
            stored["certification_flags"][
                "dense_packet_divergence_free_annular_unit_energy_proved"
            ]
        )
        self.assertFalse(
            stored["certification_flags"][
                "complete_signed_flux_occupation_bound_proved"
            ]
        )


if __name__ == "__main__":
    unittest.main()
