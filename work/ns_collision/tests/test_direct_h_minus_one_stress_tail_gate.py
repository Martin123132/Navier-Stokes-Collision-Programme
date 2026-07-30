from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from direct_h_minus_one_stress_tail_gate_audit import (
    DEFAULT_OUTPUT,
    _dyadic_overlap_audit,
    _endpoint_pulse_admissibility_audit,
    _finite_sequence_audit,
    _physical_space_product_certificate,
)


class DirectHMinusOneStressTailGateTests(unittest.TestCase):
    def test_exact_five_shell_overlap_constants(self) -> None:
        audit = _dyadic_overlap_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertEqual(audit["energy_overlap_constant"], 5)
        self.assertEqual(
            audit["first_moment_overlap_constant"]["exact"],
            "31/4",
        )
        self.assertEqual(
            audit["second_moment_overlap_constant"]["exact"],
            "341/16",
        )
        self.assertEqual(
            audit["high_tail_first_moment_constant"]["exact"],
            "31",
        )
        self.assertEqual(
            audit["integrated_squared_tail_constant"]["exact"],
            "155",
        )

    def test_finite_sequences_obey_every_inequality(self) -> None:
        audit = _finite_sequence_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertLessEqual(
            audit["maximum_energy_overlap_ratio"],
            1.0,
        )
        self.assertLessEqual(
            audit["maximum_first_moment_tail_ratio"],
            1.0,
        )
        self.assertLessEqual(
            audit["maximum_cauchy_ratio"],
            1.0 + 1.0e-14,
        )
        self.assertLessEqual(
            audit["maximum_final_tail_ratio"],
            1.0 + 1.0e-14,
        )

    def test_product_chain_uses_endpoint_sobolev_duality(self) -> None:
        certificate = _physical_space_product_certificate()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertIn("L^(6/5)", certificate["sobolev_duality"])
        self.assertIn("H^(1/2)", certificate["shell_Bernstein_step"])
        self.assertIn("155", certificate["leray_payment"])

    def test_old_pulse_exceeds_actual_stress_amplitude(self) -> None:
        audit = _endpoint_pulse_admissibility_audit()
        self.assertTrue(audit["all_checks_pass"])
        self.assertEqual(
            audit["exact_pulse_end_response"],
            "H^(1/2)(1-e^(-1))",
        )
        self.assertEqual(
            audit["first_audited_carrier_violating_unit_bound"],
            4,
        )
        self.assertGreater(
            audit["rows"][-1]["response_at_pulse_end"],
            1.0,
        )

    def test_stored_correction_scope(self) -> None:
        stored = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "direct_H_minus_one_high_high_stress_tail_certified",
        )
        flags = stored["certification_flags"]
        self.assertTrue(
            flags["actual_high_high_stress_H_minus_1_tail_vanishes"]
        )
        self.assertTrue(
            flags["H_minus_1_high_high_stress_Galerkin_passage_proved"]
        )
        self.assertTrue(
            flags["prior_scalar_envelope_no_go_remains_logically_valid"]
        )
        self.assertFalse(
            flags["prior_pulse_admissible_as_actual_Reynolds_stress"]
        )
        self.assertFalse(
            flags["complete_cubic_local_energy_passage_proved"]
        )
        self.assertFalse(
            flags["suitable_weak_solution_closure_proved"]
        )
        self.assertFalse(flags["Navier_Stokes_global_regularity_proved"])


if __name__ == "__main__":
    unittest.main()
