from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "work/ns_collision/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dense_spaced_continuum_positivity_audit import (
    DEFAULT_CARRIER_MULTIPLE,
    Interval,
    _central_symbol_self_audit,
    _certificate,
    _complete_tau_derivative_interval,
    _leading_stress_interval,
    _rounding_self_audit,
    _tau_zero_identity_self_audit,
)


RESULT = (
    ROOT
    / "work/ns_collision/results/"
    "dense_spaced_continuum_positivity_audit_v1.json"
)


class DenseSpacedContinuumPositivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stored = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_directed_interval_elementary_containment(self) -> None:
        audit = _rounding_self_audit()
        self.assertTrue(audit["all_checks_pass"])
        quotient = Interval(2.0, 3.0) / Interval(4.0, 5.0)
        self.assertLessEqual(quotient.lower, 0.4)
        self.assertGreaterEqual(quotient.upper, 0.75)

    def test_relaxed_continuum_leading_interval_is_positive(self) -> None:
        leading = _leading_stress_interval(DEFAULT_CARRIER_MULTIPLE)
        self.assertTrue(leading["all_checks_pass"])
        self.assertGreater(
            leading["scaled_coefficient_interval"][0],
            0.14,
        )
        self.assertIn("contains the true polytope", leading["relaxed_domain"])
        center = _central_symbol_self_audit(DEFAULT_CARRIER_MULTIPLE)
        self.assertTrue(center["all_checks_pass"])
        self.assertEqual(center["exact_scaled_center_coefficient"], "3/16")

    def test_complete_low_frequency_correction_is_paid(self) -> None:
        tau_zero = _tau_zero_identity_self_audit(
            DEFAULT_CARRIER_MULTIPLE
        )
        self.assertTrue(tau_zero["all_checks_pass"])
        self.assertEqual(tau_zero["normalized_high_wave_sum"], [0, 0, 0])
        self.assertEqual(
            tau_zero["exact_affine_sum_coefficients"],
            [["0"] * 7 for _ in range(3)],
        )
        self.assertEqual(
            tau_zero[
                "partition_dot_unnormalized_low_polarization"
            ],
            0,
        )
        correction = _complete_tau_derivative_interval(
            DEFAULT_CARRIER_MULTIPLE
        )
        self.assertTrue(correction["all_checks_pass"])
        self.assertLess(
            correction["mean_value_correction_upper"],
            5.0e-5,
        )

    def test_complete_continuum_coefficient_has_strict_margin(self) -> None:
        certificate = _certificate(DEFAULT_CARRIER_MULTIPLE)
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(certificate["physical_carrier"], "R=16384M")
        self.assertGreater(
            certificate[
                "complete_actual_positive_quartet_coefficient_lower"
            ],
            0.10,
        )

    def test_stored_certificate_scope(self) -> None:
        stored = self.stored
        self.assertTrue(stored["all_positive_checks_pass"])
        self.assertEqual(
            stored["status"],
            "complete_continuum_positive_quartic_coefficient_certified",
        )
        certificate = stored["certificate"]
        self.assertEqual(
            certificate["carrier_multiple_relative_to_offset_box"],
            DEFAULT_CARRIER_MULTIPLE,
        )
        self.assertGreater(
            certificate[
                "complete_actual_positive_quartet_coefficient_lower"
            ],
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
