from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_h006_chebyshev_scaling_coefficients_v1.json"
)


class ChebyshevScalingCoefficientsTest(unittest.TestCase):
    def test_production_certificate(self):
        payload = json.loads(RESULT.read_text(encoding="ascii"))
        self.assertTrue(payload["all_scaling_coefficient_checks_pass"])
        self.assertTrue(
            payload["matrix_scaling"][
                "exact_spectrum_inside_scaling_interval_certified"
            ]
        )
        coefficients = payload["coefficient_intervals"]
        self.assertTrue(
            coefficients[
                "degree_320_exact_coefficients_and_infinite_tail_certified"
            ]
        )
        self.assertLess(
            coefficients["scipy_coefficient_l1_error_upper"],
            7.1e-16,
        )
        self.assertEqual(len(coefficients["rows"]), 321)
        self.assertLess(coefficients["tail"]["upper"], 7.1e-17)
        self.assertFalse(payload["sparse_recurrence_roundoff_enclosed"])
        self.assertFalse(payload["screen_updated"])


if __name__ == "__main__":
    unittest.main()
