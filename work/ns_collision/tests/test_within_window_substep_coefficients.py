import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_h006_within_window_substep_coefficients_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WithinWindowSubstepCoefficientsTest(unittest.TestCase):
    def test_production_certificate(self) -> None:
        result = json.loads(RESULT.read_text(encoding="ascii"))
        premise = result["premise_artifacts"]
        scaling_path = ROOT / premise["production_scaling_result"]

        self.assertEqual(
            _sha256(scaling_path),
            premise["production_scaling_result_sha256"],
        )
        self.assertTrue(result["all_substep_coefficient_checks_pass"])
        self.assertTrue(all(result["checks"]))
        coefficients = result["coefficient_intervals"]
        self.assertEqual(coefficients["time"], 0.0375)
        self.assertEqual(coefficients["degree"], 112)
        self.assertEqual(coefficients["substeps_per_3_over_8_window"], 10)
        self.assertEqual(coefficients["finite_window_count"], 15)
        self.assertEqual(coefficients["total_substeps_through_time_6"], 160)
        self.assertEqual(len(coefficients["rows"]), 113)
        self.assertTrue(
            coefficients["all_exact_coefficient_intervals_ordered"]
        )
        self.assertTrue(
            coefficients["scipy_implementation_discrepancy_certified"]
        )
        self.assertTrue(
            coefficients[
                "degree_112_exact_coefficients_and_infinite_tail_certified"
            ]
        )
        self.assertLess(
            coefficients["scipy_coefficient_l1_error_upper"], 1.0e-14
        )
        self.assertLess(coefficients["tail"]["upper"], 3.0e-19)
        self.assertFalse(result["within_window_suprema_certified"])
        self.assertFalse(result["screen_updated"])


if __name__ == "__main__":
    unittest.main()
