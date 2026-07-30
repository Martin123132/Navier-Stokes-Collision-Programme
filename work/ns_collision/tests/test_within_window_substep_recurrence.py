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
    / "neutral_strip_h006_within_window_substep_recurrence_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WithinWindowSubstepRecurrenceTest(unittest.TestCase):
    def test_production_certificate(self) -> None:
        result = json.loads(RESULT.read_text(encoding="ascii"))
        premises = result["premise_artifacts"]
        for path_key, hash_key in (
            (
                "substep_coefficient_result",
                "substep_coefficient_result_sha256",
            ),
            (
                "production_recurrence_result",
                "production_recurrence_result_sha256",
            ),
        ):
            self.assertEqual(
                _sha256(ROOT / premises[path_key]),
                premises[hash_key],
            )

        self.assertTrue(result["all_substep_recurrence_checks_pass"])
        self.assertTrue(all(result["checks"]))
        operator = result["operator"]
        self.assertEqual(operator["substep"], 0.0375)
        self.assertEqual(operator["degree"], 112)
        self.assertEqual(operator["total_substeps_through_time_6"], 160)
        self.assertLess(
            operator["total_one_substep_operator_error_upper"], 3.0e-12
        )
        self.assertLess(
            operator["computational_substep_operator_norm_upper"], 0.932
        )
        repeated = result["repeated_propagation"]
        self.assertLess(repeated["maximum_operator_error_upper"], 2.0e-11)
        self.assertLess(repeated["terminal_operator_error_upper"], 1.0e-12)
        self.assertTrue(result["sparse_recurrence_roundoff_enclosed"])
        self.assertFalse(result["within_window_grid_propagated"])
        self.assertFalse(result["screen_updated"])


if __name__ == "__main__":
    unittest.main()
