from __future__ import annotations

import json
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_h006_chebyshev_recurrence_roundoff_v1.json"
)


class ChebyshevRecurrenceRoundoffTest(unittest.TestCase):
    def test_production_certificate(self):
        payload = json.loads(RESULT.read_text(encoding="ascii"))
        self.assertTrue(payload["all_recurrence_roundoff_checks_pass"])
        self.assertTrue(
            payload[
                "one_step_full_state_chebyshev_action_roundoff_certified"
            ]
        )
        self.assertTrue(
            payload["operator"][
                "computational_scaled_spectrum_inside_unit_interval_certified"
            ]
        )
        self.assertFalse(
            payload["stability"][
                "raw_exponential_norm_recurrence_used"
            ]
        )
        self.assertLess(
            payload["maximum_error_components"][
                "total_one_step_state_action_error_upper"
            ],
            1.0e-7,
        )
        cache = payload["computed_state_cache"]
        cache_path = ROOT / Path(cache["path"])
        self.assertEqual(
            hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            cache["sha256"],
        )
        self.assertFalse(payload["boundary_output_roundoff_enclosed"])
        self.assertFalse(payload["screen_updated"])


if __name__ == "__main__":
    unittest.main()
