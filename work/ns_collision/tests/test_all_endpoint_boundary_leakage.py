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
    / "neutral_strip_h006_all_endpoint_boundary_leakage_v1.json"
)


class AllEndpointBoundaryLeakageTest(unittest.TestCase):
    def test_production_certificate(self):
        payload = json.loads(RESULT.read_text(encoding="ascii"))
        self.assertTrue(payload["all_endpoint_boundary_checks_pass"])
        self.assertTrue(payload["all_sixteen_boundary_endpoints_certified"])
        self.assertEqual(len(payload["endpoint_rows"]), 16)
        self.assertTrue(payload["endpoint_uppers_strictly_decrease"])
        self.assertLess(
            payload["finite_endpoint_only_screen_charge_upper"], 4.0e-4
        )
        self.assertLess(
            payload["endpoint_only_combined_screen_diagnostic_upper"],
            0.971,
        )
        self.assertFalse(payload["within_window_suprema_certified"])
        self.assertFalse(payload["screen_updated"])


if __name__ == "__main__":
    unittest.main()
