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
    / "neutral_strip_h006_first_endpoint_boundary_leakage_v1.json"
)


class FirstEndpointBoundaryLeakageTest(unittest.TestCase):
    def test_production_certificate(self):
        payload = json.loads(RESULT.read_text(encoding="ascii"))
        self.assertTrue(payload["all_first_endpoint_boundary_checks_pass"])
        self.assertTrue(
            payload[
                "first_endpoint_stored_chain_boundary_leakage_certified"
            ]
        )
        endpoint = payload["endpoint"]
        self.assertLess(
            endpoint["maximum_boundary_l2_difference_upper"], 7.0e-4
        )
        self.assertGreater(
            endpoint["maximum_boundary_l2_difference_upper"], 6.0e-4
        )
        self.assertFalse(payload["within_window_supremum_certified"])
        self.assertFalse(payload["screen_updated"])


if __name__ == "__main__":
    unittest.main()
