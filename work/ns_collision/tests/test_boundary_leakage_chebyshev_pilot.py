from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "work"
    / "ns_collision"
    / "scripts"
    / "neutral_strip_boundary_leakage_chebyshev_pilot.py"
)
RESULT = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_h006_boundary_leakage_chebyshev_pilot_v1.json"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "boundary_leakage_chebyshev_pilot_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BoundaryLeakageChebyshevPilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_script()

    def test_two_block_floor_supports_scaling_lower(self):
        floor = self.module._global_two_block_floor(
            2.36, 102.7, 6.44525405835444
        )
        self.assertGreater(floor, 1.9)
        self.assertLess(floor, 2.0)

    def test_scalar_chebyshev_action(self):
        lower = 1.9
        upper = 8.0
        time_value = 0.375
        coefficients, _ = self.module._chebyshev_coefficients(
            time_value, 80, lower, upper
        )
        eigenvalue = 4.25
        scaled_value = (
            2.0 * eigenvalue - (upper + lower)
        ) / (upper - lower)
        result, _ = self.module._chebyshev_step(
            csr_matrix([[scaled_value]]),
            np.asarray([[1.0]]),
            coefficients,
        )
        self.assertAlmostEqual(
            float(result[0, 0]),
            float(np.exp(-time_value * eigenvalue)),
            places=14,
        )

    def test_production_result_is_fail_closed(self):
        payload = json.loads(RESULT.read_text(encoding="ascii"))
        self.assertEqual(
            payload["kind"],
            "neutral_strip_boundary_leakage_chebyshev_pilot",
        )
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["completed_step_count"], 16)
        self.assertTrue(payload["all_pilot_integrity_checks_pass"])
        self.assertFalse(payload["boundary_leakage_certificate"])
        self.assertFalse(payload["screen_updated"])
        self.assertFalse(
            payload["chebyshev_diagnostics"][
                "sparse_recurrence_roundoff_enclosed"
            ]
        )
        self.assertEqual(len(payload["endpoint_rows"]), 16)
        first = payload["endpoint_rows"][0]
        self.assertGreater(
            first["maximum_boundary_l2_difference"], 6.0e-4
        )
        self.assertLess(
            first["maximum_boundary_l2_difference"], 6.5e-4
        )


if __name__ == "__main__":
    unittest.main()
