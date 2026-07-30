import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "work"
    / "ns_collision"
    / "scripts"
    / "neutral_strip_modified_two_block_leakage_certificate.py"
)
RESULT = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_h006_modified_two_block_leakage_v1.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "modified_two_block_leakage_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load modified leakage checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModifiedTwoBlockLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_dense_eigenvalue_enclosure(self):
        matrix = np.diag([1.25, 2.5, 4.0])
        bounds = self.module._symmetric_eigen_bounds(matrix, 1.0e-12)
        self.assertLessEqual(bounds["minimum_lower"], 1.25)
        self.assertGreaterEqual(bounds["maximum_upper"], 4.0)
        self.assertGreater(bounds["minimum_lower"], 1.24)

    def test_svd_enclosure(self):
        matrix = np.diag([3.0, 2.0, 1.0])
        bounds = self.module._svd_spectral_upper(matrix, 1.0e-12)
        self.assertGreaterEqual(bounds["spectral_norm_upper"], 3.0)
        self.assertLess(bounds["spectral_norm_upper"], 3.000001)

    def test_production_certificate_when_present(self):
        if not RESULT.is_file():
            self.skipTest("production certificate has not been run")
        payload = json.loads(RESULT.read_text(encoding="ascii"))
        self.assertTrue(
            payload["all_modified_two_block_leakage_checks_pass"]
        )
        parameters = payload["certified_parameters"]
        self.assertTrue(parameters["all_two_block_parameters_certified"])
        self.assertEqual(parameters["high_block_floor"], 102.7)
        self.assertGreaterEqual(parameters["low_block_floor"], 2.36)
        self.assertLess(parameters["off_block_coupling_upper"], 6.5)
        self.assertFalse(payload["boundary_output_smoothing_composed"])


if __name__ == "__main__":
    unittest.main()
