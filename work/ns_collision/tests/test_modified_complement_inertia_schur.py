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
    / "neutral_strip_modified_complement_inertia_schur_certificate.py"
)
RESULT = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_h006_modified_complement_inertia_schur_v1.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "modified_complement_certificate_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load modified-complement checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModifiedComplementInertiaSchurTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_abstract_inertia_schur_gate(self):
        hamiltonian = np.diag([1.0, 2.0, 5.0, 7.0])
        constraint = np.eye(4)[:, :2]
        shift = 4.0
        pencil = hamiltonian - shift * np.eye(4)
        schur = constraint.T @ np.linalg.solve(pencil, constraint)
        self.assertEqual(int(np.sum(np.linalg.eigvalsh(pencil) < 0.0)), 2)
        self.assertLess(float(np.linalg.eigvalsh(schur)[-1]), 0.0)
        self.assertTrue(
            self.module._inertia_schur_floor_certified(
                4,
                2,
                2,
                float(np.linalg.eigvalsh(schur)[-1]),
            )
        )
        complement = np.eye(4)[:, 2:]
        restricted = complement.T @ hamiltonian @ complement
        self.assertGreater(float(np.linalg.eigvalsh(restricted)[0]), shift)

    def test_production_certificate_when_present(self):
        if not RESULT.is_file():
            self.skipTest("production certificate has not been run")
        payload = json.loads(RESULT.read_text(encoding="ascii"))
        self.assertTrue(payload["all_modified_complement_checks_pass"])
        self.assertEqual(
            payload["lower_inertia_row"]["summary"][
                "negative_pivot_count"
            ],
            240,
        )
        self.assertEqual(
            payload["upper_inertia_row"]["summary"][
                "negative_pivot_count"
            ],
            240,
        )
        self.assertLess(
            payload["schur_complement"]["exact_schur_maximum_upper"],
            0.0,
        )
        self.assertEqual(
            payload["theorem"]["modified_complement_floor_lower"],
            102.7,
        )


if __name__ == "__main__":
    unittest.main()
