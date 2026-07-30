import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULT = (
    ROOT
    / "work"
    / "ns_collision"
    / "results"
    / "neutral_strip_h006_within_window_source_grid_propagation_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WithinWindowSourceGridPropagationTest(unittest.TestCase):
    def test_production_certificate(self) -> None:
        result = json.loads(RESULT.read_text(encoding="ascii"))
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["all_propagation_integrity_checks_pass"])
        self.assertTrue(all(result["checks"]))
        self.assertTrue(result["actual_source_grid_points_certified"])
        self.assertFalse(result["within_window_suprema_certified"])
        self.assertFalse(result["screen_updated"])
        self.assertEqual(result["completed_blocks"], 16)
        self.assertEqual(result["completed_substeps"], 160)
        self.assertEqual(result["recorded_grid_point_count"], 151)

        checkpoint = result["checkpoint"]
        checkpoint_path = ROOT / checkpoint["npz_path"]
        metadata_path = ROOT / checkpoint["metadata_path"]
        self.assertEqual(_sha256(checkpoint_path), checkpoint["npz_sha256"])
        self.assertEqual(_sha256(metadata_path), checkpoint["metadata_sha256"])
        with np.load(checkpoint_path, allow_pickle=False) as cached:
            self.assertEqual(int(cached["completed_substeps"].item()), 160)
            self.assertEqual(cached["full_state"].shape, (15211, 112))
            self.assertEqual(
                cached["reduced_coordinates"].shape, (240, 112)
            )

        for key, value in result["premise_artifacts"].items():
            if key.endswith("_sha256") and key not in {
                "mass_sha256",
                "stiffness_sha256",
                "retained_vectors_sha256",
            }:
                path_key = key.removesuffix("_sha256")
                if path_key in result["premise_artifacts"]:
                    self.assertEqual(
                        _sha256(ROOT / result["premise_artifacts"][path_key]),
                        value,
                    )

        rows = result["grid_rows"]
        self.assertEqual(rows[0]["substep"], 10)
        self.assertEqual(rows[-1]["substep"], 160)
        self.assertAlmostEqual(rows[0]["time"], 0.375)
        self.assertAlmostEqual(rows[-1]["time"], 6.0)
        self.assertTrue(
            all(
                row["maximum_boundary_l2_difference_upper"] > 0.0
                for row in rows
            )
        )
        crosschecks = result["production_endpoint_crosschecks"]
        self.assertEqual(len(crosschecks), 16)
        self.assertTrue(
            all(row["certified_intervals_overlap"] for row in crosschecks)
        )


if __name__ == "__main__":
    unittest.main()
