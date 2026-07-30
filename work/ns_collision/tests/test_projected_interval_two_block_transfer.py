from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "work" / "ns_collision" / "scripts" / (
    "neutral_strip_projected_interval_two_block_transfer.py"
)
RESULT = ROOT / "work" / "ns_collision" / "results" / (
    "neutral_strip_h006_projected_interval_two_block_transfer_v1.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "projected_interval_two_block_transfer_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProjectedIntervalTwoBlockTransferTests(unittest.TestCase):
    def test_two_block_scalar_regression(self) -> None:
        module = _load_module()
        alpha = 2.0
        beta = 9.0
        coupling = 1.0
        generator = np.array(
            [[-alpha, coupling], [coupling, -beta]],
            dtype=float,
        )
        for time_value in (0.125, 0.375, 0.75, 1.5):
            row = module._two_block_bounds(
                alpha,
                beta,
                coupling,
                time_value,
            )
            exact = expm(time_value * generator) @ np.array([1.0, 0.0])
            low_feedback = max(exact[0] - np.exp(-alpha * time_value), 0.0)
            self.assertLessEqual(
                abs(exact[1]),
                row["high_component_upper"] + 2.0e-14,
            )
            self.assertLessEqual(
                low_feedback,
                row["low_feedback_upper"] + 2.0e-14,
            )
            self.assertLess(
                row["high_component_upper"],
                row["gap_free_high_component_upper"],
            )

    def test_production_artifact_is_fail_closed(self) -> None:
        payload = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(
            payload[
                "all_projected_interval_two_block_transfer_checks_pass"
            ]
        )
        self.assertTrue(
            payload[
                "finite_common_circle_riesz_gram_projected_algebra_certified"
            ]
        )
        self.assertTrue(payload["exact_polygon_low_projector_certified"])
        self.assertFalse(payload["modified_off_block_leakage_certified"])
        self.assertFalse(payload["continuum_Ritz_transfer_certified"])
        self.assertFalse(
            payload["polygon_to_circle_domain_transfer_certified"]
        )
        transfer = payload["exact_form_boundary_and_source_transfer"]
        self.assertLess(transfer["upgraded_complete_screen_upper"], 1.0)
        self.assertGreater(
            transfer["upgraded_complete_screen_headroom_lower"],
            0.029,
        )
        theorem = payload["two_block_damped_leakage_theorem"]
        self.assertTrue(theorem["theorem_implemented"])
        self.assertFalse(
            theorem["modified_chain_high_block_floor_certified"]
        )
        self.assertFalse(
            theorem["reference_complement_floor_substituted_for_modified_floor"]
        )


if __name__ == "__main__":
    unittest.main()
