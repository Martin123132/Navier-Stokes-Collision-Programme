import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csc_matrix


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name,
        SCRIPT_DIR / filename,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREFIX = _load(
    "symbolic_transition_prefix_test_base",
    "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
)
MODULE = _load(
    "weighted_hypercircle_symbolic_transition_map",
    "neutral_strip_weighted_hypercircle_symbolic_transition_map.py",
)


def test_symbolic_map_reports_blocks_and_next_transition():
    center = np.asarray(
        [
            [5.0, 0.2, 0.3, 0.4],
            [0.2, 4.0, 0.5, 0.6],
            [0.3, 0.5, 3.0, 0.7],
            [0.4, 0.6, 0.7, 2.0],
        ]
    )
    order = np.arange(4, dtype=np.int64)
    problem = PREFIX.PrefixProblem(
        center=csc_matrix(center).tocsr(),
        radius=csc_matrix(np.zeros_like(center)).tocsr(),
        scale=np.ones(4),
        order=order,
        positions=order.copy(),
        lower=csc_matrix(np.eye(4)),
        central_pivots=np.diag(center),
    )
    preparation = {
        "matrix_inventory": {
            "edge_count": 1,
            "triangle_count": 1,
            "state_count": 1,
        }
    }
    result = MODULE.summarize_map(
        problem,
        preparation,
        maximum_pivots=4,
        checkpoints=(2, 4),
        prior_scan_pivots=2,
    )
    assert result["final_block_counts"] == {
        "edge_metric": 1,
        "triangle_constraint": 1,
        "state": 1,
        "source_triangle": 1,
    }
    assert result["first_block_pivots_within_scan"] == {
        "edge_metric": 0,
        "triangle_constraint": 1,
        "state": 2,
        "source_triangle": 3,
    }
    assert result["next_transition_pivot"] == 2
    assert result["recommended_next_bounded_pivot_count"] == 4
    assert not result["profile"]["arithmetic_signs_certified"]
