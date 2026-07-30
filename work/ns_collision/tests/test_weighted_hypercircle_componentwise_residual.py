import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csc_matrix


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_componentwise_residual.py"
)
RESULTS_DIR = ROOT / "work/ns_collision/results"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_componentwise_propagation_matches_explicit_dense_majorant():
    componentwise = _load_module(
        "test_componentwise_residual_module",
        SCRIPT,
    )
    prefix = componentwise._load_module(
        "test_componentwise_prefix_module",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )
    arithmetic = prefix.DirectedDecimal(50)
    lower_dense = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [-0.2, 0.3, 1.0],
        ],
        dtype=float,
    )
    lower = csc_matrix(lower_dense)
    residual_dense = np.asarray(
        [
            [0.10, 0.02, 0.00],
            [0.02, 0.20, 0.01],
            [0.00, 0.01, 0.30],
        ],
        dtype=float,
    )
    residual_entries = [
        (row, column, arithmetic.nearest.create_decimal_from_float(
            float(residual_dense[row, column])
        ))
        for row in range(3)
        for column in range(row + 1)
        if residual_dense[row, column] != 0.0
    ]
    result = componentwise._componentwise_bound_from_residual_entries(
        lower,
        residual_entries,
        arithmetic,
    )
    strictly_lower = np.abs(lower_dense - np.eye(3))
    majorant = np.linalg.inv(np.eye(3) - strictly_lower)
    explicit = majorant @ residual_dense @ majorant.T
    observed = np.asarray(
        [float(value) for value in result["row_sums"]],
        dtype=float,
    )
    assert np.allclose(observed, explicit @ np.ones(3), rtol=1e-14)
    assert np.isclose(
        float(result["maximum_row_sum"]),
        np.linalg.norm(explicit, ord=np.inf),
        rtol=1e-14,
    )


def test_stored_componentwise64040_result_is_hash_bound_and_certified():
    result = json.loads(
        (
            RESULTS_DIR
            / "neutral_strip_h006_hypercircle_"
            "standalone_componentwise_residual64040_v1.json"
        ).read_text(encoding="ascii")
    )
    assert result["all_current_stage_checks_pass"]
    assert result["status"] == "standalone_prefix_inertia_certified"
    assert result["validation_mode"] == "standalone_hash_bound"
    assert result["checks"]["separated_reference_and_residual_reproduced"]
    assert result["checks"]["separated_bound_reproduced"]
    assert result["checks"][
        "componentwise_bound_not_larger_than_separated_bound"
    ]
    assert not result["directed_LDL_dependency"]["required"]
    assert not result["directed_LDL_dependency"]["audit_loaded"]
    assert result["certification_flags"][
        "standalone_bounded_prefix_inertia_certified"
    ]
    assert not result["certification_flags"][
        "full_123816_pivot_inertia_certified"
    ]
    assert not result["certification_flags"][
        "continuum_spectrum_below_60_captured"
    ]
    assert not result["certification_flags"][
        "navier_stokes_regularity_certified"
    ]


def test_stored_componentwise64064_result_is_hash_bound_and_certified():
    result = json.loads(
        (
            RESULTS_DIR
            / "neutral_strip_h006_hypercircle_"
            "standalone_componentwise_residual64064_v1.json"
        ).read_text(encoding="ascii")
    )
    assert result["all_current_stage_checks_pass"]
    assert result["status"] == "standalone_prefix_inertia_certified"
    assert result["checks"]["separated_reference_and_residual_reproduced"]
    assert result["checks"]["separated_bound_reproduced"]
    assert result["checks"][
        "componentwise_bound_not_larger_than_separated_bound"
    ]
    assert result["certification_flags"][
        "standalone_bounded_prefix_inertia_certified"
    ]
    assert result["certificate"]["reference_diagonal_signs"] == {
        "negative": 32500,
        "positive": 31564,
        "zero": 0,
    }
    assert not result["certification_flags"][
        "full_123816_pivot_inertia_certified"
    ]
