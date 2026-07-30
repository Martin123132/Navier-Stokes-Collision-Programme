import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse import csc_matrix, diags


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "neutral_strip_weighted_hypercircle_central_factorization_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "weighted_hypercircle_central_factorization_audit",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_decimal_beta_squared_is_enclosed():
    exact = MODULE.BETA_DECIMAL * MODULE.BETA_DECIMAL
    lower, upper = MODULE._exact_decimal_interval(exact)
    from decimal import Decimal

    assert Decimal.from_float(lower) < exact
    assert Decimal.from_float(upper) > exact


def test_symmetric_ruiz_scaling_preserves_symmetry():
    matrix = csc_matrix(
        np.asarray(
            [
                [4.0, 2.0, 0.0],
                [2.0, -3.0, 1.0],
                [0.0, 1.0, 0.25],
            ]
        )
    )
    scale, diagnostics = MODULE._symmetric_ruiz_scaling(matrix, 5)
    scaled = MODULE._symmetric_scale(matrix, scale)
    assert MODULE._exactly_symmetric(scaled)
    assert diagnostics["scale_ratio"] >= 1.0
    assert diagnostics["maximum_final_row_maximum"] < 1.1


def test_small_central_factor_matches_dense_inertia():
    dense = np.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, -3.0, 0.5, 0.0],
            [0.0, 0.5, 2.0, 1.0],
            [0.0, 0.0, 1.0, -2.0],
        ]
    )
    matrix = csc_matrix(dense)
    radius = diags(np.full(4, 1.0e-14), format="csc")
    values = eigvalsh(dense)
    expected = {
        "positive": int(np.count_nonzero(values > 0.0)),
        "negative": int(np.count_nonzero(values < 0.0)),
        "zero": int(np.count_nonzero(values == 0.0)),
    }
    row = MODULE._factor_case(
        matrix,
        radius,
        {
            "label": "small",
            "permc_spec": "MMD_AT_PLUS_A",
            "scaling": "none",
        },
        expected,
    )
    assert row["status"] == "success"
    assert row["row_and_column_permutations_equal"]
    assert row["LDL_relation_relative_infinity_defect"] < 1.0e-14
    assert row["central_solve_relative_backward_error"] < 1.0e-14
    assert row["U_diagonal_counts_match_target"]


def test_stored_full_mesh_audit_selects_only_symmetric_mmd_path():
    results_dir = Path(__file__).resolve().parents[1] / "results"
    result_path = (
        results_dir
        / "neutral_strip_h006_hypercircle_central_factorization_audit_v1.json"
    )
    checkpoint_path = (
        results_dir
        / "neutral_strip_h006_hypercircle_central_factorization_checkpoint_v1.json"
    )
    result = MODULE.json.loads(result_path.read_text(encoding="ascii"))
    checkpoint = MODULE.json.loads(
        checkpoint_path.read_text(encoding="ascii")
    )

    assert result["status"] == "complete"
    assert result["all_current_stage_checks_pass"]
    assert result["complete_case_set"]
    assert len(result["factorization_cases"]) == 8
    assert (
        result["artifacts"]["checkpoint_sha256"]
        == MODULE._sha256_file(checkpoint_path)
    )
    assert checkpoint["contract"] == result["contract"]
    assert (
        checkpoint["factorization_cases"]
        == result["factorization_cases"]
    )

    mmd_rows = result["factorization_cases"][:6]
    assert all(row["permc_spec"] == "MMD_AT_PLUS_A" for row in mmd_rows)
    assert all(
        row["row_and_column_permutations_equal"] for row in mmd_rows
    )
    assert all(row["U_diagonal_counts_match_target"] for row in mmd_rows)
    assert all(
        row["U_diagonal_counts"]
        == {"positive": 61908, "negative": 61908, "zero": 0}
        for row in mmd_rows
    )
    assert all(
        row["LDL_relation_relative_infinity_defect"] < 1.0e-10
        for row in mmd_rows
    )

    rejected = result["factorization_cases"][6:]
    assert result["rejected_ordering_cases"] == [
        "mmd_ata_symmetric_ruiz_8",
        "colamd_symmetric_ruiz_8",
    ]
    assert all(
        not row["row_and_column_permutations_equal"] for row in rejected
    )
    assert all(
        row["central_solve_relative_backward_error"] > 0.02
        for row in rejected
    )
    assert all(
        row["minimum_absolute_U_diagonal"] <= np.finfo(float).eps
        for row in rejected
    )

    assert (
        result["recommended_central_case"]
        == "mmd_at_plus_a_symmetric_ruiz_10"
    )
    assert (
        result["recommended_scale_sha256"]
        == "bef73a763a5ee24b85651a3c761b940b0fde2f8e04294a7d7e9b0c085c7ca9c2"
    )
    assert (
        result["recommended_permutation_sha256"]
        == "fc613374bffd7bba84293e3c302e56d0ef945a0530443b04dde5ba079adb36db"
    )
    assert (
        result["recommended_factor_pattern_sha256"]
        == "dc941205f68286d3f318a58670fd0ddf14bf63afdaf48202dcc9e0291238103b"
    )
    viability = result["viability_assessment"]
    assert viability["central_target_pivot_count_observed"]
    assert viability["floating_inverse_estimate_times_radius_below_one"]
    assert not viability["global_norm_roundoff_proxy_closes"]
    assert viability["componentwise_or_verified_residual_method_required"]
    assert not viability["verified_directed_sparse_inertia_method_ready"]

    flags = result["certification_flags"]
    assert flags["central_threshold_pencil_factorization_observed"]
    assert not flags["full_mesh_threshold_inertia_certified"]
    assert not flags["kappa_h_verified_upper_bound"]
    assert not flags["continuum_spectrum_below_60_captured"]
