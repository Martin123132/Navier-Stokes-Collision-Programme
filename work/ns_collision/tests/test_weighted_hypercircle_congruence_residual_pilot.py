import importlib.util
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


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
    "congruence_residual_prefix_test_base",
    "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
)
MODULE = _load(
    "weighted_hypercircle_congruence_residual_pilot",
    "neutral_strip_weighted_hypercircle_congruence_residual_pilot.py",
)


def _problem(center_values: np.ndarray, radius_values: np.ndarray):
    dimension = len(center_values)
    center = csc_matrix(center_values).tocsr()
    radius = csc_matrix(radius_values).tocsr()
    center.sort_indices()
    radius.sort_indices()
    order = np.arange(dimension, dtype=np.int64)
    return PREFIX.PrefixProblem(
        center=center,
        radius=radius,
        scale=np.ones(dimension, dtype=float),
        order=order,
        positions=order.copy(),
        lower=csc_matrix(np.eye(dimension)),
        central_pivots=np.diag(center_values).copy(),
    )


def test_congruence_residual_certifies_small_indefinite_family():
    center = np.asarray(
        [
            [4.0, 0.2, 0.1],
            [0.2, -3.0, 0.3],
            [0.1, 0.3, 2.0],
        ]
    )
    problem = _problem(center, np.full_like(center, 1.0e-12))
    result = MODULE.certify_problem(
        problem,
        maximum_pivots=3,
        decimal_precision=60,
        prefix_module=PREFIX,
    )
    assert result["identity_factor_permutations"]
    assert result["interval_family_inertia_certified"]
    assert result["reference_diagonal_signs"] == {
        "negative": 1,
        "positive": 2,
        "zero": 0,
    }
    assert len(result["reference_L_sha256"]) == 64
    assert len(result["reference_D_sha256"]) == 64
    assert len(result["reference_factor_sha256"]) == 64
    assert float(
        result[
            "transformed_bound_to_minimum_diagonal_upper_decimal"
        ]
    ) < 1.0


def test_congruence_residual_fails_closed_for_wide_family():
    center = np.asarray(
        [
            [4.0, 0.2, 0.1],
            [0.2, -3.0, 0.3],
            [0.1, 0.3, 2.0],
        ]
    )
    problem = _problem(center, np.full_like(center, 5.0))
    result = MODULE.certify_problem(
        problem,
        maximum_pivots=3,
        decimal_precision=60,
        prefix_module=PREFIX,
    )
    assert not result["interval_family_inertia_certified"]
    assert float(
        result[
            "transformed_bound_to_minimum_diagonal_upper_decimal"
        ]
    ) > 1.0


def test_absolute_triangular_inverse_bounds_dominate_numeric_norms():
    matrix = csc_matrix(
        np.asarray(
            [
                [4.0, 0.2, 0.1],
                [0.2, -3.0, 0.3],
                [0.1, 0.3, 2.0],
            ]
        )
    )
    factor = splu(
        matrix,
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    lower = factor.L.tocsc()
    lower.sort_indices()
    arithmetic = PREFIX.DirectedDecimal(60)
    infinity_bound, one_bound = (
        MODULE._absolute_triangular_inverse_norm_bounds(
            lower,
            arithmetic,
        )
    )
    inverse = np.linalg.inv(lower.toarray())
    assert float(infinity_bound) >= np.linalg.norm(inverse, ord=np.inf)
    assert float(one_bound) >= np.linalg.norm(inverse, ord=1)


def test_reference_product_intervals_enclose_exact_binary_factor_product():
    matrix = csc_matrix(
        np.asarray(
            [
                [4.0, 0.2, 0.1, -0.15],
                [0.2, -3.0, 0.3, 0.05],
                [0.1, 0.3, 2.0, 0.25],
                [-0.15, 0.05, 0.25, -1.5],
            ]
        )
    )
    factor = splu(
        matrix,
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    lower = factor.L.tocsc()
    lower.sort_indices()
    diagonal = np.asarray(factor.U.diagonal(), dtype=float)
    intervals = MODULE._reference_product_intervals(
        lower,
        diagonal,
        PREFIX.DirectedDecimal(80),
    )
    dense_lower = lower.toarray()
    for row in range(matrix.shape[0]):
        for column in range(row + 1):
            exact = sum(
                (
                    Fraction.from_float(float(dense_lower[row, pivot]))
                    * Fraction.from_float(float(diagonal[pivot]))
                    * Fraction.from_float(float(dense_lower[column, pivot]))
                )
                for pivot in range(matrix.shape[0])
            )
            interval = intervals[(row, column)]
            assert Fraction(interval[0]) <= exact <= Fraction(interval[1])


def test_congruence_residual_rejects_asymmetric_interval_input():
    center = np.asarray(
        [
            [2.0, 0.25],
            [0.5, -1.0],
        ]
    )
    problem = _problem(center, np.zeros_like(center))
    try:
        MODULE.certify_problem(
            problem,
            maximum_pivots=2,
            decimal_precision=60,
            prefix_module=PREFIX,
        )
    except RuntimeError as error:
        assert "not exactly symmetric" in str(error)
    else:
        raise AssertionError("asymmetric interval input was accepted")


def test_congruence_residual_honors_cpu_park_callback():
    center = np.asarray(
        [
            [4.0, 0.2],
            [0.2, -3.0],
        ]
    )
    problem = _problem(center, np.full_like(center, 1.0e-12))
    try:
        MODULE.certify_problem(
            problem,
            maximum_pivots=2,
            decimal_precision=60,
            prefix_module=PREFIX,
            cpu_park_callback=lambda: True,
        )
    except RuntimeError as error:
        assert "CPU park requested" in str(error)
    else:
        raise AssertionError("CPU park callback was ignored")


def test_higher_precision_residual_bound_nests():
    center = np.asarray(
        [
            [4.0, 0.2, 0.1],
            [0.2, -3.0, 0.3],
            [0.1, 0.3, 2.0],
        ]
    )
    problem = _problem(center, np.full_like(center, 1.0e-12))
    precision_40 = MODULE.certify_problem(
        problem,
        maximum_pivots=3,
        decimal_precision=40,
        prefix_module=PREFIX,
    )
    precision_80 = MODULE.certify_problem(
        problem,
        maximum_pivots=3,
        decimal_precision=80,
        prefix_module=PREFIX,
    )
    assert precision_40["interval_family_inertia_certified"]
    assert precision_80["interval_family_inertia_certified"]
    assert float(
        precision_80[
            "transformed_residual_two_norm_upper_decimal"
        ]
    ) <= float(
        precision_40[
            "transformed_residual_two_norm_upper_decimal"
        ]
    )
