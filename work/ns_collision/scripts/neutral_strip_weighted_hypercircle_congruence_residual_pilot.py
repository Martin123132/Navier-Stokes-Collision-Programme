#!/usr/bin/env python3
"""Test an independent congruence-residual inertia certificate.

For a fixed exact reference ``M = L D L^T`` and every symmetric interval
matrix ``A``, write

    A = L (D + L^-1 (A - M) L^-T) L^T.

If a directed upper bound for the spectral norm of the transformed residual
is smaller than ``min(abs(diag(D)))``, Weyl's theorem and Sylvester's law of
inertia certify that every matrix in the interval family has the signs of
``D``.  This route never divides by an interval pivot.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPLETE_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_complete_assembly_v1.json"
)
DEFAULT_MATRICES = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_assembly_matrices_v1.npz"
)
DEFAULT_GAUSSIAN_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
)
DEFAULT_GAUSSIAN_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
)
DEFAULT_DIRECTED_AUDIT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_transition2304_audit_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_congruence_residual_pilot2304_v1.json"
)

DEFAULT_MAXIMUM_PIVOTS = 2304
DEFAULT_DECIMAL_PRECISION = 60
DAYTIME_BASELINE_CPU_LIMIT = 60.0


def _load_prefix_module():
    path = (
        SCRIPT_DIR
        / "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "congruence_residual_prefix_base",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scaled_central_prefix(problem, maximum_pivots: int) -> csc_matrix:
    original = problem.order[:maximum_pivots]
    prefix = problem.center[original, :][:, original].tocoo(copy=True)
    row_scale = problem.scale[original[prefix.row]]
    column_scale = problem.scale[original[prefix.col]]
    prefix.data = prefix.data * (row_scale * column_scale)
    result = prefix.tocsc()
    result.sort_indices()
    return result


def _exactly_symmetric(matrix) -> bool:
    difference = (matrix - matrix.transpose()).tocsr()
    difference.eliminate_zeros()
    return difference.nnz == 0


def _validate_problem_contract(problem) -> None:
    dimension = problem.dimension
    if problem.center.shape != (dimension, dimension):
        raise RuntimeError("central matrix is not square")
    if problem.radius.shape != (dimension, dimension):
        raise RuntimeError("radius matrix shape does not match the center")
    if not _exactly_symmetric(problem.center):
        raise RuntimeError("central matrix is not exactly symmetric")
    if not _exactly_symmetric(problem.radius):
        raise RuntimeError("radius matrix is not exactly symmetric")
    if (
        not np.all(np.isfinite(problem.center.data))
        or not np.all(np.isfinite(problem.radius.data))
        or np.any(problem.radius.data < 0.0)
    ):
        raise RuntimeError("input interval matrix data are invalid")
    if (
        problem.scale.shape != (dimension,)
        or not np.all(np.isfinite(problem.scale))
        or np.any(problem.scale <= 0.0)
    ):
        raise RuntimeError("congruence scale is invalid")
    expected = np.arange(dimension, dtype=np.int64)
    if (
        problem.order.shape != (dimension,)
        or problem.positions.shape != (dimension,)
        or not np.array_equal(np.sort(problem.order), expected)
        or not np.array_equal(problem.positions[problem.order], expected)
    ):
        raise RuntimeError("elimination order and inverse positions disagree")


def _validate_unit_lower(lower: csc_matrix) -> None:
    if lower.shape[0] != lower.shape[1]:
        raise RuntimeError("reference factor L is not square")
    if not np.all(np.isfinite(lower.data)):
        raise RuntimeError("reference factor L contains non-finite values")
    diagonal = np.asarray(lower.diagonal(), dtype=float)
    if (
        len(diagonal) != lower.shape[0]
        or not np.array_equal(diagonal, np.ones(lower.shape[0]))
    ):
        raise RuntimeError("reference factor L is not unit lower triangular")
    for column in range(lower.shape[1]):
        rows = lower.indices[
            int(lower.indptr[column]) : int(lower.indptr[column + 1])
        ]
        if np.any(rows < column):
            raise RuntimeError("reference factor L contains an upper entry")


def _add_interval(
    entries: dict[tuple[int, int], tuple[Decimal, Decimal]],
    key: tuple[int, int],
    term: tuple[Decimal, Decimal],
    arithmetic,
) -> None:
    entries[key] = arithmetic.add(
        entries.get(key, (arithmetic.zero, arithmetic.zero)),
        term,
    )


def _reference_product_intervals(
    lower: csc_matrix,
    diagonal: np.ndarray,
    arithmetic,
    cpu_park_callback: Callable[[], bool] | None = None,
) -> dict[tuple[int, int], tuple[Decimal, Decimal]]:
    _validate_unit_lower(lower)
    entries: dict[tuple[int, int], tuple[Decimal, Decimal]] = {}
    term_count = 0
    for column in range(lower.shape[1]):
        start = int(lower.indptr[column])
        stop = int(lower.indptr[column + 1])
        rows = lower.indices[start:stop]
        values = lower.data[start:stop]
        diagonal_point = Decimal.from_float(float(diagonal[column]))
        diagonal_interval = (diagonal_point, diagonal_point)
        for first_offset, first_row_value in enumerate(rows):
            first_row = int(first_row_value)
            first_point = Decimal.from_float(float(values[first_offset]))
            first_interval = (first_point, first_point)
            for second_offset in range(first_offset + 1):
                second_row = int(rows[second_offset])
                second_point = Decimal.from_float(
                    float(values[second_offset])
                )
                second_interval = (second_point, second_point)
                term = arithmetic.multiply(
                    first_interval,
                    diagonal_interval,
                )
                term = arithmetic.multiply(term, second_interval)
                key = (
                    max(first_row, second_row),
                    min(first_row, second_row),
                )
                _add_interval(entries, key, term, arithmetic)
                term_count += 1
                if (
                    cpu_park_callback is not None
                    and term_count % 50000 == 0
                    and cpu_park_callback()
                ):
                    raise RuntimeError(
                        "daytime CPU park requested during reference product"
                    )
    return entries


def _absolute_triangular_inverse_norm_bounds(
    lower: csc_matrix,
    arithmetic,
    cpu_park_callback: Callable[[], bool] | None = None,
) -> tuple[Decimal, Decimal]:
    _validate_unit_lower(lower)
    dimension = lower.shape[0]
    lower_csr = lower.tocsr()
    lower_csr.sort_indices()
    row_sums = [arithmetic.one] * dimension
    for row in range(dimension):
        value = arithmetic.one
        for pointer in range(
            int(lower_csr.indptr[row]),
            int(lower_csr.indptr[row + 1]),
        ):
            column = int(lower_csr.indices[pointer])
            if column >= row:
                continue
            coefficient = Decimal.from_float(
                abs(float(lower_csr.data[pointer]))
            )
            value = arithmetic.upper.add(
                value,
                arithmetic.upper.multiply(
                    coefficient,
                    row_sums[column],
                ),
            )
        row_sums[row] = value
        if (
            cpu_park_callback is not None
            and (row + 1) % 4096 == 0
            and cpu_park_callback()
        ):
            raise RuntimeError(
                "daytime CPU park requested during inverse majorant"
            )

    column_sums = [arithmetic.one] * dimension
    for column in range(dimension - 1, -1, -1):
        value = arithmetic.one
        for pointer in range(
            int(lower.indptr[column]),
            int(lower.indptr[column + 1]),
        ):
            row = int(lower.indices[pointer])
            if row <= column:
                continue
            coefficient = Decimal.from_float(
                abs(float(lower.data[pointer]))
            )
            value = arithmetic.upper.add(
                value,
                arithmetic.upper.multiply(
                    coefficient,
                    column_sums[row],
                ),
            )
        column_sums[column] = value
        if (
            cpu_park_callback is not None
            and column % 4096 == 0
            and cpu_park_callback()
        ):
            raise RuntimeError(
                "daytime CPU park requested during inverse majorant"
            )
    return max(row_sums), max(column_sums)


def certify_problem(
    problem,
    maximum_pivots: int,
    decimal_precision: int,
    prefix_module,
    cpu_park_callback: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if not 0 < maximum_pivots <= problem.dimension:
        raise ValueError("maximum pivots is out of range")
    _validate_problem_contract(problem)
    arithmetic = prefix_module.DirectedDecimal(decimal_precision)
    started = time.perf_counter()
    central_prefix = _scaled_central_prefix(
        problem,
        maximum_pivots,
    )
    factor = splu(
        central_prefix,
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    identity = np.arange(maximum_pivots, dtype=factor.perm_r.dtype)
    identity_permutations = bool(
        np.array_equal(factor.perm_r, identity)
        and np.array_equal(factor.perm_c, identity)
    )
    if not identity_permutations:
        raise RuntimeError(
            "leading central factorization introduced a permutation"
        )
    lower = factor.L.tocsc()
    lower.sort_indices()
    _validate_unit_lower(lower)
    diagonal = np.asarray(factor.U.diagonal(), dtype=float)
    if (
        len(diagonal) != maximum_pivots
        or not np.all(np.isfinite(diagonal))
        or np.any(diagonal == 0.0)
    ):
        raise RuntimeError("central reference diagonal is invalid")
    if cpu_park_callback is not None and cpu_park_callback():
        raise RuntimeError(
            "daytime CPU park requested before interval propagation"
        )

    reference_entries = _reference_product_intervals(
        lower,
        diagonal,
        arithmetic,
        cpu_park_callback,
    )
    input_keys = {(index, index) for index in range(maximum_pivots)}
    for column in range(maximum_pivots):
        input_keys.update(
            (row, column)
            for row in prefix_module._input_neighbors_after(
                problem,
                column,
            )
            if row < maximum_pivots
        )
    residual_keys = input_keys.union(reference_entries)
    row_sums = [arithmetic.zero] * maximum_pivots
    maximum_entry = arithmetic.zero
    maximum_entry_key: tuple[int, int] | None = None
    for residual_index, (row, column) in enumerate(
        sorted(residual_keys),
        start=1,
    ):
        input_interval = prefix_module._input_interval(
            problem,
            arithmetic,
            row,
            column,
        )
        reference_interval = reference_entries.get(
            (row, column),
            (arithmetic.zero, arithmetic.zero),
        )
        residual = arithmetic.subtract(
            input_interval,
            reference_interval,
        )
        magnitude = arithmetic.absolute_upper(residual)
        row_sums[row] = arithmetic.upper.add(
            row_sums[row],
            magnitude,
        )
        if row != column:
            row_sums[column] = arithmetic.upper.add(
                row_sums[column],
                magnitude,
            )
        if magnitude > maximum_entry:
            maximum_entry = magnitude
            maximum_entry_key = (row, column)
        if (
            cpu_park_callback is not None
            and residual_index % 10000 == 0
            and cpu_park_callback()
        ):
            raise RuntimeError(
                "daytime CPU park requested during residual assembly"
            )
    residual_infinity_norm = max(row_sums)

    inverse_infinity, inverse_one = (
        _absolute_triangular_inverse_norm_bounds(
            lower,
            arithmetic,
            cpu_park_callback,
        )
    )
    transformed_bound = arithmetic.upper.multiply(
        inverse_infinity,
        inverse_one,
    )
    transformed_bound = arithmetic.upper.multiply(
        transformed_bound,
        residual_infinity_norm,
    )
    minimum_diagonal = min(
        abs(Decimal.from_float(float(value))) for value in diagonal
    )
    ratio = arithmetic.upper.divide(
        transformed_bound,
        minimum_diagonal,
    )
    certified = transformed_bound < minimum_diagonal
    positive = int(np.count_nonzero(diagonal > 0.0))
    negative = int(np.count_nonzero(diagonal < 0.0))

    return {
        "method": (
            "directed residual around exact binary64 L D L^T reference, "
            "with positive absolute-triangular inverse recurrences"
        ),
        "proof_basis": {
            "reference_congruence": (
                "A = L (D + L^-1 (A - L D L^T) L^-T) L^T"
            ),
            "inverse_majorant": (
                "|L^-1| <= (I - |L-I|)^-1 for unit lower triangular L"
            ),
            "spectral_bound": (
                "||L^-1 E L^-T||_2 <= "
                "||L^-1||_1 ||L^-1||_inf ||E||_inf for symmetric E"
            ),
            "sign_gate": (
                "directed transformed-residual bound is strictly smaller "
                "than every absolute reference diagonal"
            ),
            "conclusion": "Weyl sign preservation plus Sylvester inertia",
            "interval_pivot_divisions_used": False,
        },
        "validated_assumptions": {
            "input_center_exactly_symmetric": True,
            "input_radius_exactly_symmetric_nonnegative": True,
            "positive_congruence_scale": True,
            "order_and_inverse_positions_form_a_permutation": True,
            "reference_L_unit_lower_triangular": True,
            "reference_D_finite_and_nonzero": True,
        },
        "decimal_precision": decimal_precision,
        "dimension": maximum_pivots,
        "identity_factor_permutations": identity_permutations,
        "reference_L_nnz": int(lower.nnz),
        "reference_L_sha256": prefix_module._sha256_arrays(
            lower.indptr,
            lower.indices,
            lower.data,
        ),
        "reference_D_sha256": prefix_module._sha256_arrays(diagonal),
        "reference_factor_sha256": prefix_module._sha256_arrays(
            lower.indptr,
            lower.indices,
            lower.data,
            diagonal,
        ),
        "reference_product_lower_entry_count": len(reference_entries),
        "residual_lower_entry_count": len(residual_keys),
        "maximum_residual_entry_upper_decimal": str(maximum_entry),
        "maximum_residual_entry_coordinate": (
            list(maximum_entry_key)
            if maximum_entry_key is not None
            else None
        ),
        "residual_infinity_norm_upper_decimal": str(
            residual_infinity_norm
        ),
        "absolute_L_inverse_infinity_norm_upper_decimal": str(
            inverse_infinity
        ),
        "absolute_L_inverse_one_norm_upper_decimal": str(inverse_one),
        "transformed_residual_two_norm_upper_decimal": str(
            transformed_bound
        ),
        "minimum_absolute_reference_diagonal_decimal": str(
            minimum_diagonal
        ),
        "transformed_bound_to_minimum_diagonal_upper_decimal": str(
            ratio
        ),
        "reference_diagonal_signs": {
            "negative": negative,
            "positive": positive,
            "zero": 0,
        },
        "interval_family_inertia_certified": certified,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_pilot(
    complete_result_path: Path = DEFAULT_COMPLETE_RESULT,
    matrices_path: Path = DEFAULT_MATRICES,
    gaussian_result_path: Path = DEFAULT_GAUSSIAN_RESULT,
    gaussian_checkpoint_path: Path = DEFAULT_GAUSSIAN_CHECKPOINT,
    directed_audit_path: Path = DEFAULT_DIRECTED_AUDIT,
    maximum_pivots: int = DEFAULT_MAXIMUM_PIVOTS,
    decimal_precision: int = DEFAULT_DECIMAL_PRECISION,
    enforce_cpu_policy: bool = True,
) -> dict[str, Any]:
    prefix = _load_prefix_module()
    priority_set = prefix._set_below_normal_priority()
    baseline = (
        prefix._sample_cpu_baseline(5) if enforce_cpu_policy else []
    )
    baseline_mean = (
        sum(baseline) / len(baseline) if baseline else None
    )
    if (
        baseline_mean is not None
        and baseline_mean > DAYTIME_BASELINE_CPU_LIMIT
    ):
        raise RuntimeError(
            "daytime baseline CPU exceeds the one-worker launch limit: "
            f"{baseline_mean:.3f}%"
        )
    problem, preparation = prefix._prepare_production_problem(
        complete_result_path,
        matrices_path,
        gaussian_result_path,
        gaussian_checkpoint_path,
    )
    directed = json.loads(
        directed_audit_path.read_text(encoding="ascii")
    )
    if (
        directed["directed_LDL_prefix"]["completed_pivot_count"]
        != maximum_pivots
        or not directed["certification_flags"][
            "bounded_prefix_directed_LDL_certified"
        ]
    ):
        raise RuntimeError(
            "matching directed-LDL prefix certificate is unavailable"
        )
    certificate = certify_problem(
        problem,
        maximum_pivots,
        decimal_precision,
        prefix,
    )
    directed_signs = {
        "negative": directed["directed_LDL_prefix"][
            "negative_pivot_count"
        ],
        "positive": directed["directed_LDL_prefix"][
            "positive_pivot_count"
        ],
        "zero": 0,
    }
    signs_match = (
        certificate["reference_diagonal_signs"] == directed_signs
    )
    checks = {
        **preparation["preparation_checks"],
        "matching_directed_prefix_is_certified": True,
        "reference_signs_match_directed_prefix": signs_match,
        "residual_certificate_decision_recorded": True,
        "full_inertia_claim_remains_false": True,
    }
    return {
        "kind": "hypercircle-congruence-residual-inertia-pilot",
        "status": (
            "independent_prefix_inertia_certified"
            if certificate["interval_family_inertia_certified"]
            and signs_match
            else "independent_route_does_not_close"
        ),
        "all_current_stage_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "certificate": certificate,
        "comparison_with_directed_LDL": {
            "directed_signs": directed_signs,
            "reference_signs": certificate[
                "reference_diagonal_signs"
            ],
            "signs_match": signs_match,
            "methods_are_independent_at_interval_propagation_level": True,
            "shared_inputs_scaling_and_order": True,
        },
        "certification_flags": {
            "independent_bounded_prefix_inertia_certified": bool(
                certificate["interval_family_inertia_certified"]
                and signs_match
            ),
            "full_123816_pivot_inertia_certified": False,
            "continuum_spectrum_below_60_captured": False,
        },
        "preparation": preparation,
        "runtime": {
            "below_normal_priority_set": priority_set,
            "baseline_cpu_samples_percent": baseline,
            "baseline_cpu_mean_percent": baseline_mean,
        },
        "artifacts": {
            "directed_LDL_audit": str(
                directed_audit_path
            ).replace("\\", "/"),
            "directed_LDL_audit_sha256": prefix._sha256_file(
                directed_audit_path
            ),
        },
        "next_required_step": (
            "Cross-check the residual certificate at higher Decimal "
            "precision and on adversarial small matrices before using it as "
            "independent evidence for any larger prefix."
            if certificate["interval_family_inertia_certified"]
            else "The congruence-residual route is fail-closed at this "
            "prefix; inspect the inverse-norm and residual contributions "
            "before changing the reference factorization."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directed-audit",
        type=Path,
        default=DEFAULT_DIRECTED_AUDIT,
    )
    parser.add_argument(
        "--maximum-pivots",
        type=int,
        default=DEFAULT_MAXIMUM_PIVOTS,
    )
    parser.add_argument(
        "--decimal-precision",
        type=int,
        default=DEFAULT_DECIMAL_PRECISION,
    )
    parser.add_argument("--skip-cpu-policy", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_pilot(
        directed_audit_path=args.directed_audit,
        maximum_pivots=args.maximum_pivots,
        decimal_precision=args.decimal_precision,
        enforce_cpu_policy=not args.skip_cpu_policy,
    )
    prefix = _load_prefix_module()
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
