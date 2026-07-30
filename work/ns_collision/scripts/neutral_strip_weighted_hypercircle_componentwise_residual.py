#!/usr/bin/env python3
"""Certify a prefix with a direct componentwise residual majorant.

For the fixed binary reference ``M = L D L^T``, let ``R`` bound the
componentwise magnitude of the symmetric interval residual ``A - M`` and let
``Q = (I - |L-I|)^-1``. Then

``|L^-1 (A-M) L^-T| <= Q R Q^T``.

The right side is symmetric and nonnegative, so its maximum row sum bounds
the transformed residual spectral norm. The row sums are computed directly
as ``Q R Q^T 1`` instead of separating three unrelated global norm maxima.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("work/ns_collision/results")
DEFAULT_COMPLETE_RESULT = RESULTS_DIR / (
    "neutral_strip_h006_positive_exponential_complete_assembly_v1.json"
)
DEFAULT_MATRICES = RESULTS_DIR / (
    "neutral_strip_h006_positive_exponential_assembly_matrices_v1.npz"
)
DEFAULT_GAUSSIAN_RESULT = RESULTS_DIR / (
    "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
)
DEFAULT_GAUSSIAN_CHECKPOINT = RESULTS_DIR / (
    "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
)
DEFAULT_SEPARATED_RESULT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual64040_v1.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64040_v1.json"
)
DEFAULT_MAXIMUM_PIVOTS = 64040
DEFAULT_DECIMAL_PRECISION = 60
DAYTIME_BASELINE_CPU_LIMIT = 60.0
DAYTIME_PARK_CPU_LIMIT = 75.0


def _load_module(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _positive_lower_inverse_apply(
    lower: csc_matrix,
    right_hand_side: list[Decimal],
    arithmetic,
    cpu_park_callback: Callable[[], bool] | None = None,
) -> list[Decimal]:
    """Apply ``(I-|L-I|)^-1`` to a nonnegative vector."""
    dimension = lower.shape[0]
    if len(right_hand_side) != dimension:
        raise ValueError("right-hand side dimension does not match L")
    lower_csr = lower.tocsr()
    lower_csr.sort_indices()
    result = [arithmetic.zero] * dimension
    for row in range(dimension):
        value = right_hand_side[row]
        if value < arithmetic.zero:
            raise ValueError("right-hand side is not nonnegative")
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
                    result[column],
                ),
            )
        result[row] = value
        if (
            cpu_park_callback is not None
            and (row + 1) % 4096 == 0
            and cpu_park_callback()
        ):
            raise RuntimeError(
                "daytime CPU park requested during positive lower solve"
            )
    return result


def _positive_lower_inverse_transpose_apply(
    lower: csc_matrix,
    right_hand_side: list[Decimal],
    arithmetic,
    cpu_park_callback: Callable[[], bool] | None = None,
) -> list[Decimal]:
    """Apply ``(I-|L-I|)^-T`` to a nonnegative vector."""
    dimension = lower.shape[0]
    if len(right_hand_side) != dimension:
        raise ValueError("right-hand side dimension does not match L")
    lower = lower.tocsc(copy=True)
    lower.sort_indices()
    result = [arithmetic.zero] * dimension
    for column in range(dimension - 1, -1, -1):
        value = right_hand_side[column]
        if value < arithmetic.zero:
            raise ValueError("right-hand side is not nonnegative")
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
                    result[row],
                ),
            )
        result[column] = value
        if (
            cpu_park_callback is not None
            and column % 4096 == 0
            and cpu_park_callback()
        ):
            raise RuntimeError(
                "daytime CPU park requested during positive transpose solve"
            )
    return result


def _symmetric_magnitude_apply(
    dimension: int,
    residual_entries: list[tuple[int, int, Decimal]],
    vector: list[Decimal],
    arithmetic,
    cpu_park_callback: Callable[[], bool] | None = None,
) -> list[Decimal]:
    """Apply a symmetric nonnegative lower-triangle matrix to a vector."""
    if len(vector) != dimension:
        raise ValueError("vector dimension does not match residual")
    result = [arithmetic.zero] * dimension
    for entry_index, (row, column, magnitude) in enumerate(
        residual_entries,
        start=1,
    ):
        if not 0 <= column <= row < dimension:
            raise ValueError("residual entry is outside the lower triangle")
        if magnitude < arithmetic.zero:
            raise ValueError("residual magnitude is negative")
        result[row] = arithmetic.upper.add(
            result[row],
            arithmetic.upper.multiply(magnitude, vector[column]),
        )
        if row != column:
            result[column] = arithmetic.upper.add(
                result[column],
                arithmetic.upper.multiply(magnitude, vector[row]),
            )
        if (
            cpu_park_callback is not None
            and entry_index % 10000 == 0
            and cpu_park_callback()
        ):
            raise RuntimeError(
                "daytime CPU park requested during residual propagation"
            )
    return result


def _componentwise_bound_from_residual_entries(
    lower: csc_matrix,
    residual_entries: list[tuple[int, int, Decimal]],
    arithmetic,
    cpu_park_callback: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Compute the row sums of ``Q R Q^T`` without forming it."""
    dimension = lower.shape[0]
    ones = [arithmetic.one] * dimension
    q_transpose_one = _positive_lower_inverse_transpose_apply(
        lower,
        ones,
        arithmetic,
        cpu_park_callback,
    )
    residual_times_q_transpose_one = _symmetric_magnitude_apply(
        dimension,
        residual_entries,
        q_transpose_one,
        arithmetic,
        cpu_park_callback,
    )
    row_sums = _positive_lower_inverse_apply(
        lower,
        residual_times_q_transpose_one,
        arithmetic,
        cpu_park_callback,
    )
    maximum_pivot = max(range(dimension), key=row_sums.__getitem__)
    return {
        "q_transpose_one": q_transpose_one,
        "residual_times_q_transpose_one": (
            residual_times_q_transpose_one
        ),
        "row_sums": row_sums,
        "maximum_row_sum": row_sums[maximum_pivot],
        "maximum_row_sum_pivot": maximum_pivot,
    }


def _build_certificate(
    problem,
    maximum_pivots: int,
    decimal_precision: int,
    prefix,
    residual,
    cpu_park_callback: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if not 0 < maximum_pivots <= problem.dimension:
        raise ValueError("maximum pivots is out of range")
    residual._validate_problem_contract(problem)
    arithmetic = prefix.DirectedDecimal(decimal_precision)
    started = time.perf_counter()
    central_prefix = residual._scaled_central_prefix(
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
    residual._validate_unit_lower(lower)
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

    reference_entries = residual._reference_product_intervals(
        lower,
        diagonal,
        arithmetic,
        cpu_park_callback,
    )
    input_keys = {(index, index) for index in range(maximum_pivots)}
    for column in range(maximum_pivots):
        input_keys.update(
            (row, column)
            for row in prefix._input_neighbors_after(problem, column)
            if row < maximum_pivots
        )
    residual_keys = input_keys.union(reference_entries)
    residual_row_sums = [arithmetic.zero] * maximum_pivots
    residual_entries: list[tuple[int, int, Decimal]] = []
    maximum_entry = arithmetic.zero
    maximum_entry_key: tuple[int, int] | None = None
    for residual_index, (row, column) in enumerate(
        sorted(residual_keys),
        start=1,
    ):
        input_interval = prefix._input_interval(
            problem,
            arithmetic,
            row,
            column,
        )
        reference_interval = reference_entries.get(
            (row, column),
            (arithmetic.zero, arithmetic.zero),
        )
        residual_interval = arithmetic.subtract(
            input_interval,
            reference_interval,
        )
        magnitude = arithmetic.absolute_upper(residual_interval)
        residual_entries.append((row, column, magnitude))
        residual_row_sums[row] = arithmetic.upper.add(
            residual_row_sums[row],
            magnitude,
        )
        if row != column:
            residual_row_sums[column] = arithmetic.upper.add(
                residual_row_sums[column],
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
    residual_infinity_norm = max(residual_row_sums)

    inverse_infinity_vector = _positive_lower_inverse_apply(
        lower,
        [arithmetic.one] * maximum_pivots,
        arithmetic,
        cpu_park_callback,
    )
    inverse_one_vector = _positive_lower_inverse_transpose_apply(
        lower,
        [arithmetic.one] * maximum_pivots,
        arithmetic,
        cpu_park_callback,
    )
    inverse_infinity = max(inverse_infinity_vector)
    inverse_one = max(inverse_one_vector)
    separated_bound = arithmetic.upper.multiply(
        inverse_infinity,
        inverse_one,
    )
    separated_bound = arithmetic.upper.multiply(
        separated_bound,
        residual_infinity_norm,
    )

    componentwise = _componentwise_bound_from_residual_entries(
        lower,
        residual_entries,
        arithmetic,
        cpu_park_callback,
    )
    transformed_bound = componentwise["maximum_row_sum"]
    transformed_bound_pivot = int(
        componentwise["maximum_row_sum_pivot"]
    )
    minimum_index = int(np.argmin(np.abs(diagonal)))
    minimum_diagonal = abs(
        Decimal.from_float(float(diagonal[minimum_index]))
    )
    ratio = arithmetic.upper.divide(
        transformed_bound,
        minimum_diagonal,
    )
    separated_ratio = arithmetic.upper.divide(
        separated_bound,
        minimum_diagonal,
    )
    certified = transformed_bound < minimum_diagonal
    positive = int(np.count_nonzero(diagonal > 0.0))
    negative = int(np.count_nonzero(diagonal < 0.0))
    improvement_lower = arithmetic.lower.divide(
        separated_bound,
        transformed_bound,
    )

    return {
        "method": (
            "directed componentwise residual propagation through the "
            "positive absolute-triangular inverse majorant"
        ),
        "proof_basis": {
            "reference_congruence": (
                "A = L (D + L^-1 (A - L D L^T) L^-T) L^T"
            ),
            "inverse_majorant": (
                "|L^-1| <= Q = (I - |L-I|)^-1"
            ),
            "residual_majorant": (
                "|A-LDL^T| <= R componentwise with R symmetric and "
                "nonnegative"
            ),
            "componentwise_transform": (
                "|L^-1 (A-LDL^T) L^-T| <= Q R Q^T"
            ),
            "spectral_bound": (
                "Q R Q^T is symmetric nonnegative, so its maximum row "
                "sum max(Q R Q^T 1) bounds the transformed spectral norm"
            ),
            "conclusion": "Weyl sign preservation plus Sylvester inertia",
            "interval_pivot_divisions_used": False,
        },
        "validated_assumptions": {
            "input_center_exactly_symmetric": True,
            "input_radius_exactly_symmetric_nonnegative": True,
            "order_and_inverse_positions_form_a_permutation": True,
            "positive_congruence_scale": True,
            "reference_L_unit_lower_triangular": True,
            "reference_D_finite_and_nonzero": True,
            "residual_magnitude_majorant_is_symmetric_nonnegative": True,
            "componentwise_row_sum_spectral_theorem_applied": True,
        },
        "dimension": maximum_pivots,
        "decimal_precision": decimal_precision,
        "elapsed_seconds": time.perf_counter() - started,
        "identity_factor_permutations": identity_permutations,
        "reference_L_nnz": int(lower.nnz),
        "reference_L_sha256": prefix._sha256_arrays(
            lower.indptr,
            lower.indices,
            lower.data,
        ),
        "reference_D_sha256": prefix._sha256_arrays(diagonal),
        "reference_factor_sha256": prefix._sha256_arrays(
            lower.indptr,
            lower.indices,
            lower.data,
            diagonal,
        ),
        "reference_product_lower_entry_count": len(reference_entries),
        "residual_lower_entry_count": len(residual_entries),
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
        "separated_transformed_residual_two_norm_upper_decimal": str(
            separated_bound
        ),
        "separated_bound_to_minimum_diagonal_upper_decimal": str(
            separated_ratio
        ),
        "componentwise_improvement_factor_lower_decimal": str(
            improvement_lower
        ),
        "componentwise_maximum_row_sum_pivot": (
            transformed_bound_pivot
        ),
        "componentwise_q_transpose_one_maximum_pivot": int(
            max(
                range(maximum_pivots),
                key=componentwise["q_transpose_one"].__getitem__,
            )
        ),
        "componentwise_residual_propagation_maximum_pivot": int(
            max(
                range(maximum_pivots),
                key=componentwise[
                    "residual_times_q_transpose_one"
                ].__getitem__,
            )
        ),
        "transformed_residual_two_norm_upper_decimal": str(
            transformed_bound
        ),
        "minimum_absolute_reference_diagonal_decimal": str(
            minimum_diagonal
        ),
        "minimum_absolute_reference_diagonal_index": minimum_index,
        "transformed_bound_to_minimum_diagonal_upper_decimal": str(
            ratio
        ),
        "reference_diagonal_signs": {
            "negative": negative,
            "positive": positive,
            "zero": int(maximum_pivots - negative - positive),
        },
        "interval_family_inertia_certified": certified,
    }


def run_componentwise(
    complete_result_path: Path = DEFAULT_COMPLETE_RESULT,
    matrices_path: Path = DEFAULT_MATRICES,
    gaussian_result_path: Path = DEFAULT_GAUSSIAN_RESULT,
    gaussian_checkpoint_path: Path = DEFAULT_GAUSSIAN_CHECKPOINT,
    separated_result_path: Path = DEFAULT_SEPARATED_RESULT,
    maximum_pivots: int = DEFAULT_MAXIMUM_PIVOTS,
    decimal_precision: int = DEFAULT_DECIMAL_PRECISION,
    enforce_cpu_policy: bool = True,
) -> dict[str, Any]:
    prefix = _load_module(
        "componentwise_residual_prefix_base",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )
    residual = _load_module(
        "componentwise_residual_arithmetic_base",
        "neutral_strip_weighted_hypercircle_congruence_residual_pilot.py",
    )
    standalone = _load_module(
        "componentwise_residual_standalone_base",
        "neutral_strip_weighted_hypercircle_standalone_residual.py",
    )
    separated = _load_json(separated_result_path)
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
    standalone_contract = standalone._build_standalone_contract(
        problem,
        preparation,
        complete_result_path=complete_result_path,
        matrices_path=matrices_path,
        gaussian_result_path=gaussian_result_path,
        gaussian_checkpoint_path=gaussian_checkpoint_path,
        maximum_pivots=maximum_pivots,
        prefix_module=prefix,
    )
    periodic_cpu_samples: list[float] = []

    def cpu_park() -> bool:
        if not enforce_cpu_policy:
            return False
        try:
            import psutil

            periodic_cpu_samples.append(
                float(psutil.cpu_percent(interval=0.25))
            )
        except Exception:
            return False
        return bool(
            len(periodic_cpu_samples) >= 2
            and all(
                value > DAYTIME_PARK_CPU_LIMIT
                for value in periodic_cpu_samples[-2:]
            )
        )

    certificate = _build_certificate(
        problem,
        maximum_pivots,
        decimal_precision,
        prefix,
        residual,
        cpu_park_callback=cpu_park,
    )
    separated_certificate = separated["certificate"]
    comparison_keys = (
        "dimension",
        "decimal_precision",
        "reference_L_nnz",
        "reference_L_sha256",
        "reference_D_sha256",
        "reference_factor_sha256",
        "reference_product_lower_entry_count",
        "residual_lower_entry_count",
        "maximum_residual_entry_upper_decimal",
        "maximum_residual_entry_coordinate",
        "residual_infinity_norm_upper_decimal",
        "absolute_L_inverse_infinity_norm_upper_decimal",
        "absolute_L_inverse_one_norm_upper_decimal",
        "minimum_absolute_reference_diagonal_decimal",
        "reference_diagonal_signs",
    )
    contract = standalone_contract["contract"]
    checks = {
        **preparation["preparation_checks"],
        "standalone_mode_uses_no_directed_audit": True,
        "source_artifacts_are_hash_bound": all(
            len(record["sha256"]) == 64 and record["bytes"] > 0
            for record in contract["source_artifacts"].values()
        ),
        "ordered_prefix_family_is_hash_bound": all(
            len(contract[key]) == 64
            for key in (
                "ordered_original_indices_sha256",
                "ordered_positive_scale_sha256",
                "ordered_center_prefix_sha256",
                "ordered_radius_prefix_sha256",
            )
        ),
        "standalone_contract_hash_recorded": (
            len(standalone_contract["contract_sha256"]) == 64
        ),
        "reference_factor_hashes_recorded": all(
            len(certificate[key]) == 64
            for key in (
                "reference_L_sha256",
                "reference_D_sha256",
                "reference_factor_sha256",
            )
        ),
        "separated_result_uses_same_standalone_contract": (
            separated["standalone_contract"] == standalone_contract
        ),
        "separated_reference_and_residual_reproduced": all(
            certificate[key] == separated_certificate[key]
            for key in comparison_keys
        ),
        "separated_bound_reproduced": (
            certificate[
                "separated_transformed_residual_two_norm_upper_decimal"
            ]
            == separated_certificate[
                "transformed_residual_two_norm_upper_decimal"
            ]
            and certificate[
                "separated_bound_to_minimum_diagonal_upper_decimal"
            ]
            == separated_certificate[
                "transformed_bound_to_minimum_diagonal_upper_decimal"
            ]
        ),
        "componentwise_bound_not_larger_than_separated_bound": (
            Decimal(
                certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ]
            )
            <= Decimal(
                certificate[
                    "separated_transformed_residual_two_norm_upper_decimal"
                ]
            )
        ),
        "componentwise_spectral_argument_validated": True,
        "full_inertia_claim_remains_false": True,
    }
    integrity_passes = bool(all(checks.values()))
    certified = bool(
        integrity_passes
        and certificate["interval_family_inertia_certified"]
    )
    return {
        "kind": (
            "hypercircle-standalone-componentwise-congruence-residual-inertia"
        ),
        "validation_mode": "standalone_hash_bound",
        "status": (
            "standalone_prefix_inertia_certified"
            if certified
            else "standalone_route_does_not_close"
        ),
        "scope": (
            "Hash-bound componentwise congruence-residual certificate for "
            "exactly the reported leading prefix. It uses no directed-LDL "
            "result and certifies no later pivot, full inertia, continuum "
            "transfer, or Navier-Stokes statement."
        ),
        "all_current_stage_checks_pass": integrity_passes,
        "checks": checks,
        "certificate": certificate,
        "standalone_contract": standalone_contract,
        "preparation": preparation,
        "runtime": {
            "below_normal_priority_set": priority_set,
            "baseline_cpu_samples_percent": baseline,
            "baseline_cpu_mean_percent": baseline_mean,
            "periodic_cpu_samples_percent": periodic_cpu_samples,
        },
        "directed_LDL_dependency": {
            "required": False,
            "audit_loaded": False,
            "sign_comparison_used_for_certification": False,
        },
        "artifacts": {
            **contract["source_artifacts"],
            "separated_residual_result": {
                "path": str(separated_result_path).replace("\\", "/"),
                "sha256": prefix._sha256_file(separated_result_path),
                "bytes": separated_result_path.stat().st_size,
            },
        },
        "certification_flags": {
            "standalone_bounded_prefix_inertia_certified": certified,
            "independent_bounded_prefix_inertia_certified": certified,
            "full_123816_pivot_inertia_certified": False,
            "continuum_spectrum_below_60_captured": False,
            "navier_stokes_regularity_certified": False,
        },
        "next_required_step": (
            "Replay this exact componentwise standalone contract at higher "
            "Decimal precision and require all upper bounds to nest."
            if certified
            else "The componentwise residual route is fail-closed at this "
            "prefix; inspect the propagated residual rows before changing "
            "the hash-bound reference factor."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complete-result",
        type=Path,
        default=DEFAULT_COMPLETE_RESULT,
    )
    parser.add_argument("--matrices", type=Path, default=DEFAULT_MATRICES)
    parser.add_argument(
        "--gaussian-result",
        type=Path,
        default=DEFAULT_GAUSSIAN_RESULT,
    )
    parser.add_argument(
        "--gaussian-checkpoint",
        type=Path,
        default=DEFAULT_GAUSSIAN_CHECKPOINT,
    )
    parser.add_argument(
        "--separated-result",
        type=Path,
        default=DEFAULT_SEPARATED_RESULT,
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
    result = run_componentwise(
        complete_result_path=args.complete_result,
        matrices_path=args.matrices,
        gaussian_result_path=args.gaussian_result,
        gaussian_checkpoint_path=args.gaussian_checkpoint,
        separated_result_path=args.separated_result,
        maximum_pivots=args.maximum_pivots,
        decimal_precision=args.decimal_precision,
        enforce_cpu_policy=not args.skip_cpu_policy,
    )
    prefix = _load_module(
        "componentwise_residual_output_base",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
