from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import mpmath
import numpy as np


WINDOW = 0.375
TERMINAL_TIME = 6.0
DEFAULT_SUBSTEP = 0.0125


def _load_certificate_module():
    script = Path(__file__).resolve().with_name(
        "neutral_strip_common_circle_source_time_slab_certificate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "common_circle_time_slab_for_roundoff",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _down(value: float) -> float:
    return float(np.nextafter(float(value), -math.inf))


def _up_array(values: np.ndarray, operation_count: int = 8) -> np.ndarray:
    nonnegative = np.maximum(np.asarray(values, dtype=float), 0.0)
    inflated = nonnegative * (1.0 + _gamma(operation_count))
    return np.nextafter(inflated, math.inf)


def _gamma(operation_count: int) -> float:
    unit = np.finfo(float).eps
    product = operation_count * unit
    if product >= 0.01:
        raise RuntimeError("roundoff operation count is too large")
    return _up(product / (1.0 - product))


def _nonnegative_matmul_upper(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    if left.shape[1] != right.shape[0]:
        raise ValueError("matrix dimensions do not align")
    operation_count = 2 * left.shape[1] + 8
    gamma = _gamma(operation_count)
    product = np.asarray(left, dtype=float) @ np.asarray(right, dtype=float)
    return np.nextafter(
        np.maximum(product, 0.0) / (1.0 - gamma),
        math.inf,
    )


def _matmul_roundoff_upper(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    absolute_product = _nonnegative_matmul_upper(
        np.abs(left),
        np.abs(right),
    )
    return _up_array(
        _gamma(2 * left.shape[1] + 8) * absolute_product,
        8,
    )


def _interval_decay(
    values: np.ndarray,
    time_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    mpmath.iv.dps = 60
    time_numerator, time_denominator = float(time_value).as_integer_ratio()
    time_interval = (
        mpmath.iv.mpf(time_numerator) / time_denominator
    )
    center = np.exp(-np.asarray(values) * time_value)
    error = np.empty_like(center)
    for index, value in enumerate(np.asarray(values)):
        numerator, denominator = float(value).as_integer_ratio()
        value_interval = mpmath.iv.mpf(numerator) / denominator
        enclosed = mpmath.iv.exp(-value_interval * time_interval)
        lower = np.nextafter(float(enclosed.a), -math.inf)
        upper = np.nextafter(float(enclosed.b), math.inf)
        error[index] = _up(
            max(float(center[index]) - lower, upper - float(center[index]))
        )
    return center, error


def _action_with_error(
    modes: np.ndarray,
    source: np.ndarray,
    decay: np.ndarray,
    decay_error: np.ndarray,
    mode_error: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scaled = modes * decay[None, :]
    central = scaled @ source
    product_error = _matmul_roundoff_upper(scaled, source)

    scaling_error = _up_array(
        np.abs(modes) * decay_error[None, :]
        + _gamma(4) * np.abs(scaled),
        12,
    )
    propagated_scaling_error = _nonnegative_matmul_upper(
        scaling_error,
        np.abs(source),
    )
    total_error = product_error + propagated_scaling_error
    if mode_error is not None:
        source_interval_magnitude = (
            np.abs(source) * (np.abs(decay) + decay_error)[:, None]
        )
        total_error += _nonnegative_matmul_upper(
            mode_error,
            source_interval_magnitude,
        )
    return central, _up_array(total_error, 16)


def _gershgorin_lower(matrix: np.ndarray) -> float:
    diagonal = np.diag(matrix)
    off_sum = np.sum(np.abs(matrix), axis=1) - np.abs(diagonal)
    return _down(float(np.min(diagonal - off_sum)))


def _gershgorin_norm_upper(matrix: np.ndarray) -> float:
    return _up(float(np.max(np.sum(np.abs(matrix), axis=1))))


def _mode_error_data(data: dict[str, object]) -> dict[str, object]:
    trace_mass = np.asarray(data["trace_mass"])
    density_modes = np.asarray(data["reference_density_modes"])
    load_modes = np.asarray(data["reference_load_modes"])
    mass_product = trace_mass @ density_modes
    mass_product_error = _matmul_roundoff_upper(
        trace_mass,
        density_modes,
    )
    subtraction_error = _up_array(
        _gamma(6) * (np.abs(mass_product) + np.abs(load_modes)),
        8,
    )
    residual_entry_upper = _up_array(
        np.abs(mass_product - load_modes)
        + mass_product_error
        + subtraction_error,
        16,
    )
    residual_frobenius_upper = _up(
        float(np.linalg.norm(residual_entry_upper, "fro"))
        * (1.0 + _gamma(residual_entry_upper.size + 8))
    )
    mass_lower = _gershgorin_lower(trace_mass)
    if mass_lower <= 0.0:
        raise RuntimeError("the boundary trace mass lower bound is not positive")
    solve_error_operator_upper = _up(
        residual_frobenius_upper / mass_lower
    )

    cross = np.asarray(data["cross_gram"])
    pushed = np.asarray(data["pushed_gram"])
    stored_cross_modes = np.asarray(data["reference_cross_density_modes"])
    stored_pushed_modes = np.asarray(data["reference_pushed_density_modes"])
    cross_product = cross @ density_modes
    pushed_product = pushed @ density_modes
    cross_mode_error = _up_array(
        np.abs(stored_cross_modes - cross_product)
        + _matmul_roundoff_upper(cross, density_modes),
        12,
    )
    pushed_mode_error = _up_array(
        np.abs(stored_pushed_modes - pushed_product)
        + _matmul_roundoff_upper(pushed, density_modes),
        12,
    )
    return {
        "boundary_mass_gershgorin_lower": mass_lower,
        "riesz_residual_frobenius_upper": residual_frobenius_upper,
        "riesz_solve_error_operator_upper": solve_error_operator_upper,
        "cross_mode_error": cross_mode_error,
        "pushed_mode_error": pushed_mode_error,
        "pushed_gram_norm_upper": _gershgorin_norm_upper(pushed),
        "maximum_cross_mode_entry_error": float(np.max(cross_mode_error)),
        "maximum_pushed_mode_entry_error": float(np.max(pushed_mode_error)),
    }


def _sum_upper(values: np.ndarray, axis: int = 0) -> np.ndarray:
    count = values.shape[axis]
    central = np.sum(np.maximum(values, 0.0), axis=axis)
    return np.nextafter(
        central / (1.0 - _gamma(2 * count + 8)),
        math.inf,
    )


def _endpoint_roundoff_row(
    data: dict[str, object],
    mode_errors: dict[str, object],
    time_value: float,
) -> dict[str, float | int | bool]:
    reference_decay, reference_decay_error = _interval_decay(
        np.asarray(data["reference_values"]),
        time_value,
    )
    modified_decay, modified_decay_error = _interval_decay(
        np.asarray(data["modified_values"]),
        time_value,
    )
    source = np.asarray(data["reference_source_modes"])
    modified_source = np.asarray(data["modified_source_modes"])

    density, density_error = _action_with_error(
        np.asarray(data["reference_density_modes"]),
        source,
        reference_decay,
        reference_decay_error,
    )
    cross_density, cross_density_error = _action_with_error(
        np.asarray(data["reference_cross_density_modes"]),
        source,
        reference_decay,
        reference_decay_error,
        np.asarray(mode_errors["cross_mode_error"]),
    )
    pushed_density, pushed_density_error = _action_with_error(
        np.asarray(data["reference_pushed_density_modes"]),
        source,
        reference_decay,
        reference_decay_error,
        np.asarray(mode_errors["pushed_mode_error"]),
    )
    modified, modified_error = _action_with_error(
        np.asarray(data["modified_load_modes"]),
        modified_source,
        modified_decay,
        modified_decay_error,
    )

    inverse_arc = float(data["inverse_arc"])
    modified_squared = inverse_arc * np.sum(modified * modified, axis=0)
    mixed = np.sum(modified * cross_density, axis=0)
    reference_squared = np.sum(density * pushed_density, axis=0)
    discrepancy_squared = (
        modified_squared - 2.0 * mixed + reference_squared
    )

    modified_roundoff = _up_array(
        _gamma(2 * modified.shape[0] + 12)
        * inverse_arc
        * _sum_upper(np.abs(modified) * np.abs(modified)),
        12,
    )
    modified_perturbation = _up_array(
        inverse_arc
        * _sum_upper(
            2.0 * np.abs(modified) * modified_error
            + modified_error * modified_error
        ),
        12,
    )
    mixed_roundoff = _up_array(
        _gamma(2 * modified.shape[0] + 12)
        * _sum_upper(np.abs(modified) * np.abs(cross_density)),
        12,
    )
    mixed_perturbation = _up_array(
        _sum_upper(
            np.abs(modified) * cross_density_error
            + np.abs(cross_density) * modified_error
            + modified_error * cross_density_error
        ),
        12,
    )
    reference_roundoff = _up_array(
        _gamma(2 * density.shape[0] + 12)
        * _sum_upper(np.abs(density) * np.abs(pushed_density)),
        12,
    )
    reference_perturbation = _up_array(
        _sum_upper(
            np.abs(density) * pushed_density_error
            + np.abs(pushed_density) * density_error
            + density_error * pushed_density_error
        ),
        12,
    )
    combination_roundoff = _up_array(
        _gamma(16)
        * (
            np.abs(modified_squared)
            + 2.0 * np.abs(mixed)
            + np.abs(reference_squared)
        ),
        12,
    )
    squared_error = _up_array(
        modified_roundoff
        + modified_perturbation
        + 2.0 * (mixed_roundoff + mixed_perturbation)
        + reference_roundoff
        + reference_perturbation
        + combination_roundoff,
        24,
    )

    state_interval_magnitude = (
        np.abs(source)
        * (np.abs(reference_decay) + reference_decay_error)[:, None]
    )
    state_norm_upper = np.linalg.norm(
        state_interval_magnitude,
        axis=0,
    ) * (1.0 + _gamma(2 * source.shape[0] + 8))
    riesz_solve_norm_error = _up_array(
        math.sqrt(float(mode_errors["pushed_gram_norm_upper"]))
        * float(mode_errors["riesz_solve_error_operator_upper"])
        * state_norm_upper,
        12,
    )

    central_norm = np.sqrt(np.maximum(discrepancy_squared, 0.0))
    upper_norm = np.nextafter(
        np.sqrt(np.maximum(discrepancy_squared + squared_error, 0.0)),
        math.inf,
    )
    upper_norm = _up_array(upper_norm + riesz_solve_norm_error, 12)
    norm_error = _up_array(
        np.maximum(upper_norm - central_norm, 0.0),
        8,
    )
    component_scale = np.sqrt(np.maximum(modified_squared, 0.0)) + np.sqrt(
        np.maximum(reference_squared, 0.0)
    )
    existing_guard = _up(
        float(data["rounding_guard_relative"])
        * max(float(np.max(component_scale)), 1.0)
    )
    worst_column = int(np.argmax(norm_error))
    maximum_error = float(norm_error[worst_column])
    return {
        "time": time_value,
        "maximum_central_column_discrepancy": float(np.max(central_norm)),
        "maximum_directed_roundoff_norm_error_upper": maximum_error,
        "existing_arithmetic_guard": existing_guard,
        "guard_margin": existing_guard - maximum_error,
        "worst_roundoff_column_index": worst_column,
        "worst_roundoff_column_central_discrepancy": float(
            central_norm[worst_column]
        ),
        "worst_roundoff_column_upper": float(upper_norm[worst_column]),
        "maximum_squared_error_upper": float(np.max(squared_error)),
        "maximum_reference_decay_interval_radius": float(
            np.max(reference_decay_error)
        ),
        "maximum_modified_decay_interval_radius": float(
            np.max(modified_decay_error)
        ),
        "guard_dominates": bool(existing_guard >= maximum_error),
    }


def _mp_exact_float(value: float):
    numerator, denominator = float(value).as_integer_ratio()
    return mpmath.mpf(numerator) / denominator


def _high_precision_spot_check(
    data: dict[str, object],
    time_value: float,
    column: int,
) -> dict[str, object]:
    mpmath.mp.dps = 80
    time_mp = _mp_exact_float(time_value)
    reference_values = np.asarray(data["reference_values"])
    modified_values = np.asarray(data["modified_values"])
    reference_decay = [
        mpmath.exp(-_mp_exact_float(value) * time_mp)
        for value in reference_values
    ]
    modified_decay = [
        mpmath.exp(-_mp_exact_float(value) * time_mp)
        for value in modified_values
    ]
    reference_modes = np.asarray(data["reference_density_modes"])
    reference_source = np.asarray(data["reference_source_modes"])
    modified_modes = np.asarray(data["modified_load_modes"])
    modified_source = np.asarray(data["modified_source_modes"])

    density = []
    for row in range(reference_modes.shape[0]):
        density.append(
            mpmath.fsum(
                _mp_exact_float(reference_modes[row, mode])
                * reference_decay[mode]
                * _mp_exact_float(reference_source[mode, column])
                for mode in range(reference_modes.shape[1])
            )
        )
    modified = []
    for row in range(modified_modes.shape[0]):
        modified.append(
            mpmath.fsum(
                _mp_exact_float(modified_modes[row, mode])
                * modified_decay[mode]
                * _mp_exact_float(modified_source[mode, column])
                for mode in range(modified_modes.shape[1])
            )
        )
    cross = np.asarray(data["cross_gram"])
    pushed = np.asarray(data["pushed_gram"])
    cross_density = [
        mpmath.fsum(
            _mp_exact_float(cross[row, index]) * density[index]
            for index in range(len(density))
            if cross[row, index] != 0.0
        )
        for row in range(cross.shape[0])
    ]
    pushed_density = [
        mpmath.fsum(
            _mp_exact_float(pushed[row, index]) * density[index]
            for index in range(len(density))
            if pushed[row, index] != 0.0
        )
        for row in range(pushed.shape[0])
    ]
    inverse_arc = _mp_exact_float(float(data["inverse_arc"]))
    squared = (
        inverse_arc * mpmath.fsum(value * value for value in modified)
        - 2
        * mpmath.fsum(
            modified[index] * cross_density[index]
            for index in range(len(modified))
        )
        + mpmath.fsum(
            density[index] * pushed_density[index]
            for index in range(len(density))
        )
    )
    norm = mpmath.sqrt(max(squared, mpmath.mpf("0")))
    return {
        "time": time_value,
        "column": column,
        "precision_decimal_digits": mpmath.mp.dps,
        "common_circle_norm_decimal": mpmath.nstr(norm, 60),
        "common_circle_norm_float": float(norm),
    }


def audit(
    spacing: float,
    mode_count: int,
    quadrature_order: int,
    eigen_cache_path: Path,
    substep: float,
) -> dict[str, object]:
    started = time.perf_counter()
    certificate = _load_certificate_module()
    priority_set = certificate._set_below_normal_priority()
    data = certificate._assemble_frozen_block(
        spacing,
        mode_count,
        quadrature_order,
        eigen_cache_path,
    )
    data["rounding_guard_relative"] = certificate.ROUNDING_GUARD_RELATIVE
    mode_errors = _mode_error_data(data)
    endpoint_count = int(round((TERMINAL_TIME - WINDOW) / substep)) + 1
    endpoint_rows = []
    for index in range(endpoint_count):
        point = round(WINDOW + index * substep, 14)
        endpoint_rows.append(
            _endpoint_roundoff_row(data, mode_errors, point)
        )

    maximum_error = max(
        float(row["maximum_directed_roundoff_norm_error_upper"])
        for row in endpoint_rows
    )
    minimum_margin = min(float(row["guard_margin"]) for row in endpoint_rows)
    worst_row = max(
        endpoint_rows,
        key=lambda row: float(
            row["maximum_directed_roundoff_norm_error_upper"]
        ),
    )
    high_precision_spot = _high_precision_spot_check(
        data,
        float(worst_row["time"]),
        int(worst_row["worst_roundoff_column_index"]),
    )
    spot_difference = abs(
        float(high_precision_spot["common_circle_norm_float"])
        - float(worst_row["worst_roundoff_column_central_discrepancy"])
    )
    high_precision_spot["central_float_absolute_difference"] = spot_difference
    high_precision_spot["covered_by_directed_roundoff_upper"] = bool(
        spot_difference
        <= float(worst_row["maximum_directed_roundoff_norm_error_upper"])
    )
    checks = [
        bool(data["cache_info"]["loaded"]),
        priority_set,
        endpoint_count == 451,
        float(mode_errors["boundary_mass_gershgorin_lower"]) > 0.0,
        float(mode_errors["riesz_solve_error_operator_upper"]) < 1.0e-9,
        all(bool(row["guard_dominates"]) for row in endpoint_rows),
        minimum_margin > 0.0,
        maximum_error < 5.0e-11,
        high_precision_spot["covered_by_directed_roundoff_upper"],
    ]
    return {
        "model": "binary-frozen common-circle endpoint roundoff audit",
        "spacing": spacing,
        "state_count": data["state_count"],
        "entry_count": data["entry_count"],
        "inner_boundary_vertex_count": data["vertex_count"],
        "retained_mode_count": data["retained_count"],
        "quadrature_order": quadrature_order,
        "substep": substep,
        "first_endpoint": WINDOW,
        "last_endpoint": TERMINAL_TIME,
        "endpoint_count": endpoint_count,
        "reference_eigensystem_cache": data["cache_info"],
        "below_normal_priority_set": priority_set,
        "exponential_interval_backend": "mpmath.iv at 60 decimal digits",
        "roundoff_model": (
            "IEEE binary64 gamma_n bounds with outward nextafter inflation"
        ),
        "mode_error_data": {
            key: value
            for key, value in mode_errors.items()
            if not isinstance(value, np.ndarray)
        },
        "endpoint_rows": endpoint_rows,
        "maximum_directed_roundoff_norm_error_upper": maximum_error,
        "minimum_existing_guard_margin": minimum_margin,
        "worst_endpoint": worst_row,
        "independent_high_precision_spot_check": high_precision_spot,
        "existing_guard_dominates_all_derived_roundoff": bool(
            minimum_margin >= 0.0
        ),
        "binary_frozen_endpoint_arithmetic_directed_enclosed": True,
        "boundary_geometry_input_treated_as_exact_binary64": True,
        "reference_modes_and_eigenvalues_treated_as_exact_binary64": True,
        "modified_projected_modes_treated_as_exact_binary64": True,
        "assembled_coefficient_matrices_interval_enclosed": False,
        "reference_generalized_eigenpairs_interval_enclosed": False,
        "continuum_Ritz_projector_error_certified": False,
        "scope": (
            "The audit encloses endpoint evaluation roundoff relative to the "
            "stored binary64 coefficient data. It does not enclose the "
            "finite-element assembly or prove that the stored eigensystems "
            "enclose the exact discrete operators."
        ),
        "all_endpoint_roundoff_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, default=0.06)
    parser.add_argument("--mode-count", type=int, default=240)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--substep", type=float, default=DEFAULT_SUBSTEP)
    parser.add_argument("--eigen-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.spacing,
        args.mode_count,
        args.quadrature_order,
        args.eigen_cache,
        args.substep,
    )
    if args.output is not None:
        _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
