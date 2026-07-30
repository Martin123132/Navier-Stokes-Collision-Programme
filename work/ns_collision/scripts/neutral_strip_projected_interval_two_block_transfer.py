from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import mpmath
import numpy as np
from scipy.sparse import coo_matrix


WINDOW = 0.375
TERMINAL_TIME = 6.0
FORM_FLOOR = 4.832287335665
FIRST_WINDOW_INTERVAL_FACTOR = 0.9523841939624662
LI_YAU_HIGH_INTERVAL_FACTOR = 0.009609366961522469
FROZEN_LOW_BLOCK_INTERVAL_FACTOR = 0.008090916167796953
DEFAULT_EIGEN_CACHE = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k241_reference_eigensystem_v1.npz"
)
DEFAULT_ASSEMBLY_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
)
DEFAULT_ASSEMBLY_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
)
DEFAULT_EIGENSYSTEM_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k240_eigensystem_residual_audit_v1.json"
)
DEFAULT_SPECTRUM_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_exact_polygon_indexed_spectrum_transfer_v1.json"
)


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _down(value: float) -> float:
    return float(np.nextafter(float(value), -math.inf))


def _gamma(operation_count: int) -> float:
    product = operation_count * np.finfo(float).eps
    if product >= 0.01:
        raise RuntimeError("roundoff operation count is too large")
    return _up(product / (1.0 - product))


def _mp_exact(value: float):
    numerator, denominator = float(value).as_integer_ratio()
    return mpmath.mpf(numerator) / denominator


def _mp_up(value) -> float:
    return _up(float(value))


def _mp_down(value) -> float:
    return _down(float(value))


def _frobenius_magnitude_upper(
    central: np.ndarray,
    entry_error: np.ndarray | None = None,
) -> float:
    magnitude = np.abs(np.asarray(central, dtype=float))
    if entry_error is not None:
        magnitude = magnitude + np.maximum(
            np.asarray(entry_error, dtype=float),
            0.0,
        )
    count = magnitude.size
    squared_sum = float(np.sum(magnitude * magnitude))
    squared_upper = _up(squared_sum / (1.0 - _gamma(2 * count + 8)))
    return _up(math.sqrt(max(squared_upper, 0.0)))


def _maximum_column_norm_upper(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    count = values.shape[0]
    squared = np.sum(np.abs(values) * np.abs(values), axis=0)
    squared = np.nextafter(
        squared / (1.0 - _gamma(2 * count + 8)),
        math.inf,
    )
    return _up(float(np.max(np.sqrt(np.maximum(squared, 0.0)))))


def _iv_bounds(value) -> tuple[float, float]:
    return (
        _down(float(value.a)),
        _up(float(value.b)),
    )


def _interval_difference_upper(
    stored: float,
    exact_interval: tuple[float, float],
) -> float:
    lower, upper = exact_interval
    return _up(max(abs(stored - lower), abs(stored - upper)))


def _circulant_template_defect(
    matrix: np.ndarray,
    diagonal: float,
    neighbor: float,
) -> float:
    size = matrix.shape[0]
    template = np.zeros_like(matrix)
    for index in range(size):
        template[index, index] = diagonal
        template[index, (index - 1) % size] = neighbor
        template[index, (index + 1) % size] = neighbor
    return _up(float(np.max(np.abs(np.asarray(matrix) - template))))


def _boundary_geometry_certificate(
    data: dict[str, object],
) -> dict[str, object]:
    mpmath.iv.dps = 100
    vertex_count = int(data["vertex_count"])
    alpha = mpmath.iv.pi / vertex_count
    arc = 2 * alpha
    edge = 2 * mpmath.iv.sin(alpha)
    cosine = mpmath.iv.cos(alpha)
    tangent = mpmath.iv.tan(alpha)
    pushed_diagonal_local = cosine**2 * (
        2 * tangent / 3 + 4 * tangent**3 / 15
    )
    pushed_neighbor = cosine**2 * (
        tangent / 3 + tangent**3 / 15
    )
    intervals = {
        "trace_diagonal": _iv_bounds(2 * edge / 3),
        "trace_neighbor": _iv_bounds(edge / 6),
        "pushed_diagonal": _iv_bounds(2 * pushed_diagonal_local),
        "pushed_neighbor": _iv_bounds(pushed_neighbor),
        "cross_diagonal": _iv_bounds((edge / arc) * mpmath.mpf(3) / 4),
        "cross_neighbor": _iv_bounds((edge / arc) / 8),
        "inverse_arc": _iv_bounds(1 / arc),
        "exact_arc": _iv_bounds(arc),
    }

    trace = np.asarray(data["trace_mass"])
    pushed = np.asarray(data["pushed_gram"])
    cross = np.asarray(data["cross_gram"])
    inverse_arc_stored = float(data["inverse_arc"])
    stored = {
        "trace_diagonal": float(trace[0, 0]),
        "trace_neighbor": float(trace[0, 1]),
        "pushed_diagonal": float(pushed[0, 0]),
        "pushed_neighbor": float(pushed[0, 1]),
        "cross_diagonal": float(cross[0, 0]),
        "cross_neighbor": float(cross[0, 1]),
        "inverse_arc": inverse_arc_stored,
    }
    errors = {
        key: _interval_difference_upper(stored[key], intervals[key])
        for key in stored
    }
    template_defects = {
        "trace": _circulant_template_defect(
            trace,
            stored["trace_diagonal"],
            stored["trace_neighbor"],
        ),
        "pushed": _circulant_template_defect(
            pushed,
            stored["pushed_diagonal"],
            stored["pushed_neighbor"],
        ),
        "cross": _circulant_template_defect(
            cross,
            stored["cross_diagonal"],
            stored["cross_neighbor"],
        ),
    }

    exact_trace_min_lower = _down(
        intervals["trace_diagonal"][0]
        - 2.0 * intervals["trace_neighbor"][1]
    )
    exact_pushed_max_upper = _up(
        intervals["pushed_diagonal"][1]
        + 2.0 * intervals["pushed_neighbor"][1]
    )
    stored_trace_min_lower = _down(
        stored["trace_diagonal"] - 2.0 * abs(stored["trace_neighbor"])
    )
    trace_difference_norm_upper = _up(
        errors["trace_diagonal"] + 2.0 * errors["trace_neighbor"]
    )
    pushed_difference_norm_upper = _up(
        errors["pushed_diagonal"] + 2.0 * errors["pushed_neighbor"]
    )
    cross_difference_norm_upper = _up(
        errors["cross_diagonal"] + 2.0 * errors["cross_neighbor"]
    )
    riesz_load_to_circle_upper = _up(
        math.sqrt(exact_pushed_max_upper) / exact_trace_min_lower
    )
    riesz_geometry_difference_upper = _up(
        math.sqrt(exact_pushed_max_upper)
        * trace_difference_norm_upper
        / (exact_trace_min_lower * stored_trace_min_lower)
    )

    mpmath.mp.dps = 100
    alpha_mp = mpmath.pi / vertex_count
    edge_mp = 2 * mpmath.sin(alpha_mp)
    cosine_mp = mpmath.cos(alpha_mp)
    tangent_mp = mpmath.tan(alpha_mp)
    pd_mp = cosine_mp**2 * (
        2 * tangent_mp / 3 + 4 * tangent_mp**3 / 15
    )
    po_mp = cosine_mp**2 * (
        tangent_mp / 3 + tangent_mp**3 / 15
    )
    ratios = []
    for index in range(vertex_count):
        theta = 2 * mpmath.pi * index / vertex_count
        mass_eigenvalue = edge_mp * (2 + mpmath.cos(theta)) / 3
        pushed_eigenvalue = 2 * pd_mp + 2 * po_mp * mpmath.cos(theta)
        ratios.append(pushed_eigenvalue / mass_eigenvalue)
    push_factor_upper = _mp_up(mpmath.sqrt(max(ratios)))

    checks = [
        vertex_count == 112,
        exact_trace_min_lower > 0.0,
        stored_trace_min_lower > 0.0,
        max(template_defects.values()) < 1.0e-300,
        push_factor_upper < 1.00004,
        riesz_load_to_circle_upper < 12.667,
        trace_difference_norm_upper < 1.0e-16,
        pushed_difference_norm_upper < 1.0e-16,
        cross_difference_norm_upper < 2.0e-16,
        errors["inverse_arc"] < 1.0e-13,
    ]
    return {
        "vertex_count": vertex_count,
        "exact_scalar_intervals": {
            key: list(value) for key, value in intervals.items()
        },
        "stored_scalar_values": stored,
        "stored_to_exact_scalar_error_uppers": errors,
        "circulant_template_defects": template_defects,
        "exact_trace_mass_minimum_eigenvalue_lower": exact_trace_min_lower,
        "stored_trace_mass_minimum_eigenvalue_lower": stored_trace_min_lower,
        "exact_pushed_gram_maximum_eigenvalue_upper": exact_pushed_max_upper,
        "trace_mass_difference_spectral_upper": trace_difference_norm_upper,
        "pushed_gram_difference_spectral_upper": (
            pushed_difference_norm_upper
        ),
        "cross_gram_difference_spectral_upper": cross_difference_norm_upper,
        "exact_riesz_load_to_circle_l2_operator_upper": (
            riesz_load_to_circle_upper
        ),
        "riesz_geometry_difference_operator_upper": (
            riesz_geometry_difference_upper
        ),
        "exact_p1_pushforward_l2_factor_upper": push_factor_upper,
        "boundary_riesz_gram_push_cross_interval_certified": bool(all(checks)),
        "checks": checks,
    }


def _two_block_bounds(
    low_floor: float,
    high_floor: float,
    coupling: float,
    time_value: float,
) -> dict[str, float]:
    if min(low_floor, high_floor, coupling, time_value) < 0.0:
        raise ValueError("two-block inputs must be nonnegative")
    mpmath.mp.dps = 100
    alpha = _mp_exact(low_floor)
    beta = _mp_exact(high_floor)
    epsilon = _mp_exact(coupling)
    t = _mp_exact(time_value)
    center = (alpha + beta) / 2
    half_gap = (beta - alpha) / 2
    root = mpmath.sqrt(half_gap**2 + epsilon**2)
    if root == 0:
        high = mpmath.mpf("0")
    else:
        high = (
            epsilon
            * mpmath.exp(-center * t)
            * mpmath.sinh(root * t)
            / root
        )
    reduced_low = mpmath.exp(-alpha * t)
    full_low_comparison = mpmath.exp(-center * t) * (
        mpmath.cosh(root * t)
        + (half_gap / root) * mpmath.sinh(root * t)
        if root != 0
        else 1
    )
    low_feedback = max(full_low_comparison - reduced_low, 0)
    return {
        "time": time_value,
        "low_floor": low_floor,
        "high_floor": high_floor,
        "coupling": coupling,
        "high_component_upper": _mp_up(high),
        "low_feedback_upper": _mp_up(low_feedback),
        "orthogonal_total_difference_upper": _mp_up(
            mpmath.sqrt(high**2 + low_feedback**2)
        ),
        "gap_free_high_component_upper": _mp_up(epsilon * t),
    }


def _time_factor_maximum(
    decay: float,
    start: float,
    end: float,
) -> float:
    critical = 1.0 / decay
    point = min(max(critical, start), end)
    return _up(point * math.exp(-decay * point))


def audit(
    eigen_cache_path: Path,
    assembly_checkpoint_path: Path,
    assembly_result_path: Path,
    eigensystem_result_path: Path,
    spectrum_result_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    certificate = _load_module(
        "neutral_strip_common_circle_source_time_slab_certificate.py",
        "projected_interval_source_certificate",
    )
    residual_module = _load_module(
        "neutral_strip_common_circle_eigensystem_residual_audit.py",
        "projected_interval_residual_helpers",
    )
    priority_set = certificate._set_below_normal_priority()
    data = certificate._assemble_frozen_block(
        0.06,
        240,
        12,
        eigen_cache_path,
    )
    assembly = _load_json(assembly_result_path)
    eigensystem = _load_json(eigensystem_result_path)
    spectrum = _load_json(spectrum_result_path)
    assembly_cache = np.load(assembly_checkpoint_path, allow_pickle=False)

    state_count = int(data["state_count"])
    vertex_count = int(data["vertex_count"])
    vectors = np.asarray(data["reference_all_vectors"])[:, :240]
    values = np.asarray(data["reference_values"])
    source_modes = np.asarray(data["reference_source_modes"])
    boundary = coo_matrix(
        (
            assembly_cache["boundary_values"],
            (
                assembly_cache["boundary_rows"],
                assembly_cache["boundary_columns"],
            ),
        ),
        shape=(state_count, vertex_count),
    ).tocsr()
    boundary.sum_duplicates()
    boundary_mass = coo_matrix(
        (
            assembly_cache["boundary_mass_values"],
            (
                assembly_cache["boundary_mass_rows"],
                assembly_cache["boundary_mass_columns"],
            ),
        ),
        shape=(state_count, vertex_count),
    ).tocsr()
    boundary_mass.sum_duplicates()

    boundary_modes, boundary_mode_error = residual_module._sparse_action_with_error(
        boundary.transpose().tocsr(),
        vectors,
    )
    boundary_mass_modes, boundary_mass_mode_error = (
        residual_module._sparse_action_with_error(
            boundary_mass.transpose().tocsr(),
            vectors,
        )
    )
    scaled_boundary_mass, scaled_boundary_mass_error = (
        residual_module._scaled_columns_with_error(
            values,
            boundary_mass_modes,
            boundary_mass_mode_error,
        )
    )
    frozen_load_modes = boundary_modes + scaled_boundary_mass
    frozen_load_error = np.nextafter(
        boundary_mode_error
        + scaled_boundary_mass_error
        + _gamma(6)
        * (np.abs(boundary_modes) + np.abs(scaled_boundary_mass)),
        math.inf,
    )
    stored_load_reconstruction_difference = _up(
        float(
            np.max(
                np.abs(
                    frozen_load_modes
                    - np.asarray(data["reference_load_modes"])
                )
            )
        )
    )
    frozen_load_norm_upper = _frobenius_magnitude_upper(
        frozen_load_modes,
        frozen_load_error,
    )
    boundary_mass_mode_norm_upper = _frobenius_magnitude_upper(
        boundary_mass_modes,
        boundary_mass_mode_error,
    )
    source_column_norm_upper = _maximum_column_norm_upper(source_modes)
    modified_load_norm_upper = _frobenius_magnitude_upper(
        np.asarray(data["modified_load_modes"])
    )
    modified_source_column_norm_upper = _maximum_column_norm_upper(
        np.asarray(data["modified_source_modes"])
    )
    reference_density_norm_upper = _frobenius_magnitude_upper(
        np.asarray(data["reference_density_modes"])
    )

    reference = eigensystem["reference_generalized_eigensystem"]
    assembly_errors = assembly["assembled_error_bounds"]
    form_bounds = assembly["form_bounds"]
    stored_gram_defect = float(
        reference["orthogonality"]["directed_frobenius_defect_upper"]
    )
    stored_residual_block = float(
        reference["inverse_mass_residual_block_frobenius_upper"]
    )
    mass_form_error = float(
        form_bounds["absolute_mass_error_relative_to_stored_mass_form"]
    )
    stiffness_form_error = float(
        form_bounds["absolute_stiffness_error_in_stored_mass_form_units"]
    )
    lambda_min_stored = float(values[0])
    lambda_max_stored = float(values[-1])
    mass_row_sum_lower = float(reference["mass_row_sum_lower_minimum"])
    mass_coercivity_lower = float(
        reference["mass_coercivity"][
            "global_row_lumped_coercivity_lower"
        ]
    )

    mpmath.mp.dps = 100
    gram_stored_mp = _mp_exact(stored_gram_defect)
    mass_error_mp = _mp_exact(mass_form_error)
    stiffness_error_mp = _mp_exact(stiffness_form_error)
    lambda_max_mp = _mp_exact(lambda_max_stored)
    gram_exact = gram_stored_mp + mass_error_mp * (1 + gram_stored_mp)
    residual_stored_mp = _mp_exact(stored_residual_block)
    residual_in_stored_metric = residual_stored_mp + mpmath.sqrt(
        1 + gram_stored_mp
    ) * (stiffness_error_mp + mass_error_mp * lambda_max_mp)
    residual_in_exact_metric = residual_in_stored_metric / mpmath.sqrt(
        1 - mass_error_mp
    )
    reduced_generator_difference = (
        mpmath.sqrt(1 + gram_exact)
        * residual_in_exact_metric
        / (1 - gram_exact)
    )
    exact_gram_defect_upper = _mp_up(gram_exact)
    exact_residual_upper = _mp_up(residual_in_exact_metric)
    generator_difference_upper = _mp_up(reduced_generator_difference)
    exact_gram_condition_sqrt_upper = _mp_up(
        mpmath.sqrt((1 + gram_exact) / (1 - gram_exact))
    )
    exact_gram_inverse_upper = _mp_up(1 / (1 - gram_exact))

    vector_euclidean_operator_upper = _mp_up(
        mpmath.sqrt(
            (1 + gram_stored_mp)
            / (
                _mp_exact(mass_coercivity_lower)
                * _mp_exact(mass_row_sum_lower)
            )
        )
    )
    boundary_error_frobenius = float(
        assembly_errors["boundary"]["frobenius_error_upper"]
    )
    boundary_mass_error_frobenius = float(
        assembly_errors["boundary_mass"]["frobenius_error_upper"]
    )
    exact_generator_norm_upper = _up(
        lambda_max_stored + generator_difference_upper
    )
    exact_load_map_difference_upper = _up(
        boundary_error_frobenius * vector_euclidean_operator_upper
        + boundary_mass_error_frobenius
        * vector_euclidean_operator_upper
        * exact_generator_norm_upper
        + boundary_mass_mode_norm_upper * generator_difference_upper
    )

    geometry = _boundary_geometry_certificate(data)
    circle_riesz_upper = float(
        geometry["exact_riesz_load_to_circle_l2_operator_upper"]
    )
    exact_decay_lower = float(
        spectrum["rows"][0][
            "exact_polygon_indexed_interval_lower"
        ]
    )
    frozen_decay = lambda_min_stored
    slow_decay = min(exact_decay_lower, frozen_decay)
    source_exact_coordinate_upper = _up(
        exact_gram_inverse_upper * source_column_norm_upper
    )
    source_coordinate_difference_upper = _up(
        exact_gram_defect_upper
        * exact_gram_inverse_upper
        * source_column_norm_upper
    )

    geometry_errors = geometry["stored_to_exact_scalar_error_uppers"]
    geometry_pushed_error = float(
        geometry["pushed_gram_difference_spectral_upper"]
    )
    geometry_cross_error = float(
        geometry["cross_gram_difference_spectral_upper"]
    )
    geometry_inverse_arc_error = float(geometry_errors["inverse_arc"])
    geometry_riesz_difference = float(
        geometry["riesz_geometry_difference_operator_upper"]
    )

    def coefficient_error_upper(time_value: float) -> dict[str, float]:
        exact_semigroup = _up(
            exact_gram_condition_sqrt_upper
            * math.exp(-exact_decay_lower * time_value)
        )
        frozen_semigroup = _up(math.exp(-frozen_decay * time_value))
        exponential_difference = _up(
            exact_gram_condition_sqrt_upper
            * generator_difference_upper
            * time_value
            * math.exp(-slow_decay * time_value)
        )
        load_term = _up(
            exact_load_map_difference_upper
            * exact_semigroup
            * source_exact_coordinate_upper
        )
        dynamics_term = _up(
            frozen_load_norm_upper
            * exponential_difference
            * source_exact_coordinate_upper
        )
        source_term = _up(
            frozen_load_norm_upper
            * frozen_semigroup
            * source_coordinate_difference_upper
        )
        coefficient_error = _up(
            circle_riesz_upper * (load_term + dynamics_term + source_term)
        )

        modified_load_upper = _up(
            modified_load_norm_upper
            * modified_source_column_norm_upper
            * math.exp(
                -float(np.min(data["modified_values"])) * time_value
            )
        )
        reference_density_upper = _up(
            reference_density_norm_upper
            * source_column_norm_upper
            * frozen_semigroup
        )
        reference_load_upper = _up(
            frozen_load_norm_upper
            * source_column_norm_upper
            * frozen_semigroup
        )
        quadratic_geometry_error = _up(
            geometry_inverse_arc_error * modified_load_upper**2
            + 2.0
            * geometry_cross_error
            * modified_load_upper
            * reference_density_upper
            + geometry_pushed_error * reference_density_upper**2
        )
        norm_evaluation_error = _up(
            math.sqrt(max(quadratic_geometry_error, 0.0))
        )
        riesz_geometry_error = _up(
            geometry_riesz_difference * reference_load_upper
        )
        geometry_error = _up(
            norm_evaluation_error + riesz_geometry_error
        )
        return {
            "time": time_value,
            "load_map_error": load_term,
            "reduced_dynamics_error": dynamics_term,
            "source_metric_error": source_term,
            "coefficient_error_common_circle_l2_upper": coefficient_error,
            "boundary_geometry_norm_error_upper": geometry_error,
            "total_projected_endpoint_error_upper": _up(
                coefficient_error + geometry_error
            ),
        }

    window_rows = []
    finite_raw_sum = 0.0
    for window_index in range(1, 16):
        start = window_index * WINDOW
        end = (window_index + 1) * WINDOW
        start_row = coefficient_error_upper(start)
        load_sup = float(start_row["load_map_error"])
        source_sup = float(start_row["source_metric_error"])
        geometry_sup = float(start_row["boundary_geometry_norm_error_upper"])
        dynamics_time_factor = _time_factor_maximum(
            slow_decay,
            start,
            end,
        )
        dynamics_sup = _up(
            frozen_load_norm_upper
            * exact_gram_condition_sqrt_upper
            * generator_difference_upper
            * dynamics_time_factor
            * source_exact_coordinate_upper
        )
        coefficient_sup = _up(
            circle_riesz_upper * (load_sup + dynamics_sup + source_sup)
        )
        endpoint_sup = _up(coefficient_sup + geometry_sup)
        axial_upper = certificate._axial_l2_global_upper(start)
        raw_upper = _up(axial_upper * endpoint_sup)
        finite_raw_sum = _up(finite_raw_sum + raw_upper)
        window_rows.append(
            {
                "window_index": window_index,
                "start": start,
                "end": end,
                "load_map_error_upper": load_sup,
                "reduced_dynamics_error_upper": dynamics_sup,
                "source_metric_error_upper": source_sup,
                "boundary_geometry_error_upper": geometry_sup,
                "projected_endpoint_error_upper": endpoint_sup,
                "axial_l2_upper": axial_upper,
                "raw_interval_error_upper": raw_upper,
            }
        )

    tail_start = coefficient_error_upper(TERMINAL_TIME)
    modified_decay = float(np.min(data["modified_values"]))
    tail_decay = min(slow_decay, modified_decay)
    tail_ratio = _up(
        (1.0 + WINDOW / TERMINAL_TIME)
        * math.exp(-tail_decay * WINDOW)
    )
    if tail_ratio >= 1.0:
        raise RuntimeError("projected endpoint tail ratio does not contract")
    tail_endpoint_sum = _up(
        float(tail_start["total_projected_endpoint_error_upper"])
        / (1.0 - tail_ratio)
    )
    tail_raw_sum = _up(
        certificate._axial_l2_global_upper(TERMINAL_TIME)
        * tail_endpoint_sum
    )
    total_raw_sum = _up(finite_raw_sum + tail_raw_sum)
    projected_interval_factor = _up(
        (WINDOW + 1.0 / FORM_FLOOR) * total_raw_sum
    )
    upgraded_low_block_factor = _up(
        FROZEN_LOW_BLOCK_INTERVAL_FACTOR + projected_interval_factor
    )
    upgraded_total_screen = _up(
        FIRST_WINDOW_INTERVAL_FACTOR
        + LI_YAU_HIGH_INTERVAL_FACTOR
        + upgraded_low_block_factor
    )

    exact_complement_floor = float(
        spectrum[
            "exact_polygon_complement_generalized_eigenvalue_lower_bound"
        ]
    )
    ritz_upper = _up(lambda_max_stored + generator_difference_upper)
    projector_gap_lower = _down(exact_complement_floor - ritz_upper)
    normalized_invariance_residual = _up(
        exact_residual_upper / math.sqrt(1.0 - exact_gram_defect_upper)
    )
    projector_sine_upper = _up(
        normalized_invariance_residual / projector_gap_lower
    )
    source_dual_norm_upper = _up(vector_euclidean_operator_upper)
    projector_source_state_error_upper = _up(
        projector_sine_upper * source_dual_norm_upper
    )

    two_block_regression_rows = [
        _two_block_bounds(2.0, 9.0, 1.0, time_value)
        for time_value in (0.125, 0.375, 0.75, 1.5)
    ]
    diagnostic_reference_floor_row = _two_block_bounds(
        float(np.min(data["modified_values"])),
        exact_complement_floor,
        6.343703098841749,
        WINDOW,
    )

    premise_hashes = {
        "eigen_cache": _sha256_file(eigen_cache_path),
        "assembly_checkpoint": _sha256_file(assembly_checkpoint_path),
        "assembly_result": _sha256_file(assembly_result_path),
        "eigensystem_result": _sha256_file(eigensystem_result_path),
        "indexed_spectrum_result": _sha256_file(spectrum_result_path),
    }
    checks = [
        priority_set,
        bool(data["cache_info"]["loaded"]),
        bool(assembly["finite_element_assembly_interval_enclosed"]),
        bool(eigensystem["all_eigensystem_residual_checks_pass"]),
        bool(
            spectrum[
                "all_exact_polygon_indexed_spectrum_transfer_checks_pass"
            ]
        ),
        stored_load_reconstruction_difference < 1.0e-300,
        bool(
            geometry[
                "boundary_riesz_gram_push_cross_interval_certified"
            ]
        ),
        exact_gram_defect_upper < 1.1e-9,
        generator_difference_upper < 6.6e-9,
        exact_load_map_difference_upper < 6.1e-9,
        projector_gap_lower > 0.6,
        projector_sine_upper < 1.1e-8,
        projected_interval_factor < 2.0e-5,
        upgraded_total_screen < 1.0,
        upgraded_total_screen < 0.97011,
        tail_ratio < 0.5,
    ]
    return {
        "model": (
            "exact-polygon projected finite-block interval transfer and "
            "two-block damped comparison"
        ),
        "spacing": 0.06,
        "retained_mode_count": 240,
        "state_count": state_count,
        "inner_boundary_vertex_count": vertex_count,
        "premise_artifacts": {
            "eigen_cache": str(eigen_cache_path),
            "assembly_checkpoint": str(assembly_checkpoint_path),
            "assembly_result": str(assembly_result_path),
            "eigensystem_result": str(eigensystem_result_path),
            "indexed_spectrum_result": str(spectrum_result_path),
            "sha256": premise_hashes,
        },
        "boundary_geometry": geometry,
        "stored_binary_reconstruction": {
            "reference_load_mode_maximum_difference": (
                stored_load_reconstruction_difference
            ),
            "reference_load_operator_frobenius_upper": (
                frozen_load_norm_upper
            ),
            "boundary_mass_mode_frobenius_upper": (
                boundary_mass_mode_norm_upper
            ),
            "reference_source_maximum_column_norm_upper": (
                source_column_norm_upper
            ),
            "modified_load_operator_frobenius_upper": (
                modified_load_norm_upper
            ),
            "modified_source_maximum_column_norm_upper": (
                modified_source_column_norm_upper
            ),
        },
        "exact_form_reduced_generator": {
            "stored_mass_gram_defect_upper": stored_gram_defect,
            "exact_mass_gram_defect_upper": exact_gram_defect_upper,
            "exact_mass_gram_inverse_upper": exact_gram_inverse_upper,
            "exact_mass_gram_condition_sqrt_upper": (
                exact_gram_condition_sqrt_upper
            ),
            "stored_inverse_mass_residual_block_frobenius_upper": (
                stored_residual_block
            ),
            "exact_inverse_mass_residual_block_upper": (
                exact_residual_upper
            ),
            "exact_reduced_generator_difference_from_frozen_diagonal_upper": (
                generator_difference_upper
            ),
            "frozen_reference_minimum_eigenvalue": lambda_min_stored,
            "frozen_reference_maximum_retained_eigenvalue": (
                lambda_max_stored
            ),
            "exact_polygon_minimum_eigenvalue_lower": exact_decay_lower,
            "exact_polygon_complement_floor": exact_complement_floor,
            "exact_reduced_ritz_maximum_upper": ritz_upper,
            "retained_complement_gap_lower": projector_gap_lower,
            "normalized_invariance_residual_upper": (
                normalized_invariance_residual
            ),
            "exact_low_projector_sine_upper": projector_sine_upper,
            "source_dual_norm_global_upper": source_dual_norm_upper,
            "exact_projector_source_state_error_global_upper": (
                projector_source_state_error_upper
            ),
            "exact_polygon_low_subspace_projector_certified": True,
            "individual_exact_eigenvectors_interval_enclosed": False,
        },
        "exact_form_boundary_and_source_transfer": {
            "stored_vector_euclidean_operator_upper": (
                vector_euclidean_operator_upper
            ),
            "exact_boundary_load_map_difference_upper": (
                exact_load_map_difference_upper
            ),
            "source_exact_coordinate_upper": source_exact_coordinate_upper,
            "source_coordinate_difference_upper": (
                source_coordinate_difference_upper
            ),
            "common_circle_projected_endpoint_rows": window_rows,
            "finite_raw_interval_error_sum_upper": finite_raw_sum,
            "tail_start": tail_start,
            "tail_ratio_upper": tail_ratio,
            "post_terminal_raw_interval_error_sum_upper": tail_raw_sum,
            "total_raw_interval_error_sum_upper": total_raw_sum,
            "added_projected_interval_factor_upper": (
                projected_interval_factor
            ),
            "frozen_low_block_interval_factor": (
                FROZEN_LOW_BLOCK_INTERVAL_FACTOR
            ),
            "upgraded_low_block_interval_factor_upper": (
                upgraded_low_block_factor
            ),
            "upgraded_complete_screen_upper": upgraded_total_screen,
            "upgraded_complete_screen_headroom_lower": _down(
                1.0 - upgraded_total_screen
            ),
            "fixed_trial_space_projected_endpoint_transfer_certified": True,
            "exact_spectral_projector_endpoint_composition_certified": False,
        },
        "two_block_damped_leakage_theorem": {
            "statement": (
                "For H=[[A,B*],[B,C]] self-adjoint with A>=alpha, "
                "C>=beta and ||B||<=epsilon, component norms are dominated "
                "by exp(t[[-alpha,epsilon],[epsilon,-beta]])."
            ),
            "regression_rows": two_block_regression_rows,
            "reference_floor_diagnostic_only": diagnostic_reference_floor_row,
            "theorem_implemented": True,
            "modified_chain_high_block_floor_certified": False,
            "reference_complement_floor_substituted_for_modified_floor": False,
            "off_block_leakage_interval_certified": False,
            "reason_leakage_remains_open": (
                "The certified 107.01775717228844 floor belongs to the exact "
                "reference polygon form. The coupling 6.343703098841749 "
                "belongs to the modified chain. A certified transfer or a "
                "direct lower bound for the modified complementary block is "
                "required before the theorem can be charged to the screen."
            ),
        },
        "finite_common_circle_riesz_gram_projected_algebra_certified": True,
        "exact_polygon_low_projector_certified": True,
        "endpoint_assembly_and_eigensystem_effect_on_fixed_trial_space_certified": (
            True
        ),
        "modified_off_block_leakage_certified": False,
        "continuum_Ritz_transfer_certified": False,
        "polygon_to_circle_domain_transfer_certified": False,
        "scope": (
            "This certificate promotes the common-circle boundary geometry, "
            "the exact-form reduced reference generator on the frozen trial "
            "space, its source metric, and the resulting added endpoint/time-"
            "slab charge. It also encloses the exact polygon low spectral "
            "subspace projector in operator angle. It does not yet compose "
            "that projector error through the boundary trace, certify the "
            "modified complementary spectral floor, transfer the finite "
            "element projector to the continuum, or compare polygon and "
            "circle domains."
        ),
        "next_required_step": (
            "Certify a lower form bound for the modified-chain complementary "
            "block, then apply the encoded two-block damped theorem together "
            "with a boundary-output smoothing estimate. After that, compose "
            "the certified exact-polygon projector source error through the "
            "trace before continuum Ritz and polygon-to-circle transfer."
        ),
        "all_projected_interval_two_block_transfer_checks_pass": bool(
            all(checks)
        ),
        "checks": checks,
        "below_normal_priority_set": priority_set,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigen-cache", type=Path, default=DEFAULT_EIGEN_CACHE)
    parser.add_argument(
        "--assembly-checkpoint",
        type=Path,
        default=DEFAULT_ASSEMBLY_CHECKPOINT,
    )
    parser.add_argument(
        "--assembly-result",
        type=Path,
        default=DEFAULT_ASSEMBLY_RESULT,
    )
    parser.add_argument(
        "--eigensystem-result",
        type=Path,
        default=DEFAULT_EIGENSYSTEM_RESULT,
    )
    parser.add_argument(
        "--spectrum-result",
        type=Path,
        default=DEFAULT_SPECTRUM_RESULT,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.eigen_cache,
        args.assembly_checkpoint,
        args.assembly_result,
        args.eigensystem_result,
        args.spectrum_result,
    )
    if args.output is not None:
        _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
