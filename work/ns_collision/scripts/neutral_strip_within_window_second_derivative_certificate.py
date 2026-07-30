#!/usr/bin/env python3
"""Certify finite-window interpolation for stored-chain boundary leakage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import mpmath
import numpy as np
from scipy.linalg import cholesky, eigh, solve, solve_triangular
from scipy.sparse import diags, eye


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EIGEN_CACHE = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k241_reference_eigensystem_v1.npz"
)
DEFAULT_TWO_BLOCK_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_modified_two_block_leakage_v1.json"
)
DEFAULT_PROJECTED_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_projected_interval_two_block_transfer_v1.json"
)
DEFAULT_COEFFICIENT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_chebyshev_scaling_coefficients_v1.json"
)
DEFAULT_RECURRENCE_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_chebyshev_recurrence_roundoff_v1.json"
)
DEFAULT_FIRST_ENDPOINT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_first_endpoint_boundary_leakage_v1.json"
)
DEFAULT_ALL_ENDPOINT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_all_endpoint_boundary_leakage_v1.json"
)
DEFAULT_GRID_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_source_grid_propagation_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_second_derivative_v1.json"
)
WINDOW = 0.375
SUBSTEP = 0.0375
SUBSTEPS_PER_WINDOW = 10
FINITE_WINDOW_COUNT = 15
DEGREE = 320
FULL_FLOOR = 1.9
REDUCED_FLOOR = 2.36
SCALING_UPPER = 8000.0
FORM_FLOOR = 4.832287335665
UNIT_ROUNDOFF = 2.0**-53


def _load_module(filename: str, module_name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_below_normal_priority() -> bool:
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return process.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        return False


def _cpu_pair() -> list[float]:
    try:
        import psutil

        return [
            float(psutil.cpu_percent(interval=1.0)),
            float(psutil.cpu_percent(interval=1.0)),
        ]
    except Exception:
        return []


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _gamma(operation_count: int) -> float:
    product = operation_count * UNIT_ROUNDOFF
    if product >= 0.01:
        raise ArithmeticError("roundoff operation count is too large")
    return float(np.nextafter(product / (1.0 - product), math.inf))


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _functional_maximum(time_value: float, floor: float) -> float:
    critical = 2.0 / time_value
    point = max(critical, floor)
    return _up(point * point * math.exp(-time_value * point))


def _functional_intervals(
    values: np.ndarray,
    time_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    mpmath.iv.dps = 70
    central = values * values * np.exp(-time_value * values)
    errors = np.empty_like(central)
    time_numerator, time_denominator = float(time_value).as_integer_ratio()
    time_interval = (
        mpmath.iv.mpf(time_numerator) / time_denominator
    )
    for index, value in enumerate(values):
        numerator, denominator = float(value).as_integer_ratio()
        exact_value = mpmath.iv.mpf(numerator) / denominator
        interval = (
            exact_value
            * exact_value
            * mpmath.iv.exp(-time_interval * exact_value)
        )
        lower = float(interval.a)
        upper = float(interval.b)
        errors[index] = _up(
            max(abs(central[index] - lower), abs(central[index] - upper))
        )
    return central, errors


def _vector_interpolation_upper(
    left_upper: float,
    right_upper: float,
    second_derivative_upper: float,
    length: float = SUBSTEP,
) -> float:
    return _up(
        max(left_upper, right_upper)
        + length * length * second_derivative_upper / 8.0
    )


def audit(
    eigen_cache: Path,
    two_block_result_path: Path,
    projected_result_path: Path,
    coefficient_result_path: Path,
    recurrence_result_path: Path,
    first_endpoint_result_path: Path,
    all_endpoint_result_path: Path,
    grid_result_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    two_block = json.loads(
        two_block_result_path.read_text(encoding="ascii")
    )
    projected = json.loads(
        projected_result_path.read_text(encoding="ascii")
    )
    coefficient = json.loads(
        coefficient_result_path.read_text(encoding="ascii")
    )
    recurrence = json.loads(
        recurrence_result_path.read_text(encoding="ascii")
    )
    first_endpoint = json.loads(
        first_endpoint_result_path.read_text(encoding="ascii")
    )
    all_endpoint = json.loads(
        all_endpoint_result_path.read_text(encoding="ascii")
    )
    grid_result = json.loads(
        grid_result_path.read_text(encoding="ascii")
    )
    if not two_block["all_modified_two_block_leakage_checks_pass"]:
        raise RuntimeError("two-block premise is not certified")
    if not coefficient["all_scaling_coefficient_checks_pass"]:
        raise RuntimeError("coefficient premise is not certified")
    if not recurrence["all_recurrence_roundoff_checks_pass"]:
        raise RuntimeError("recurrence premise is not certified")
    if not first_endpoint["all_first_endpoint_boundary_checks_pass"]:
        raise RuntimeError("first-endpoint premise is not certified")
    if not all_endpoint["all_endpoint_boundary_checks_pass"]:
        raise RuntimeError("endpoint premise is not certified")
    if not grid_result["actual_source_grid_points_certified"]:
        raise RuntimeError("source-grid premise is not certified")

    base_module = _load_module(
        "neutral_strip_modified_complement_inertia_schur_certificate.py",
        "within_window_derivative_base",
    )
    mesh_module = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "within_window_derivative_mesh",
    )
    pilot_module = _load_module(
        "neutral_strip_boundary_leakage_chebyshev_pilot.py",
        "within_window_derivative_chebyshev",
    )
    endpoint_module = _load_module(
        "neutral_strip_first_endpoint_boundary_leakage_certificate.py",
        "within_window_derivative_endpoint",
    )

    mass, stiffness, vectors, matrix_metadata = (
        base_module._assemble_modified_pencil(0.06, eigen_cache)
    )
    expected_metadata = two_block["matrix_metadata"]
    matrix_hashes_match = bool(
        matrix_metadata["mass_sha256"] == expected_metadata["mass_sha256"]
        and matrix_metadata["stiffness_sha256"]
        == expected_metadata["stiffness_sha256"]
        and matrix_metadata["retained_vectors_sha256"]
        == expected_metadata["retained_vectors_sha256"]
    )
    if not matrix_hashes_match:
        raise RuntimeError("modified-chain matrix hashes changed")

    grid = mesh_module._build_mesh(0.06)
    mass_diagonal = np.asarray(mass.diagonal(), dtype=np.float64)
    square_root_mass = np.sqrt(mass_diagonal)
    inverse_square_root_mass = 1.0 / square_root_mass
    normalized_generator = (
        diags(inverse_square_root_mass)
        @ stiffness
        @ diags(inverse_square_root_mass)
    ).tocsr()
    normalized_generator = (
        0.5
        * (
            normalized_generator
            + normalized_generator.transpose()
        )
    ).tocsr()

    entry_states = np.asarray(grid["entry_states"], dtype=np.int64)
    entry_count = len(entry_states)
    source_state = np.zeros(
        (normalized_generator.shape[0], entry_count),
        dtype=np.float64,
    )
    source_state[entry_states, np.arange(entry_count)] = (
        inverse_square_root_mass[entry_states]
    )
    inverse_arc = float(
        projected["boundary_geometry"]["stored_scalar_values"][
            "inverse_arc"
        ]
    )
    modified_boundary = (mass @ grid["inner_rate_matrix"]).tocsr()
    output_operator = (
        math.sqrt(inverse_arc)
        * modified_boundary.transpose()
        @ diags(inverse_square_root_mass)
    ).tocsr()
    output_dense_transpose = np.asarray(
        output_operator.transpose().toarray(),
        dtype=np.float64,
    )

    full_boundary_generator, full_boundary_product_error = (
        endpoint_module._sparse_action_with_error(
            normalized_generator,
            output_dense_transpose,
        )
    )
    full_source_generator, full_source_product_error = (
        endpoint_module._sparse_action_with_error(
            normalized_generator,
            source_state,
        )
    )
    boundary_generator_norm_upper = (
        endpoint_module._frobenius_upper(
            np.abs(full_boundary_generator)
        )
    )
    source_generator_norm_uppers = (
        endpoint_module._column_norms_upper(
            np.abs(full_source_generator)
        )
    )

    coefficients = np.asarray(
        [
            float(row["scipy_central"])
            for row in coefficient["coefficient_intervals"]["rows"]
        ],
        dtype=np.float64,
    )
    if len(coefficients) != DEGREE + 1:
        raise RuntimeError("degree-320 coefficient count changed")
    center = 0.5 * (SCALING_UPPER + FULL_FLOOR)
    radius = 0.5 * (SCALING_UPPER - FULL_FLOOR)
    scaled_operator = (
        normalized_generator
        - center * eye(normalized_generator.shape[0], format="csr")
    ) * (1.0 / radius)

    recurrence_operator = recurrence["operator"]
    generator_error = float(
        recurrence_operator[
            "exact_to_computational_generator_error_upper"
        ]
    )
    generator_norm_upper = float(
        recurrence_operator["computational_generator_upper"]
    )
    source_construction_error = float(
        recurrence["maximum_error_components"][
            "source_construction_error_upper"
        ]
    )
    exact_output_norm = float(
        first_endpoint["boundary_operator"]["exact_operator_norm_upper"]
    )
    output_error = float(
        first_endpoint["boundary_operator"][
            "exact_operator_difference_upper"
        ]
    )
    boundary_product_error_norm = (
        endpoint_module._frobenius_upper(
            full_boundary_product_error
        )
    )
    boundary_generator_construction_error = _up(
        generator_error * exact_output_norm
        + generator_norm_upper * output_error
        + boundary_product_error_norm
    )
    central_source_norms = endpoint_module._column_norms_upper(
        np.abs(source_state)
    )
    source_product_error_norms = (
        endpoint_module._column_norms_upper(
            full_source_product_error
        )
    )
    source_generator_construction_errors = np.nextafter(
        generator_error
        * (central_source_norms + source_construction_error)
        + generator_norm_upper * source_construction_error
        + source_product_error_norms,
        math.inf,
    )
    step_error = float(
        all_endpoint["one_step_operator_error"][
            "total_one_step_operator_error_upper"
        ]
    )
    exact_step_contraction = float(
        all_endpoint["exact_step_contraction_upper"]
    )
    computational_step_norm = float(
        all_endpoint["computational_step_operator_norm_upper"]
    )
    propagated_source_generator = full_source_generator
    full_window_derivatives: list[dict[str, float]] = []
    for window_index in range(1, FINITE_WINDOW_COUNT + 1):
        propagated_source_generator, _ = pilot_module._chebyshev_step(
            scaled_operator,
            propagated_source_generator,
            coefficients,
        )
        full_derivative_central, full_derivative_product_error = (
            endpoint_module._dense_action_with_error(
                full_boundary_generator.transpose(),
                propagated_source_generator,
            )
        )
        full_central_column_uppers = (
            endpoint_module._column_norms_upper(
                np.abs(full_derivative_central)
                + full_derivative_product_error
            )
        )
        exact_contraction = _up(
            exact_step_contraction**window_index
        )
        repeated_step_error = _up(
            window_index
            * step_error
            * computational_step_norm ** (window_index - 1)
        )
        full_model_errors = np.nextafter(
            boundary_generator_construction_error
            * exact_contraction
            * (
                source_generator_norm_uppers
                + source_generator_construction_errors
            )
            + boundary_generator_norm_upper
            * exact_contraction
            * source_generator_construction_errors
            + boundary_generator_norm_upper
            * repeated_step_error
            * source_generator_norm_uppers,
            math.inf,
        )
        full_exact_column_uppers = np.nextafter(
            full_central_column_uppers + full_model_errors,
            math.inf,
        )
        full_window_derivatives.append(
            {
                "window_index": window_index,
                "time": window_index * WINDOW,
                "maximum_central_column_upper": _up(
                    float(np.max(full_central_column_uppers))
                ),
                "repeated_semigroup_operator_error_upper": (
                    repeated_step_error
                ),
                "maximum_model_error_upper": _up(
                    float(np.max(full_model_errors))
                ),
                "exact_column_maximum_upper": _up(
                    float(np.max(full_exact_column_uppers))
                ),
            }
        )

    restricted_mass = vectors.transpose() @ (mass @ vectors)
    restricted_mass = 0.5 * (
        restricted_mass + restricted_mass.transpose()
    )
    restricted_stiffness = vectors.transpose() @ (stiffness @ vectors)
    restricted_stiffness = 0.5 * (
        restricted_stiffness + restricted_stiffness.transpose()
    )
    lower = cholesky(restricted_mass, lower=True)
    state_trial = square_root_mass[:, None] * vectors
    orthonormal_trial = solve_triangular(
        lower,
        state_trial.transpose(),
        lower=True,
    ).transpose()
    transformed = solve_triangular(
        lower,
        restricted_stiffness,
        lower=True,
    )
    transformed = solve_triangular(
        lower,
        transformed.transpose(),
        lower=True,
    ).transpose()
    transformed = 0.5 * (transformed + transformed.transpose())
    reduced_values, reduced_vectors = eigh(transformed)
    source_rhs = vectors[entry_states, :].transpose()
    source_coordinates = solve_triangular(
        lower,
        source_rhs,
        lower=True,
    )
    reduced_output, reduced_output_product_error = (
        endpoint_module._sparse_action_with_error(
            output_operator,
            orthonormal_trial,
        )
    )

    first_action, first_action_error = (
        endpoint_module._dense_action_with_error(
            reduced_vectors.transpose(),
            source_coordinates,
        )
    )
    reduced_transfer = first_endpoint["reduced_galerkin_transfer"]
    trial_solve_error = float(
        reduced_transfer["trial_triangular_solve_error_upper"]
    )
    source_solve_error = float(
        reduced_transfer["source_triangular_solve_error_upper"]
    )
    eig_generator_error = float(
        reduced_transfer["eigensystem_generator_error_upper"]
    )
    eig_gram = reduced_vectors.transpose() @ reduced_vectors
    eig_orthogonality_error = endpoint_module._frobenius_upper(
        np.abs(eig_gram - np.eye(len(reduced_values)))
        + endpoint_module._dense_product_error(
            reduced_vectors.transpose(),
            reduced_vectors,
        )
    )
    approximate_generator_norm = _up(
        float(np.max(np.abs(reduced_values)))
    )
    eig_exact_generator_norm = _up(
        approximate_generator_norm + eig_generator_error
    )
    reduced_output_norm_upper = endpoint_module._frobenius_upper(
        np.abs(reduced_output)
    )
    reduced_output_product_error_norm = (
        endpoint_module._frobenius_upper(
            reduced_output_product_error
        )
    )
    reduced_output_model_error = _up(
        reduced_output_product_error_norm
        + float(
            first_endpoint["boundary_operator"][
                "central_spectral_norm"
            ]
        )
        * trial_solve_error
    )
    source_coordinate_norm_upper = _up(
        float(
            np.max(
                endpoint_module._column_norms_upper(
                    np.abs(source_coordinates)
                )
            )
        )
    )
    parameters = two_block["certified_parameters"]
    exact_mass_minimum = float(
        parameters["restricted_mass"]["minimum_lower"]
    )
    exact_mass_maximum = float(
        parameters["restricted_mass"]["maximum_upper"]
    )
    form_condition = _up(
        math.sqrt(exact_mass_maximum / exact_mass_minimum)
    )
    exact_reduced_generator_error = float(
        reduced_transfer["exact_reduced_generator_error_upper"]
    )
    central_form_generator_norm = _up(
        form_condition
        * (
            approximate_generator_norm
            + eig_generator_error
        )
    )
    exact_form_generator_norm = _up(
        central_form_generator_norm
        + exact_reduced_generator_error
    )
    source_coefficients = solve(
        restricted_mass,
        source_rhs,
        assume_a="pos",
    )
    source_coefficient_norm_upper = _up(
        float(np.max(np.linalg.norm(source_coefficients, axis=0)))
    )
    source_coordinate_error = float(
        reduced_transfer["source_coordinate_error_upper"]
    )
    exact_source_coefficient_norm = _up(
        source_coefficient_norm_upper + source_coordinate_error
    )
    coefficient_output, coefficient_output_error = (
        endpoint_module._sparse_action_with_error(
            output_operator,
            state_trial,
        )
    )
    coefficient_output_norm_upper = (
        endpoint_module._frobenius_upper(
            np.abs(coefficient_output)
            + coefficient_output_error
        )
    )
    state_trial_construction_error = _up(
        _gamma(8) * float(np.linalg.norm(state_trial, "fro"))
    )
    coefficient_output_form_error = _up(
        output_error * math.sqrt(exact_mass_maximum)
        + float(
            first_endpoint["boundary_operator"][
                "central_spectral_norm"
            ]
        )
        * state_trial_construction_error
    )
    reduced_window_derivatives: list[dict[str, float]] = []
    for window_index in range(1, FINITE_WINDOW_COUNT + 1):
        time_value = window_index * WINDOW
        functional_values, functional_value_errors = (
            _functional_intervals(
                reduced_values,
                time_value,
            )
        )
        scaled_action = functional_values[:, None] * first_action
        scaled_action_error = np.nextafter(
            np.abs(functional_values)[:, None] * first_action_error
            + functional_value_errors[:, None] * np.abs(first_action)
            + functional_value_errors[:, None] * first_action_error
            + _gamma(6) * np.abs(scaled_action),
            math.inf,
        )
        second_action, second_action_error = (
            endpoint_module._dense_action_with_error(
                reduced_vectors,
                scaled_action,
                scaled_action_error,
            )
        )
        reduced_derivative_central, reduced_derivative_error = (
            endpoint_module._dense_action_with_error(
                reduced_output,
                second_action,
                second_action_error,
            )
        )
        reduced_numeric_column_uppers = (
            endpoint_module._column_norms_upper(
                np.abs(reduced_derivative_central)
                + reduced_derivative_error
            )
        )
        reduced_numeric_upper = _up(
            float(np.max(reduced_numeric_column_uppers))
        )

        functional_global_upper = _functional_maximum(
            time_value,
            REDUCED_FLOOR,
        )
        eig_functional_orthogonality_error = _up(
            functional_global_upper
            * eig_orthogonality_error
            / (1.0 - eig_orthogonality_error)
        )
        eig_semigroup_difference = _up(
            time_value
            * eig_generator_error
            * math.exp(-REDUCED_FLOOR * time_value)
        )
        eig_functional_error = _up(
            eig_exact_generator_norm**2
            * eig_semigroup_difference
            + eig_generator_error
            * (
                eig_exact_generator_norm
                + approximate_generator_norm
            )
            * math.exp(-REDUCED_FLOOR * time_value)
            + eig_functional_orthogonality_error
        )
        eigen_path_model_error = _up(
            reduced_output_model_error
            * functional_global_upper
            * (source_coordinate_norm_upper + source_solve_error)
            + reduced_output_norm_upper
            * eig_functional_error
            * (source_coordinate_norm_upper + source_solve_error)
            + reduced_output_norm_upper
            * (
                functional_global_upper
                + eig_functional_orthogonality_error
            )
            * source_solve_error
        )
        central_reduced_upper = _up(
            reduced_numeric_upper + eigen_path_model_error
        )

        semigroup_generator_difference = _up(
            time_value
            * form_condition**2
            * exact_reduced_generator_error
            * math.exp(-REDUCED_FLOOR * time_value)
        )
        central_form_semigroup_norm = _up(
            form_condition
            * math.exp(-REDUCED_FLOOR * time_value)
        )
        form_functional_difference = _up(
            exact_form_generator_norm**2
            * semigroup_generator_difference
            + exact_reduced_generator_error
            * (
                exact_form_generator_norm
                + central_form_generator_norm
            )
            * central_form_semigroup_norm
        )
        exact_form_functional_norm = _up(
            form_condition * functional_global_upper
        )
        central_form_functional_norm = _up(
            form_condition * functional_global_upper
        )
        exact_form_transfer_error = _up(
            coefficient_output_form_error
            * exact_form_functional_norm
            * exact_source_coefficient_norm
            + coefficient_output_norm_upper
            * form_functional_difference
            * exact_source_coefficient_norm
            + coefficient_output_norm_upper
            * central_form_functional_norm
            * source_coordinate_error
        )
        reduced_exact_upper = _up(
            central_reduced_upper + exact_form_transfer_error
        )
        reduced_window_derivatives.append(
            {
                "window_index": window_index,
                "time": time_value,
                "maximum_numeric_column_upper": reduced_numeric_upper,
                "eigen_path_model_error_upper": (
                    eigen_path_model_error
                ),
                "central_reduced_column_maximum_upper": (
                    central_reduced_upper
                ),
                "eigensystem_functional_error_upper": (
                    eig_functional_error
                ),
                "exact_form_functional_difference_upper": (
                    form_functional_difference
                ),
                "exact_form_transfer_error_upper": (
                    exact_form_transfer_error
                ),
                "exact_column_maximum_upper": reduced_exact_upper,
            }
        )

    grid_rows = grid_result["grid_rows"]
    if len(grid_rows) != 151:
        raise RuntimeError("source grid no longer has 151 points")
    window_rows: list[dict[str, Any]] = []
    maximum_interpolation_charge = 0.0
    finite_raw_sum = 0.0
    for window_index in range(1, FINITE_WINDOW_COUNT + 1):
        start = window_index * WINDOW
        full_derivative_upper = float(
            full_window_derivatives[window_index - 1][
                "exact_column_maximum_upper"
            ]
        )
        reduced_derivative_upper = float(
            reduced_window_derivatives[window_index - 1][
                "exact_column_maximum_upper"
            ]
        )
        second_derivative_upper = _up(
            full_derivative_upper + reduced_derivative_upper
        )
        subslab_rows = []
        window_boundary_upper = 0.0
        window_raw_upper = 0.0
        for subslab in range(SUBSTEPS_PER_WINDOW):
            left_index = (
                (window_index - 1) * SUBSTEPS_PER_WINDOW + subslab
            )
            left = grid_rows[left_index]
            right = grid_rows[left_index + 1]
            interpolation_charge = _up(
                SUBSTEP
                * SUBSTEP
                * second_derivative_upper
                / 8.0
            )
            boundary_upper = _vector_interpolation_upper(
                float(left["maximum_boundary_l2_difference_upper"]),
                float(right["maximum_boundary_l2_difference_upper"]),
                second_derivative_upper,
            )
            raw_upper = _up(
                float(left["axial_l2_upper"]) * boundary_upper
            )
            maximum_interpolation_charge = max(
                maximum_interpolation_charge,
                interpolation_charge,
            )
            window_boundary_upper = max(
                window_boundary_upper,
                boundary_upper,
            )
            window_raw_upper = max(window_raw_upper, raw_upper)
            subslab_rows.append(
                {
                    "subslab_index": subslab + 1,
                    "start": float(left["time"]),
                    "end": float(right["time"]),
                    "left_boundary_upper": float(
                        left["maximum_boundary_l2_difference_upper"]
                    ),
                    "right_boundary_upper": float(
                        right["maximum_boundary_l2_difference_upper"]
                    ),
                    "interpolation_charge_upper": interpolation_charge,
                    "boundary_supremum_upper": boundary_upper,
                    "axial_l2_upper": float(left["axial_l2_upper"]),
                    "raw_source_discrepancy_upper": raw_upper,
                }
            )
        finite_raw_sum = _up(finite_raw_sum + window_raw_upper)
        window_rows.append(
            {
                "window_index": window_index,
                "start": start,
                "end": (window_index + 1) * WINDOW,
                "full_second_derivative_upper": full_derivative_upper,
                "reduced_second_derivative_upper": (
                    reduced_derivative_upper
                ),
                "difference_second_derivative_upper": (
                    second_derivative_upper
                ),
                "maximum_boundary_supremum_upper": (
                    window_boundary_upper
                ),
                "raw_source_discrepancy_upper": window_raw_upper,
                "subslabs": subslab_rows,
            }
        )

    finite_screen_charge = _up(
        (WINDOW + 1.0 / FORM_FLOOR) * finite_raw_sum
    )
    existing_screen = float(
        projected["exact_form_boundary_and_source_transfer"][
            "upgraded_complete_screen_upper"
        ]
    )
    finite_combined_screen = _up(
        existing_screen + finite_screen_charge
    )
    derivative_rows_decrease = all(
        window_rows[index][
            "difference_second_derivative_upper"
        ]
        < window_rows[index - 1][
            "difference_second_derivative_upper"
        ]
        for index in range(1, len(window_rows))
    )
    first_full_derivative = full_window_derivatives[0]
    first_reduced_derivative = reduced_window_derivatives[0]
    difference_first_upper = _up(
        float(first_full_derivative["exact_column_maximum_upper"])
        + float(first_reduced_derivative["exact_column_maximum_upper"])
    )
    checks = [
        priority_set,
        matrix_hashes_match,
        len(coefficients) == DEGREE + 1,
        len(grid_rows) == 151,
        entry_count == 112,
        all(
            abs(
                float(row["time"])
                - (index + SUBSTEPS_PER_WINDOW) * SUBSTEP
            )
            < 1.0e-12
            for index, row in enumerate(grid_rows)
        ),
        len(full_window_derivatives) == FINITE_WINDOW_COUNT,
        len(reduced_window_derivatives) == FINITE_WINDOW_COUNT,
        all(
            row["exact_column_maximum_upper"] > 0.0
            for row in full_window_derivatives
        ),
        all(
            row["exact_column_maximum_upper"] > 0.0
            for row in reduced_window_derivatives
        ),
        len(window_rows) == FINITE_WINDOW_COUNT,
        all(len(row["subslabs"]) == SUBSTEPS_PER_WINDOW for row in window_rows),
        finite_combined_screen < 1.0,
    ]
    premise_artifacts = {
        "eigen_cache": str(eigen_cache),
        "eigen_cache_sha256": _sha256_file(eigen_cache),
        "two_block_result": str(two_block_result_path),
        "two_block_result_sha256": _sha256_file(two_block_result_path),
        "projected_result": str(projected_result_path),
        "projected_result_sha256": _sha256_file(projected_result_path),
        "coefficient_result": str(coefficient_result_path),
        "coefficient_result_sha256": _sha256_file(
            coefficient_result_path
        ),
        "recurrence_result": str(recurrence_result_path),
        "recurrence_result_sha256": _sha256_file(
            recurrence_result_path
        ),
        "first_endpoint_result": str(first_endpoint_result_path),
        "first_endpoint_result_sha256": _sha256_file(
            first_endpoint_result_path
        ),
        "all_endpoint_result": str(all_endpoint_result_path),
        "all_endpoint_result_sha256": _sha256_file(
            all_endpoint_result_path
        ),
        "grid_result": str(grid_result_path),
        "grid_result_sha256": _sha256_file(grid_result_path),
    }
    return {
        "kind": (
            "neutral_strip_within_window_second_derivative_certificate"
        ),
        "model": (
            "direct second-derivative enclosure for the exact stored "
            "finite-chain boundary discrepancy on fifteen 3/8 windows"
        ),
        "below_normal_priority_set": priority_set,
        "premise_artifacts": premise_artifacts,
        "matrix_hashes_match": matrix_hashes_match,
        "interpolation_theorem": {
            "statement": (
                "For a Hilbert-valued C2 function on an interval of "
                "length h, sup norm is at most the larger endpoint norm "
                "plus h^2/8 times the supremum of the second-derivative "
                "norm."
            ),
            "substep": SUBSTEP,
            "charge_multiplier": _up(SUBSTEP * SUBSTEP / 8.0),
        },
        "full_first_window_derivative": {
            **first_full_derivative,
            "boundary_generator_frobenius_norm_upper": (
                boundary_generator_norm_upper
            ),
            "maximum_source_generator_norm_upper": _up(
                float(np.max(source_generator_norm_uppers))
            ),
            "boundary_generator_construction_error_upper": (
                boundary_generator_construction_error
            ),
            "maximum_source_generator_construction_error_upper": _up(
                float(np.max(source_generator_construction_errors))
            ),
            "semigroup_step_operator_error_upper": step_error,
            "full_generator_floor_lower": FULL_FLOOR,
        },
        "full_window_derivative_rows": full_window_derivatives,
        "reduced_first_window_derivative": {
            **first_reduced_derivative,
            "reduced_generator_floor_lower": REDUCED_FLOOR,
        },
        "reduced_window_derivative_rows": reduced_window_derivatives,
        "first_window_difference_second_derivative_upper": (
            difference_first_upper
        ),
        "window_rows": window_rows,
        "direct_derivative_rows_strictly_decrease": (
            derivative_rows_decrease
        ),
        "maximum_interpolation_charge_upper": (
            maximum_interpolation_charge
        ),
        "finite_raw_source_discrepancy_sum_upper": finite_raw_sum,
        "finite_screen_charge_upper": finite_screen_charge,
        "existing_certified_screen_upper": existing_screen,
        "finite_window_combined_screen_upper": finite_combined_screen,
        "finite_window_combined_screen_below_one": bool(
            finite_combined_screen < 1.0
        ),
        "all_fifteen_within_window_suprema_certified": bool(
            all(checks)
        ),
        "post_terminal_time_tail_certified": False,
        "screen_updated": False,
        "resource_samples_after_atomic_action": _cpu_pair(),
        "checks": checks,
        "all_second_derivative_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Use the certified t=6 endpoint as the anchor for a separate "
            "post-time-6 boundary-discrepancy tail, then combine that tail "
            "with this finite-window charge before updating the screen."
        ),
        "scope": (
            "This certificate closes interpolation only for the stored "
            "binary finite chain from t=3/8 through t=6. It does not "
            "certify the post-t=6 tail, continuum transfer, circle-domain "
            "transfer, or Navier-Stokes regularity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigen-cache", type=Path, default=DEFAULT_EIGEN_CACHE)
    parser.add_argument(
        "--two-block-result",
        type=Path,
        default=DEFAULT_TWO_BLOCK_RESULT,
    )
    parser.add_argument(
        "--projected-result",
        type=Path,
        default=DEFAULT_PROJECTED_RESULT,
    )
    parser.add_argument(
        "--coefficient-result",
        type=Path,
        default=DEFAULT_COEFFICIENT_RESULT,
    )
    parser.add_argument(
        "--recurrence-result",
        type=Path,
        default=DEFAULT_RECURRENCE_RESULT,
    )
    parser.add_argument(
        "--first-endpoint-result",
        type=Path,
        default=DEFAULT_FIRST_ENDPOINT_RESULT,
    )
    parser.add_argument(
        "--all-endpoint-result",
        type=Path,
        default=DEFAULT_ALL_ENDPOINT_RESULT,
    )
    parser.add_argument(
        "--grid-result",
        type=Path,
        default=DEFAULT_GRID_RESULT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = audit(
        arguments.eigen_cache,
        arguments.two_block_result,
        arguments.projected_result,
        arguments.coefficient_result,
        arguments.recurrence_result,
        arguments.first_endpoint_result,
        arguments.all_endpoint_result,
        arguments.grid_result,
    )
    _atomic_json(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_second_derivative_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
