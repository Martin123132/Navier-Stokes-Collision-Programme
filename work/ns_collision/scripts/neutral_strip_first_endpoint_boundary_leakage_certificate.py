#!/usr/bin/env python3
"""Certify the first stored-chain boundary-leakage endpoint."""

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
from scipy.sparse import diags


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
DEFAULT_RECURRENCE_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_chebyshev_recurrence_roundoff_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_first_endpoint_boundary_leakage_v1.json"
)
WINDOW = 0.375
LOW_FLOOR = 2.36
RETAINED_COUNT = 240
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
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
    return float(
        np.nextafter(product / (1.0 - product), math.inf)
    )


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _down(value: float) -> float:
    return float(np.nextafter(float(value), -math.inf))


def _frobenius_upper(entry_magnitude_upper: np.ndarray) -> float:
    flat = np.asarray(entry_magnitude_upper, dtype=np.float64).ravel()
    squared = float(np.sum(flat * flat))
    squared = _up(squared / (1.0 - _gamma(2 * len(flat) + 8)))
    return _up(math.sqrt(max(squared, 0.0)))


def _column_norms_upper(entry_magnitude_upper: np.ndarray) -> np.ndarray:
    rows = entry_magnitude_upper.shape[0]
    squared = np.sum(entry_magnitude_upper**2, axis=0)
    squared = np.nextafter(
        squared / (1.0 - _gamma(2 * rows + 8)),
        math.inf,
    )
    return np.nextafter(np.sqrt(np.maximum(squared, 0.0)), math.inf)


def _dense_product_error(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    terms = left.shape[1]
    absolute_product = np.abs(left) @ np.abs(right)
    return np.nextafter(
        _gamma(2 * terms + 8) * absolute_product,
        math.inf,
    )


def _dense_action_with_error(
    matrix: np.ndarray,
    vectors: np.ndarray,
    input_error: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    central = matrix @ vectors
    error = _dense_product_error(matrix, vectors)
    if input_error is not None:
        error = np.nextafter(
            error
            + (
                np.abs(matrix) @ input_error
            ) / (1.0 - _gamma(2 * matrix.shape[1] + 8)),
            math.inf,
        )
    return central, error


def _sparse_action_with_error(
    matrix,
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    maximum_terms = int(np.max(np.diff(matrix.indptr)))
    central = np.asarray(matrix @ vectors)
    absolute_product = np.asarray(abs(matrix) @ np.abs(vectors))
    error = np.nextafter(
        _gamma(2 * maximum_terms + 8) * absolute_product,
        math.inf,
    )
    return central, error


def _decay_intervals(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mpmath.iv.dps = 70
    central = np.exp(-WINDOW * values)
    errors = np.empty_like(central)
    time_interval = mpmath.iv.mpf(["0.375", "0.375"])
    for index, value in enumerate(values):
        numerator, denominator = float(value).as_integer_ratio()
        exact_value = mpmath.iv.mpf(numerator) / denominator
        interval = mpmath.iv.exp(-time_interval * exact_value)
        lower = _down(float(interval.a))
        upper = _up(float(interval.b))
        errors[index] = _up(
            max(abs(central[index] - lower), abs(central[index] - upper))
        )
    return central, errors


def audit(
    eigen_cache: Path,
    two_block_result_path: Path,
    projected_result_path: Path,
    recurrence_result_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    two_block = json.loads(
        two_block_result_path.read_text(encoding="ascii")
    )
    projected = json.loads(
        projected_result_path.read_text(encoding="ascii")
    )
    recurrence = json.loads(
        recurrence_result_path.read_text(encoding="ascii")
    )
    if not two_block["all_modified_two_block_leakage_checks_pass"]:
        raise RuntimeError("two-block premise is not certified")
    if not recurrence["all_recurrence_roundoff_checks_pass"]:
        raise RuntimeError("recurrence premise is not certified")
    cache_path = Path(recurrence["computed_state_cache"]["path"])
    if (
        _sha256_file(cache_path)
        != recurrence["computed_state_cache"]["sha256"]
    ):
        raise RuntimeError("computed recurrence state cache hash changed")
    with np.load(cache_path, allow_pickle=False) as cached:
        full_state = np.asarray(
            cached["computed_one_step_state"], dtype=np.float64
        )
        source_state = np.asarray(cached["source_state"], dtype=np.float64)
        cached_entry_states = np.asarray(
            cached["entry_states"], dtype=np.int64
        )
    if (
        _sha256_array(full_state)
        != recurrence["computed_one_step_state_sha256"]
    ):
        raise RuntimeError("computed recurrence state array hash changed")

    base_module = _load_module(
        "neutral_strip_modified_complement_inertia_schur_certificate.py",
        "first_endpoint_boundary_base",
    )
    mesh_module = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "first_endpoint_boundary_mesh",
    )
    mass, stiffness, vectors, matrix_metadata = (
        base_module._assemble_modified_pencil(0.06, eigen_cache)
    )
    grid = mesh_module._build_mesh(0.06)
    entry_states = np.asarray(grid["entry_states"], dtype=np.int64)
    if not np.array_equal(entry_states, cached_entry_states):
        raise RuntimeError("entry-state ordering changed")
    mass_diagonal = np.asarray(mass.diagonal(), dtype=np.float64)
    square_root_mass = np.sqrt(mass_diagonal)
    inverse_square_root_mass = 1.0 / square_root_mass
    state_trial = square_root_mass[:, None] * vectors

    restricted_mass = vectors.transpose() @ (mass @ vectors)
    restricted_mass = 0.5 * (
        restricted_mass + restricted_mass.transpose()
    )
    restricted_stiffness = vectors.transpose() @ (stiffness @ vectors)
    restricted_stiffness = 0.5 * (
        restricted_stiffness + restricted_stiffness.transpose()
    )
    lower = cholesky(restricted_mass, lower=True)
    orthonormal_trial = solve_triangular(
        lower, state_trial.transpose(), lower=True
    ).transpose()
    transformed = solve_triangular(
        lower, restricted_stiffness, lower=True
    )
    transformed = solve_triangular(
        lower, transformed.transpose(), lower=True
    ).transpose()
    transformed = 0.5 * (transformed + transformed.transpose())
    reduced_values, reduced_vectors = eigh(transformed)
    source_rhs = vectors[entry_states, :].transpose()
    source_coordinates = solve_triangular(
        lower, source_rhs, lower=True
    )
    decay, decay_error = _decay_intervals(reduced_values)

    first_action, first_action_error = _dense_action_with_error(
        reduced_vectors.transpose(), source_coordinates
    )
    scaled_action = decay[:, None] * first_action
    scaled_action_error = np.nextafter(
        np.abs(decay)[:, None] * first_action_error
        + decay_error[:, None] * np.abs(first_action)
        + decay_error[:, None] * first_action_error
        + _gamma(6) * np.abs(scaled_action),
        math.inf,
    )
    second_action, second_action_error = _dense_action_with_error(
        reduced_vectors, scaled_action, scaled_action_error
    )
    reduced_state, reduced_evaluation_error = _dense_action_with_error(
        orthonormal_trial, second_action, second_action_error
    )
    reduced_evaluation_error_norms = _column_norms_upper(
        reduced_evaluation_error
    )

    parameters = two_block["certified_parameters"]
    restricted_mass_error = float(
        parameters["restricted_mass_projection_error_upper"]
    )
    restricted_stiffness_error = float(
        parameters["restricted_stiffness_projection_error_upper"]
    )
    exact_mass_minimum = float(
        parameters["restricted_mass"]["minimum_lower"]
    )
    exact_mass_maximum = float(
        parameters["restricted_mass"]["maximum_upper"]
    )
    factor_product = lower @ lower.transpose()
    factor_product_error = _dense_product_error(
        lower, lower.transpose()
    )
    cholesky_residual_upper = _frobenius_upper(
        np.abs(restricted_mass - factor_product)
        + factor_product_error
    )
    mass_input_error = _up(
        restricted_mass_error + cholesky_residual_upper
    )

    projected_generator = solve(
        restricted_mass, restricted_stiffness, assume_a="pos"
    )
    projected_generator_norm = _up(
        float(np.linalg.norm(projected_generator, 2))
    )
    exact_generator_error = _up(
        (
            restricted_stiffness_error
            + mass_input_error * projected_generator_norm
        )
        / exact_mass_minimum
    )
    source_coefficients = solve(
        restricted_mass, source_rhs, assume_a="pos"
    )
    source_residual = source_rhs - restricted_mass @ source_coefficients
    source_product_error = _dense_product_error(
        restricted_mass, source_coefficients
    )
    source_residual_norms = _column_norms_upper(
        np.abs(source_residual) + source_product_error
    )
    source_coefficient_norms = np.linalg.norm(
        source_coefficients, axis=0
    )
    source_coordinate_error = np.nextafter(
        (
            source_residual_norms
            + mass_input_error * source_coefficient_norms
        )
        / exact_mass_minimum,
        math.inf,
    )
    exact_source_coordinate_norms = np.nextafter(
        source_coefficient_norms + source_coordinate_error,
        math.inf,
    )

    exact_condition = _up(
        math.sqrt(exact_mass_maximum / exact_mass_minimum)
    )
    central_condition = _up(
        math.sqrt(
            float(np.linalg.eigvalsh(restricted_mass)[-1])
            / float(np.linalg.eigvalsh(restricted_mass)[0])
        )
    )
    semigroup_difference = _up(
        WINDOW
        * exact_condition
        * central_condition
        * exact_generator_error
        * math.exp(-LOW_FLOOR * WINDOW)
    )
    central_semigroup_norm = _up(
        central_condition * math.exp(-LOW_FLOOR * WINDOW)
    )
    exact_state_trial_norm = _up(math.sqrt(exact_mass_maximum))
    state_trial_construction_error = _up(
        _gamma(8) * float(np.linalg.norm(state_trial, "fro"))
    )
    reduced_form_transfer_error = np.nextafter(
        exact_state_trial_norm
        * (
            semigroup_difference * exact_source_coordinate_norms
            + central_semigroup_norm * source_coordinate_error
        )
        + state_trial_construction_error
        * central_semigroup_norm
        * source_coefficient_norms,
        math.inf,
    )

    sigma_lower = _down(
        math.sqrt(float(np.linalg.eigvalsh(restricted_mass)[0]))
    )
    trial_solve_residual = (
        state_trial - orthonormal_trial @ lower.transpose()
    )
    trial_solve_error = _up(
        _frobenius_upper(
            np.abs(trial_solve_residual)
            + _dense_product_error(
                orthonormal_trial, lower.transpose()
            )
        )
        / sigma_lower
    )
    source_solve_residual = source_rhs - lower @ source_coordinates
    source_solve_error = np.nextafter(
        _column_norms_upper(
            np.abs(source_solve_residual)
            + _dense_product_error(lower, source_coordinates)
        )
        / sigma_lower,
        math.inf,
    )
    transformed_factor_product = (
        lower @ transformed @ lower.transpose()
    )
    transformed_residual = (
        restricted_stiffness - transformed_factor_product
    )
    transformed_solve_error = _up(
        _frobenius_upper(np.abs(transformed_residual))
        / (sigma_lower**2)
    )
    reconstructed_transformed = (
        reduced_vectors * reduced_values[None, :]
    ) @ reduced_vectors.transpose()
    eig_reconstruction_error = _up(
        _frobenius_upper(
            np.abs(transformed - reconstructed_transformed)
            + _dense_product_error(
                reduced_vectors * reduced_values[None, :],
                reduced_vectors.transpose(),
            )
        )
    )
    gram = reduced_vectors.transpose() @ reduced_vectors
    eig_orthogonality_error = _up(
        _frobenius_upper(
            np.abs(gram - np.eye(RETAINED_COUNT))
            + _dense_product_error(
                reduced_vectors.transpose(), reduced_vectors
            )
        )
    )
    eig_generator_error = _up(
        transformed_solve_error
        + eig_reconstruction_error
        + float(np.linalg.norm(transformed, 2))
        * eig_orthogonality_error
        / (1.0 - eig_orthogonality_error)
    )
    orthonormal_trial_norm = _up(
        float(np.linalg.norm(orthonormal_trial, 2))
    )
    source_coordinate_maximum = _up(
        float(np.max(np.linalg.norm(source_coordinates, axis=0)))
    )
    reduced_dense_arithmetic_error = np.nextafter(
        reduced_evaluation_error_norms
        + trial_solve_error
        * math.exp(-LOW_FLOOR * WINDOW)
        * source_coordinate_maximum
        + orthonormal_trial_norm
        * math.exp(-LOW_FLOOR * WINDOW)
        * source_solve_error
        + orthonormal_trial_norm
        * WINDOW
        * eig_generator_error
        * math.exp(-LOW_FLOOR * WINDOW)
        * source_coordinate_maximum,
        math.inf,
    )

    inverse_arc_central = float(
        projected["boundary_geometry"]["stored_scalar_values"][
            "inverse_arc"
        ]
    )
    inverse_arc_interval = projected["boundary_geometry"][
        "exact_scalar_intervals"
    ]["inverse_arc"]
    modified_boundary = (
        mass @ grid["inner_rate_matrix"]
    ).tocsr()
    output_operator = (
        math.sqrt(inverse_arc_central)
        * modified_boundary.transpose()
        @ diags(inverse_square_root_mass)
    ).tocsr()
    output_operator_norm = _up(
        float(np.linalg.norm(output_operator.toarray(), 2))
    )
    square_root_arc_central = math.sqrt(inverse_arc_central)
    square_root_arc_bounds = (
        math.sqrt(float(inverse_arc_interval[0])),
        math.sqrt(float(inverse_arc_interval[1])),
    )
    geometry_relative_error = _up(
        max(
            abs(square_root_arc_central - square_root_arc_bounds[0]),
            abs(square_root_arc_central - square_root_arc_bounds[1]),
        )
        / square_root_arc_central
    )
    output_relative_error = _up(
        geometry_relative_error + _gamma(24)
    )
    output_operator_error = _up(
        output_operator_norm
        * output_relative_error
        / (1.0 - output_relative_error)
    )
    exact_output_operator_norm = _up(
        output_operator_norm + output_operator_error
    )

    central_difference = full_state - reduced_state
    central_boundary, boundary_product_error = (
        _sparse_action_with_error(output_operator, central_difference)
    )
    central_boundary_norms = np.linalg.norm(central_boundary, axis=0)
    boundary_product_error_norms = _column_norms_upper(
        boundary_product_error
    )
    central_difference_norms = np.linalg.norm(
        central_difference, axis=0
    )
    full_state_error = float(
        recurrence["maximum_error_components"][
            "total_one_step_state_action_error_upper"
        ]
    )
    total_state_difference_error = np.nextafter(
        full_state_error
        + reduced_form_transfer_error
        + reduced_dense_arithmetic_error,
        math.inf,
    )
    boundary_error = np.nextafter(
        exact_output_operator_norm * total_state_difference_error
        + output_operator_error * central_difference_norms
        + boundary_product_error_norms,
        math.inf,
    )
    boundary_upper = np.nextafter(
        central_boundary_norms + boundary_error,
        math.inf,
    )
    maximizing_entry = int(np.argmax(boundary_upper))
    maximum_central = float(np.max(central_boundary_norms))
    maximum_upper = _up(float(np.max(boundary_upper)))
    maximum_error = _up(float(np.max(boundary_error)))

    checks = [
        priority_set,
        matrix_metadata["mass_diagonal_strictly_positive"],
        matrix_metadata["stiffness_exactly_symmetric"],
        exact_generator_error < 1.1e-7,
        float(np.max(source_coordinate_error)) < 1.0e-8,
        float(np.max(reduced_form_transfer_error)) < 2.0e-7,
        float(np.max(reduced_dense_arithmetic_error)) < 1.0e-8,
        output_operator_error < 1.0e-10,
        maximum_central < 6.4e-4,
        maximum_error < 5.0e-5,
        maximum_upper < 7.0e-4,
    ]
    return {
        "kind": "neutral_strip_first_endpoint_boundary_leakage_certificate",
        "model": (
            "exact stored modified-chain point-source semigroup minus the "
            "exact frozen-trial Galerkin semigroup at t=3/8, observed in "
            "common-circle boundary L2"
        ),
        "below_normal_priority_set": priority_set,
        "premise_artifacts": {
            "eigen_cache": str(eigen_cache),
            "eigen_cache_sha256": _sha256_file(eigen_cache),
            "two_block_result": str(two_block_result_path),
            "two_block_result_sha256": _sha256_file(
                two_block_result_path
            ),
            "projected_result": str(projected_result_path),
            "projected_result_sha256": _sha256_file(
                projected_result_path
            ),
            "recurrence_result": str(recurrence_result_path),
            "recurrence_result_sha256": _sha256_file(
                recurrence_result_path
            ),
            "recurrence_state_cache": str(cache_path),
            "recurrence_state_cache_sha256": _sha256_file(cache_path),
        },
        "reduced_galerkin_transfer": {
            "restricted_mass_error_upper": restricted_mass_error,
            "restricted_stiffness_error_upper": (
                restricted_stiffness_error
            ),
            "cholesky_residual_upper": cholesky_residual_upper,
            "exact_reduced_generator_error_upper": exact_generator_error,
            "source_coordinate_error_upper": _up(
                float(np.max(source_coordinate_error))
            ),
            "semigroup_generator_difference_upper": (
                semigroup_difference
            ),
            "reduced_form_transfer_state_error_upper": _up(
                float(np.max(reduced_form_transfer_error))
            ),
            "trial_triangular_solve_error_upper": trial_solve_error,
            "source_triangular_solve_error_upper": _up(
                float(np.max(source_solve_error))
            ),
            "transformed_generator_solve_error_upper": (
                transformed_solve_error
            ),
            "eigensystem_generator_error_upper": eig_generator_error,
            "reduced_dense_arithmetic_state_error_upper": _up(
                float(np.max(reduced_dense_arithmetic_error))
            ),
        },
        "boundary_operator": {
            "central_spectral_norm": output_operator_norm,
            "geometry_relative_error_upper": geometry_relative_error,
            "exact_operator_difference_upper": output_operator_error,
            "exact_operator_norm_upper": exact_output_operator_norm,
            "maximum_sparse_product_roundoff_upper": _up(
                float(np.max(boundary_product_error_norms))
            ),
        },
        "endpoint": {
            "time": WINDOW,
            "entry_count": len(entry_states),
            "maximum_central_boundary_l2_difference": maximum_central,
            "maximum_boundary_l2_error_upper": maximum_error,
            "maximum_boundary_l2_difference_upper": maximum_upper,
            "maximizing_entry_index": maximizing_entry,
            "central_maximizing_entry_index": int(
                np.argmax(central_boundary_norms)
            ),
        },
        "first_endpoint_stored_chain_boundary_leakage_certified": bool(
            all(checks)
        ),
        "within_window_supremum_certified": False,
        "later_endpoint_boundary_leakage_certified": False,
        "post_terminal_time_tail_certified": False,
        "screen_updated": False,
        "checks": checks,
        "all_first_endpoint_boundary_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Propagate the certified one-step action and reduced-state error "
            "through the remaining 15 endpoint steps, then replace endpoint "
            "samples by within-window suprema and certify the post-time-6 "
            "tail before charging the screen."
        ),
        "scope": (
            "This certificate concerns the stored binary finite chain and "
            "its frozen 240-column trial space at one endpoint. It is not a "
            "continuum Ritz transfer, polygon-circle domain transfer, full "
            "time-slab bound, or Navier-Stokes regularity proof."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigen-cache", type=Path, default=DEFAULT_EIGEN_CACHE)
    parser.add_argument(
        "--two-block-result", type=Path, default=DEFAULT_TWO_BLOCK_RESULT
    )
    parser.add_argument(
        "--projected-result", type=Path, default=DEFAULT_PROJECTED_RESULT
    )
    parser.add_argument(
        "--recurrence-result",
        type=Path,
        default=DEFAULT_RECURRENCE_RESULT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = audit(
        arguments.eigen_cache,
        arguments.two_block_result,
        arguments.projected_result,
        arguments.recurrence_result,
    )
    _atomic_json(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
