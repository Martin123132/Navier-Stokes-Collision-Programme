#!/usr/bin/env python3
"""Certify the finite modified-chain two-block state leakage."""

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

import numpy as np
from scipy.linalg import eigh, solve, svd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPLEMENT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_modified_complement_inertia_schur_v1.json"
)
DEFAULT_EIGEN_CACHE = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k241_reference_eigensystem_v1.npz"
)
LOW_BLOCK_FLOOR = 2.36
RETAINED_COUNT = 240
UNIT_ROUNDOFF = 2.0**-53
TIME_ROWS = (0.1875, 0.375, 0.75, 1.5, 3.0, 6.0)


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


def _down(value: float) -> float:
    return float(np.nextafter(float(value), -math.inf))


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _gamma(operation_count: int) -> float:
    product = operation_count * UNIT_ROUNDOFF
    if product >= 1.0:
        raise ArithmeticError("roundoff operation count is too large")
    return product / (1.0 - product)


def _frob_upper(array: np.ndarray) -> float:
    correction = math.sqrt(1.0 - _gamma(2 * array.size + 10))
    return _up(float(np.linalg.norm(array, "fro")) / correction)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
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


def _projected_form(
    matrix,
    vectors: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    state_count = matrix.shape[0]
    maximum_row_nonzeros = int(np.max(np.diff(matrix.indptr)))
    action = matrix @ vectors
    absolute_action_upper = (
        abs(matrix) @ np.abs(vectors)
    ) / (1.0 - _gamma(2 * maximum_row_nonzeros + 10))
    action_error_upper = (
        _gamma(2 * maximum_row_nonzeros + 30)
        * absolute_action_upper
    )
    raw_projection = vectors.T @ action
    dense_absolute_product = (
        np.abs(vectors).T @ np.abs(action)
    ) / (1.0 - _gamma(2 * state_count + 10))
    propagated_action_error = (
        np.abs(vectors).T @ action_error_upper
    ) / (1.0 - _gamma(2 * state_count + 10))
    projection_error_entries = (
        _gamma(2 * state_count + 30) * dense_absolute_product
        + propagated_action_error
        + _gamma(4) * np.abs(raw_projection)
    )
    central_projection = 0.5 * (
        raw_projection + raw_projection.T
    )
    return (
        central_projection,
        _frob_upper(projection_error_entries),
        action,
        action_error_upper,
    )


def _symmetric_eigen_bounds(
    matrix: np.ndarray,
    input_error_spectral_upper: float,
) -> dict[str, float]:
    dimension = matrix.shape[0]
    eigenvalues, eigenvectors = eigh(matrix)
    reconstructed = (
        eigenvectors * eigenvalues[None, :]
    ) @ eigenvectors.T
    reconstruction_absolute_product = (
        (np.abs(eigenvectors) * np.abs(eigenvalues)[None, :])
        @ np.abs(eigenvectors).T
    ) / (1.0 - _gamma(2 * dimension + 10))
    reconstruction_error = _frob_upper(
        np.abs(matrix - reconstructed)
        + _gamma(2 * dimension + 30)
        * reconstruction_absolute_product
    )
    gram = eigenvectors.T @ eigenvectors
    gram_absolute_product = (
        np.abs(eigenvectors).T @ np.abs(eigenvectors)
    ) / (1.0 - _gamma(2 * dimension + 10))
    orthogonality_defect = _frob_upper(
        np.abs(gram - np.eye(dimension))
        + _gamma(2 * dimension + 30) * gram_absolute_product
    )
    minimum_lower = _down(
        float(eigenvalues[0]) * (1.0 - orthogonality_defect)
        - reconstruction_error
        - input_error_spectral_upper
    )
    maximum_upper = _up(
        float(eigenvalues[-1]) * (1.0 + orthogonality_defect)
        + reconstruction_error
        + input_error_spectral_upper
    )
    return {
        "central_minimum": float(eigenvalues[0]),
        "central_maximum": float(eigenvalues[-1]),
        "minimum_lower": minimum_lower,
        "maximum_upper": maximum_upper,
        "orthogonality_defect_upper": orthogonality_defect,
        "reconstruction_error_upper": reconstruction_error,
        "input_error_spectral_upper": input_error_spectral_upper,
    }


def _svd_spectral_upper(
    matrix: np.ndarray,
    input_error_spectral_upper: float,
) -> dict[str, float]:
    row_count, column_count = matrix.shape
    left, singular_values, right = svd(
        matrix, full_matrices=False, lapack_driver="gesdd"
    )
    reconstructed = (left * singular_values[None, :]) @ right
    reconstruction_absolute_product = (
        (np.abs(left) * singular_values[None, :]) @ np.abs(right)
    ) / (1.0 - _gamma(2 * column_count + 10))
    reconstruction_error = _frob_upper(
        np.abs(matrix - reconstructed)
        + _gamma(2 * column_count + 30)
        * reconstruction_absolute_product
    )

    left_gram = left.T @ left
    left_absolute_product = (
        np.abs(left).T @ np.abs(left)
    ) / (1.0 - _gamma(2 * row_count + 10))
    left_defect = _frob_upper(
        np.abs(left_gram - np.eye(column_count))
        + _gamma(2 * row_count + 30) * left_absolute_product
    )
    right_gram = right @ right.T
    right_absolute_product = (
        np.abs(right) @ np.abs(right).T
    ) / (1.0 - _gamma(2 * column_count + 10))
    right_defect = _frob_upper(
        np.abs(right_gram - np.eye(column_count))
        + _gamma(2 * column_count + 30) * right_absolute_product
    )
    spectral_upper = _up(
        float(singular_values[0])
        * math.sqrt((1.0 + left_defect) * (1.0 + right_defect))
        + reconstruction_error
        + input_error_spectral_upper
    )
    return {
        "central_spectral_norm": float(singular_values[0]),
        "spectral_norm_upper": spectral_upper,
        "left_orthogonality_defect_upper": left_defect,
        "right_orthogonality_defect_upper": right_defect,
        "reconstruction_error_upper": reconstruction_error,
        "input_error_spectral_upper": input_error_spectral_upper,
    }


def _certify_parameters(
    mass,
    stiffness,
    vectors: np.ndarray,
) -> dict[str, object]:
    (
        restricted_mass,
        restricted_mass_error,
        mass_action,
        mass_action_error,
    ) = _projected_form(mass, vectors)
    (
        restricted_stiffness,
        restricted_stiffness_error,
        stiffness_action,
        stiffness_action_error,
    ) = _projected_form(stiffness, vectors)
    mass_bounds = _symmetric_eigen_bounds(
        restricted_mass, restricted_mass_error
    )

    shifted_low_form = (
        restricted_stiffness - LOW_BLOCK_FLOOR * restricted_mass
    )
    shifted_low_error = _up(
        restricted_stiffness_error
        + LOW_BLOCK_FLOOR * restricted_mass_error
        + _gamma(6)
        * (
            float(np.linalg.norm(restricted_stiffness, 2))
            + LOW_BLOCK_FLOOR
            * float(np.linalg.norm(restricted_mass, 2))
        )
    )
    shifted_low_bounds = _symmetric_eigen_bounds(
        shifted_low_form, shifted_low_error
    )
    low_floor_certified = bool(
        mass_bounds["minimum_lower"] > 0.0
        and shifted_low_bounds["minimum_lower"] > 0.0
    )

    projected_generator = solve(
        restricted_mass, restricted_stiffness, assume_a="pos"
    )
    central_solve_residual = (
        restricted_stiffness
        - restricted_mass @ projected_generator
    )
    dense_absolute_product = (
        np.abs(restricted_mass) @ np.abs(projected_generator)
    ) / (1.0 - _gamma(2 * RETAINED_COUNT + 10))
    dense_product_roundoff = _frob_upper(
        _gamma(2 * RETAINED_COUNT + 30)
        * dense_absolute_product
        + _gamma(4)
        * (
            np.abs(restricted_stiffness)
            + np.abs(restricted_mass @ projected_generator)
        )
    )
    exact_solve_residual_upper = _up(
        float(np.linalg.norm(central_solve_residual, 2))
        + restricted_stiffness_error
        + restricted_mass_error
        * float(np.linalg.norm(projected_generator, 2))
        + dense_product_roundoff
    )
    projected_generator_error_upper = _up(
        exact_solve_residual_upper / mass_bounds["minimum_lower"]
    )

    projected_mass_action = mass_action @ projected_generator
    projected_mass_absolute_product = (
        np.abs(mass_action) @ np.abs(projected_generator)
    ) / (1.0 - _gamma(2 * RETAINED_COUNT + 10))
    invariance_residual = (
        stiffness_action - projected_mass_action
    )
    invariance_error_entries = (
        stiffness_action_error
        + (
            mass_action_error @ np.abs(projected_generator)
        )
        / (1.0 - _gamma(2 * RETAINED_COUNT + 10))
        + _gamma(2 * RETAINED_COUNT + 30)
        * projected_mass_absolute_product
        + _gamma(5)
        * (
            np.abs(stiffness_action)
            + np.abs(projected_mass_action)
        )
    )
    mass_diagonal = np.asarray(mass.diagonal(), dtype=np.float64)
    weighted_residual = (
        invariance_residual / np.sqrt(mass_diagonal)[:, None]
    )
    weighted_residual_input_error = _frob_upper(
        invariance_error_entries
        / np.sqrt(mass_diagonal)[:, None]
        + _gamma(4) * np.abs(weighted_residual)
    )
    weighted_residual_bound = _svd_spectral_upper(
        weighted_residual, weighted_residual_input_error
    )
    exact_invariance_residual_upper = _up(
        weighted_residual_bound["spectral_norm_upper"]
        + math.sqrt(mass_bounds["maximum_upper"])
        * projected_generator_error_upper
    )
    off_block_coupling_upper = _up(
        exact_invariance_residual_upper
        / math.sqrt(mass_bounds["minimum_lower"])
    )

    return {
        "restricted_mass": mass_bounds,
        "shifted_low_form": shifted_low_bounds,
        "low_block_floor": LOW_BLOCK_FLOOR,
        "low_block_floor_certified": low_floor_certified,
        "restricted_mass_projection_error_upper": (
            restricted_mass_error
        ),
        "restricted_stiffness_projection_error_upper": (
            restricted_stiffness_error
        ),
        "projected_generator_exact_solve_residual_upper": (
            exact_solve_residual_upper
        ),
        "projected_generator_error_upper": (
            projected_generator_error_upper
        ),
        "weighted_invariance_residual": weighted_residual_bound,
        "exact_weighted_invariance_residual_upper": (
            exact_invariance_residual_upper
        ),
        "off_block_coupling_upper": off_block_coupling_upper,
        "off_block_coupling_certified": bool(
            low_floor_certified and off_block_coupling_upper < 6.5
        ),
    }


def audit(
    spacing: float,
    eigen_cache: Path,
    complement_result_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    complement = json.loads(
        complement_result_path.read_text(encoding="ascii")
    )
    if not complement["all_modified_complement_checks_pass"]:
        raise RuntimeError("modified complement premise is not certified")
    for key in ("lower_inertia_row", "upper_inertia_row"):
        row_path = Path(complement[key]["path"])
        if _sha256_file(row_path) != complement[key]["file_sha256"]:
            raise RuntimeError(f"modified complement row hash changed: {row_path}")

    base_module = _load_module(
        "neutral_strip_modified_complement_inertia_schur_certificate.py",
        "modified_two_block_complement_base",
    )
    mass, stiffness, vectors, matrix_metadata = (
        base_module._assemble_modified_pencil(spacing, eigen_cache)
    )
    if (
        matrix_metadata["mass_sha256"]
        != complement["pencil"]["mass_sha256"]
        or matrix_metadata["stiffness_sha256"]
        != complement["pencil"]["stiffness_sha256"]
        or matrix_metadata["retained_vectors_sha256"]
        != complement["pencil"]["retained_vectors_sha256"]
    ):
        raise RuntimeError("modified two-block matrices changed")

    parameters = _certify_parameters(mass, stiffness, vectors)
    high_floor = float(
        complement["theorem"]["modified_complement_floor_lower"]
    )
    coupling = float(parameters["off_block_coupling_upper"])
    transfer_module = _load_module(
        "neutral_strip_projected_interval_two_block_transfer.py",
        "modified_two_block_transfer_formula",
    )
    rows = [
        transfer_module._two_block_bounds(
            LOW_BLOCK_FLOOR, high_floor, coupling, time_value
        )
        for time_value in TIME_ROWS
    ]
    all_parameters_certified = bool(
        parameters["low_block_floor_certified"]
        and parameters["off_block_coupling_certified"]
        and complement["theorem"][
            "modified_complement_floor_certified"
        ]
    )
    checks = [
        priority_set,
        all_parameters_certified,
        float(parameters["restricted_mass"]["minimum_lower"]) > 1.0,
        float(
            parameters["shifted_low_form"]["minimum_lower"]
        )
        > 0.01,
        coupling < 6.5,
        high_floor == 102.7,
        len(rows) == len(TIME_ROWS),
        all(
            float(row["high_component_upper"])
            < float(row["gap_free_high_component_upper"])
            for row in rows
        ),
    ]
    return {
        "model": "certified modified-chain two-block state leakage",
        "spacing": spacing,
        "below_normal_priority_set": priority_set,
        "premise_artifacts": {
            "modified_complement_result": str(complement_result_path),
            "modified_complement_result_sha256": _sha256_file(
                complement_result_path
            ),
            "reference_eigen_cache": str(eigen_cache),
            "reference_eigen_cache_sha256": _sha256_file(eigen_cache),
        },
        "matrix_metadata": matrix_metadata,
        "certified_parameters": {
            **parameters,
            "high_block_floor": high_floor,
            "high_block_floor_certified": True,
            "all_two_block_parameters_certified": (
                all_parameters_certified
            ),
        },
        "two_block_state_rows": rows,
        "modified_two_block_state_leakage_certified": (
            all_parameters_certified
        ),
        "boundary_output_smoothing_composed": False,
        "checks": checks,
        "all_modified_two_block_leakage_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Split each output time in half, apply the certified state "
            "leakage on the first half and an interval boundary-flux "
            "operator norm on the second half, then add the low-projector "
            "source/trace mismatch."
        ),
        "scope": (
            "This result certifies finite modified-chain state-space "
            "two-block leakage. It is not yet a boundary-output charge and "
            "does not address continuum Ritz or polygon-circle transfer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, default=0.06)
    parser.add_argument("--eigen-cache", type=Path, default=DEFAULT_EIGEN_CACHE)
    parser.add_argument(
        "--complement-result",
        type=Path,
        default=DEFAULT_COMPLEMENT_RESULT,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.spacing, args.eigen_cache, args.complement_result
    )
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
