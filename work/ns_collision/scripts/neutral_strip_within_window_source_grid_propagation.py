#!/usr/bin/env python3
"""Propagate and certify the 160-substep actual-source boundary grid."""

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

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular
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
    "neutral_strip_h006_within_window_substep_coefficients_v1.json"
)
DEFAULT_RECURRENCE_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_substep_recurrence_v1.json"
)
DEFAULT_FIRST_ENDPOINT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_first_endpoint_boundary_leakage_v1.json"
)
DEFAULT_ALL_ENDPOINT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_all_endpoint_boundary_leakage_v1.json"
)
DEFAULT_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_source_grid_checkpoint_v1.npz"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_source_grid_propagation_v1.json"
)
SUBSTEP = 0.0375
SUBSTEPS_PER_BLOCK = 10
TOTAL_BLOCKS = 16
TOTAL_SUBSTEPS = 160
FIRST_RECORDED_SUBSTEP = 10
DEGREE = 112
WINDOW = 0.375
LOW_FLOOR = 2.36
SCALING_LOWER = 1.9
SCALING_UPPER = 8000.0
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


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
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


def _checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".json")


def _write_checkpoint(
    checkpoint_path: Path,
    signature: dict[str, Any],
    completed_substeps: int,
    full_state: np.ndarray,
    reduced_coordinates: np.ndarray,
    rows: list[dict[str, Any]],
    endpoint_crosschecks: list[dict[str, Any]],
    resource_samples: list[dict[str, Any]],
) -> None:
    _atomic_npz(
        checkpoint_path,
        {
            "completed_substeps": np.asarray(
                completed_substeps, dtype=np.int64
            ),
            "full_state": np.asarray(full_state, dtype=np.float64),
            "reduced_coordinates": np.asarray(
                reduced_coordinates, dtype=np.float64
            ),
        },
    )
    _atomic_json(
        _checkpoint_metadata_path(checkpoint_path),
        {
            "kind": "neutral_strip_within_window_source_grid_checkpoint",
            "schema_version": 1,
            "signature": signature,
            "completed_substeps": completed_substeps,
            "completed_blocks": completed_substeps // SUBSTEPS_PER_BLOCK,
            "rows": rows,
            "endpoint_crosschecks": endpoint_crosschecks,
            "resource_samples": resource_samples,
            "npz_path": str(checkpoint_path),
            "npz_sha256": _sha256_file(checkpoint_path),
        },
    )


def _load_checkpoint(
    checkpoint_path: Path,
    signature: dict[str, Any],
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
] | None:
    metadata_path = _checkpoint_metadata_path(checkpoint_path)
    if not checkpoint_path.is_file() and not metadata_path.is_file():
        return None
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("source-grid checkpoint pair is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    if metadata.get("signature") != signature:
        raise RuntimeError("source-grid checkpoint signature changed")
    if metadata.get("npz_sha256") != _sha256_file(checkpoint_path):
        raise RuntimeError("source-grid checkpoint NPZ hash mismatch")
    with np.load(checkpoint_path, allow_pickle=False) as cached:
        completed_substeps = int(cached["completed_substeps"].item())
        full_state = np.asarray(cached["full_state"], dtype=np.float64)
        reduced_coordinates = np.asarray(
            cached["reduced_coordinates"], dtype=np.float64
        )
    if completed_substeps != int(metadata["completed_substeps"]):
        raise RuntimeError("source-grid checkpoint step mismatch")
    if completed_substeps % SUBSTEPS_PER_BLOCK:
        raise RuntimeError("source-grid checkpoint is not block atomic")
    expected_row_count = max(
        completed_substeps - FIRST_RECORDED_SUBSTEP + 1, 0
    )
    rows = list(metadata["rows"])
    if len(rows) != expected_row_count:
        raise RuntimeError("source-grid checkpoint row count changed")
    return (
        completed_substeps,
        full_state,
        reduced_coordinates,
        rows,
        list(metadata["endpoint_crosschecks"]),
        list(metadata["resource_samples"]),
    )


def _cpu_pair() -> list[float]:
    try:
        import psutil

        return [
            float(psutil.cpu_percent(interval=1.0)),
            float(psutil.cpu_percent(interval=1.0)),
        ]
    except Exception:
        return []


def _reduced_error_factor(time_value: float) -> float:
    if time_value < WINDOW:
        raise ValueError("reduced transfer is used only after t=3/8")
    return _up(
        (time_value / WINDOW)
        * math.exp(-LOW_FLOOR * (time_value - WINDOW))
    )


def audit(
    eigen_cache: Path,
    two_block_result_path: Path,
    projected_result_path: Path,
    coefficient_result_path: Path,
    recurrence_result_path: Path,
    first_endpoint_result_path: Path,
    all_endpoint_result_path: Path,
    checkpoint_path: Path,
    maximum_blocks: int,
    cpu_threshold: float,
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
    production_recurrence_path = Path(
        recurrence["premise_artifacts"]["production_recurrence_result"]
    )
    if (
        _sha256_file(production_recurrence_path)
        != recurrence["premise_artifacts"][
            "production_recurrence_result_sha256"
        ]
    ):
        raise RuntimeError("production recurrence premise hash changed")
    production_recurrence = json.loads(
        production_recurrence_path.read_text(encoding="ascii")
    )
    first_endpoint = json.loads(
        first_endpoint_result_path.read_text(encoding="ascii")
    )
    all_endpoint = json.loads(
        all_endpoint_result_path.read_text(encoding="ascii")
    )
    if not two_block["all_modified_two_block_leakage_checks_pass"]:
        raise RuntimeError("two-block premise is not certified")
    if not coefficient["all_substep_coefficient_checks_pass"]:
        raise RuntimeError("substep coefficients are not certified")
    if not recurrence["all_substep_recurrence_checks_pass"]:
        raise RuntimeError("substep recurrence is not certified")
    if not first_endpoint["all_first_endpoint_boundary_checks_pass"]:
        raise RuntimeError("first endpoint is not certified")
    if not all_endpoint["all_endpoint_boundary_checks_pass"]:
        raise RuntimeError("all production endpoints are not certified")

    base_module = _load_module(
        "neutral_strip_modified_complement_inertia_schur_certificate.py",
        "within_window_source_grid_base",
    )
    mesh_module = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "within_window_source_grid_mesh",
    )
    source_module = _load_module(
        "neutral_strip_common_circle_source_time_slab_certificate.py",
        "within_window_source_grid_source",
    )
    pilot_module = _load_module(
        "neutral_strip_boundary_leakage_chebyshev_pilot.py",
        "within_window_source_grid_chebyshev",
    )
    endpoint_module = _load_module(
        "neutral_strip_first_endpoint_boundary_leakage_certificate.py",
        "within_window_source_grid_endpoint",
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
        0.5 * (normalized_generator + normalized_generator.transpose())
    ).tocsr()

    restricted_mass = vectors.T @ (mass @ vectors)
    restricted_mass = 0.5 * (
        restricted_mass + restricted_mass.transpose()
    )
    lower_factor = cholesky(restricted_mass, lower=True)
    orthonormal_trial = solve_triangular(
        lower_factor,
        (square_root_mass[:, None] * vectors).transpose(),
        lower=True,
    ).transpose()
    trial_norm_upper = endpoint_module._frobenius_upper(
        np.abs(orthonormal_trial)
    )
    reduced_generator = orthonormal_trial.transpose() @ (
        normalized_generator @ orthonormal_trial
    )
    reduced_generator = 0.5 * (
        reduced_generator + reduced_generator.transpose()
    )
    reduced_values, reduced_vectors = eigh(reduced_generator)

    entry_states = np.asarray(grid["entry_states"], dtype=np.int64)
    entry_count = len(entry_states)
    initial_full_state = np.zeros(
        (normalized_generator.shape[0], entry_count), dtype=np.float64
    )
    initial_full_state[entry_states, np.arange(entry_count)] = (
        inverse_square_root_mass[entry_states]
    )
    initial_reduced_coordinates = (
        orthonormal_trial[entry_states, :].transpose()
        * inverse_square_root_mass[entry_states][None, :]
    )
    initial_reduced_modes = (
        reduced_vectors.transpose() @ initial_reduced_coordinates
    )

    modified_boundary = (mass @ grid["inner_rate_matrix"]).tocsr()
    inverse_arc = float(
        projected["boundary_geometry"]["stored_scalar_values"][
            "inverse_arc"
        ]
    )
    output_operator = (
        math.sqrt(inverse_arc)
        * modified_boundary.transpose()
        @ diags(inverse_square_root_mass)
    ).tocsr()
    reduced_output_operator, reduced_output_construction_error = (
        endpoint_module._sparse_action_with_error(
            output_operator, orthonormal_trial
        )
    )

    center = 0.5 * (SCALING_UPPER + SCALING_LOWER)
    radius = 0.5 * (SCALING_UPPER - SCALING_LOWER)
    scaled_operator = (
        normalized_generator
        - center * eye(normalized_generator.shape[0], format="csr")
    ) * (1.0 / radius)
    coefficients = np.asarray(
        [
            float(row["scipy_central"])
            for row in coefficient["coefficient_intervals"]["rows"]
        ],
        dtype=np.float64,
    )
    if len(coefficients) != DEGREE + 1:
        raise RuntimeError("degree-112 coefficient count changed")

    signature = {
        "schema_version": 1,
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
        "production_recurrence_result": str(
            production_recurrence_path
        ),
        "production_recurrence_result_sha256": _sha256_file(
            production_recurrence_path
        ),
        "first_endpoint_result": str(first_endpoint_result_path),
        "first_endpoint_result_sha256": _sha256_file(
            first_endpoint_result_path
        ),
        "all_endpoint_result": str(all_endpoint_result_path),
        "all_endpoint_result_sha256": _sha256_file(
            all_endpoint_result_path
        ),
        "mass_sha256": matrix_metadata["mass_sha256"],
        "stiffness_sha256": matrix_metadata["stiffness_sha256"],
        "retained_vectors_sha256": matrix_metadata[
            "retained_vectors_sha256"
        ],
        "substep": SUBSTEP,
        "degree": DEGREE,
        "total_substeps": TOTAL_SUBSTEPS,
        "entry_count": entry_count,
    }
    loaded = _load_checkpoint(checkpoint_path, signature)
    if loaded is None:
        completed_substeps = 0
        full_state = initial_full_state
        reduced_coordinates = initial_reduced_coordinates
        rows: list[dict[str, Any]] = []
        endpoint_crosschecks: list[dict[str, Any]] = []
        resource_samples: list[dict[str, Any]] = []
    else:
        (
            completed_substeps,
            full_state,
            reduced_coordinates,
            rows,
            endpoint_crosschecks,
            resource_samples,
        ) = loaded

    substep_operator = recurrence["operator"]
    step_error = float(
        substep_operator["total_one_substep_operator_error_upper"]
    )
    computational_norm = float(
        substep_operator["computational_substep_operator_norm_upper"]
    )
    maximum_source_norm = float(
        production_recurrence["operator"]["maximum_source_state_norm"]
    )
    source_construction_relative_error = _gamma(6)
    first_reduced_form_error = float(
        first_endpoint["reduced_galerkin_transfer"][
            "reduced_form_transfer_state_error_upper"
        ]
    )
    first_reduced_dense_error = float(
        first_endpoint["reduced_galerkin_transfer"][
            "reduced_dense_arithmetic_state_error_upper"
        ]
    )
    exact_output_norm = float(
        first_endpoint["boundary_operator"]["exact_operator_norm_upper"]
    )
    output_operator_error = float(
        first_endpoint["boundary_operator"][
            "exact_operator_difference_upper"
        ]
    )
    production_endpoint_rows = all_endpoint["endpoint_rows"]

    target_blocks = min(
        TOTAL_BLOCKS,
        completed_substeps // SUBSTEPS_PER_BLOCK + maximum_blocks,
    )
    park_reason = None
    while completed_substeps // SUBSTEPS_PER_BLOCK < target_blocks:
        block_target = completed_substeps + SUBSTEPS_PER_BLOCK
        while completed_substeps < block_target:
            full_state, _ = pilot_module._chebyshev_step(
                scaled_operator,
                full_state,
                coefficients,
            )
            completed_substeps += 1
            time_value = completed_substeps * SUBSTEP
            reduced_coordinates = reduced_vectors @ (
                np.exp(-time_value * reduced_values)[:, None]
                * initial_reduced_modes
            )
            if completed_substeps < FIRST_RECORDED_SUBSTEP:
                continue

            full_boundary, full_boundary_error = (
                endpoint_module._sparse_action_with_error(
                    output_operator, full_state
                )
            )
            reduced_boundary, reduced_boundary_error = (
                endpoint_module._dense_action_with_error(
                    reduced_output_operator, reduced_coordinates
                )
            )
            reduced_construction_action = (
                np.abs(reduced_output_construction_error)
                @ np.abs(reduced_coordinates)
            ) / (
                1.0
                - _gamma(2 * reduced_output_operator.shape[1] + 8)
            )
            reduced_boundary_error = np.nextafter(
                reduced_boundary_error + reduced_construction_action,
                math.inf,
            )
            boundary_difference = full_boundary - reduced_boundary
            subtraction_error = np.nextafter(
                _gamma(1)
                * (np.abs(full_boundary) + np.abs(reduced_boundary)),
                math.inf,
            )
            product_error_norms = endpoint_module._column_norms_upper(
                np.nextafter(
                    full_boundary_error
                    + reduced_boundary_error
                    + subtraction_error,
                    math.inf,
                )
            )
            central_boundary_norms = np.linalg.norm(
                boundary_difference, axis=0
            )
            full_state_norms = endpoint_module._column_norms_upper(
                np.abs(full_state)
            )
            reduced_coordinate_norms = (
                endpoint_module._column_norms_upper(
                    np.abs(reduced_coordinates)
                )
            )
            central_state_difference_uppers = np.nextafter(
                full_state_norms
                + trial_norm_upper * reduced_coordinate_norms,
                math.inf,
            )

            full_state_error = _up(
                maximum_source_norm
                * (
                    computational_norm**completed_substeps
                    * source_construction_relative_error
                    + completed_substeps
                    * step_error
                    * computational_norm ** (completed_substeps - 1)
                )
            )
            reduced_factor = _reduced_error_factor(time_value)
            reduced_form_error = _up(
                first_reduced_form_error * reduced_factor
            )
            reduced_dense_error = _up(
                first_reduced_dense_error * reduced_factor
            )
            total_state_error = _up(
                full_state_error
                + reduced_form_error
                + reduced_dense_error
            )
            norm_evaluation_errors = np.nextafter(
                _gamma(2 * output_operator.shape[0] + 8)
                * central_boundary_norms,
                math.inf,
            )
            boundary_errors = np.nextafter(
                exact_output_norm * total_state_error
                + output_operator_error
                * central_state_difference_uppers
                + product_error_norms
                + norm_evaluation_errors,
                math.inf,
            )
            boundary_uppers = np.nextafter(
                central_boundary_norms + boundary_errors, math.inf
            )
            maximizing_entry = int(np.argmax(boundary_uppers))
            row = {
                "substep": completed_substeps,
                "time": time_value,
                "maximum_central_boundary_l2_difference": float(
                    np.max(central_boundary_norms)
                ),
                "maximum_boundary_l2_error_upper": _up(
                    float(np.max(boundary_errors))
                ),
                "maximum_boundary_l2_difference_upper": _up(
                    float(np.max(boundary_uppers))
                ),
                "maximizing_entry_index": maximizing_entry,
                "full_repeated_state_error_upper": full_state_error,
                "reduced_form_state_error_upper": reduced_form_error,
                "reduced_dense_state_error_upper": reduced_dense_error,
                "total_state_difference_error_upper": total_state_error,
                "maximum_output_product_roundoff_upper": _up(
                    float(np.max(product_error_norms))
                ),
                "maximum_central_state_difference_norm_upper": _up(
                    float(np.max(central_state_difference_uppers))
                ),
                "axial_l2_upper": float(
                    source_module._axial_l2_global_upper(time_value)
                ),
            }
            rows.append(row)

            if completed_substeps % SUBSTEPS_PER_BLOCK == 0:
                endpoint_index = (
                    completed_substeps // SUBSTEPS_PER_BLOCK - 1
                )
                endpoint_row = production_endpoint_rows[endpoint_index]
                central_difference = abs(
                    row["maximum_central_boundary_l2_difference"]
                    - float(
                        endpoint_row[
                            "central_boundary_l2_difference"
                        ]
                    )
                )
                combined_error = _up(
                    row["maximum_boundary_l2_error_upper"]
                    + float(endpoint_row["boundary_l2_error_upper"])
                )
                endpoint_crosschecks.append(
                    {
                        "endpoint_step": endpoint_index + 1,
                        "time": time_value,
                        "central_maximum_difference": central_difference,
                        "combined_certified_error_upper": combined_error,
                        "guard_margin": _up(
                            combined_error - central_difference
                        ),
                        "certified_intervals_overlap": bool(
                            central_difference <= combined_error
                        ),
                    }
                )

        samples = _cpu_pair()
        resource_samples.append(
            {
                "completed_block": (
                    completed_substeps // SUBSTEPS_PER_BLOCK
                ),
                "total_cpu_percent": samples,
            }
        )
        _write_checkpoint(
            checkpoint_path,
            signature,
            completed_substeps,
            full_state,
            reduced_coordinates,
            rows,
            endpoint_crosschecks,
            resource_samples,
        )
        if (
            len(samples) == 2
            and samples[0] > cpu_threshold
            and samples[1] > cpu_threshold
        ):
            park_reason = (
                "Two consecutive post-block total-CPU samples exceeded "
                f"{cpu_threshold} percent."
            )
            break

    completed_blocks = completed_substeps // SUBSTEPS_PER_BLOCK
    expected_row_count = max(
        completed_substeps - FIRST_RECORDED_SUBSTEP + 1, 0
    )
    row_ordered = all(
        int(row["substep"]) == FIRST_RECORDED_SUBSTEP + index
        for index, row in enumerate(rows)
    )
    all_crosschecks_pass = all(
        row["certified_intervals_overlap"]
        for row in endpoint_crosschecks
    )
    checkpoint_metadata_path = _checkpoint_metadata_path(checkpoint_path)
    checks = [
        priority_set,
        matrix_hashes_match,
        coefficient["coefficient_intervals"]["degree"] == DEGREE,
        recurrence["operator"]["substep"] == SUBSTEP,
        completed_substeps % SUBSTEPS_PER_BLOCK == 0,
        completed_substeps <= TOTAL_SUBSTEPS,
        len(rows) == expected_row_count,
        row_ordered,
        len(endpoint_crosschecks) == completed_blocks,
        all_crosschecks_pass,
        checkpoint_path.is_file(),
        checkpoint_metadata_path.is_file(),
    ]
    complete = completed_substeps == TOTAL_SUBSTEPS
    if complete:
        checks.extend(
            [
                len(rows) == 151,
                len(endpoint_crosschecks) == 16,
                rows[0]["time"] == WINDOW,
                abs(rows[-1]["time"] - 6.0) < 1.0e-14,
                max(
                    row["maximum_boundary_l2_difference_upper"]
                    for row in rows
                )
                < 7.0e-4,
            ]
        )
    return {
        "kind": (
            "neutral_strip_within_window_source_grid_propagation_certificate"
        ),
        "model": (
            "certified 112-source full-minus-reduced common-circle boundary "
            "values on the 3/80 grid from t=3/8 through t=6"
        ),
        "status": "complete" if complete else "parked",
        "park_reason": park_reason,
        "below_normal_priority_set": priority_set,
        "completed_blocks": completed_blocks,
        "target_blocks": TOTAL_BLOCKS,
        "completed_substeps": completed_substeps,
        "target_substeps": TOTAL_SUBSTEPS,
        "recorded_grid_point_count": len(rows),
        "checkpoint": {
            "npz_path": str(checkpoint_path),
            "npz_sha256": _sha256_file(checkpoint_path),
            "metadata_path": str(checkpoint_metadata_path),
            "metadata_sha256": _sha256_file(checkpoint_metadata_path),
            "loaded": loaded is not None,
        },
        "premise_artifacts": signature,
        "grid": {
            "substep": SUBSTEP,
            "substeps_per_3_over_8_block": SUBSTEPS_PER_BLOCK,
            "total_blocks_from_time_0_to_6": TOTAL_BLOCKS,
            "finite_window_count_from_time_3_over_8_to_6": 15,
            "first_recorded_time": WINDOW,
            "last_target_time": 6.0,
        },
        "error_model": {
            "full_substep_operator_error_upper": step_error,
            "full_computational_substep_norm_upper": computational_norm,
            "maximum_source_state_norm": maximum_source_norm,
            "source_construction_relative_error_upper": (
                source_construction_relative_error
            ),
            "first_endpoint_reduced_form_error_upper": (
                first_reduced_form_error
            ),
            "first_endpoint_reduced_dense_error_upper": (
                first_reduced_dense_error
            ),
            "later_reduced_error_factor": (
                "(t/(3/8))*exp(-2.36*(t-3/8))"
            ),
            "exact_boundary_operator_norm_upper": exact_output_norm,
            "boundary_operator_difference_upper": output_operator_error,
        },
        "grid_rows": rows,
        "production_endpoint_crosschecks": endpoint_crosschecks,
        "all_production_endpoint_crosschecks_pass": (
            all_crosschecks_pass
        ),
        "resource_samples": resource_samples,
        "actual_source_grid_points_certified": bool(
            complete and all(checks)
        ),
        "within_window_suprema_certified": False,
        "post_terminal_time_tail_certified": False,
        "screen_updated": False,
        "checks": checks,
        "all_propagation_integrity_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "resume_command": (
            "python work/ns_collision/scripts/"
            "neutral_strip_within_window_source_grid_propagation.py "
            "--maximum-blocks 16"
        ),
        "next_required_step": (
            "After all 151 grid points are certified, bound the full and "
            "reduced boundary-discrepancy second derivatives on each finite "
            "window and add h^2/8 times that bound to adjacent grid values."
        ),
        "scope": (
            "This certificate controls only the stored finite-chain grid "
            "points. It does not infer between-grid monotonicity, certify "
            "the interpolation remainder or terminal tail, transfer to the "
            "continuum or circle, or prove Navier-Stokes regularity."
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
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-blocks", type=int, default=1)
    parser.add_argument("--cpu-threshold", type=float, default=75.0)
    arguments = parser.parse_args()
    if arguments.maximum_blocks <= 0:
        raise ValueError("--maximum-blocks must be positive")
    payload = audit(
        arguments.eigen_cache,
        arguments.two_block_result,
        arguments.projected_result,
        arguments.coefficient_result,
        arguments.recurrence_result,
        arguments.first_endpoint_result,
        arguments.all_endpoint_result,
        arguments.checkpoint,
        arguments.maximum_blocks,
        arguments.cpu_threshold,
    )
    _atomic_json(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_propagation_integrity_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
