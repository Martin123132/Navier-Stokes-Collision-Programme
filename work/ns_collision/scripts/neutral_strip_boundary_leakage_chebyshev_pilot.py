#!/usr/bin/env python3
"""Run a resumable source-oriented boundary-leakage Chebyshev pilot."""

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
from scipy.special import ive


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EIGEN_CACHE = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k241_reference_eigensystem_v1.npz"
)
DEFAULT_COMPLEMENT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_modified_complement_inertia_schur_v1.json"
)
DEFAULT_TWO_BLOCK_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_modified_two_block_leakage_v1.json"
)
DEFAULT_PROJECTED_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_projected_interval_two_block_transfer_v1.json"
)
DEFAULT_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_boundary_leakage_chebyshev_pilot_checkpoint_v1.npz"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_boundary_leakage_chebyshev_pilot_v1.json"
)
WINDOW = 0.375
STEP_COUNT = 16
DEGREE = 320
SCALING_LOWER = 1.9
SCALING_UPPER = 8000.0
FORM_FLOOR = 4.832287335665
CONVERGENCE_DEGREES = (200, 240, 280, 320)


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


def _global_two_block_floor(
    low_floor: float,
    high_floor: float,
    coupling: float,
) -> float:
    midpoint = 0.5 * (low_floor + high_floor)
    half_gap = 0.5 * (high_floor - low_floor)
    return midpoint - math.hypot(half_gap, coupling)


def _chebyshev_coefficients(
    time_value: float,
    degree: int,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, float]:
    radius = 0.5 * (upper - lower)
    argument = time_value * radius
    orders = np.arange(degree + 1)
    coefficients = np.exp(-time_value * lower) * ive(orders, argument)
    coefficients[1:] *= 2.0
    coefficients[1::2] *= -1.0
    tail_orders = np.arange(degree + 1, max(degree + 2001, 2501))
    sampled_tail = float(
        2.0
        * np.exp(-time_value * lower)
        * np.sum(ive(tail_orders, argument))
    )
    return coefficients, sampled_tail


def _chebyshev_step(
    scaled_operator,
    state: np.ndarray,
    coefficients: np.ndarray,
    snapshot_degrees: set[int] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    degree = len(coefficients) - 1
    if degree < 1:
        raise ValueError("degree must be at least one")
    first = np.asarray(state, dtype=np.float64)
    second = np.asarray(scaled_operator @ first)
    result = coefficients[0] * first + coefficients[1] * second
    snapshots: dict[int, np.ndarray] = {}
    requested = snapshot_degrees or set()
    if 1 in requested:
        snapshots[1] = result.copy()
    for index in range(1, degree):
        following = 2.0 * np.asarray(scaled_operator @ second) - first
        result += coefficients[index + 1] * following
        first, second = second, following
        current_degree = index + 1
        if current_degree in requested:
            snapshots[current_degree] = result.copy()
    return result, snapshots


def _checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".json")


def _write_checkpoint(
    checkpoint_path: Path,
    signature: dict[str, Any],
    completed_step: int,
    full_state: np.ndarray,
    reduced_coordinates: np.ndarray,
    rows: list[dict[str, Any]],
    convergence_rows: list[dict[str, Any]],
) -> None:
    _atomic_npz(
        checkpoint_path,
        {
            "completed_step": np.asarray(completed_step, dtype=np.int64),
            "full_state": np.asarray(full_state, dtype=np.float64),
            "reduced_coordinates": np.asarray(
                reduced_coordinates, dtype=np.float64
            ),
        },
    )
    _atomic_json(
        _checkpoint_metadata_path(checkpoint_path),
        {
            "kind": "boundary_leakage_chebyshev_pilot_checkpoint",
            "signature": signature,
            "completed_step": completed_step,
            "rows": rows,
            "first_step_degree_convergence": convergence_rows,
            "npz_path": str(checkpoint_path),
            "npz_sha256": _sha256_file(checkpoint_path),
        },
    )


def _load_checkpoint(
    checkpoint_path: Path,
    signature: dict[str, Any],
) -> tuple[int, np.ndarray, np.ndarray, list[dict], list[dict]] | None:
    metadata_path = _checkpoint_metadata_path(checkpoint_path)
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    if metadata.get("signature") != signature:
        return None
    if metadata.get("npz_sha256") != _sha256_file(checkpoint_path):
        raise RuntimeError("Chebyshev checkpoint NPZ hash mismatch")
    with np.load(checkpoint_path, allow_pickle=False) as cached:
        completed_step = int(cached["completed_step"].item())
        full_state = np.asarray(cached["full_state"], dtype=np.float64)
        reduced_coordinates = np.asarray(
            cached["reduced_coordinates"], dtype=np.float64
        )
    if completed_step != int(metadata["completed_step"]):
        raise RuntimeError("Chebyshev checkpoint step mismatch")
    return (
        completed_step,
        full_state,
        reduced_coordinates,
        list(metadata["rows"]),
        list(metadata["first_step_degree_convergence"]),
    )


def audit(
    eigen_cache: Path,
    complement_result_path: Path,
    two_block_result_path: Path,
    projected_result_path: Path,
    checkpoint_path: Path,
    maximum_steps: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    complement = json.loads(
        complement_result_path.read_text(encoding="ascii")
    )
    two_block = json.loads(
        two_block_result_path.read_text(encoding="ascii")
    )
    projected = json.loads(
        projected_result_path.read_text(encoding="ascii")
    )
    if not complement["all_modified_complement_checks_pass"]:
        raise RuntimeError("modified-complement premise is not certified")
    if not two_block["all_modified_two_block_leakage_checks_pass"]:
        raise RuntimeError("two-block premise is not certified")
    expected_complement_hash = two_block["premise_artifacts"][
        "modified_complement_result_sha256"
    ]
    if _sha256_file(complement_result_path) != expected_complement_hash:
        raise RuntimeError("modified-complement result hash changed")

    base_module = _load_module(
        "neutral_strip_modified_complement_inertia_schur_certificate.py",
        "boundary_leakage_chebyshev_base",
    )
    mesh_module = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "boundary_leakage_chebyshev_mesh",
    )
    source_module = _load_module(
        "neutral_strip_common_circle_source_time_slab_certificate.py",
        "boundary_leakage_chebyshev_source",
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
    reduced_generator = orthonormal_trial.transpose() @ (
        normalized_generator @ orthonormal_trial
    )
    reduced_generator = 0.5 * (
        reduced_generator + reduced_generator.transpose()
    )
    reduced_values, reduced_vectors = eigh(reduced_generator)

    entry_states = np.asarray(grid["entry_states"], dtype=np.int64)
    entry_count = len(entry_states)
    full_state = np.zeros(
        (normalized_generator.shape[0], entry_count), dtype=np.float64
    )
    full_state[entry_states, np.arange(entry_count)] = (
        inverse_square_root_mass[entry_states]
    )
    reduced_coordinates = (
        orthonormal_trial[entry_states, :].transpose()
        * inverse_square_root_mass[entry_states][None, :]
    )

    modified_boundary = (
        mass @ grid["inner_rate_matrix"]
    ).tocsr()
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

    coupling_parameters = two_block["certified_parameters"]
    low_floor = float(coupling_parameters["low_block_floor"])
    high_floor = float(coupling_parameters["high_block_floor"])
    coupling = float(coupling_parameters["off_block_coupling_upper"])
    global_floor = _global_two_block_floor(
        low_floor, high_floor, coupling
    )
    central_gershgorin_upper = float(
        np.max(np.asarray(abs(normalized_generator).sum(axis=1)).ravel())
    )
    center = 0.5 * (SCALING_UPPER + SCALING_LOWER)
    radius = 0.5 * (SCALING_UPPER - SCALING_LOWER)
    scaled_operator = (
        normalized_generator
        - center * eye(normalized_generator.shape[0], format="csr")
    ) * (1.0 / radius)
    coefficients, sampled_tail = _chebyshev_coefficients(
        WINDOW, DEGREE, SCALING_LOWER, SCALING_UPPER
    )
    reduced_step = (
        reduced_vectors
        * np.exp(-WINDOW * reduced_values)[None, :]
    ) @ reduced_vectors.transpose()

    signature = {
        "schema_version": 1,
        "eigen_cache_sha256": _sha256_file(eigen_cache),
        "complement_result_sha256": _sha256_file(complement_result_path),
        "two_block_result_sha256": _sha256_file(two_block_result_path),
        "projected_result_sha256": _sha256_file(projected_result_path),
        "mass_sha256": matrix_metadata["mass_sha256"],
        "stiffness_sha256": matrix_metadata["stiffness_sha256"],
        "retained_vectors_sha256": matrix_metadata[
            "retained_vectors_sha256"
        ],
        "window": WINDOW,
        "degree": DEGREE,
        "scaling_lower": SCALING_LOWER,
        "scaling_upper": SCALING_UPPER,
        "entry_count": entry_count,
    }
    loaded = _load_checkpoint(checkpoint_path, signature)
    if loaded is None:
        completed_step = 0
        rows: list[dict[str, Any]] = []
        convergence_rows: list[dict[str, Any]] = []
    else:
        (
            completed_step,
            full_state,
            reduced_coordinates,
            rows,
            convergence_rows,
        ) = loaded

    target_step = min(STEP_COUNT, completed_step + maximum_steps)
    while completed_step < target_step:
        requested = (
            set(CONVERGENCE_DEGREES) if completed_step == 0 else set()
        )
        full_state, snapshots = _chebyshev_step(
            scaled_operator,
            full_state,
            coefficients,
            requested,
        )
        reduced_coordinates = reduced_step @ reduced_coordinates
        reduced_state = orthonormal_trial @ reduced_coordinates
        difference = full_state - reduced_state
        boundary_difference = np.asarray(output_operator @ difference)
        boundary_norms = np.linalg.norm(boundary_difference, axis=0)
        state_norms = np.linalg.norm(difference, axis=0)
        completed_step += 1
        time_value = completed_step * WINDOW
        rows.append(
            {
                "step": completed_step,
                "time": time_value,
                "maximum_boundary_l2_difference": float(
                    np.max(boundary_norms)
                ),
                "maximizing_entry_index": int(np.argmax(boundary_norms)),
                "maximum_state_l2_difference": float(np.max(state_norms)),
                "axial_l2_upper": float(
                    source_module._axial_l2_global_upper(time_value)
                ),
            }
        )
        if completed_step == 1:
            reference_output = boundary_difference
            reference_norms = boundary_norms
            for snapshot_degree in CONVERGENCE_DEGREES:
                snapshot_difference = (
                    snapshots[snapshot_degree] - reduced_state
                )
                snapshot_output = np.asarray(
                    output_operator @ snapshot_difference
                )
                snapshot_norms = np.linalg.norm(
                    snapshot_output, axis=0
                )
                _, degree_tail = _chebyshev_coefficients(
                    WINDOW,
                    snapshot_degree,
                    SCALING_LOWER,
                    SCALING_UPPER,
                )
                convergence_rows.append(
                    {
                        "degree": snapshot_degree,
                        "maximum_boundary_l2_difference": float(
                            np.max(snapshot_norms)
                        ),
                        "maximizing_entry_index": int(
                            np.argmax(snapshot_norms)
                        ),
                        "maximum_output_vector_difference_from_degree_320": (
                            float(
                                np.max(
                                    np.linalg.norm(
                                        snapshot_output
                                        - reference_output,
                                        axis=0,
                                    )
                                )
                            )
                        ),
                        "maximum_output_norm_difference_from_degree_320": (
                            float(
                                np.max(
                                    np.abs(
                                        snapshot_norms - reference_norms
                                    )
                                )
                            )
                        ),
                        "sampled_scalar_tail": degree_tail,
                    }
                )
        _write_checkpoint(
            checkpoint_path,
            signature,
            completed_step,
            full_state,
            reduced_coordinates,
            rows,
            convergence_rows,
        )

    finite_rows = rows[: min(len(rows), STEP_COUNT - 1)]
    finite_weighted_endpoint_sum = float(
        sum(
            float(row["axial_l2_upper"])
            * float(row["maximum_boundary_l2_difference"])
            for row in finite_rows
        )
    )
    finite_endpoint_charge = (
        WINDOW + 1.0 / FORM_FLOOR
    ) * finite_weighted_endpoint_sum
    orthogonality_defect = float(
        np.linalg.norm(
            orthonormal_trial.transpose() @ orthonormal_trial
            - np.eye(orthonormal_trial.shape[1]),
            2,
        )
    )
    checks = [
        priority_set,
        matrix_hashes_match,
        global_floor > SCALING_LOWER,
        central_gershgorin_upper < SCALING_UPPER,
        orthogonality_defect < 2.0e-12,
        len(rows) == completed_step,
        completed_step <= STEP_COUNT,
        all(
            int(row["step"]) == index + 1
            for index, row in enumerate(rows)
        ),
    ]
    return {
        "kind": "neutral_strip_boundary_leakage_chebyshev_pilot",
        "model": (
            "stored modified-chain full point-source semigroup minus its "
            "240-dimensional low-trial semigroup, observed in common-circle "
            "boundary L2"
        ),
        "status": "complete" if completed_step == STEP_COUNT else "parked",
        "below_normal_priority_set": priority_set,
        "completed_step_count": completed_step,
        "target_step_count": STEP_COUNT,
        "checkpoint": {
            "npz_path": str(checkpoint_path),
            "npz_sha256": _sha256_file(checkpoint_path),
            "metadata_path": str(
                _checkpoint_metadata_path(checkpoint_path)
            ),
            "metadata_sha256": _sha256_file(
                _checkpoint_metadata_path(checkpoint_path)
            ),
            "loaded": loaded is not None,
        },
        "premise_artifacts": signature,
        "operator_diagnostics": {
            "state_count": int(normalized_generator.shape[0]),
            "entry_count": entry_count,
            "retained_count": int(orthonormal_trial.shape[1]),
            "certified_two_block_global_floor_lower": global_floor,
            "chebyshev_scaling_lower": SCALING_LOWER,
            "central_binary64_gershgorin_upper": (
                central_gershgorin_upper
            ),
            "chebyshev_scaling_upper": SCALING_UPPER,
            "trial_orthogonality_defect": orthogonality_defect,
            "output_operator_central_spectral_norm": float(
                np.linalg.norm(output_operator.toarray(), 2)
            ),
        },
        "chebyshev_diagnostics": {
            "window": WINDOW,
            "degree": DEGREE,
            "sampled_scalar_tail": sampled_tail,
            "first_step_degree_convergence": convergence_rows,
            "coefficient_evaluation_interval_certified": False,
            "sparse_recurrence_roundoff_enclosed": False,
            "scaling_upper_directed_interval_certified": False,
        },
        "endpoint_rows": rows,
        "finite_endpoint_only_weighted_sum_pilot": (
            finite_weighted_endpoint_sum
        ),
        "finite_endpoint_only_screen_charge_pilot": finite_endpoint_charge,
        "endpoint_supremum_between_grid_times_certified": False,
        "post_terminal_tail_certified": False,
        "boundary_leakage_certificate": False,
        "screen_updated": False,
        "checks": checks,
        "all_pilot_integrity_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Enclose the scaling upper bound, Bessel coefficients, sparse "
            "Chebyshev recurrence roundoff, reduced action, and output "
            "multiplication. Then replace endpoint samples by window suprema "
            "and certify the post-time-6 tail before charging the screen."
        ),
        "scope": (
            "This is a reproducible floating feasibility audit for the "
            "stored binary finite chain. It is not an interval certificate, "
            "a continuum Ritz transfer, a polygon-to-circle transfer, or a "
            "Navier-Stokes regularity proof."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigen-cache", type=Path, default=DEFAULT_EIGEN_CACHE)
    parser.add_argument(
        "--complement-result",
        type=Path,
        default=DEFAULT_COMPLEMENT_RESULT,
    )
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
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-steps", type=int, default=STEP_COUNT)
    arguments = parser.parse_args()
    if arguments.maximum_steps <= 0:
        raise ValueError("--maximum-steps must be positive")
    payload = audit(
        arguments.eigen_cache,
        arguments.complement_result,
        arguments.two_block_result,
        arguments.projected_result,
        arguments.checkpoint,
        arguments.maximum_steps,
    )
    _atomic_json(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
