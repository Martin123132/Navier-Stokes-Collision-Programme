#!/usr/bin/env python3
"""Certify the stored-chain boundary-discrepancy tail after t=6."""

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
from scipy.linalg import cholesky, solve_triangular


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
DEFAULT_FIRST_ENDPOINT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_first_endpoint_boundary_leakage_v1.json"
)
DEFAULT_GRID_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_source_grid_propagation_v1.json"
)
DEFAULT_FINITE_WINDOW_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_second_derivative_v1.json"
)
DEFAULT_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_within_window_source_grid_checkpoint_v1.npz"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_post_terminal_boundary_tail_v1.json"
)
WINDOW = 0.375
TERMINAL_TIME = 6.0
FULL_FLOOR = 1.9
REDUCED_FLOOR = 2.36
FORM_FLOOR = 4.832287335665


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


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _exp_upper(exponent: float) -> float:
    mpmath.iv.dps = 70
    numerator, denominator = float(exponent).as_integer_ratio()
    exact_exponent = mpmath.iv.mpf(numerator) / denominator
    return _up(float(mpmath.iv.exp(exact_exponent).b))


def audit(
    eigen_cache: Path,
    two_block_result_path: Path,
    projected_result_path: Path,
    first_endpoint_result_path: Path,
    grid_result_path: Path,
    finite_window_result_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    two_block = json.loads(
        two_block_result_path.read_text(encoding="ascii")
    )
    projected = json.loads(
        projected_result_path.read_text(encoding="ascii")
    )
    first_endpoint = json.loads(
        first_endpoint_result_path.read_text(encoding="ascii")
    )
    grid_result = json.loads(
        grid_result_path.read_text(encoding="ascii")
    )
    finite_window = json.loads(
        finite_window_result_path.read_text(encoding="ascii")
    )
    if not two_block["all_modified_two_block_leakage_checks_pass"]:
        raise RuntimeError("two-block premise is not certified")
    if not first_endpoint["all_first_endpoint_boundary_checks_pass"]:
        raise RuntimeError("boundary-output premise is not certified")
    if not grid_result["actual_source_grid_points_certified"]:
        raise RuntimeError("terminal source-grid premise is not certified")
    if not finite_window["all_fifteen_within_window_suprema_certified"]:
        raise RuntimeError("finite-window premise is not certified")

    checkpoint = grid_result["checkpoint"]
    expected_checkpoint_path = Path(checkpoint["npz_path"])
    if checkpoint_path != expected_checkpoint_path:
        raise RuntimeError("terminal checkpoint path changed")
    checkpoint_hash = _sha256_file(checkpoint_path)
    if checkpoint_hash != checkpoint["npz_sha256"]:
        raise RuntimeError("terminal checkpoint hash changed")
    metadata_path = Path(checkpoint["metadata_path"])
    if _sha256_file(metadata_path) != checkpoint["metadata_sha256"]:
        raise RuntimeError("terminal checkpoint metadata hash changed")
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    if metadata["completed_substeps"] != 160:
        raise RuntimeError("terminal checkpoint is incomplete")

    base_module = _load_module(
        "neutral_strip_modified_complement_inertia_schur_certificate.py",
        "post_terminal_tail_base",
    )
    endpoint_module = _load_module(
        "neutral_strip_first_endpoint_boundary_leakage_certificate.py",
        "post_terminal_tail_endpoint",
    )
    source_module = _load_module(
        "neutral_strip_common_circle_source_time_slab_certificate.py",
        "post_terminal_tail_source",
    )
    mass, _, vectors, matrix_metadata = (
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

    mass_diagonal = np.asarray(mass.diagonal(), dtype=np.float64)
    square_root_mass = np.sqrt(mass_diagonal)
    restricted_mass = vectors.transpose() @ (mass @ vectors)
    restricted_mass = 0.5 * (
        restricted_mass + restricted_mass.transpose()
    )
    lower = cholesky(restricted_mass, lower=True)
    orthonormal_trial = solve_triangular(
        lower,
        (square_root_mass[:, None] * vectors).transpose(),
        lower=True,
    ).transpose()

    with np.load(checkpoint_path, allow_pickle=False) as cached:
        completed_substeps = int(cached["completed_substeps"].item())
        full_state = np.asarray(cached["full_state"], dtype=np.float64)
        reduced_coordinates = np.asarray(
            cached["reduced_coordinates"],
            dtype=np.float64,
        )
    if completed_substeps != 160:
        raise RuntimeError("terminal NPZ is incomplete")
    if full_state.shape != (15211, 112):
        raise RuntimeError("terminal full-state shape changed")
    if reduced_coordinates.shape != (240, 112):
        raise RuntimeError("terminal reduced-coordinate shape changed")

    terminal_row = grid_result["grid_rows"][-1]
    if abs(float(terminal_row["time"]) - TERMINAL_TIME) > 1.0e-14:
        raise RuntimeError("terminal grid row changed")
    full_central_norms = endpoint_module._column_norms_upper(
        np.abs(full_state)
    )
    reduced_state, reduced_product_error = (
        endpoint_module._dense_action_with_error(
            orthonormal_trial,
            reduced_coordinates,
        )
    )
    reduced_central_norms = endpoint_module._column_norms_upper(
        np.abs(reduced_state) + reduced_product_error
    )
    full_state_error = float(
        terminal_row["full_repeated_state_error_upper"]
    )
    reduced_state_error = _up(
        float(terminal_row["reduced_form_state_error_upper"])
        + float(terminal_row["reduced_dense_state_error_upper"])
    )
    full_exact_norms = np.nextafter(
        full_central_norms + full_state_error,
        math.inf,
    )
    reduced_exact_norms = np.nextafter(
        reduced_central_norms + reduced_state_error,
        math.inf,
    )
    exact_output_norm = float(
        first_endpoint["boundary_operator"]["exact_operator_norm_upper"]
    )
    full_terminal_amplitudes = np.nextafter(
        exact_output_norm * full_exact_norms,
        math.inf,
    )
    reduced_terminal_amplitudes = np.nextafter(
        exact_output_norm * reduced_exact_norms,
        math.inf,
    )

    full_tail_ratio = _exp_upper(-FULL_FLOOR * WINDOW)
    reduced_tail_ratio = _exp_upper(-REDUCED_FLOOR * WINDOW)
    if max(full_tail_ratio, reduced_tail_ratio) >= 1.0:
        raise RuntimeError("tail ratio is not contractive")
    full_geometric_sums = np.nextafter(
        full_terminal_amplitudes / (1.0 - full_tail_ratio),
        math.inf,
    )
    reduced_geometric_sums = np.nextafter(
        reduced_terminal_amplitudes
        / (1.0 - reduced_tail_ratio),
        math.inf,
    )
    axial_tail_upper = float(
        source_module._axial_l2_global_upper(TERMINAL_TIME)
    )
    stored_axial_upper = float(terminal_row["axial_l2_upper"])
    if axial_tail_upper > stored_axial_upper:
        raise RuntimeError("terminal axial premise rounded downward")
    per_source_raw_tail_uppers = np.nextafter(
        stored_axial_upper
        * (full_geometric_sums + reduced_geometric_sums),
        math.inf,
    )
    maximizing_source = int(np.argmax(per_source_raw_tail_uppers))
    raw_tail_upper = _up(float(np.max(per_source_raw_tail_uppers)))
    tail_screen_charge = _up(
        (WINDOW + 1.0 / FORM_FLOOR) * raw_tail_upper
    )
    finite_screen_charge = float(
        finite_window["finite_screen_charge_upper"]
    )
    existing_screen = float(
        finite_window["existing_certified_screen_upper"]
    )
    complete_screen = _up(
        existing_screen + finite_screen_charge + tail_screen_charge
    )
    complete_headroom = float(
        np.nextafter(1.0 - complete_screen, -math.inf)
    )

    checks = [
        priority_set,
        matrix_hashes_match,
        completed_substeps == 160,
        full_state.shape == (15211, 112),
        reduced_coordinates.shape == (240, 112),
        float(terminal_row["time"]) == TERMINAL_TIME,
        FULL_FLOOR > 0.0,
        REDUCED_FLOOR > 0.0,
        full_tail_ratio < 1.0,
        reduced_tail_ratio < 1.0,
        axial_tail_upper <= stored_axial_upper,
        raw_tail_upper > 0.0,
        tail_screen_charge > 0.0,
        complete_screen < 1.0,
    ]
    premise_artifacts = {
        "eigen_cache": str(eigen_cache),
        "eigen_cache_sha256": _sha256_file(eigen_cache),
        "two_block_result": str(two_block_result_path),
        "two_block_result_sha256": _sha256_file(two_block_result_path),
        "projected_result": str(projected_result_path),
        "projected_result_sha256": _sha256_file(projected_result_path),
        "first_endpoint_result": str(first_endpoint_result_path),
        "first_endpoint_result_sha256": _sha256_file(
            first_endpoint_result_path
        ),
        "grid_result": str(grid_result_path),
        "grid_result_sha256": _sha256_file(grid_result_path),
        "finite_window_result": str(finite_window_result_path),
        "finite_window_result_sha256": _sha256_file(
            finite_window_result_path
        ),
        "terminal_checkpoint": str(checkpoint_path),
        "terminal_checkpoint_sha256": checkpoint_hash,
        "terminal_checkpoint_metadata": str(metadata_path),
        "terminal_checkpoint_metadata_sha256": _sha256_file(
            metadata_path
        ),
    }
    return {
        "kind": (
            "neutral_strip_post_terminal_boundary_tail_certificate"
        ),
        "model": (
            "source-oriented geometric boundary-discrepancy tail from "
            "the exact stored full and reduced terminal state norms"
        ),
        "below_normal_priority_set": priority_set,
        "premise_artifacts": premise_artifacts,
        "matrix_hashes_match": matrix_hashes_match,
        "terminal_time": TERMINAL_TIME,
        "tail_first_window_index": 16,
        "terminal_state_enclosure": {
            "maximum_full_central_state_norm_upper": _up(
                float(np.max(full_central_norms))
            ),
            "full_state_error_upper": full_state_error,
            "maximum_full_exact_state_norm_upper": _up(
                float(np.max(full_exact_norms))
            ),
            "maximum_reduced_central_state_norm_upper": _up(
                float(np.max(reduced_central_norms))
            ),
            "reduced_state_error_upper": reduced_state_error,
            "maximum_reduced_exact_state_norm_upper": _up(
                float(np.max(reduced_exact_norms))
            ),
            "exact_boundary_output_norm_upper": exact_output_norm,
            "maximum_full_terminal_boundary_amplitude_upper": _up(
                float(np.max(full_terminal_amplitudes))
            ),
            "maximum_reduced_terminal_boundary_amplitude_upper": _up(
                float(np.max(reduced_terminal_amplitudes))
            ),
        },
        "geometric_tail": {
            "window_length": WINDOW,
            "full_decay_floor_lower": FULL_FLOOR,
            "reduced_decay_floor_lower": REDUCED_FLOOR,
            "full_window_ratio_upper": full_tail_ratio,
            "reduced_window_ratio_upper": reduced_tail_ratio,
            "axial_tail_upper": stored_axial_upper,
            "maximum_full_geometric_boundary_sum_upper": _up(
                float(np.max(full_geometric_sums))
            ),
            "maximum_reduced_geometric_boundary_sum_upper": _up(
                float(np.max(reduced_geometric_sums))
            ),
            "maximizing_source_index": maximizing_source,
            "post_terminal_raw_sum_upper": raw_tail_upper,
        },
        "screen_composition": {
            "existing_certified_screen_upper": existing_screen,
            "finite_window_boundary_charge_upper": (
                finite_screen_charge
            ),
            "post_terminal_boundary_charge_upper": tail_screen_charge,
            "complete_stored_chain_screen_upper": complete_screen,
            "complete_stored_chain_screen_headroom_lower": (
                complete_headroom
            ),
            "complete_stored_chain_screen_below_one": bool(
                complete_screen < 1.0
            ),
        },
        "post_terminal_time_tail_certified": bool(all(checks)),
        "stored_finite_chain_boundary_leakage_screen_complete": bool(
            all(checks)
        ),
        "screen_updated": bool(all(checks)),
        "checks": checks,
        "all_post_terminal_tail_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Audit the remaining continuum Ritz-projector and "
            "polygon-to-circle domain-transfer gates before treating the "
            "stored-chain screen as a continuum Navier-Stokes estimate."
        ),
        "scope": (
            "This certificate closes only the post-t=6 tail and stored "
            "finite-chain boundary-leakage screen. It does not certify "
            "continuum Ritz transfer, polygon-circle domain transfer, or "
            "Navier-Stokes regularity."
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
        "--first-endpoint-result",
        type=Path,
        default=DEFAULT_FIRST_ENDPOINT_RESULT,
    )
    parser.add_argument(
        "--grid-result",
        type=Path,
        default=DEFAULT_GRID_RESULT,
    )
    parser.add_argument(
        "--finite-window-result",
        type=Path,
        default=DEFAULT_FINITE_WINDOW_RESULT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = audit(
        arguments.eigen_cache,
        arguments.two_block_result,
        arguments.projected_result,
        arguments.first_endpoint_result,
        arguments.grid_result,
        arguments.finite_window_result,
        arguments.checkpoint,
    )
    _atomic_json(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_post_terminal_tail_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
