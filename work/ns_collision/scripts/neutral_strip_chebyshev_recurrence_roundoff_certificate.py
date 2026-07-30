#!/usr/bin/env python3
"""Certify one-step sparse Chebyshev recurrence roundoff."""

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
from scipy.sparse import diags, eye


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EIGEN_CACHE = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k241_reference_eigensystem_v1.npz"
)
DEFAULT_COEFFICIENT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_chebyshev_scaling_coefficients_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_chebyshev_recurrence_roundoff_v1.json"
)
DEFAULT_STATE_CACHE = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_chebyshev_recurrence_roundoff_v1.npz"
)
WINDOW = 0.375
DEGREE = 320
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
    return float(
        np.nextafter(product / (1.0 - product), math.inf)
    )


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _down(value: float) -> float:
    return float(np.nextafter(float(value), -math.inf))


def _column_norms_upper(entry_magnitude_upper: np.ndarray) -> np.ndarray:
    rows = entry_magnitude_upper.shape[0]
    squared = entry_magnitude_upper * entry_magnitude_upper
    sums = np.sum(squared, axis=0)
    sums = np.nextafter(
        sums / (1.0 - _gamma(2 * rows + 8)),
        math.inf,
    )
    return np.nextafter(np.sqrt(np.maximum(sums, 0.0)), math.inf)


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


def audit(
    eigen_cache: Path,
    coefficient_result_path: Path,
    state_cache_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    coefficient_result = json.loads(
        coefficient_result_path.read_text(encoding="ascii")
    )
    if not coefficient_result["all_scaling_coefficient_checks_pass"]:
        raise RuntimeError("scaling/coefficient premise is not certified")
    scaling = coefficient_result["matrix_scaling"]
    coefficients_result = coefficient_result["coefficient_intervals"]
    if not scaling[
        "exact_spectrum_inside_scaling_interval_certified"
    ]:
        raise RuntimeError("Chebyshev scaling interval is not certified")
    if not coefficients_result[
        "degree_320_exact_coefficients_and_infinite_tail_certified"
    ]:
        raise RuntimeError("Chebyshev coefficients are not certified")

    base_module = _load_module(
        "neutral_strip_modified_complement_inertia_schur_certificate.py",
        "chebyshev_recurrence_base",
    )
    mesh_module = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "chebyshev_recurrence_mesh",
    )
    mass, stiffness, _, matrix_metadata = (
        base_module._assemble_modified_pencil(0.06, eigen_cache)
    )
    grid = mesh_module._build_mesh(0.06)
    mass_diagonal = np.asarray(mass.diagonal(), dtype=np.float64)
    inverse_square_root_mass = 1.0 / np.sqrt(mass_diagonal)
    normalized_generator = (
        diags(inverse_square_root_mass)
        @ stiffness
        @ diags(inverse_square_root_mass)
    ).tocsr()
    normalized_generator = (
        0.5 * (normalized_generator + normalized_generator.transpose())
    ).tocsr()

    center = 0.5 * (SCALING_UPPER + SCALING_LOWER)
    radius = 0.5 * (SCALING_UPPER - SCALING_LOWER)
    scaled_operator = (
        normalized_generator
        - center * eye(normalized_generator.shape[0], format="csr")
    ) * (1.0 / radius)
    scaled_operator = scaled_operator.tocsr()
    maximum_scaled_row_nonzeros = int(
        np.max(np.diff(scaled_operator.indptr))
    )
    central_scaled_row_sum = float(
        np.max(np.asarray(abs(scaled_operator).sum(axis=1)).ravel())
    )
    scaled_construction_error_upper = _up(
        _gamma(12)
        * (
            float(
                scaling[
                    "central_binary64_gershgorin_upper"
                ]
            )
            + center
        )
        / radius
    )
    equivalent_generator_scaling_error_upper = _up(
        radius * scaled_construction_error_upper
    )
    target_generator_error_upper = _up(
        float(
            scaling["normalized_generator_construction_error_upper"]
        )
        + equivalent_generator_scaling_error_upper
    )
    central_generator_floor_lower = _down(
        float(scaling["certified_two_block_global_floor_lower"])
        - target_generator_error_upper
    )
    central_generator_upper = _up(
        float(
            scaling[
                "exact_stored_normalized_generator_gershgorin_upper"
            ]
        )
        + target_generator_error_upper
    )
    stored_scaled_spectrum_contained = bool(
        central_generator_floor_lower > SCALING_LOWER
        and central_generator_upper < SCALING_UPPER
    )

    entry_states = np.asarray(grid["entry_states"], dtype=np.int64)
    entry_count = len(entry_states)
    source_state = np.zeros(
        (normalized_generator.shape[0], entry_count),
        dtype=np.float64,
    )
    source_state[entry_states, np.arange(entry_count)] = (
        inverse_square_root_mass[entry_states]
    )
    source_norms = np.linalg.norm(source_state, axis=0)
    maximum_source_norm = _up(float(np.max(source_norms)))

    coefficient_rows = coefficients_result["rows"]
    if len(coefficient_rows) != DEGREE + 1:
        raise RuntimeError("coefficient row count changed")
    coefficients = np.asarray(
        [float(row["scipy_central"]) for row in coefficient_rows],
        dtype=np.float64,
    )

    first = source_state.copy()
    second, first_action_error_entries = _sparse_action_with_error(
        scaled_operator, first
    )
    first_action_error_norms = _column_norms_upper(
        first_action_error_entries
    )
    recurrence_local_error_norms = np.zeros(
        (DEGREE + 1, entry_count), dtype=np.float64
    )
    recurrence_local_error_norms[1, :] = first_action_error_norms

    result = coefficients[0] * first + coefficients[1] * second
    absolute_term_sum = (
        abs(coefficients[0]) * np.abs(first)
        + abs(coefficients[1]) * np.abs(second)
    )
    maximum_computed_recurrence_state_norm = max(
        float(np.max(np.linalg.norm(first, axis=0))),
        float(np.max(np.linalg.norm(second, axis=0))),
    )
    for order in range(1, DEGREE):
        action, action_error_entries = _sparse_action_with_error(
            scaled_operator, second
        )
        following = 2.0 * action - first
        recurrence_arithmetic_error = np.nextafter(
            2.0 * action_error_entries
            + _gamma(6) * (2.0 * np.abs(action) + np.abs(first)),
            math.inf,
        )
        recurrence_local_error_norms[order + 1, :] = (
            _column_norms_upper(recurrence_arithmetic_error)
        )
        result += coefficients[order + 1] * following
        absolute_term_sum += (
            abs(coefficients[order + 1]) * np.abs(following)
        )
        maximum_computed_recurrence_state_norm = max(
            maximum_computed_recurrence_state_norm,
            float(np.max(np.linalg.norm(following, axis=0))),
        )
        first, second = second, following

    recurrence_state_error_bounds = np.zeros_like(
        recurrence_local_error_norms
    )
    for order in range(1, DEGREE + 1):
        bound = order * first_action_error_norms
        for local_order in range(2, order + 1):
            bound += (
                order - local_order + 1
            ) * recurrence_local_error_norms[local_order, :]
        recurrence_state_error_bounds[order, :] = np.nextafter(
            bound / (1.0 - _gamma(2 * order + 16)),
            math.inf,
        )

    recurrence_polynomial_error = np.sum(
        np.abs(coefficients)[:, None]
        * recurrence_state_error_bounds,
        axis=0,
    )
    recurrence_polynomial_error = np.nextafter(
        recurrence_polynomial_error
        / (1.0 - _gamma(2 * (DEGREE + 1) + 16)),
        math.inf,
    )
    accumulation_error_entries = np.nextafter(
        _gamma(2 * (DEGREE + 1) + 16) * absolute_term_sum,
        math.inf,
    )
    accumulation_error_norms = _column_norms_upper(
        accumulation_error_entries
    )

    coefficient_l1_error = float(
        coefficients_result["scipy_coefficient_l1_error_upper"]
    )
    exact_tail_upper = float(
        coefficients_result["tail"]["upper"]
    )
    coefficient_action_error = np.nextafter(
        coefficient_l1_error * source_norms, math.inf
    )
    truncation_error = np.nextafter(
        exact_tail_upper * source_norms, math.inf
    )
    source_construction_error = np.nextafter(
        _gamma(6) * source_norms, math.inf
    )
    operator_perturbation_error = np.nextafter(
        WINDOW
        * target_generator_error_upper
        * math.exp(-WINDOW * SCALING_LOWER)
        * source_norms,
        math.inf,
    )
    total_state_action_error = np.nextafter(
        recurrence_polynomial_error
        + accumulation_error_norms
        + coefficient_action_error
        + truncation_error
        + source_construction_error
        + operator_perturbation_error,
        math.inf,
    )

    maxima = {
        "initial_sparse_action_roundoff_upper": _up(
            float(np.max(first_action_error_norms))
        ),
        "local_recurrence_roundoff_upper": _up(
            float(np.max(recurrence_local_error_norms[2:, :]))
        ),
        "chebyshev_state_recurrence_error_upper": _up(
            float(np.max(recurrence_polynomial_error))
        ),
        "polynomial_accumulation_roundoff_upper": _up(
            float(np.max(accumulation_error_norms))
        ),
        "scipy_coefficient_action_error_upper": _up(
            float(np.max(coefficient_action_error))
        ),
        "exact_polynomial_tail_action_error_upper": _up(
            float(np.max(truncation_error))
        ),
        "source_construction_error_upper": _up(
            float(np.max(source_construction_error))
        ),
        "exact_to_computational_generator_action_error_upper": _up(
            float(np.max(operator_perturbation_error))
        ),
        "total_one_step_state_action_error_upper": _up(
            float(np.max(total_state_action_error))
        ),
    }
    checks = [
        priority_set,
        matrix_metadata["mass_diagonal_strictly_positive"],
        matrix_metadata["stiffness_exactly_symmetric"],
        stored_scaled_spectrum_contained,
        maximum_scaled_row_nonzeros <= 10,
        central_scaled_row_sum < 2.0,
        maximum_source_norm < 66.0,
        maxima["initial_sparse_action_roundoff_upper"] < 1.0e-11,
        maxima["local_recurrence_roundoff_upper"] < 1.0e-11,
        maxima["chebyshev_state_recurrence_error_upper"] < 1.0e-8,
        maxima["polynomial_accumulation_roundoff_upper"] < 1.0e-10,
        maxima["total_one_step_state_action_error_upper"] < 1.0e-7,
    ]
    _atomic_npz(
        state_cache_path,
        {
            "computed_one_step_state": result,
            "source_state": source_state,
            "entry_states": entry_states,
        },
    )
    return {
        "kind": "neutral_strip_chebyshev_recurrence_roundoff_certificate",
        "model": (
            "directed one-step sparse Chebyshev action error for all 112 "
            "stored modified-chain point sources"
        ),
        "below_normal_priority_set": priority_set,
        "premise_artifacts": {
            "eigen_cache": str(eigen_cache),
            "eigen_cache_sha256": _sha256_file(eigen_cache),
            "scaling_coefficient_result": str(
                coefficient_result_path
            ),
            "scaling_coefficient_result_sha256": _sha256_file(
                coefficient_result_path
            ),
        },
        "operator": {
            "state_count": int(normalized_generator.shape[0]),
            "entry_count": entry_count,
            "degree": DEGREE,
            "window": WINDOW,
            "maximum_scaled_row_nonzeros": maximum_scaled_row_nonzeros,
            "central_scaled_absolute_row_sum_upper": (
                central_scaled_row_sum
            ),
            "scaled_operator_construction_error_upper": (
                scaled_construction_error_upper
            ),
            "equivalent_generator_scaling_error_upper": (
                equivalent_generator_scaling_error_upper
            ),
            "exact_to_computational_generator_error_upper": (
                target_generator_error_upper
            ),
            "computational_generator_floor_lower": (
                central_generator_floor_lower
            ),
            "computational_generator_upper": central_generator_upper,
            "computational_scaled_spectrum_inside_unit_interval_certified": (
                stored_scaled_spectrum_contained
            ),
            "maximum_source_state_norm": maximum_source_norm,
            "maximum_computed_chebyshev_state_norm": (
                maximum_computed_recurrence_state_norm
            ),
        },
        "stability": {
            "identity": (
                "e_n=U_(n-1)(X)e_1+sum_(r=2)^n "
                "U_(n-r)(X)delta_r"
            ),
            "u_polynomial_bound": (
                "||U_j(X)||<=j+1 for self-adjoint spectrum in [-1,1]"
            ),
            "raw_exponential_norm_recurrence_used": False,
        },
        "maximum_error_components": maxima,
        "computed_one_step_state_sha256": _sha256_array(result),
        "computed_state_cache": {
            "path": str(state_cache_path),
            "sha256": _sha256_file(state_cache_path),
        },
        "one_step_full_state_chebyshev_action_roundoff_certified": bool(
            all(checks)
        ),
        "reduced_semigroup_roundoff_enclosed": False,
        "boundary_output_roundoff_enclosed": False,
        "sixteen_step_error_propagation_enclosed": False,
        "within_window_suprema_certified": False,
        "post_terminal_time_tail_certified": False,
        "screen_updated": False,
        "checks": checks,
        "all_recurrence_roundoff_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Enclose the reduced 240-dimensional action and common-circle "
            "boundary multiplication, then propagate the one-step error "
            "through the 16 resumable steps."
        ),
        "scope": (
            "This certificate closes one-step full-state sparse action "
            "roundoff for the stored finite chain. It does not yet certify "
            "the reduced subtraction, boundary output, time-window suprema, "
            "continuum transfer, polygon-circle transfer, or regularity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigen-cache", type=Path, default=DEFAULT_EIGEN_CACHE)
    parser.add_argument(
        "--coefficient-result",
        type=Path,
        default=DEFAULT_COEFFICIENT_RESULT,
    )
    parser.add_argument(
        "--state-cache", type=Path, default=DEFAULT_STATE_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = audit(
        arguments.eigen_cache,
        arguments.coefficient_result,
        arguments.state_cache,
    )
    _atomic_json(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
