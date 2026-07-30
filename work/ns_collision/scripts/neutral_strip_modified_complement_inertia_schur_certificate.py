#!/usr/bin/env python3
"""Certify the modified-chain complementary spectral floor."""

from __future__ import annotations

import argparse
import gc
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
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EIGEN_CACHE = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_q12_k241_reference_eigensystem_v1.npz"
)
DEFAULT_LOWER_ROW = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_modified_complement_inertia_lower_v1.json"
)
DEFAULT_UPPER_ROW = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_modified_complement_inertia_upper_v1.json"
)
TARGET_FLOOR = 102.7
RESOLVENT_HALF_WIDTH = 0.1
RETAINED_COUNT = 240
DECIMAL_PRECISION = 220
ORDERING = "MMD_AT_PLUS_A"
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
    flat_size = int(array.size)
    correction = math.sqrt(1.0 - _gamma(2 * flat_size + 10))
    return _up(float(np.linalg.norm(array, "fro")) / correction)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _matrix_fingerprint(matrix: csr_matrix) -> str:
    matrix = matrix.tocsr()
    matrix.sort_indices()
    return _sha256_arrays(
        matrix.indptr,
        matrix.indices,
        matrix.data,
    )


def _atomic_json(path: Path, payload: Any) -> None:
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


def _is_exactly_symmetric(matrix: csr_matrix) -> bool:
    matrix = matrix.tocsr()
    matrix.sort_indices()
    transpose = matrix.transpose().tocsr()
    transpose.sort_indices()
    return bool(
        np.array_equal(matrix.indptr, transpose.indptr)
        and np.array_equal(matrix.indices, transpose.indices)
        and np.array_equal(matrix.data, transpose.data)
    )


def _assemble_modified_pencil(
    spacing: float,
    eigen_cache: Path,
) -> tuple[csr_matrix, csr_matrix, np.ndarray, dict[str, object]]:
    mesh_module = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "modified_complement_boundary_mesh",
    )
    grid = mesh_module._build_mesh(spacing)
    mass_diagonal = np.asarray(grid["state_mass"], dtype=np.float64)
    mass = diags(mass_diagonal).tocsr()
    stiffness = (-mass @ grid["generator"]).tocsr()
    stiffness = (0.5 * (stiffness + stiffness.transpose())).tocsr()
    stiffness.sort_indices()
    mass.sort_indices()

    with np.load(eigen_cache, allow_pickle=False) as cached:
        if int(cached["state_count"].item()) != mass.shape[0]:
            raise RuntimeError("reference eigen cache has the wrong state count")
        if int(cached["requested"].item()) < RETAINED_COUNT:
            raise RuntimeError("reference eigen cache has too few columns")
        vectors = np.asarray(cached["eigenvectors"], dtype=np.float64)[
            :, :RETAINED_COUNT
        ]
        cache_fingerprint = str(
            cached["matrix_fingerprint_sha256"].item()
        )

    metadata = {
        "state_count": int(mass.shape[0]),
        "retained_count": RETAINED_COUNT,
        "minimum_mass_diagonal": float(np.min(mass_diagonal)),
        "maximum_mass_diagonal": float(np.max(mass_diagonal)),
        "mass_diagonal_strictly_positive": bool(
            np.min(mass_diagonal) > 0.0
        ),
        "mass_sha256": _matrix_fingerprint(mass),
        "stiffness_sha256": _matrix_fingerprint(stiffness),
        "stiffness_exactly_symmetric": _is_exactly_symmetric(stiffness),
        "reference_eigen_cache": str(eigen_cache),
        "reference_eigen_cache_sha256": _sha256_file(eigen_cache),
        "reference_matrix_fingerprint_sha256": cache_fingerprint,
        "retained_vectors_sha256": _sha256_arrays(vectors),
    }
    return mass, stiffness, vectors, metadata


def _row_cache_path(row_path: Path) -> Path:
    return row_path.with_suffix(".npz")


def _load_inertia_row(
    row_path: Path,
    expected: dict[str, object],
) -> dict[str, object] | None:
    cache_path = _row_cache_path(row_path)
    if not row_path.is_file() or not cache_path.is_file():
        return None
    try:
        payload = json.loads(row_path.read_text(encoding="ascii"))
    except (OSError, ValueError):
        return None
    for key, value in expected.items():
        if payload.get(key) != value:
            return None
    if payload.get("cache_sha256") != _sha256_file(cache_path):
        return None
    with np.load(cache_path, allow_pickle=False) as cached:
        signs = np.asarray(cached["pivot_sign"])
        if signs.shape != (int(expected["state_count"]),):
            return None
        if int(np.sum(signs < 0)) != RETAINED_COUNT:
            return None
    summary = payload.get("summary", {})
    if not summary.get("all_directed_ldl_checks_pass", False):
        return None
    if int(summary.get("negative_pivot_count", -1)) != RETAINED_COUNT:
        return None
    payload["loaded_from_checkpoint"] = True
    return payload


def _compute_inertia_row(
    mass: csr_matrix,
    stiffness: csr_matrix,
    shift: float,
    row_path: Path,
    common_metadata: dict[str, object],
) -> dict[str, object]:
    expected = {
        "kind": "modified_complement_directed_inertia_row",
        "schema_version": 1,
        "state_count": int(mass.shape[0]),
        "retained_count": RETAINED_COUNT,
        "shift": shift,
        "decimal_precision": DECIMAL_PRECISION,
        "ordering": ORDERING,
        "mass_sha256": common_metadata["mass_sha256"],
        "stiffness_sha256": common_metadata["stiffness_sha256"],
    }
    loaded = _load_inertia_row(row_path, expected)
    if loaded is not None:
        return loaded

    inertia_module = _load_module(
        "neutral_strip_common_circle_sparse_inertia_audit.py",
        f"modified_complement_inertia_{str(shift).replace('.', '_')}",
    )
    summary, cache = inertia_module._directed_ldl_decimal(
        stiffness,
        mass,
        shift,
        0,
        ORDERING,
        DECIMAL_PRECISION,
    )
    cache_path = _row_cache_path(row_path)
    _atomic_npz(cache_path, cache)
    payload = {
        **expected,
        "summary": summary,
        "cache_path": str(cache_path),
        "cache_sha256": _sha256_file(cache_path),
        "loaded_from_checkpoint": False,
    }
    _atomic_json(row_path, payload)
    del cache
    gc.collect()
    return payload


def _schur_certificate(
    mass: csr_matrix,
    stiffness: csr_matrix,
    vectors: np.ndarray,
    target_floor: float,
    spectral_gap_lower: float,
) -> dict[str, object]:
    mass_diagonal = np.asarray(mass.diagonal(), dtype=np.float64)
    right_hand_side = mass_diagonal[:, None] * vectors
    central_pencil = (stiffness - target_floor * mass).tocsc()
    factor = splu(
        central_pencil,
        permc_spec=ORDERING,
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    solution = factor.solve(right_hand_side)

    action = stiffness @ solution
    shifted_mass_action = (
        target_floor * mass_diagonal[:, None] * solution
    )
    central_residual = (
        right_hand_side - action + shifted_mass_action
    )
    maximum_row_nonzeros = int(np.max(np.diff(stiffness.indptr)))
    sparse_product_correction = 1.0 / (
        1.0 - _gamma(2 * maximum_row_nonzeros + 10)
    )
    absolute_terms = (
        np.abs(right_hand_side)
        + sparse_product_correction * (abs(stiffness) @ np.abs(solution))
        + abs(target_floor)
        * mass_diagonal[:, None]
        * np.abs(solution)
    )
    residual_upper = np.nextafter(
        np.abs(central_residual)
        + _gamma(2 * maximum_row_nonzeros + 30) * absolute_terms,
        math.inf,
    )
    weighted_residual_upper = (
        residual_upper / np.sqrt(mass_diagonal)[:, None]
    )
    weighted_residual_frobenius_upper = _frob_upper(
        weighted_residual_upper
    )

    weighted_trial = np.sqrt(mass_diagonal)[:, None] * vectors
    weighted_trial_frobenius_upper = _frob_upper(weighted_trial) * (
        1.0 + _gamma(4)
    )
    solve_error_spectral_upper = _up(
        weighted_trial_frobenius_upper
        * weighted_residual_frobenius_upper
        / spectral_gap_lower
    )

    raw_schur = right_hand_side.T @ solution
    central_schur = 0.5 * (raw_schur + raw_schur.T)
    dense_absolute_product = (
        np.abs(right_hand_side).T @ np.abs(solution)
    ) / (1.0 - _gamma(2 * mass.shape[0] + 10))
    product_error_entries = (
        _gamma(2 * mass.shape[0] + 30) * dense_absolute_product
        + _gamma(4) * np.abs(raw_schur)
    )
    schur_product_error_spectral_upper = _frob_upper(
        product_error_entries
    )

    eigenvalues, eigenvectors = eigh(central_schur)
    reconstructed = (
        eigenvectors * eigenvalues[None, :]
    ) @ eigenvectors.T
    reconstruction_absolute_product = (
        (np.abs(eigenvectors) * np.abs(eigenvalues)[None, :])
        @ np.abs(eigenvectors).T
    ) / (1.0 - _gamma(2 * RETAINED_COUNT + 10))
    reconstruction_error_entries = (
        np.abs(central_schur - reconstructed)
        + _gamma(2 * RETAINED_COUNT + 30)
        * reconstruction_absolute_product
    )
    reconstruction_error_upper = _frob_upper(
        reconstruction_error_entries
    )

    orthogonality_action = eigenvectors.T @ eigenvectors
    orthogonality_absolute_product = (
        np.abs(eigenvectors).T @ np.abs(eigenvectors)
    ) / (1.0 - _gamma(2 * RETAINED_COUNT + 10))
    orthogonality_error_entries = (
        np.abs(
            orthogonality_action - np.eye(RETAINED_COUNT)
        )
        + _gamma(2 * RETAINED_COUNT + 30)
        * orthogonality_absolute_product
    )
    orthogonality_defect_upper = _frob_upper(
        orthogonality_error_entries
    )
    if orthogonality_defect_upper >= 1.0:
        raise ArithmeticError("Schur eigensystem orthogonality bound failed")
    approximate_maximum = float(eigenvalues[-1])
    if approximate_maximum >= 0.0:
        raise ArithmeticError("central Schur matrix is not negative definite")
    central_schur_maximum_upper = _up(
        approximate_maximum * (1.0 - orthogonality_defect_upper)
        + reconstruction_error_upper
    )
    exact_schur_maximum_upper = _up(
        central_schur_maximum_upper
        + schur_product_error_spectral_upper
        + solve_error_spectral_upper
    )

    return {
        "target_floor": target_floor,
        "spectral_gap_lower": spectral_gap_lower,
        "maximum_stiffness_row_nonzeros": maximum_row_nonzeros,
        "central_solve_residual_frobenius": float(
            np.linalg.norm(central_residual, "fro")
        ),
        "weighted_residual_frobenius_upper": (
            weighted_residual_frobenius_upper
        ),
        "weighted_trial_frobenius_upper": (
            weighted_trial_frobenius_upper
        ),
        "solve_error_spectral_upper": solve_error_spectral_upper,
        "schur_product_error_spectral_upper": (
            schur_product_error_spectral_upper
        ),
        "central_schur_minimum_eigenvalue": float(eigenvalues[0]),
        "central_schur_maximum_eigenvalue": approximate_maximum,
        "schur_eigensystem_orthogonality_defect_upper": (
            orthogonality_defect_upper
        ),
        "schur_reconstruction_error_upper": (
            reconstruction_error_upper
        ),
        "central_schur_maximum_upper": central_schur_maximum_upper,
        "exact_schur_maximum_upper": exact_schur_maximum_upper,
        "exact_schur_negative_definite_certified": bool(
            exact_schur_maximum_upper < 0.0
        ),
        "central_schur_sha256": _sha256_arrays(central_schur),
        "central_solution_sha256": _sha256_arrays(solution),
    }


def _inertia_schur_floor_certified(
    dimension: int,
    constraint_count: int,
    negative_inertia_count: int,
    exact_schur_maximum_upper: float,
) -> bool:
    return bool(
        dimension > constraint_count > 0
        and negative_inertia_count == constraint_count
        and exact_schur_maximum_upper < 0.0
    )


def audit(
    spacing: float,
    eigen_cache: Path,
    lower_row_path: Path,
    upper_row_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    mass, stiffness, vectors, metadata = _assemble_modified_pencil(
        spacing, eigen_cache
    )
    lower_shift = _down(TARGET_FLOOR - RESOLVENT_HALF_WIDTH)
    upper_shift = _up(TARGET_FLOOR + RESOLVENT_HALF_WIDTH)
    lower_row = _compute_inertia_row(
        mass, stiffness, lower_shift, lower_row_path, metadata
    )
    upper_row = _compute_inertia_row(
        mass, stiffness, upper_shift, upper_row_path, metadata
    )
    lower_summary = lower_row["summary"]
    upper_summary = upper_row["summary"]
    same_inertia = bool(
        lower_summary["all_directed_ldl_checks_pass"]
        and upper_summary["all_directed_ldl_checks_pass"]
        and int(lower_summary["negative_pivot_count"]) == RETAINED_COUNT
        and int(upper_summary["negative_pivot_count"]) == RETAINED_COUNT
    )
    spectral_gap_lower = _down(
        min(
            TARGET_FLOOR - lower_shift,
            upper_shift - TARGET_FLOOR,
        )
    )
    schur = _schur_certificate(
        mass,
        stiffness,
        vectors,
        TARGET_FLOOR,
        spectral_gap_lower,
    )
    floor_certified = bool(
        same_inertia
        and _inertia_schur_floor_certified(
            mass.shape[0],
            RETAINED_COUNT,
            RETAINED_COUNT,
            float(schur["exact_schur_maximum_upper"]),
        )
    )
    checks = [
        priority_set,
        metadata["mass_diagonal_strictly_positive"],
        metadata["stiffness_exactly_symmetric"],
        lower_summary["all_directed_ldl_checks_pass"],
        upper_summary["all_directed_ldl_checks_pass"],
        int(lower_summary["completed_pivot_count"]) == mass.shape[0],
        int(upper_summary["completed_pivot_count"]) == mass.shape[0],
        int(lower_summary["negative_pivot_count"]) == RETAINED_COUNT,
        int(upper_summary["negative_pivot_count"]) == RETAINED_COUNT,
        spectral_gap_lower > 0.099,
        schur["exact_schur_negative_definite_certified"],
        floor_certified,
    ]
    return {
        "model": "modified-chain constrained complement inertia-Schur certificate",
        "spacing": spacing,
        "below_normal_priority_set": priority_set,
        "pencil": metadata,
        "target_complement_floor": TARGET_FLOOR,
        "resolvent_interval": [lower_shift, upper_shift],
        "resolvent_spectral_gap_lower": spectral_gap_lower,
        "lower_inertia_row": {
            "path": str(lower_row_path),
            "file_sha256": _sha256_file(lower_row_path),
            "loaded_from_checkpoint": lower_row["loaded_from_checkpoint"],
            "summary": lower_summary,
        },
        "upper_inertia_row": {
            "path": str(upper_row_path),
            "file_sha256": _sha256_file(upper_row_path),
            "loaded_from_checkpoint": upper_row["loaded_from_checkpoint"],
            "summary": upper_summary,
        },
        "schur_complement": schur,
        "theorem": {
            "statement": (
                "Let H be self-adjoint, W have k columns, and "
                "J=H-beta I. If J has exactly k negative eigenvalues "
                "and W^T J^(-1) W is negative definite, then "
                "H-beta I is positive definite on ker(W^T)."
            ),
            "proof_mechanism": (
                "The KKT matrix has inertia inertia(J)+"
                "inertia(-W^T J^(-1)W) by Schur complementation, "
                "and also inertia(H restricted to ker(W^T)-beta I)"
                "+(k,k,0) by an orthogonal constraint decomposition."
            ),
            "constraint": "V^T M_tilde u=0",
            "modified_complement_floor_certified": floor_certified,
            "modified_complement_floor_lower": (
                TARGET_FLOOR if floor_certified else None
            ),
        },
        "checks": checks,
        "all_modified_complement_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
        "next_required_step": (
            "Combine the certified beta=102.7 complement floor with a "
            "directed low-block floor and off-block coupling, then propagate "
            "the two-block leakage through half-time boundary smoothing."
        ),
        "scope": (
            "This certificate concerns the stored binary modified chain and "
            "the M_tilde-orthogonal complement of the frozen 240-column "
            "reference trial matrix. It does not transfer the finite chain "
            "to the continuum or compare the polygon and circle domains."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, default=0.06)
    parser.add_argument("--eigen-cache", type=Path, default=DEFAULT_EIGEN_CACHE)
    parser.add_argument("--lower-row", type=Path, default=DEFAULT_LOWER_ROW)
    parser.add_argument("--upper-row", type=Path, default=DEFAULT_UPPER_ROW)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.spacing,
        args.eigen_cache,
        args.lower_row,
        args.upper_row,
    )
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
