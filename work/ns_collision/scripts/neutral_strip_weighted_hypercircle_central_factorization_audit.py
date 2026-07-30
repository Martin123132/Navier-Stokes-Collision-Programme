#!/usr/bin/env python3
"""Probe central factorization viability for the complete hypercircle pencil."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from decimal import Decimal
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import scipy
from scipy.sparse import (
    bmat,
    coo_matrix,
    csc_matrix,
    csr_matrix,
    diags,
)
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import LinearOperator, onenormest, splu


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPLETE_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_complete_assembly_v1.json"
)
DEFAULT_MATRICES = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_assembly_matrices_v1.npz"
)
DEFAULT_GAUSSIAN_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
)
DEFAULT_GAUSSIAN_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
)
DEFAULT_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_central_factorization_checkpoint_v1.json"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_central_factorization_audit_v1.json"
)

BETA_DECIMAL = Decimal("0.045")
RUIZ_SWEEP = (2, 4, 6, 8, 10)
DAYTIME_CPU_THRESHOLD = 75.0
EXPECTED_DIMENSION = 123816
EXPECTED_PENCIL_NNZ = 798384
EXPECTED_INERTIA = {"positive": 61908, "negative": 61908, "zero": 0}
CASE_SPECS = (
    {
        "label": "mmd_at_plus_a_unscaled",
        "permc_spec": "MMD_AT_PLUS_A",
        "scaling": "none",
    },
    {
        "label": "mmd_at_plus_a_symmetric_ruiz_2",
        "permc_spec": "MMD_AT_PLUS_A",
        "scaling": "symmetric_ruiz",
        "ruiz_iterations": 2,
    },
    {
        "label": "mmd_at_plus_a_symmetric_ruiz_4",
        "permc_spec": "MMD_AT_PLUS_A",
        "scaling": "symmetric_ruiz",
        "ruiz_iterations": 4,
    },
    {
        "label": "mmd_at_plus_a_symmetric_ruiz_6",
        "permc_spec": "MMD_AT_PLUS_A",
        "scaling": "symmetric_ruiz",
        "ruiz_iterations": 6,
    },
    {
        "label": "mmd_at_plus_a_symmetric_ruiz_8",
        "permc_spec": "MMD_AT_PLUS_A",
        "scaling": "symmetric_ruiz",
        "ruiz_iterations": 8,
    },
    {
        "label": "mmd_at_plus_a_symmetric_ruiz_10",
        "permc_spec": "MMD_AT_PLUS_A",
        "scaling": "symmetric_ruiz",
        "ruiz_iterations": 10,
    },
    {
        "label": "mmd_ata_symmetric_ruiz_8",
        "permc_spec": "MMD_ATA",
        "scaling": "symmetric_ruiz",
        "ruiz_iterations": 8,
    },
    {
        "label": "colamd_symmetric_ruiz_8",
        "permc_spec": "COLAMD",
        "scaling": "symmetric_ruiz",
        "ruiz_iterations": 8,
    },
)


def _load_module(filename: str, module_name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = module
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


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
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


def _exact_decimal_interval(value: Decimal) -> tuple[float, float]:
    center = float(value)
    lower = center
    upper = center
    if Decimal.from_float(lower) > value:
        lower = float(np.nextafter(lower, -math.inf))
    if Decimal.from_float(upper) < value:
        upper = float(np.nextafter(upper, math.inf))
    return (
        float(np.nextafter(lower, -math.inf)),
        float(np.nextafter(upper, math.inf)),
    )


def _maximum_abs(matrix: csc_matrix | csr_matrix) -> float:
    if matrix.nnz == 0:
        return 0.0
    return float(np.max(np.abs(matrix.data)))


def _row_sum_norm(matrix: csc_matrix | csr_matrix) -> float:
    if matrix.shape[0] == 0:
        return 0.0
    values = np.asarray(np.abs(matrix).sum(axis=1)).reshape(-1)
    return float(np.max(values))


def _column_sum_norm(matrix: csc_matrix | csr_matrix) -> float:
    if matrix.shape[1] == 0:
        return 0.0
    values = np.asarray(np.abs(matrix).sum(axis=0)).reshape(-1)
    return float(np.max(values))


def _gamma(operation_count: int) -> float:
    epsilon = np.finfo(float).eps
    product = operation_count * epsilon
    if product >= 1.0:
        return math.inf
    return product / (1.0 - product)


def _sparse_storage_bytes(matrix: csc_matrix | csr_matrix) -> int:
    return int(
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    )


def _exactly_symmetric(matrix: csc_matrix | csr_matrix) -> bool:
    left = matrix.tocsr()
    right = matrix.transpose().tocsr()
    left.sort_indices()
    right.sort_indices()
    return bool(
        np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data)
    )


def _assemble_gaussian_stiffness(
    checkpoint_path: Path,
    result: dict[str, Any],
    base,
) -> tuple[csr_matrix, csr_matrix, dict[str, Any]]:
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        contract = json.loads(str(checkpoint["contract_json"].item()))
        next_position = int(checkpoint["next_selected_position"].item())
        rows = np.asarray(checkpoint["stiffness_rows"], dtype=np.int64)
        columns = np.asarray(
            checkpoint["stiffness_columns"], dtype=np.int64
        )
        values = np.asarray(checkpoint["stiffness_values"], dtype=float)
        errors = np.asarray(checkpoint["stiffness_errors"], dtype=float)
    state_count = int(contract["state_count"])
    triangle_count = int(contract["selected_triangle_count"])
    if next_position != triangle_count:
        raise RuntimeError("Gaussian stiffness checkpoint is incomplete")
    if not (
        len(rows) == len(columns) == len(values) == len(errors)
        and np.all(np.isfinite(values))
        and np.all(np.isfinite(errors))
        and np.all(errors >= 0.0)
    ):
        raise RuntimeError("Gaussian stiffness contribution arrays are invalid")
    coordinates = (rows, columns)
    shape = (state_count, state_count)
    central = coo_matrix((values, coordinates), shape=shape).tocsr()
    raw_errors = coo_matrix((errors, coordinates), shape=shape).tocsr()
    absolute_values = coo_matrix(
        (np.abs(values), coordinates),
        shape=shape,
    ).tocsr()
    counts = coo_matrix(
        (np.ones(len(values)), coordinates),
        shape=shape,
    ).tocsr()
    maximum_count = int(np.max(counts.data))
    enclosed_errors = base._inflate_sparse_contribution_sum(
        raw_errors,
        absolute_values,
        maximum_count,
    )
    central.sort_indices()
    enclosed_errors.sort_indices()
    if not (
        np.array_equal(central.indptr, enclosed_errors.indptr)
        and np.array_equal(central.indices, enclosed_errors.indices)
    ):
        raise RuntimeError("Gaussian stiffness center/error structures differ")
    fingerprint = base._sparse_matrix_fingerprint(central)
    expected = result["stored_matrix_reconstruction"][
        "expected_per_matrix_fingerprints_sha256"
    ]["stiffness"]
    if fingerprint != expected:
        raise RuntimeError("Gaussian stiffness fingerprint mismatch")
    return central, enclosed_errors, {
        "contract": contract,
        "contribution_count": len(values),
        "maximum_contribution_count": maximum_count,
        "nnz": int(central.nnz),
        "fingerprint_sha256": fingerprint,
    }


def _shifted_source_block(
    centers: np.ndarray,
    errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    beta_squared = BETA_DECIMAL * BETA_DECIMAL
    beta_squared_center = float(beta_squared)
    beta_squared_interval = _exact_decimal_interval(beta_squared)
    lower_w = np.nextafter(centers - errors, -math.inf)
    upper_w = np.nextafter(centers + errors, math.inf)
    if np.any(lower_w <= 0.0):
        raise RuntimeError("source-mass interval is not positive")
    product_lower = np.nextafter(
        beta_squared_interval[0] * lower_w,
        -math.inf,
    )
    product_upper = np.nextafter(
        beta_squared_interval[1] * upper_w,
        math.inf,
    )
    product_center = beta_squared_center * centers
    product_error = np.nextafter(
        np.maximum(
            np.abs(product_center - product_lower),
            np.abs(product_upper - product_center),
        ),
        math.inf,
    )
    return product_center, product_error, beta_squared_interval


def _load_complete_pencil(
    complete_result_path: Path,
    matrices_path: Path,
    gaussian_result_path: Path,
    gaussian_checkpoint_path: Path,
) -> tuple[csc_matrix, csc_matrix, dict[str, Any]]:
    complete_result = json.loads(
        complete_result_path.read_text(encoding="ascii")
    )
    gaussian_result = json.loads(
        gaussian_result_path.read_text(encoding="ascii")
    )
    if not complete_result["all_current_stage_checks_pass"]:
        raise RuntimeError("complete hypercircle assembly did not pass")
    if not gaussian_result["finite_element_assembly_interval_enclosed"]:
        raise RuntimeError("Gaussian stiffness assembly did not pass")
    if (
        _sha256_file(matrices_path)
        != complete_result["artifacts"]["matrices_sha256"]
    ):
        raise RuntimeError("complete matrix archive hash mismatch")

    base = _load_module(
        "neutral_strip_gaussian_weighted_assembly_interval_audit.py",
        "central_factorization_gaussian_base",
    )
    stiffness, stiffness_errors, stiffness_inventory = (
        _assemble_gaussian_stiffness(
            gaussian_checkpoint_path,
            gaussian_result,
            base,
        )
    )

    with np.load(matrices_path, allow_pickle=False) as archive:
        contract = json.loads(str(archive["contract_json"].item()))
        edge_count = int(contract["edge_count"])
        state_count = int(contract["state_count"])
        triangle_count = int(contract["triangle_count"])
        p_indptr = np.asarray(archive["p_indptr"], dtype=np.int64)
        p_indices = np.asarray(archive["p_indices"], dtype=np.int64)
        p_values = np.asarray(archive["p_values"], dtype=float)
        p_errors = np.asarray(archive["p_errors"], dtype=float)
        w_values = np.asarray(archive["w_values"], dtype=float)
        w_errors = np.asarray(archive["w_errors"], dtype=float)
        d_values = np.asarray(archive["d_values"], dtype=float)
        d_errors = np.asarray(archive["d_errors"], dtype=float)
        b_rows = np.asarray(archive["b_rows"], dtype=np.int64)
        b_columns = np.asarray(archive["b_columns"], dtype=np.int64)
        b_values = np.asarray(archive["b_values"], dtype=float)
        b_errors = np.asarray(archive["b_errors"], dtype=float)
        n_rows = np.asarray(archive["n_rows"], dtype=np.int64)
        n_columns = np.asarray(archive["n_columns"], dtype=np.int64)
        n_values = np.asarray(archive["n_values"], dtype=np.int8)

    gaussian_contract = stiffness_inventory["contract"]
    if (
        contract["mesh_fingerprint_sha256"]
        != gaussian_contract["mesh_fingerprint_sha256"]
        or state_count != int(gaussian_contract["state_count"])
        or triangle_count != int(gaussian_contract["total_triangle_count"])
    ):
        raise RuntimeError("Gaussian and hypercircle contracts differ")
    if stiffness.shape != (state_count, state_count):
        raise RuntimeError("Gaussian stiffness dimension mismatch")

    p_shape = (edge_count, edge_count)
    p_center = csr_matrix(
        (p_values, p_indices, p_indptr),
        shape=p_shape,
    )
    p_radius = csr_matrix(
        (p_errors, p_indices, p_indptr),
        shape=p_shape,
    )
    b_shape = (state_count, triangle_count)
    b_center = coo_matrix(
        (b_values, (b_rows, b_columns)),
        shape=b_shape,
    ).tocsr()
    b_radius = coo_matrix(
        (b_errors, (b_rows, b_columns)),
        shape=b_shape,
    ).tocsr()
    n_matrix = coo_matrix(
        (n_values.astype(float), (n_rows, n_columns)),
        shape=(triangle_count, edge_count),
    ).tocsr()
    shifted_w, shifted_w_errors, beta_squared_interval = (
        _shifted_source_block(w_values, w_errors)
    )
    area = diags(d_values, format="csc")
    area_radius = diags(d_errors, format="csc")
    shifted_source = diags(shifted_w, format="csc")
    shifted_source_radius = diags(shifted_w_errors, format="csc")

    zero_es = csc_matrix((edge_count, state_count))
    zero_et = csc_matrix((edge_count, triangle_count))
    zero_ts = csc_matrix((triangle_count, state_count))
    central = bmat(
        [
            [p_center, n_matrix.T, zero_es, zero_et],
            [n_matrix, None, zero_ts, area],
            [zero_es.T, zero_ts.T, stiffness, -b_center],
            [zero_et.T, area, -b_center.T, -shifted_source],
        ],
        format="csc",
    )
    radius = bmat(
        [
            [p_radius, None, zero_es, zero_et],
            [None, None, zero_ts, area_radius],
            [zero_es.T, zero_ts.T, stiffness_errors, b_radius],
            [zero_et.T, area_radius, b_radius.T, shifted_source_radius],
        ],
        format="csc",
    )
    central.sort_indices()
    radius.sort_indices()
    if central.shape != (EXPECTED_DIMENSION, EXPECTED_DIMENSION):
        raise RuntimeError("threshold-pencil dimension mismatch")
    if central.nnz != EXPECTED_PENCIL_NNZ:
        raise RuntimeError("threshold-pencil nonzero count mismatch")
    if not _exactly_symmetric(central):
        raise RuntimeError("central threshold pencil is not exactly symmetric")
    if not _exactly_symmetric(radius):
        raise RuntimeError("threshold-pencil radius is not exactly symmetric")
    if np.any(radius.data < 0.0) or not np.all(np.isfinite(radius.data)):
        raise RuntimeError("threshold-pencil radius is invalid")

    return central, radius, {
        "mesh_fingerprint_sha256": contract["mesh_fingerprint_sha256"],
        "edge_count": edge_count,
        "state_count": state_count,
        "triangle_count": triangle_count,
        "dimension": central.shape[0],
        "central_nnz": int(central.nnz),
        "radius_nnz": int(radius.nnz),
        "central_exactly_symmetric": True,
        "radius_exactly_symmetric": True,
        "Gaussian_stiffness": stiffness_inventory,
        "beta": str(BETA_DECIMAL),
        "beta_squared": str(BETA_DECIMAL * BETA_DECIMAL),
        "beta_squared_binary_interval": list(beta_squared_interval),
        "central_storage_bytes": _sparse_storage_bytes(central),
        "radius_storage_bytes": _sparse_storage_bytes(radius),
        "central_infinity_norm": _row_sum_norm(central),
        "radius_infinity_norm": _row_sum_norm(radius),
        "radius_maximum_entry": _maximum_abs(radius),
    }


def _symmetric_scale(
    matrix: csc_matrix,
    scale: np.ndarray,
) -> csc_matrix:
    coordinates = matrix.tocoo(copy=True)
    pair_scale = scale[coordinates.row] * scale[coordinates.col]
    coordinates.data = coordinates.data * pair_scale
    result = coordinates.tocsc()
    result.sort_indices()
    return result


def _symmetric_ruiz_scaling(
    matrix: csc_matrix,
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scale = np.ones(matrix.shape[0], dtype=float)
    history = []
    for iteration in range(iterations):
        scaled = _symmetric_scale(matrix, scale)
        row_maximum = np.asarray(
            np.abs(scaled).max(axis=1).toarray()
        ).reshape(-1)
        if np.any(row_maximum <= 0.0) or not np.all(
            np.isfinite(row_maximum)
        ):
            raise RuntimeError("Ruiz scaling found an invalid row maximum")
        history.append(
            {
                "iteration": iteration + 1,
                "minimum_row_maximum": float(np.min(row_maximum)),
                "maximum_row_maximum": float(np.max(row_maximum)),
                "row_maximum_ratio": float(
                    np.max(row_maximum) / np.min(row_maximum)
                ),
            }
        )
        update = 1.0 / np.sqrt(row_maximum)
        scale *= update
    final = _symmetric_scale(matrix, scale)
    final_row_maximum = np.asarray(
        np.abs(final).max(axis=1).toarray()
    ).reshape(-1)
    return scale, {
        "iterations": iterations,
        "minimum_scale": float(np.min(scale)),
        "maximum_scale": float(np.max(scale)),
        "scale_ratio": float(np.max(scale) / np.min(scale)),
        "scale_sha256": _sha256_arrays(scale),
        "minimum_final_row_maximum": float(np.min(final_row_maximum)),
        "maximum_final_row_maximum": float(np.max(final_row_maximum)),
        "history": history,
    }


def _graph_diagnostics(matrix: csc_matrix) -> dict[str, Any]:
    pattern = matrix.copy()
    pattern.data = np.ones_like(pattern.data)
    pattern = pattern.maximum(pattern.transpose()).tocsr()
    degrees = np.diff(pattern.indptr)
    coordinates = pattern.tocoo()
    original_bandwidth = int(
        np.max(np.abs(coordinates.row - coordinates.col))
    )
    permutation = reverse_cuthill_mckee(pattern, symmetric_mode=True)
    positions = np.empty(len(permutation), dtype=np.int64)
    positions[permutation] = np.arange(len(permutation), dtype=np.int64)
    rcm_bandwidth = int(
        np.max(
            np.abs(
                positions[coordinates.row] - positions[coordinates.col]
            )
        )
    )
    return {
        "minimum_graph_degree": int(np.min(degrees)),
        "median_graph_degree": float(np.median(degrees)),
        "maximum_graph_degree": int(np.max(degrees)),
        "original_bandwidth": original_bandwidth,
        "reverse_Cuthill_McKee_bandwidth": rcm_bandwidth,
    }


def _solve_residual(
    matrix: csc_matrix,
    factor,
) -> float:
    indices = np.arange(matrix.shape[0], dtype=float)
    right_hand_side = (
        np.sin(0.5 + indices * 0.001)
        + 0.25 * np.cos(0.25 + indices * 0.0007)
    )
    solution = factor.solve(right_hand_side)
    residual = matrix @ solution - right_hand_side
    denominator = (
        _row_sum_norm(matrix) * float(np.max(np.abs(solution)))
        + float(np.max(np.abs(right_hand_side)))
    )
    return float(np.max(np.abs(residual)) / denominator)


def _inverse_one_norm_estimate(factor, dimension: int) -> float:
    operator = LinearOperator(
        (dimension, dimension),
        matvec=lambda vector: factor.solve(np.asarray(vector)),
        rmatvec=lambda vector: factor.solve(
            np.asarray(vector),
            trans="T",
        ),
        matmat=lambda vectors: factor.solve(np.asarray(vectors)),
        rmatmat=lambda vectors: factor.solve(
            np.asarray(vectors),
            trans="T",
        ),
        dtype=float,
    )
    return float(onenormest(operator, t=2, itmax=5))


def _factor_case(
    central: csc_matrix,
    radius: csc_matrix,
    case: dict[str, Any],
    expected_inertia: dict[str, int],
) -> dict[str, Any]:
    try:
        import psutil

        process = psutil.Process()
        rss_before = int(process.memory_info().rss)
        available_before = int(psutil.virtual_memory().available)
    except Exception:
        process = None
        rss_before = 0
        available_before = 0
    started = time.perf_counter()
    factor = splu(
        central,
        permc_spec=case["permc_spec"],
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    factor_seconds = time.perf_counter() - started
    rss_with_factor = (
        int(process.memory_info().rss) if process is not None else 0
    )

    lower = factor.L.tocsc()
    upper = factor.U.tocsc()
    diagonal = np.asarray(upper.diagonal(), dtype=float)
    if len(diagonal) != central.shape[0] or not np.all(
        np.isfinite(diagonal)
    ):
        raise RuntimeError("factor diagonal is incomplete or non-finite")
    absolute_diagonal = np.abs(diagonal)
    positive = int(np.count_nonzero(diagonal > 0.0))
    negative = int(np.count_nonzero(diagonal < 0.0))
    zero = int(np.count_nonzero(diagonal == 0.0))
    permutations_equal = bool(
        np.array_equal(factor.perm_r, factor.perm_c)
    )

    ldl_reference = diags(diagonal, format="csc") @ lower.transpose()
    ldl_defect = upper - ldl_reference
    ldl_defect.eliminate_zeros()
    upper_inf = _row_sum_norm(upper)
    ldl_relative_inf = (
        _row_sum_norm(ldl_defect) / upper_inf if upper_inf > 0.0 else 0.0
    )
    solve_residual = _solve_residual(central, factor)
    inverse_one_norm = _inverse_one_norm_estimate(
        factor,
        central.shape[0],
    )
    matrix_one_norm = _column_sum_norm(central)
    radius_one_norm = _column_sum_norm(radius)
    lower_inf = _row_sum_norm(lower)
    maximum_factor_row_nnz = max(
        int(np.max(np.diff(lower.tocsr().indptr))),
        int(np.max(np.diff(upper.tocsr().indptr))),
    )
    roundoff_proxy = float(
        np.nextafter(
            _gamma(2 * central.shape[0] + 64) * lower_inf * upper_inf,
            math.inf,
        )
    )
    minimum_pivot = float(np.min(absolute_diagonal))
    maximum_pivot = float(np.max(absolute_diagonal))
    factor_nnz = int(lower.nnz + upper.nnz)
    factor_storage = (
        _sparse_storage_bytes(lower)
        + _sparse_storage_bytes(upper)
        + factor.perm_r.nbytes
        + factor.perm_c.nbytes
    )
    target_counts_match = bool(
        positive == expected_inertia["positive"]
        and negative == expected_inertia["negative"]
        and zero == expected_inertia["zero"]
    )
    return {
        "label": case["label"],
        "status": "success",
        "permc_spec": case["permc_spec"],
        "scaling": case["scaling"],
        "Ruiz_iterations": case.get("ruiz_iterations", 0),
        "factor_seconds": factor_seconds,
        "central_L_plus_U_nnz": factor_nnz,
        "central_factor_fill_ratio": factor_nnz / central.nnz,
        "factor_storage_bytes": factor_storage,
        "row_permutation_sha256": _sha256_arrays(factor.perm_r),
        "column_permutation_sha256": _sha256_arrays(factor.perm_c),
        "factor_pattern_sha256": _sha256_arrays(
            lower.indptr,
            lower.indices,
            upper.indptr,
            upper.indices,
        ),
        "U_diagonal_sha256": _sha256_arrays(diagonal),
        "rss_before_bytes": rss_before,
        "rss_with_factor_bytes": rss_with_factor,
        "rss_growth_bytes": max(0, rss_with_factor - rss_before),
        "available_memory_before_bytes": available_before,
        "row_and_column_permutations_equal": permutations_equal,
        "LDL_relation_relative_infinity_defect": ldl_relative_inf,
        "central_solve_relative_backward_error": solve_residual,
        "U_diagonal_counts": {
            "positive": positive,
            "negative": negative,
            "zero": zero,
        },
        "U_diagonal_counts_match_target": target_counts_match,
        "minimum_absolute_U_diagonal": minimum_pivot,
        "maximum_absolute_U_diagonal": maximum_pivot,
        "U_diagonal_dynamic_range": (
            maximum_pivot / minimum_pivot
            if minimum_pivot > 0.0
            else math.inf
        ),
        "maximum_absolute_input_entry": _maximum_abs(central),
        "maximum_absolute_U_entry": _maximum_abs(upper),
        "element_growth_factor": (
            _maximum_abs(upper) / _maximum_abs(central)
        ),
        "L_infinity_norm": lower_inf,
        "U_infinity_norm": upper_inf,
        "maximum_factor_row_nnz": maximum_factor_row_nnz,
        "input_radius_one_norm": radius_one_norm,
        "inverse_one_norm_estimate": inverse_one_norm,
        "inverse_one_norm_estimate_is_not_a_verified_upper_bound": True,
        "central_condition_one_norm_estimate": (
            matrix_one_norm * inverse_one_norm
        ),
        "input_interval_inverse_amplification_estimate": (
            radius_one_norm * inverse_one_norm
        ),
        "factor_roundoff_backward_error_proxy": roundoff_proxy,
        "input_radius_to_minimum_pivot_ratio": (
            radius_one_norm / minimum_pivot
            if minimum_pivot > 0.0
            else math.inf
        ),
        "roundoff_proxy_to_minimum_pivot_ratio": (
            roundoff_proxy / minimum_pivot
            if minimum_pivot > 0.0
            else math.inf
        ),
    }


def _cpu_sample(samples: list[float]) -> tuple[float | None, bool]:
    try:
        import psutil

        value = float(psutil.cpu_percent(interval=None))
    except Exception:
        return None, False
    samples.append(value)
    high = len(samples) >= 2 and all(
        sample > DAYTIME_CPU_THRESHOLD for sample in samples[-2:]
    )
    return value, high


def _load_checkpoint(
    path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = json.loads(path.read_text(encoding="ascii"))
    if checkpoint["contract"] != contract:
        raise RuntimeError("central factorization checkpoint contract mismatch")
    rows = checkpoint["factorization_cases"]
    expected_labels = [case["label"] for case in contract["case_specs"]]
    actual_labels = [row["label"] for row in rows]
    if actual_labels != expected_labels[: len(actual_labels)]:
        raise RuntimeError("factorization checkpoint case order mismatch")
    return checkpoint


def _write_checkpoint(
    path: Path,
    contract: dict[str, Any],
    rows: list[dict[str, Any]],
    cpu_samples: list[float],
) -> None:
    _atomic_json(
        path,
        {
            "kind": "hypercircle-central-factorization-checkpoint",
            "contract": contract,
            "factorization_cases": rows,
            "cpu_samples_percent": cpu_samples,
            "completed_case_count": len(rows),
        },
    )


def run_audit(
    complete_result_path: Path,
    matrices_path: Path,
    gaussian_result_path: Path,
    gaussian_checkpoint_path: Path,
    checkpoint_path: Path,
    maximum_cases: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    central, radius, inventory = _load_complete_pencil(
        complete_result_path,
        matrices_path,
        gaussian_result_path,
        gaussian_checkpoint_path,
    )
    graph = _graph_diagnostics(central)
    scaled_variants: dict[int, tuple[csc_matrix, csc_matrix]] = {}
    scaling_sweep: dict[str, dict[str, Any]] = {}
    for iterations in RUIZ_SWEEP:
        scale, scaling = _symmetric_ruiz_scaling(central, iterations)
        scaled_central = _symmetric_scale(central, scale)
        scaled_radius = _symmetric_scale(radius, scale)
        if not (
            _exactly_symmetric(scaled_central)
            and _exactly_symmetric(scaled_radius)
        ):
            raise RuntimeError("symmetric scaling did not preserve symmetry")
        scaling["scaled_central_infinity_norm"] = _row_sum_norm(
            scaled_central
        )
        scaling["scaled_radius_infinity_norm"] = _row_sum_norm(
            scaled_radius
        )
        scaled_variants[iterations] = (scaled_central, scaled_radius)
        scaling_sweep[str(iterations)] = scaling

    contract = {
        "schema_version": 1,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "complete_result_sha256": _sha256_file(complete_result_path),
        "matrix_archive_sha256": _sha256_file(matrices_path),
        "Gaussian_result_sha256": _sha256_file(gaussian_result_path),
        "Gaussian_checkpoint_sha256": _sha256_file(
            gaussian_checkpoint_path
        ),
        "mesh_fingerprint_sha256": inventory[
            "mesh_fingerprint_sha256"
        ],
        "dimension": central.shape[0],
        "central_nnz": int(central.nnz),
        "beta_decimal": str(BETA_DECIMAL),
        "Ruiz_iterations": list(RUIZ_SWEEP),
        "case_specs": list(CASE_SPECS),
    }
    if checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path, contract)
        rows = checkpoint["factorization_cases"]
        cpu_samples = checkpoint["cpu_samples_percent"]
        resumed = True
    else:
        rows = []
        cpu_samples = []
        resumed = False
    try:
        import psutil

        psutil.cpu_percent(interval=None)
    except Exception:
        pass

    cases_completed_this_run = 0
    parked_for_cpu = False
    for case in CASE_SPECS[len(rows) :]:
        if case["scaling"] == "none":
            matrix, matrix_radius = central, radius
        else:
            matrix, matrix_radius = scaled_variants[
                int(case["ruiz_iterations"])
            ]
        try:
            row = _factor_case(
                matrix,
                matrix_radius,
                case,
                EXPECTED_INERTIA,
            )
        except Exception as error:
            row = {
                "label": case["label"],
                "status": "failed",
                "permc_spec": case["permc_spec"],
                "scaling": case["scaling"],
                "Ruiz_iterations": case.get("ruiz_iterations", 0),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
        rows.append(row)
        cases_completed_this_run += 1
        _, high = _cpu_sample(cpu_samples)
        _write_checkpoint(
            checkpoint_path,
            contract,
            rows,
            cpu_samples,
        )
        gc.collect()
        if high:
            parked_for_cpu = True
            break
        if maximum_cases > 0 and cases_completed_this_run >= maximum_cases:
            break

    complete = len(rows) == len(CASE_SPECS)
    successful = [row for row in rows if row["status"] == "success"]
    symmetric_ldl_rows = [
        row
        for row in successful
        if row["row_and_column_permutations_equal"]
        and row["LDL_relation_relative_infinity_defect"] < 1.0e-10
    ]
    target_rows = [
        row
        for row in symmetric_ldl_rows
        if row["U_diagonal_counts_match_target"]
    ]
    recommended = (
        min(
            target_rows,
            key=lambda row: (
                row["roundoff_proxy_to_minimum_pivot_ratio"],
                row["input_interval_inverse_amplification_estimate"],
                row["central_factor_fill_ratio"],
            ),
        )
        if target_rows
        else None
    )
    rejected_orderings = [
        row["label"]
        for row in successful
        if not (
            row["row_and_column_permutations_equal"]
            and row["LDL_relation_relative_infinity_defect"] < 1.0e-10
            and row["U_diagonal_counts_match_target"]
        )
    ]
    best_fill = (
        min(
            successful,
            key=lambda row: row["central_factor_fill_ratio"],
        )
        if successful
        else None
    )
    best_interval_amplification = (
        min(
            successful,
            key=lambda row: row[
                "input_interval_inverse_amplification_estimate"
            ],
        )
        if successful
        else None
    )
    status = "complete"
    if not complete:
        status = (
            "parked_for_daytime_cpu"
            if parked_for_cpu
            else "parked_at_requested_case_limit"
        )
    replayed_checkpoint = _load_checkpoint(checkpoint_path, contract)
    checkpoint_replays = bool(
        replayed_checkpoint["factorization_cases"] == rows
        and replayed_checkpoint["cpu_samples_percent"] == cpu_samples
        and replayed_checkpoint["completed_case_count"] == len(rows)
    )
    checks = {
        "complete_matrix_contracts_match": True,
        "central_dimension_matches": central.shape[0] == EXPECTED_DIMENSION,
        "central_nnz_matches": central.nnz == EXPECTED_PENCIL_NNZ,
        "central_matrix_exactly_symmetric": _exactly_symmetric(central),
        "radius_matrix_exactly_symmetric_nonnegative": bool(
            _exactly_symmetric(radius) and np.all(radius.data >= 0.0)
        ),
        "symmetric_Ruiz_scaling_preserves_symmetry": bool(
            all(
                _exactly_symmetric(center)
                and _exactly_symmetric(enclosure)
                for center, enclosure in scaled_variants.values()
            )
        ),
        "checkpoint_content_replays": checkpoint_replays,
        "all_requested_cases_succeeded": bool(
            complete and len(successful) == len(CASE_SPECS)
        ),
        "at_least_one_symmetric_LDL_like_factorization": bool(
            symmetric_ldl_rows
        ),
        "at_least_one_central_target_pivot_count_observed": bool(
            target_rows
        ),
    }
    integrity_checks = [
        checks["complete_matrix_contracts_match"],
        checks["central_dimension_matches"],
        checks["central_nnz_matches"],
        checks["central_matrix_exactly_symmetric"],
        checks["radius_matrix_exactly_symmetric_nonnegative"],
        checks["symmetric_Ruiz_scaling_preserves_symmetry"],
        checks["checkpoint_content_replays"],
    ]
    if complete:
        integrity_checks.extend(
            [
                checks["all_requested_cases_succeeded"],
                checks["at_least_one_symmetric_LDL_like_factorization"],
                checks[
                    "at_least_one_central_target_pivot_count_observed"
                ],
            ]
        )
    all_checks_pass = bool(all(integrity_checks))

    return {
        "kind": "neutral-strip-weighted-hypercircle-central-factorization-audit",
        "status": status,
        "contract": contract,
        "resumed_from_checkpoint": resumed,
        "cases_completed_this_run": cases_completed_this_run,
        "complete_case_set": complete,
        "matrix_inventory": inventory,
        "graph_diagnostics": graph,
        "symmetric_Ruiz_scaling_sweep": scaling_sweep,
        "factorization_cases": rows,
        "best_observed_fill_case": (
            best_fill["label"] if best_fill is not None else None
        ),
        "best_observed_interval_amplification_case": (
            best_interval_amplification["label"]
            if best_interval_amplification is not None
            else None
        ),
        "recommended_central_case": (
            recommended["label"] if recommended is not None else None
        ),
        "recommended_scale_sha256": (
            scaling_sweep[
                str(int(recommended["Ruiz_iterations"]))
            ]["scale_sha256"]
            if recommended is not None
            and int(recommended["Ruiz_iterations"]) > 0
            else None
        ),
        "recommended_permutation_sha256": (
            recommended["row_permutation_sha256"]
            if recommended is not None
            and recommended["row_and_column_permutations_equal"]
            else None
        ),
        "recommended_factor_pattern_sha256": (
            recommended["factor_pattern_sha256"]
            if recommended is not None
            else None
        ),
        "rejected_ordering_cases": rejected_orderings,
        "viability_assessment": {
            "central_factorization_resource_path_observed": bool(
                successful
            ),
            "central_target_pivot_count_observed": bool(target_rows),
            "recommended_central_case": (
                recommended["label"] if recommended is not None else None
            ),
            "recommended_input_interval_amplification_estimate": (
                recommended[
                    "input_interval_inverse_amplification_estimate"
                ]
                if recommended is not None
                else None
            ),
            "recommended_roundoff_proxy_to_minimum_pivot_ratio": (
                recommended["roundoff_proxy_to_minimum_pivot_ratio"]
                if recommended is not None
                else None
            ),
            "floating_inverse_estimate_times_radius_below_one": bool(
                best_interval_amplification is not None
                and best_interval_amplification[
                    "input_interval_inverse_amplification_estimate"
                ]
                < 1.0
            ),
            "global_norm_roundoff_proxy_closes": bool(
                recommended is not None
                and recommended[
                    "roundoff_proxy_to_minimum_pivot_ratio"
                ]
                < 1.0
            ),
            "componentwise_or_verified_residual_method_required": True,
            "rejected_ordering_cases": rejected_orderings,
            "verified_directed_sparse_inertia_method_ready": False,
            "interpretation": (
                "Central factors are resource diagnostics only. Pivot signs "
                "are reported only when the row/column permutations agree "
                "and U is numerically D*L^T. The inverse-norm and roundoff "
                "budgets are floating feasibility estimates; in particular, "
                "onenormest is not a verified inverse-norm upper bound. None "
                "of these values is a directed inertia certificate."
            ),
        },
        "checks": checks,
        "all_current_stage_checks_pass": all_checks_pass,
        "certification_flags": {
            "complete_mesh_matrix_entries_enclosed": True,
            "central_threshold_pencil_factorization_observed": bool(
                successful
            ),
            "full_mesh_threshold_inertia_certified": False,
            "kappa_h_verified_upper_bound": False,
            "global_weighted_Ritz_projection_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
        },
        "runtime": {
            "below_normal_priority_set": priority_set,
            "cpu_samples_percent": cpu_samples,
            "maximum_cpu_percent": (
                max(cpu_samples) if cpu_samples else None
            ),
            "cumulative_factor_seconds": sum(
                row["factor_seconds"] for row in successful
            ),
            "maximum_factor_storage_bytes": (
                max(row["factor_storage_bytes"] for row in successful)
                if successful
                else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "artifacts": {
            "complete_assembly_result": str(
                complete_result_path
            ).replace("\\", "/"),
            "complete_assembly_result_sha256": _sha256_file(
                complete_result_path
            ),
            "matrix_archive": str(matrices_path).replace("\\", "/"),
            "matrix_archive_sha256": _sha256_file(matrices_path),
            "Gaussian_result": str(gaussian_result_path).replace("\\", "/"),
            "Gaussian_result_sha256": _sha256_file(
                gaussian_result_path
            ),
            "Gaussian_checkpoint": str(
                gaussian_checkpoint_path
            ).replace("\\", "/"),
            "Gaussian_checkpoint_sha256": _sha256_file(
                gaussian_checkpoint_path
            ),
            "checkpoint": str(checkpoint_path).replace("\\", "/"),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
        },
        "next_required_step": (
            "Freeze MMD_AT_PLUS_A with ten-step symmetric Ruiz scaling. "
            "Build a bounded componentwise directed-LDL or verified-residual "
            "pilot on its fixed fill pattern; the global norm roundoff proxy "
            "does not close, and MMD_ATA/COLAMD must not be used for signs."
            if complete
            else "Resume the next independent factorization case from the "
            "hash-bound JSON checkpoint."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complete-result",
        type=Path,
        default=DEFAULT_COMPLETE_RESULT,
    )
    parser.add_argument(
        "--matrices",
        type=Path,
        default=DEFAULT_MATRICES,
    )
    parser.add_argument(
        "--gaussian-result",
        type=Path,
        default=DEFAULT_GAUSSIAN_RESULT,
    )
    parser.add_argument(
        "--gaussian-checkpoint",
        type=Path,
        default=DEFAULT_GAUSSIAN_CHECKPOINT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--maximum-cases", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.maximum_cases < 0:
        raise SystemExit("--maximum-cases must be nonnegative")
    result = run_audit(
        args.complete_result,
        args.matrices,
        args.gaussian_result,
        args.gaussian_checkpoint,
        args.checkpoint,
        args.maximum_cases,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
