#!/usr/bin/env python3
"""Resumable complete-mesh interval assembly for the hypercircle pencil."""

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
from scipy.sparse import coo_matrix, csr_matrix
import triangle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PILOT_RESULT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_rt_interval_pilot512_v1.json"
)
DEFAULT_CHECKPOINT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_assembly_checkpoint_v1.npz"
)
DEFAULT_CHECKPOINT_METADATA = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_assembly_checkpoint_v1.json"
)
DEFAULT_MATRICES = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_assembly_matrices_v1.npz"
)
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_positive_exponential_complete_assembly_v1.json"
)

SPACING = 0.06
X_HALF_WIDTH = 4.2
STRIP_HALF_WIDTH = 2.1
TAYLOR_DEGREE = 22
QUADRATURE_ORDER = 12
CROSS_CHECK_ORDER = 18
CROSS_CHECK_STRIDE = 257
DEFAULT_CHECKPOINT_INTERVAL = 512
DAYTIME_CPU_THRESHOLD = 75.0


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


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _entry_error(value: float, interval: tuple[float, float], base) -> float:
    return base._up(
        max(abs(value - interval[0]), abs(value - interval[1]))
    )


def _distance_to_interval(
    value: float,
    interval: tuple[float, float],
) -> float:
    if value < interval[0]:
        return interval[0] - value
    if value > interval[1]:
        return value - interval[1]
    return 0.0


def _mesh_and_topology(pilot, base) -> dict[str, Any]:
    boundary = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "positive_exponential_complete_boundary",
    )
    mesh_input, _, _ = boundary._mesh_input(
        SPACING,
        X_HALF_WIDTH,
        STRIP_HALF_WIDTH,
    )
    mesh = triangle.triangulate(
        mesh_input,
        f"pYq28a{0.45 * SPACING**2:.17g}Q",
    )
    vertices = np.asarray(mesh["vertices"], dtype=float)
    triangles = np.asarray(mesh["triangles"], dtype=int)
    boundary_segments = np.asarray(mesh["segments"], dtype=int)
    boundary_vertices = set(int(value) for value in boundary_segments.ravel())
    state_vertices = np.asarray(
        [
            index
            for index in range(len(vertices))
            if index not in boundary_vertices
        ],
        dtype=int,
    )
    state_lookup = {
        int(vertex): state for state, vertex in enumerate(state_vertices)
    }

    edge_indices: dict[tuple[int, int], int] = {}
    local_edges = np.empty((len(triangles), 3), dtype=np.int64)
    local_signs = np.empty((len(triangles), 3), dtype=np.int8)
    incidences: dict[int, list[int]] = {}
    for triangle_index, element in enumerate(triangles):
        for opposite in range(3):
            edge = tuple(
                sorted(
                    (
                        int(element[(opposite + 1) % 3]),
                        int(element[(opposite + 2) % 3]),
                    )
                )
            )
            edge_index = edge_indices.setdefault(edge, len(edge_indices))
            sign = pilot._edge_sign(
                vertices,
                edge,
                int(element[opposite]),
            )
            local_edges[triangle_index, opposite] = edge_index
            local_signs[triangle_index, opposite] = sign
            incidences.setdefault(edge_index, []).append(sign)
    incidence_valid = all(len(values) in (1, 2) for values in incidences.values())
    interior_signs_cancel = all(
        sum(values) == 0 for values in incidences.values() if len(values) == 2
    )
    return {
        "vertices": vertices,
        "triangles": triangles,
        "state_vertices": state_vertices,
        "state_lookup": state_lookup,
        "local_edges": local_edges,
        "local_signs": local_signs,
        "edge_count": len(edge_indices),
        "mesh_fingerprint_sha256": base._mesh_fingerprint(
            vertices,
            triangles,
        ),
        "edge_incidence_valid": incidence_valid,
        "interior_signs_cancel": interior_signs_cancel,
    }


def _empty_state() -> dict[str, Any]:
    return {
        "next_triangle": 0,
        "p_rows": [],
        "p_columns": [],
        "p_values": [],
        "p_errors": [],
        "w_values": [],
        "w_errors": [],
        "d_values": [],
        "d_errors": [],
        "b_rows": [],
        "b_columns": [],
        "b_values": [],
        "b_errors": [],
        "n_rows": [],
        "n_columns": [],
        "n_values": [],
        "metrics": {
            "q12_containment_checks": 0,
            "q12_containment_failures": 0,
            "q18_containment_checks": 0,
            "q18_containment_failures": 0,
            "maximum_q12_distance": {
                "source_mass": 0.0,
                "rt_mass": 0.0,
            },
            "maximum_q18_distance": {
                "source_mass": 0.0,
                "rt_mass": 0.0,
            },
            "maximum_q12_q18_difference": {
                "source_mass": 0.0,
                "rt_mass": 0.0,
            },
            "maximum_local_interval_width": {
                "source_mass": 0.0,
                "rt_mass": 0.0,
                "area": 0.0,
                "load": 0.0,
            },
            "minimum_source_mass_lower": math.inf,
            "minimum_rt_diagonal_lower": math.inf,
            "maximum_cpu_percent": 0.0,
            "cpu_samples_percent": [],
        },
    }


def _checkpoint_arrays(
    state: dict[str, Any],
    contract_json: str,
) -> dict[str, np.ndarray]:
    return {
        "contract_json": np.asarray(contract_json),
        "next_triangle": np.asarray(state["next_triangle"], dtype=np.int64),
        "p_rows": np.asarray(state["p_rows"], dtype=np.int64),
        "p_columns": np.asarray(state["p_columns"], dtype=np.int64),
        "p_values": np.asarray(state["p_values"], dtype=float),
        "p_errors": np.asarray(state["p_errors"], dtype=float),
        "w_values": np.asarray(state["w_values"], dtype=float),
        "w_errors": np.asarray(state["w_errors"], dtype=float),
        "d_values": np.asarray(state["d_values"], dtype=float),
        "d_errors": np.asarray(state["d_errors"], dtype=float),
        "b_rows": np.asarray(state["b_rows"], dtype=np.int64),
        "b_columns": np.asarray(state["b_columns"], dtype=np.int64),
        "b_values": np.asarray(state["b_values"], dtype=float),
        "b_errors": np.asarray(state["b_errors"], dtype=float),
        "n_rows": np.asarray(state["n_rows"], dtype=np.int64),
        "n_columns": np.asarray(state["n_columns"], dtype=np.int64),
        "n_values": np.asarray(state["n_values"], dtype=np.int8),
        "metrics_json": np.asarray(
            json.dumps(
                state["metrics"],
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }


def _write_checkpoint(
    checkpoint_path: Path,
    metadata_path: Path,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    contract_json = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    )
    _atomic_npz(
        checkpoint_path,
        _checkpoint_arrays(state, contract_json),
    )
    metadata = {
        "kind": "positive-exponential-complete-assembly-checkpoint",
        "contract": contract,
        "next_triangle": state["next_triangle"],
        "checkpoint_npz": str(checkpoint_path).replace("\\", "/"),
        "checkpoint_npz_sha256": _sha256_file(checkpoint_path),
        "metrics": state["metrics"],
    }
    _atomic_json(metadata_path, metadata)


def _load_checkpoint(
    checkpoint_path: Path,
    metadata_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    if metadata["contract"] != contract:
        raise RuntimeError("assembly checkpoint contract mismatch")
    if _sha256_file(checkpoint_path) != metadata["checkpoint_npz_sha256"]:
        raise RuntimeError("assembly checkpoint hash mismatch")
    expected_contract_json = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    )
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        if str(checkpoint["contract_json"].item()) != expected_contract_json:
            raise RuntimeError("NPZ checkpoint contract mismatch")
        state = {
            "next_triangle": int(checkpoint["next_triangle"].item()),
            "p_rows": checkpoint["p_rows"].tolist(),
            "p_columns": checkpoint["p_columns"].tolist(),
            "p_values": checkpoint["p_values"].tolist(),
            "p_errors": checkpoint["p_errors"].tolist(),
            "w_values": checkpoint["w_values"].tolist(),
            "w_errors": checkpoint["w_errors"].tolist(),
            "d_values": checkpoint["d_values"].tolist(),
            "d_errors": checkpoint["d_errors"].tolist(),
            "b_rows": checkpoint["b_rows"].tolist(),
            "b_columns": checkpoint["b_columns"].tolist(),
            "b_values": checkpoint["b_values"].tolist(),
            "b_errors": checkpoint["b_errors"].tolist(),
            "n_rows": checkpoint["n_rows"].tolist(),
            "n_columns": checkpoint["n_columns"].tolist(),
            "n_values": checkpoint["n_values"].tolist(),
            "metrics": json.loads(str(checkpoint["metrics_json"].item())),
        }
    if state["next_triangle"] != int(metadata["next_triangle"]):
        raise RuntimeError("checkpoint triangle position mismatch")
    if not 0 <= state["next_triangle"] <= int(contract["triangle_count"]):
        raise RuntimeError("checkpoint triangle position is out of range")
    if state["metrics"] != metadata["metrics"]:
        raise RuntimeError("checkpoint metric copies differ")
    expected_p = 9 * state["next_triangle"]
    expected_n = 3 * state["next_triangle"]
    if not (
        len(state["p_rows"])
        == len(state["p_columns"])
        == len(state["p_values"])
        == len(state["p_errors"])
        == expected_p
    ):
        raise RuntimeError("checkpoint RT0 contribution count mismatch")
    if not (
        len(state["w_values"])
        == len(state["w_errors"])
        == len(state["d_values"])
        == len(state["d_errors"])
        == state["next_triangle"]
    ):
        raise RuntimeError("checkpoint diagonal contribution count mismatch")
    if not (
        len(state["b_rows"])
        == len(state["b_columns"])
        == len(state["b_values"])
        == len(state["b_errors"])
    ):
        raise RuntimeError("checkpoint load contribution count mismatch")
    if state["b_columns"] and (
        min(state["b_columns"]) < 0
        or max(state["b_columns"]) >= state["next_triangle"]
    ):
        raise RuntimeError("checkpoint load triangle index is out of range")
    if state["b_rows"] and (
        min(state["b_rows"]) < 0
        or max(state["b_rows"]) >= int(contract["state_count"])
    ):
        raise RuntimeError("checkpoint load state index is out of range")
    if not (
        len(state["n_rows"])
        == len(state["n_columns"])
        == len(state["n_values"])
        == expected_n
    ):
        raise RuntimeError("checkpoint divergence contribution count mismatch")
    if state["n_rows"] and (
        min(state["n_rows"]) < 0
        or max(state["n_rows"]) >= state["next_triangle"]
    ):
        raise RuntimeError("checkpoint divergence triangle index is out of range")
    if state["n_columns"] and (
        min(state["n_columns"]) < 0
        or max(state["n_columns"]) >= int(contract["edge_count"])
    ):
        raise RuntimeError("checkpoint divergence edge index is out of range")
    if state["p_rows"] and (
        min(state["p_rows"]) < 0
        or max(state["p_rows"]) >= int(contract["edge_count"])
        or min(state["p_columns"]) < 0
        or max(state["p_columns"]) >= int(contract["edge_count"])
    ):
        raise RuntimeError("checkpoint RT0 edge index is out of range")
    numeric_keys = (
        "p_values",
        "p_errors",
        "w_values",
        "w_errors",
        "d_values",
        "d_errors",
        "b_values",
        "b_errors",
    )
    if any(
        not np.all(np.isfinite(np.asarray(state[key], dtype=float)))
        for key in numeric_keys
    ):
        raise RuntimeError("checkpoint contains a non-finite matrix contribution")
    if any(
        np.any(np.asarray(state[key], dtype=float) < 0.0)
        for key in ("p_errors", "w_errors", "d_errors", "b_errors")
    ):
        raise RuntimeError("checkpoint contains a negative enclosure radius")
    if state["n_values"] and not np.all(
        np.abs(np.asarray(state["n_values"], dtype=int)) == 1
    ):
        raise RuntimeError("checkpoint divergence entry is not plus or minus one")
    return state


def _cpu_sample(metrics: dict[str, Any]) -> tuple[float | None, bool]:
    try:
        import psutil

        value = float(psutil.cpu_percent(interval=None))
    except Exception:
        return None, False
    samples = metrics["cpu_samples_percent"]
    samples.append(value)
    metrics["maximum_cpu_percent"] = max(
        float(metrics["maximum_cpu_percent"]),
        value,
    )
    high = len(samples) >= 2 and all(
        sample > DAYTIME_CPU_THRESHOLD for sample in samples[-2:]
    )
    return value, high


def _assemble_sparse_matrices(
    state: dict[str, Any],
    topology: dict[str, Any],
    base,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    edge_count = int(topology["edge_count"])
    state_count = len(topology["state_vertices"])
    triangle_count = len(topology["triangles"])
    p_shape = (edge_count, edge_count)
    p_values = coo_matrix(
        (
            np.asarray(state["p_values"]),
            (
                np.asarray(state["p_rows"]),
                np.asarray(state["p_columns"]),
            ),
        ),
        shape=p_shape,
    ).tocsr()
    p_errors = coo_matrix(
        (
            np.asarray(state["p_errors"]),
            (
                np.asarray(state["p_rows"]),
                np.asarray(state["p_columns"]),
            ),
        ),
        shape=p_shape,
    ).tocsr()
    p_absolute_contributions = coo_matrix(
        (
            np.abs(np.asarray(state["p_values"])),
            (
                np.asarray(state["p_rows"]),
                np.asarray(state["p_columns"]),
            ),
        ),
        shape=p_shape,
    ).tocsr()
    p_values.sort_indices()
    p_errors.sort_indices()
    p_absolute_contributions.sort_indices()
    p_errors = base._inflate_sparse_contribution_sum(
        p_errors,
        p_absolute_contributions,
        2,
    )
    p_errors.sort_indices()
    p_structures_align = bool(
        np.array_equal(p_values.indptr, p_errors.indptr)
        and np.array_equal(p_values.indices, p_errors.indices)
    )
    if not p_structures_align:
        raise RuntimeError("central and error RT0 sparse structures do not align")

    p_transpose = p_values.transpose().tocsr()
    p_transpose.sort_indices()
    pe_transpose = p_errors.transpose().tocsr()
    pe_transpose.sort_indices()
    p_symmetric = bool(
        np.array_equal(p_values.indptr, p_transpose.indptr)
        and np.array_equal(p_values.indices, p_transpose.indices)
        and np.array_equal(p_values.data, p_transpose.data)
    )
    p_errors_symmetric = bool(
        np.array_equal(p_errors.indptr, pe_transpose.indptr)
        and np.array_equal(p_errors.indices, pe_transpose.indices)
        and np.array_equal(p_errors.data, pe_transpose.data)
    )

    b_rows = np.asarray(state["b_rows"], dtype=np.int64)
    b_columns = np.asarray(state["b_columns"], dtype=np.int64)
    b_values = np.asarray(state["b_values"], dtype=float)
    b_errors = np.asarray(state["b_errors"], dtype=float)
    n_rows = np.asarray(state["n_rows"], dtype=np.int64)
    n_columns = np.asarray(state["n_columns"], dtype=np.int64)
    n_values = np.asarray(state["n_values"], dtype=np.int8)
    arrays = {
        "contract_json": np.asarray(
            json.dumps(
                {
                    "schema_version": 1,
                    "mesh_fingerprint_sha256": topology[
                        "mesh_fingerprint_sha256"
                    ],
                    "edge_count": edge_count,
                    "state_count": state_count,
                    "triangle_count": triangle_count,
                    "Taylor_degree": TAYLOR_DEGREE,
                    "quadrature_order": QUADRATURE_ORDER,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "p_indptr": p_values.indptr.astype(np.int64),
        "p_indices": p_values.indices.astype(np.int64),
        "p_values": p_values.data,
        "p_errors": p_errors.data,
        "w_values": np.asarray(state["w_values"], dtype=float),
        "w_errors": np.asarray(state["w_errors"], dtype=float),
        "d_values": np.asarray(state["d_values"], dtype=float),
        "d_errors": np.asarray(state["d_errors"], dtype=float),
        "b_rows": b_rows,
        "b_columns": b_columns,
        "b_values": b_values,
        "b_errors": b_errors,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_values": n_values,
    }
    diagnostics = {
        "P_shape": list(p_shape),
        "P_nnz": int(p_values.nnz),
        "P_central_and_error_structures_align": p_structures_align,
        "P_exactly_symmetric": p_symmetric,
        "P_error_matrix_exactly_symmetric": p_errors_symmetric,
        "P_maximum_aggregated_entry_error": float(
            np.max(p_errors.data)
        ),
        "W_count": len(arrays["w_values"]),
        "D_count": len(arrays["d_values"]),
        "B_shape": [state_count, triangle_count],
        "B_nnz": len(b_values),
        "N_shape": [triangle_count, edge_count],
        "N_nnz": len(n_values),
        "N_values_exactly_plus_or_minus_one": bool(
            np.all(np.abs(n_values.astype(int)) == 1)
        ),
        "matrix_arrays_sha256": _sha256_arrays(
            arrays["p_indptr"],
            arrays["p_indices"],
            arrays["p_values"],
            arrays["p_errors"],
            arrays["w_values"],
            arrays["w_errors"],
            arrays["d_values"],
            arrays["d_errors"],
            b_rows,
            b_columns,
            b_values,
            b_errors,
            n_rows,
            n_columns,
            n_values,
        ),
    }
    return arrays, diagnostics


def run_assembly(
    pilot_result_path: Path,
    checkpoint_path: Path,
    checkpoint_metadata_path: Path,
    matrices_path: Path,
    checkpoint_interval: int,
    maximum_chunks: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    priority_set = _set_below_normal_priority()
    pilot_result = json.loads(pilot_result_path.read_text(encoding="ascii"))
    if not pilot_result[
        "all_positive_exponential_interval_pilot_checks_pass"
    ]:
        raise RuntimeError("positive-exponential interval pilot did not pass")
    pilot = _load_module(
        "neutral_strip_positive_exponential_rt_interval_pilot.py",
        "positive_exponential_complete_pilot",
    )
    hypercircle = _load_module(
        "neutral_strip_weighted_hypercircle_pilot.py",
        "positive_exponential_complete_hypercircle",
    )
    base = _load_module(
        "neutral_strip_gaussian_weighted_assembly_interval_audit.py",
        "positive_exponential_complete_base",
    )
    topology = _mesh_and_topology(hypercircle, base)
    triangles = topology["triangles"]
    vertices = topology["vertices"]
    state_lookup = topology["state_lookup"]
    local_edges = topology["local_edges"]
    local_signs = topology["local_signs"]
    total_triangles = len(triangles)
    contract = {
        "schema_version": 1,
        "spacing": SPACING,
        "Taylor_degree": TAYLOR_DEGREE,
        "quadrature_order": QUADRATURE_ORDER,
        "cross_check_order": CROSS_CHECK_ORDER,
        "cross_check_stride": CROSS_CHECK_STRIDE,
        "mesh_fingerprint_sha256": topology["mesh_fingerprint_sha256"],
        "triangle_count": total_triangles,
        "state_count": len(topology["state_vertices"]),
        "edge_count": topology["edge_count"],
    }
    if contract["mesh_fingerprint_sha256"] != pilot_result["mesh"][
        "mesh_fingerprint_sha256"
    ]:
        raise RuntimeError("pilot and assembly mesh fingerprints differ")
    if checkpoint_path.exists() or checkpoint_metadata_path.exists():
        if not (checkpoint_path.exists() and checkpoint_metadata_path.exists()):
            raise RuntimeError("only one checkpoint file exists")
        state = _load_checkpoint(
            checkpoint_path,
            checkpoint_metadata_path,
            contract,
        )
        resumed = True
    else:
        state = _empty_state()
        resumed = False

    q_nodes, q_weights = base._mapped_nodes(QUADRATURE_ORDER)
    high_nodes, high_weights = base._mapped_nodes(CROSS_CHECK_ORDER)
    try:
        import psutil

        psutil.cpu_percent(interval=None)
    except Exception:
        pass
    chunks_completed_this_run = 0
    parked_for_cpu = False

    while state["next_triangle"] < total_triangles:
        chunk_stop = min(
            total_triangles,
            state["next_triangle"] + checkpoint_interval,
        )
        for triangle_index in range(state["next_triangle"], chunk_stop):
            element = triangles[triangle_index]
            points = vertices[element]
            exact = pilot._local_interval_forms(
                points,
                TAYLOR_DEGREE,
                base,
            )
            q_source, q_rt = pilot._positive_quadrature_local_forms(
                points,
                q_nodes,
                q_weights,
            )
            metrics = state["metrics"]
            source_distance = _distance_to_interval(
                q_source,
                exact["source_mass"],
            )
            metrics["q12_containment_checks"] += 1
            metrics["q12_containment_failures"] += int(source_distance > 0.0)
            metrics["maximum_q12_distance"]["source_mass"] = max(
                metrics["maximum_q12_distance"]["source_mass"],
                source_distance,
            )
            metrics["maximum_local_interval_width"]["source_mass"] = max(
                metrics["maximum_local_interval_width"]["source_mass"],
                exact["source_mass"][1] - exact["source_mass"][0],
            )
            metrics["minimum_source_mass_lower"] = min(
                metrics["minimum_source_mass_lower"],
                exact["source_mass"][0],
            )
            state["w_values"].append(q_source)
            state["w_errors"].append(
                _entry_error(q_source, exact["source_mass"], base)
            )

            edge_matrix = np.column_stack(
                (points[1] - points[0], points[2] - points[0])
            )
            binary_area = 0.5 * abs(float(np.linalg.det(edge_matrix)))
            binary_load = binary_area / 3.0
            state["d_values"].append(binary_area)
            state["d_errors"].append(
                _entry_error(binary_area, exact["area"], base)
            )
            metrics["maximum_local_interval_width"]["area"] = max(
                metrics["maximum_local_interval_width"]["area"],
                exact["area"][1] - exact["area"][0],
            )
            metrics["maximum_local_interval_width"]["load"] = max(
                metrics["maximum_local_interval_width"]["load"],
                exact["load"][1] - exact["load"][0],
            )

            for local_row, vertex in enumerate(element):
                state_index = state_lookup.get(int(vertex))
                if state_index is not None:
                    state["b_rows"].append(state_index)
                    state["b_columns"].append(triangle_index)
                    state["b_values"].append(binary_load)
                    state["b_errors"].append(
                        _entry_error(binary_load, exact["load"], base)
                    )

            for local_row in range(3):
                edge_row = int(local_edges[triangle_index, local_row])
                sign_row = int(local_signs[triangle_index, local_row])
                state["n_rows"].append(triangle_index)
                state["n_columns"].append(edge_row)
                state["n_values"].append(sign_row)
                metrics["minimum_rt_diagonal_lower"] = min(
                    metrics["minimum_rt_diagonal_lower"],
                    exact["rt_mass"][local_row][local_row][0],
                )
                for local_column in range(3):
                    edge_column = int(
                        local_edges[triangle_index, local_column]
                    )
                    sign = (
                        sign_row
                        * int(local_signs[triangle_index, local_column])
                    )
                    interval = exact["rt_mass"][local_row][local_column]
                    central = sign * float(q_rt[local_row, local_column])
                    signed_interval = (
                        interval
                        if sign > 0
                        else (-interval[1], -interval[0])
                    )
                    state["p_rows"].append(edge_row)
                    state["p_columns"].append(edge_column)
                    state["p_values"].append(central)
                    state["p_errors"].append(
                        _entry_error(central, signed_interval, base)
                    )
                    if local_row <= local_column:
                        distance = _distance_to_interval(
                            float(q_rt[local_row, local_column]),
                            interval,
                        )
                        metrics["q12_containment_checks"] += 1
                        metrics["q12_containment_failures"] += int(
                            distance > 0.0
                        )
                        metrics["maximum_q12_distance"]["rt_mass"] = max(
                            metrics["maximum_q12_distance"]["rt_mass"],
                            distance,
                        )
                        metrics["maximum_local_interval_width"][
                            "rt_mass"
                        ] = max(
                            metrics["maximum_local_interval_width"][
                                "rt_mass"
                            ],
                            interval[1] - interval[0],
                        )

            if (
                triangle_index % CROSS_CHECK_STRIDE == 0
                or triangle_index + 1 == total_triangles
            ):
                high_source, high_rt = (
                    pilot._positive_quadrature_local_forms(
                        points,
                        high_nodes,
                        high_weights,
                    )
                )
                distance = _distance_to_interval(
                    high_source,
                    exact["source_mass"],
                )
                metrics["q18_containment_checks"] += 1
                metrics["q18_containment_failures"] += int(distance > 0.0)
                metrics["maximum_q18_distance"]["source_mass"] = max(
                    metrics["maximum_q18_distance"]["source_mass"],
                    distance,
                )
                metrics["maximum_q12_q18_difference"]["source_mass"] = max(
                    metrics["maximum_q12_q18_difference"]["source_mass"],
                    abs(q_source - high_source),
                )
                for row in range(3):
                    for column in range(row, 3):
                        interval = exact["rt_mass"][row][column]
                        distance = _distance_to_interval(
                            float(high_rt[row, column]),
                            interval,
                        )
                        metrics["q18_containment_checks"] += 1
                        metrics["q18_containment_failures"] += int(
                            distance > 0.0
                        )
                        metrics["maximum_q18_distance"]["rt_mass"] = max(
                            metrics["maximum_q18_distance"]["rt_mass"],
                            distance,
                        )
                        metrics["maximum_q12_q18_difference"][
                            "rt_mass"
                        ] = max(
                            metrics["maximum_q12_q18_difference"][
                                "rt_mass"
                            ],
                            abs(q_rt[row, column] - high_rt[row, column]),
                        )

        state["next_triangle"] = chunk_stop
        _cpu_sample(state["metrics"])
        _write_checkpoint(
            checkpoint_path,
            checkpoint_metadata_path,
            state,
            contract,
        )
        chunks_completed_this_run += 1
        samples = state["metrics"]["cpu_samples_percent"]
        if len(samples) >= 2 and all(
            value > DAYTIME_CPU_THRESHOLD for value in samples[-2:]
        ):
            parked_for_cpu = True
            break
        if maximum_chunks > 0 and chunks_completed_this_run >= maximum_chunks:
            break

    complete = state["next_triangle"] == total_triangles
    status = "complete"
    if not complete:
        status = (
            "parked_for_daytime_cpu"
            if parked_for_cpu
            else "parked_at_requested_chunk_limit"
        )

    matrices_sha256 = None
    sparse_diagnostics = None
    all_checks_pass = False
    if complete:
        matrix_arrays, sparse_diagnostics = _assemble_sparse_matrices(
            state,
            topology,
            base,
        )
        _atomic_npz(matrices_path, matrix_arrays)
        matrices_sha256 = _sha256_file(matrices_path)
        metrics = state["metrics"]
        checks = {
            "mesh_topology_valid": bool(
                topology["edge_incidence_valid"]
                and topology["interior_signs_cancel"]
            ),
            "all_q12_values_contained": (
                metrics["q12_containment_failures"] == 0
            ),
            "all_sampled_q18_values_contained": (
                metrics["q18_containment_failures"] == 0
            ),
            "source_mass_intervals_positive": (
                metrics["minimum_source_mass_lower"] > 0.0
            ),
            "RT_diagonal_intervals_positive": (
                metrics["minimum_rt_diagonal_lower"] > 0.0
            ),
            "P_exactly_symmetric": sparse_diagnostics[
                "P_exactly_symmetric"
            ],
            "P_central_and_error_structures_align": sparse_diagnostics[
                "P_central_and_error_structures_align"
            ],
            "P_errors_exactly_symmetric": sparse_diagnostics[
                "P_error_matrix_exactly_symmetric"
            ],
            "P_nnz_matches": sparse_diagnostics["P_nnz"] == 232421,
            "B_nnz_matches": sparse_diagnostics["B_nnz"] == 91124,
            "N_nnz_matches": sparse_diagnostics["N_nnz"] == 92862,
            "N_entries_exact": sparse_diagnostics[
                "N_values_exactly_plus_or_minus_one"
            ],
            "W_and_D_counts_match": (
                sparse_diagnostics["W_count"] == total_triangles
                and sparse_diagnostics["D_count"] == total_triangles
            ),
        }
        all_checks_pass = bool(all(checks.values()))
    else:
        checks = {
            "atomic_checkpoint_written": bool(
                checkpoint_path.exists()
                and checkpoint_metadata_path.exists()
            ),
            "checkpoint_hash_replays": bool(
                checkpoint_path.exists()
                and checkpoint_metadata_path.exists()
                and _sha256_file(checkpoint_path)
                == json.loads(
                    checkpoint_metadata_path.read_text(encoding="ascii")
                )["checkpoint_npz_sha256"]
            ),
        }
        all_checks_pass = bool(all(checks.values()))

    return {
        "kind": "neutral-strip-positive-exponential-complete-assembly",
        "status": status,
        "contract": contract,
        "resumed_from_checkpoint": resumed,
        "chunks_completed_this_run": chunks_completed_this_run,
        "next_triangle": state["next_triangle"],
        "complete_mesh": complete,
        "metrics": state["metrics"],
        "sparse_matrix_diagnostics": sparse_diagnostics,
        "checks": checks,
        "all_current_stage_checks_pass": all_checks_pass,
        "certification_flags": {
            "complete_mesh_RT0_P0_matrix_entries_enclosed": bool(
                complete and all_checks_pass
            ),
            "complete_mesh_P_W_D_B_checkpoint_hash_bound": bool(
                complete and all_checks_pass and matrices_sha256 is not None
            ),
            "full_mesh_threshold_inertia_certified": False,
            "kappa_h_verified_upper_bound": False,
            "global_weighted_Ritz_projection_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
        },
        "artifacts": {
            "pilot_result": str(pilot_result_path).replace("\\", "/"),
            "pilot_result_sha256": _sha256_file(pilot_result_path),
            "checkpoint": str(checkpoint_path).replace("\\", "/"),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_metadata": str(
                checkpoint_metadata_path
            ).replace("\\", "/"),
            "checkpoint_metadata_sha256": _sha256_file(
                checkpoint_metadata_path
            ),
            "matrices": (
                str(matrices_path).replace("\\", "/")
                if complete
                else None
            ),
            "matrices_sha256": matrices_sha256,
        },
        "next_required_step": (
            "Run a full central binary threshold-pencil factorization and "
            "symbolic-fill audit only after this complete assembly passes."
            if complete
            else "Resume from the hash-verified atomic triangle checkpoint."
        ),
        "below_normal_priority_set": priority_set,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-result",
        type=Path,
        default=DEFAULT_PILOT_RESULT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--checkpoint-metadata",
        type=Path,
        default=DEFAULT_CHECKPOINT_METADATA,
    )
    parser.add_argument(
        "--matrices",
        type=Path,
        default=DEFAULT_MATRICES,
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )
    parser.add_argument(
        "--maximum-chunks",
        type=int,
        default=0,
        help="zero runs to completion; positive values park after N chunks",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.checkpoint_interval <= 0:
        raise SystemExit("--checkpoint-interval must be positive")
    result = run_assembly(
        args.pilot_result,
        args.checkpoint,
        args.checkpoint_metadata,
        args.matrices,
        args.checkpoint_interval,
        args.maximum_chunks,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
