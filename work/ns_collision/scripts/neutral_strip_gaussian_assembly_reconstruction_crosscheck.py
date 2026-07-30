"""Bitwise cross-check checkpointed q12 matrices against the original assembler."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix


def _load_module(filename: str, module_name: str):
    script = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
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


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def audit(
    spacing: float,
    quadrature_order: int,
    checkpoint_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    below_normal_priority_set = _set_below_normal_priority()
    boundary = _load_module(
        "neutral_strip_reversible_boundary_fem_pilot.py",
        "assembly_crosscheck_boundary",
    )
    consistency = _load_module(
        "neutral_strip_reversible_fem_consistency_gate.py",
        "assembly_crosscheck_consistency",
    )
    assembly = _load_module(
        "neutral_strip_gaussian_weighted_assembly_interval_audit.py",
        "assembly_crosscheck_interval",
    )
    spectral = _load_module(
        "neutral_strip_parabolic_spectral_split_audit.py",
        "assembly_crosscheck_spectral",
    )

    grid = boundary._build_mesh(spacing)
    reference = consistency._reference_forms(
        grid,
        quadrature_order,
        mass_coercivity_alpha=0.15,
    )
    original = {
        "mass": reference["mass"].tocsr(),
        "stiffness": reference["stiffness"].tocsr(),
        "boundary": reference["boundary_coupling"].tocsr(),
        "boundary_mass": reference["boundary_mass_coupling"].tocsr(),
    }
    rows: dict[str, dict[str, object]] = {}
    reconstructed = {}
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        checkpoint_position = int(
            checkpoint["next_selected_position"].item()
        )
        for name, matrix in original.items():
            rebuilt = coo_matrix(
                (
                    checkpoint[f"{name}_values"],
                    (
                        checkpoint[f"{name}_rows"],
                        checkpoint[f"{name}_columns"],
                    ),
                ),
                shape=matrix.shape,
            ).tocsr()
            reconstructed[name] = rebuilt
            structure_matches = bool(
                np.array_equal(rebuilt.indptr, matrix.indptr)
                and np.array_equal(rebuilt.indices, matrix.indices)
            )
            data_matches = bool(np.array_equal(rebuilt.data, matrix.data))
            difference = abs(rebuilt - matrix)
            rows[name] = {
                "csr_structure_matches": structure_matches,
                "data_bitwise_equal": data_matches,
                "bitwise_equal": structure_matches and data_matches,
                "maximum_absolute_difference": (
                    float(np.max(difference.data))
                    if len(difference.data)
                    else 0.0
                ),
                "original_q12_fingerprint_sha256": (
                    assembly._sparse_matrix_fingerprint(matrix)
                ),
                "checkpoint_reconstruction_fingerprint_sha256": (
                    assembly._sparse_matrix_fingerprint(rebuilt)
                ),
            }

    pair_fingerprint = spectral._sparse_eigensystem_fingerprint(
        reconstructed["mass"],
        reconstructed["stiffness"],
    )
    all_four_match = all(
        row["bitwise_equal"] for row in rows.values()
    )
    expected_hashes_match = all(
        rows[name]["original_q12_fingerprint_sha256"]
        == assembly.EXPECTED_MATRIX_FINGERPRINTS[name]
        for name in rows
    )
    checks = [
        checkpoint_position == len(grid["triangles"]),
        all_four_match,
        expected_hashes_match,
        pair_fingerprint
        == "c80fdc5aa494fe1118aec4c38045df76151b2ccde3fded18f4af1161de7efd48",
    ]
    return {
        "model": (
            "independent original-q12 versus interval-checkpoint "
            "matrix reconstruction"
        ),
        "spacing": spacing,
        "quadrature_order": quadrature_order,
        "triangle_count": int(len(grid["triangles"])),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_next_triangle": checkpoint_position,
        "below_normal_priority_set": below_normal_priority_set,
        "matrix_rows": rows,
        "all_four_matrices_bitwise_equal": all_four_match,
        "all_expected_per_matrix_fingerprints_match": (
            expected_hashes_match
        ),
        "mass_stiffness_pair_fingerprint_sha256": pair_fingerprint,
        "all_reconstruction_cross_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=0.06)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "work/ns_collision/results/"
            "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.spacing,
        args.quadrature_order,
        args.checkpoint,
    )
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
