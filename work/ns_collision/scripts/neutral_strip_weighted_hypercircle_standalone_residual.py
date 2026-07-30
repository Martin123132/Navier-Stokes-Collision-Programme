#!/usr/bin/env python3
"""Certify a bounded prefix by a standalone congruence-residual proof.

The proof consumes the frozen interval pencil directly. It does not require
or inspect a directed-LDL audit. Exact source-file, ordered-prefix, scale,
permutation, and binary reference-factor hashes bind every result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


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
DEFAULT_OUTPUT = Path(
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual2304_v1.json"
)
DEFAULT_MAXIMUM_PIVOTS = 2304
DEFAULT_DECIMAL_PRECISION = 60
DAYTIME_BASELINE_CPU_LIMIT = 60.0
DAYTIME_PARK_CPU_LIMIT = 75.0


def _load_module(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_prefix_module():
    return _load_module(
        "standalone_residual_prefix_base",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )


def _load_residual_module():
    return _load_module(
        "standalone_residual_arithmetic_base",
        "neutral_strip_weighted_hypercircle_congruence_residual_pilot.py",
    )


def _sparse_fingerprint(matrix, prefix_module) -> str:
    sparse = matrix.tocsr(copy=True)
    sparse.sort_indices()
    return prefix_module._sha256_arrays(
        sparse.indptr,
        sparse.indices,
        sparse.data,
    )


def _source_artifact(
    path: Path,
    prefix_module,
) -> dict[str, Any]:
    return {
        "path": str(path).replace("\\", "/"),
        "sha256": prefix_module._sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _build_standalone_contract(
    problem,
    preparation: dict[str, Any],
    *,
    complete_result_path: Path,
    matrices_path: Path,
    gaussian_result_path: Path,
    gaussian_checkpoint_path: Path,
    maximum_pivots: int,
    prefix_module,
) -> dict[str, Any]:
    if not 0 < maximum_pivots <= problem.dimension:
        raise ValueError("maximum pivots is out of range")
    original = np.asarray(
        problem.order[:maximum_pivots],
        dtype=np.int64,
    )
    ordered_scale = np.asarray(problem.scale[original], dtype=float)
    center_prefix = problem.center[original, :][:, original]
    radius_prefix = problem.radius[original, :][:, original]
    source_artifacts = {
        "complete_assembly_result": _source_artifact(
            complete_result_path,
            prefix_module,
        ),
        "interval_matrix_archive": _source_artifact(
            matrices_path,
            prefix_module,
        ),
        "Gaussian_assembly_result": _source_artifact(
            gaussian_result_path,
            prefix_module,
        ),
        "Gaussian_assembly_checkpoint": _source_artifact(
            gaussian_checkpoint_path,
            prefix_module,
        ),
    }
    contract = {
        "algorithm_version": 1,
        "validation_mode": "standalone_hash_bound",
        "maximum_pivots": maximum_pivots,
        "full_dimension": int(problem.dimension),
        "interval_family": (
            "stored exact binary64 center plus/minus nonnegative radius, "
            "followed by the frozen positive Ruiz diagonal congruence"
        ),
        "reference_rule": (
            "natural-order zero-threshold SuperLU of the scaled binary64 "
            "central prefix, interpreted exactly as L D L^T"
        ),
        "source_artifacts": source_artifacts,
        "frozen_preparation_hashes": dict(preparation["hashes"]),
        "ordered_original_indices_sha256": (
            prefix_module._sha256_arrays(original)
        ),
        "ordered_positive_scale_sha256": (
            prefix_module._sha256_arrays(ordered_scale)
        ),
        "ordered_center_prefix_sha256": _sparse_fingerprint(
            center_prefix,
            prefix_module,
        ),
        "ordered_radius_prefix_sha256": _sparse_fingerprint(
            radius_prefix,
            prefix_module,
        ),
        "ordered_center_prefix_nnz": int(center_prefix.nnz),
        "ordered_radius_prefix_nnz": int(radius_prefix.nnz),
    }
    return {
        "contract": contract,
        "contract_sha256": prefix_module._canonical_sha256(contract),
    }


def run_standalone(
    complete_result_path: Path = DEFAULT_COMPLETE_RESULT,
    matrices_path: Path = DEFAULT_MATRICES,
    gaussian_result_path: Path = DEFAULT_GAUSSIAN_RESULT,
    gaussian_checkpoint_path: Path = DEFAULT_GAUSSIAN_CHECKPOINT,
    maximum_pivots: int = DEFAULT_MAXIMUM_PIVOTS,
    decimal_precision: int = DEFAULT_DECIMAL_PRECISION,
    enforce_cpu_policy: bool = True,
) -> dict[str, Any]:
    prefix = _load_prefix_module()
    residual = _load_residual_module()
    priority_set = prefix._set_below_normal_priority()
    baseline = (
        prefix._sample_cpu_baseline(5) if enforce_cpu_policy else []
    )
    baseline_mean = (
        sum(baseline) / len(baseline) if baseline else None
    )
    if (
        baseline_mean is not None
        and baseline_mean > DAYTIME_BASELINE_CPU_LIMIT
    ):
        raise RuntimeError(
            "daytime baseline CPU exceeds the one-worker launch limit: "
            f"{baseline_mean:.3f}%"
        )

    problem, preparation = prefix._prepare_production_problem(
        complete_result_path,
        matrices_path,
        gaussian_result_path,
        gaussian_checkpoint_path,
    )
    standalone_contract = _build_standalone_contract(
        problem,
        preparation,
        complete_result_path=complete_result_path,
        matrices_path=matrices_path,
        gaussian_result_path=gaussian_result_path,
        gaussian_checkpoint_path=gaussian_checkpoint_path,
        maximum_pivots=maximum_pivots,
        prefix_module=prefix,
    )
    periodic_cpu_samples: list[float] = []

    def cpu_park() -> bool:
        if not enforce_cpu_policy:
            return False
        try:
            import psutil

            periodic_cpu_samples.append(
                float(psutil.cpu_percent(interval=0.25))
            )
        except Exception:
            return False
        return bool(
            len(periodic_cpu_samples) >= 2
            and all(
                value > DAYTIME_PARK_CPU_LIMIT
                for value in periodic_cpu_samples[-2:]
            )
        )

    certificate = residual.certify_problem(
        problem,
        maximum_pivots,
        decimal_precision,
        prefix,
        cpu_park_callback=cpu_park,
    )
    contract = standalone_contract["contract"]
    source_hashes_recorded = all(
        len(record["sha256"]) == 64
        and record["bytes"] > 0
        for record in contract["source_artifacts"].values()
    )
    reference_hashes_recorded = all(
        len(certificate[key]) == 64
        for key in (
            "reference_L_sha256",
            "reference_D_sha256",
            "reference_factor_sha256",
        )
    )
    checks = {
        **preparation["preparation_checks"],
        "standalone_mode_uses_no_directed_audit": True,
        "source_artifacts_are_hash_bound": source_hashes_recorded,
        "ordered_prefix_family_is_hash_bound": all(
            len(contract[key]) == 64
            for key in (
                "ordered_original_indices_sha256",
                "ordered_positive_scale_sha256",
                "ordered_center_prefix_sha256",
                "ordered_radius_prefix_sha256",
            )
        ),
        "standalone_contract_hash_recorded": (
            len(standalone_contract["contract_sha256"]) == 64
        ),
        "reference_factor_hashes_recorded": reference_hashes_recorded,
        "residual_certificate_decision_recorded": True,
        "full_inertia_claim_remains_false": True,
    }
    integrity_passes = bool(all(checks.values()))
    certified = bool(
        integrity_passes
        and certificate["interval_family_inertia_certified"]
    )
    return {
        "kind": "hypercircle-standalone-congruence-residual-inertia",
        "validation_mode": "standalone_hash_bound",
        "status": (
            "standalone_prefix_inertia_certified"
            if certified
            else "standalone_route_does_not_close"
        ),
        "scope": (
            "Hash-bound congruence-residual certificate for exactly the "
            "reported leading prefix. It uses no directed-LDL result and "
            "certifies no later pivot, full inertia, continuum transfer, or "
            "Navier-Stokes statement."
        ),
        "all_current_stage_checks_pass": integrity_passes,
        "checks": checks,
        "certificate": certificate,
        "standalone_contract": standalone_contract,
        "preparation": preparation,
        "runtime": {
            "below_normal_priority_set": priority_set,
            "baseline_cpu_samples_percent": baseline,
            "baseline_cpu_mean_percent": baseline_mean,
            "periodic_cpu_samples_percent": periodic_cpu_samples,
        },
        "directed_LDL_dependency": {
            "required": False,
            "audit_loaded": False,
            "sign_comparison_used_for_certification": False,
        },
        "certification_flags": {
            "standalone_bounded_prefix_inertia_certified": certified,
            "independent_bounded_prefix_inertia_certified": certified,
            "full_123816_pivot_inertia_certified": False,
            "continuum_spectrum_below_60_captured": False,
            "navier_stokes_regularity_certified": False,
        },
        "artifacts": contract["source_artifacts"],
        "next_required_step": (
            "Replay this exact standalone contract at higher Decimal "
            "precision and require all upper bounds to nest."
            if certified
            else "The standalone residual route is fail-closed at this "
            "prefix; inspect the inverse-majorant, residual, and minimum "
            "reference diagonal before any larger run."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complete-result",
        type=Path,
        default=DEFAULT_COMPLETE_RESULT,
    )
    parser.add_argument("--matrices", type=Path, default=DEFAULT_MATRICES)
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
        "--maximum-pivots",
        type=int,
        default=DEFAULT_MAXIMUM_PIVOTS,
    )
    parser.add_argument(
        "--decimal-precision",
        type=int,
        default=DEFAULT_DECIMAL_PRECISION,
    )
    parser.add_argument("--skip-cpu-policy", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_standalone(
        complete_result_path=args.complete_result,
        matrices_path=args.matrices,
        gaussian_result_path=args.gaussian_result,
        gaussian_checkpoint_path=args.gaussian_checkpoint,
        maximum_pivots=args.maximum_pivots,
        decimal_precision=args.decimal_precision,
        enforce_cpu_policy=not args.skip_cpu_policy,
    )
    prefix = _load_prefix_module()
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
