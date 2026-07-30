#!/usr/bin/env python3
"""Audit full-matrix workload and continuum dependencies without full inertia."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("work/ns_collision/results")
DEFAULT_DIRECTED = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition32064_audit_v1.json"
)
DEFAULT_DIRECTED_REPLAY = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition32064_"
    "p80_audit_v1.json"
)
DEFAULT_DIRECTED_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition32064_"
    "precision_crosscheck_v1.json"
)
DEFAULT_LOWER_CHECKPOINT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition32064_"
    "checkpoint_v1.json"
)
DEFAULT_HIGHER_CHECKPOINT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition32064_"
    "p80_checkpoint_v1.json"
)
DEFAULT_RESIDUAL = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_congruence_residual_pilot32064_v1.json"
)
DEFAULT_RESIDUAL_REPLAY = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_congruence_residual_"
    "pilot32064_p100_v1.json"
)
DEFAULT_RESIDUAL_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_congruence_residual_"
    "precision_crosscheck32064_v1.json"
)
DEFAULT_SYMBOLIC_MAP = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_symbolic_transition_map123816_v1.json"
)
DEFAULT_CONTINUUM = RESULTS_DIR / (
    "neutral_strip_h006_continuum_ritz_dependency_audit_v1.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_full_feasibility_audit_v1.json"
)
DAYTIME_BASELINE_CPU_LIMIT = 60.0


def _load_prefix_module():
    path = (
        SCRIPT_DIR
        / "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hypercircle_full_feasibility_base",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _compact_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )


def _symbolic_points(symbolic_map: dict[str, Any]):
    rows = [(0, 0)]
    for pivot, row in symbolic_map["profile"]["checkpoints"].items():
        rows.append((int(pivot), int(row["symbolic_lower_entry_count"])))
    return sorted(set(rows))


def _interpolate_lower_entries(
    pivot: int,
    points: list[tuple[int, int]],
) -> float:
    for (left_pivot, left_count), (right_pivot, right_count) in zip(
        points,
        points[1:],
        strict=True,
    ):
        if pivot <= right_pivot:
            fraction = (
                (pivot - left_pivot) / (right_pivot - left_pivot)
                if right_pivot != left_pivot
                else 0.0
            )
            return left_count + fraction * (right_count - left_count)
    return float(points[-1][1])


def _checkpoint_projection(
    checkpoint_path: Path,
    full_pivots: int,
    full_lower_entries: int,
    points: list[tuple[int, int]],
    checkpoint_batch: int,
) -> dict[str, Any]:
    checkpoint = _load_json(checkpoint_path)
    attempt = checkpoint["current_attempt"]
    completed = int(attempt["next_pivot"])
    lower_count = len(attempt["lower_entries"])
    pivot_fields = (
        "pivot_diagnostics",
        "pivot_lower_decimal",
        "pivot_upper_decimal",
        "pivot_sign",
    )
    pivot_component_bytes = sum(
        _compact_size(attempt[key]) for key in pivot_fields
    )
    lower_component_bytes = _compact_size(attempt["lower_entries"])
    bytes_per_pivot = pivot_component_bytes / completed
    bytes_per_lower_entry = lower_component_bytes / lower_count
    projected_final = (
        bytes_per_pivot * full_pivots
        + bytes_per_lower_entry * full_lower_entries
        + 2048
    )
    checkpoints = list(range(checkpoint_batch, full_pivots, checkpoint_batch))
    checkpoints.append(full_pivots)
    projected_writes = sum(
        bytes_per_pivot * pivot
        + bytes_per_lower_entry * _interpolate_lower_entries(pivot, points)
        + 2048
        for pivot in checkpoints
    )
    return {
        "source_checkpoint": str(checkpoint_path).replace("\\", "/"),
        "source_checkpoint_bytes": checkpoint_path.stat().st_size,
        "source_completed_pivots": completed,
        "source_lower_entries": lower_count,
        "empirical_bytes_per_pivot_component": bytes_per_pivot,
        "empirical_bytes_per_lower_entry": bytes_per_lower_entry,
        "projected_full_checkpoint_bytes": projected_final,
        "projected_full_checkpoint_MiB": projected_final / (1024**2),
        "projected_checkpoint_write_count": len(checkpoints),
        "projected_cumulative_full_rewrite_bytes": projected_writes,
        "projected_cumulative_full_rewrite_GiB": (
            projected_writes / (1024**3)
        ),
        "projection_is_diagnostic_not_storage_certificate": True,
    }


def _kernel_benchmark(prefix, iterations: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for precision in (50, 80):
        arithmetic = prefix.DirectedDecimal(precision)
        left = (
            Decimal("0.1234567890123456789"),
            Decimal("0.1234567890123456791"),
        )
        right = (
            Decimal("-0.9876543210987654322"),
            Decimal("-0.9876543210987654318"),
        )
        pivot = (
            Decimal("1.234567890123456789"),
            Decimal("1.234567890123456791"),
        )
        total = (arithmetic.zero, arithmetic.zero)
        started = time.perf_counter()
        for _ in range(iterations):
            term = arithmetic.multiply(left, right)
            term = arithmetic.multiply(term, pivot)
            total = arithmetic.add(total, term)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "decimal_precision": precision,
                "representative_common_terms": iterations,
                "elapsed_seconds": elapsed,
                "representative_terms_per_second": iterations / elapsed,
                "benchmark_result_finite": (
                    total[0].is_finite() and total[1].is_finite()
                ),
            }
        )
    return rows


def run_audit(
    directed_path: Path = DEFAULT_DIRECTED,
    directed_replay_path: Path = DEFAULT_DIRECTED_REPLAY,
    directed_crosscheck_path: Path = DEFAULT_DIRECTED_CROSSCHECK,
    lower_checkpoint_path: Path = DEFAULT_LOWER_CHECKPOINT,
    higher_checkpoint_path: Path = DEFAULT_HIGHER_CHECKPOINT,
    residual_path: Path = DEFAULT_RESIDUAL,
    residual_replay_path: Path = DEFAULT_RESIDUAL_REPLAY,
    residual_crosscheck_path: Path = DEFAULT_RESIDUAL_CROSSCHECK,
    symbolic_map_path: Path = DEFAULT_SYMBOLIC_MAP,
    continuum_path: Path = DEFAULT_CONTINUUM,
    benchmark_iterations: int = 100000,
    enforce_cpu_policy: bool = True,
) -> dict[str, Any]:
    if benchmark_iterations <= 0:
        raise ValueError("benchmark iterations must be positive")
    prefix = _load_prefix_module()
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

    directed = _load_json(directed_path)
    directed_replay = _load_json(directed_replay_path)
    directed_crosscheck = _load_json(directed_crosscheck_path)
    residual = _load_json(residual_path)
    residual_replay = _load_json(residual_replay_path)
    residual_crosscheck = _load_json(residual_crosscheck_path)
    symbolic_map = _load_json(symbolic_map_path)
    continuum = _load_json(continuum_path)
    profile = symbolic_map["profile"]
    full_pivots = int(profile["maximum_pivots"])
    full_lower_entries = int(profile["symbolic_lower_entry_count"])
    full_common_terms = int(profile["total_off_diagonal_common_term_count"])
    full_reference_pairs = int(profile["reference_product_pair_term_count"])
    points = _symbolic_points(symbolic_map)
    checkpoint_batch = int(directed["contract"]["checkpoint_batch"])
    lower_projection = _checkpoint_projection(
        lower_checkpoint_path,
        full_pivots,
        full_lower_entries,
        points,
        checkpoint_batch,
    )
    higher_projection = _checkpoint_projection(
        higher_checkpoint_path,
        full_pivots,
        full_lower_entries,
        points,
        checkpoint_batch,
    )
    benchmark = _kernel_benchmark(prefix, benchmark_iterations)
    kernel_rows = []
    for row in benchmark:
        rate = row["representative_terms_per_second"]
        kernel_rows.append(
            {
                **row,
                "full_common_term_kernel_lower_bound_seconds": (
                    full_common_terms / rate
                ),
                "full_common_term_kernel_lower_bound_minutes": (
                    full_common_terms / rate / 60.0
                ),
                "excludes_set_intersections_divisions_IO_and_replay": True,
            }
        )

    checkpoints = profile["checkpoints"]
    late_start = checkpoints["114688"]
    late_common_terms = full_common_terms - int(
        late_start["total_off_diagonal_common_term_count"]
    )
    late_reference_pairs = full_reference_pairs - int(
        late_start["reference_product_pair_term_count"]
    )
    checks = {
        "directed_32064_pair_certified_and_nested": (
            directed["status"] == "certified_bounded_prefix"
            and directed_replay["status"] == "certified_bounded_prefix"
            and directed_crosscheck["all_checks_pass"]
        ),
        "residual_32064_pair_certified_and_nested": (
            residual["status"] == "independent_prefix_inertia_certified"
            and residual_replay["status"]
            == "independent_prefix_inertia_certified"
            and residual_crosscheck["all_checks_pass"]
        ),
        "two_routes_report_same_32064_signs": (
            residual["certificate"]["reference_diagonal_signs"]
            == {
                "negative": directed["directed_LDL_prefix"][
                    "negative_pivot_count"
                ],
                "positive": directed["directed_LDL_prefix"][
                    "positive_pivot_count"
                ],
                "zero": 0,
            }
        ),
        "full_symbolic_map_passes_without_arithmetic_claim": (
            symbolic_map["status"] == "pass"
            and symbolic_map["certification_flags"][
                "symbolic_profile_validated"
            ]
            and not symbolic_map["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
        ),
        "continuum_dependency_audit_remains_fail_closed": (
            continuum[
                "all_continuum_Ritz_dependency_audit_checks_pass"
            ]
            and not continuum["certification_flags"][
                "weighted_global_Ritz_projection_constant_certified"
            ]
            and not continuum["certification_flags"][
                "continuum_spectrum_below_60_captured_by_240_FE_modes"
            ]
        ),
        "full_inertia_claim_remains_false": True,
        "regularity_claim_remains_false": True,
    }
    return {
        "kind": "hypercircle-full-run-feasibility-and-dependency-audit",
        "status": "bounded_feasibility_characterized",
        "all_current_stage_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "current_certified_boundary": {
            "completed_pivots": directed["directed_LDL_prefix"][
                "completed_pivot_count"
            ],
            "fraction_of_full_dimension": (
                directed["directed_LDL_prefix"]["completed_pivot_count"]
                / full_pivots
            ),
            "negative": directed["directed_LDL_prefix"][
                "negative_pivot_count"
            ],
            "positive": directed["directed_LDL_prefix"][
                "positive_pivot_count"
            ],
            "directed_minimum_margin_decimal": directed[
                "directed_LDL_prefix"
            ]["minimum_pivot_margin_decimal"],
            "residual_bound_to_diagonal_ratio_upper_decimal": residual[
                "certificate"
            ]["transformed_bound_to_minimum_diagonal_upper_decimal"],
        },
        "full_symbolic_workload": {
            "dimension": full_pivots,
            "first_state_pivot": symbolic_map[
                "first_block_pivots_within_scan"
            ]["state"],
            "symbolic_lower_entries": full_lower_entries,
            "total_diagonal_terms": profile["total_diagonal_term_count"],
            "total_off_diagonal_common_terms": full_common_terms,
            "reference_product_pair_terms": full_reference_pairs,
            "maximum_descendants": profile["maximum_descendant_count"],
            "maximum_diagonal_terms_at_one_pivot": profile[
                "maximum_diagonal_term_count"
            ],
            "maximum_common_terms_at_one_lower_entry": profile[
                "maximum_off_diagonal_common_term_count"
            ],
            "last_9128_pivots_common_term_fraction": (
                late_common_terms / full_common_terms
            ),
            "last_9128_pivots_reference_pair_fraction": (
                late_reference_pairs / full_reference_pairs
            ),
            "workload_is_strongly_backloaded": True,
            "runtime_extrapolation_from_32064_rejected": True,
        },
        "checkpoint_storage_projection": {
            "checkpoint_batch": checkpoint_batch,
            "precision_50": lower_projection,
            "precision_80": higher_projection,
            "interpretation": (
                "Capacity is ample, but repeated monolithic checkpoint "
                "rewrites are avoidable. A full launch requires chunked or "
                "append-only hash-chained state first."
            ),
        },
        "representative_decimal_kernel_benchmark": kernel_rows,
        "launch_decision": {
            "full_directed_LDL_launch_ready": False,
            "full_congruence_residual_launch_ready": False,
            "memory_or_disk_capacity_is_the_primary_blocker": False,
            "primary_engineering_blocker": (
                "replace monolithic checkpoint rewrites and add bounded "
                "state-region pilots before the strongly backloaded tail"
            ),
            "primary_mathematical_risk": (
                "interval growth and absolute inverse-majorant growth after "
                "state variables enter at pivot 63644"
            ),
            "next_directed_transition_pivot": 33224,
            "next_directed_bounded_pivot_count": 33280,
            "first_state_entry_bounded_pivot_count": 63680,
            "recommended_sequence": [
                "Certify the bounded 33280 transition with both routes.",
                "Run an independent residual-only 63680 state-entry pilot.",
                "Redesign full checkpoints as chunked hash-chained records.",
                "Only then decide whether a full finite inertia run is "
                "scientifically justified.",
            ],
        },
        "continuum_dependency_gates": {
            "finite_full_inertia_would_not_prove_continuum_capture": True,
            "weighted_Ritz_projection_constant_strict_threshold": continuum[
                "cutoff_solution_operator_route"
            ][
                "Ritz_projection_constant_strict_threshold_lower"
            ],
            "solution_operator_error_strict_threshold": continuum[
                "cutoff_solution_operator_route"
            ][
                "solution_operator_error_strict_threshold_lower"
            ],
            "weighted_global_Ritz_projection_constant_certified": False,
            "positive_time_point_source_transfer_certified": False,
            "continuum_conormal_response_certified": False,
            "polygon_to_circle_domain_transfer_certified": False,
            "dependency_order": [
                "weighted global Ritz/solution-operator error",
                "positive-time singular-source transfer",
                "smoothed conormal-output transfer",
                "polygon-to-circle domain perturbation",
                "Navier-Stokes closure",
            ],
        },
        "certification_flags": {
            "bounded_32064_inertia_certified_by_two_routes": True,
            "full_123816_pivot_inertia_certified": False,
            "weighted_global_Ritz_projection_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
            "navier_stokes_regularity_certified": False,
        },
        "runtime": {
            "below_normal_priority_set": priority_set,
            "baseline_cpu_samples_percent": baseline,
            "baseline_cpu_mean_percent": baseline_mean,
        },
        "artifacts": {
            "directed_audit": str(directed_path).replace("\\", "/"),
            "directed_audit_sha256": prefix._sha256_file(directed_path),
            "directed_replay": str(directed_replay_path).replace("\\", "/"),
            "directed_replay_sha256": prefix._sha256_file(
                directed_replay_path
            ),
            "directed_crosscheck": str(directed_crosscheck_path).replace(
                "\\",
                "/",
            ),
            "directed_crosscheck_sha256": prefix._sha256_file(
                directed_crosscheck_path
            ),
            "residual_audit": str(residual_path).replace("\\", "/"),
            "residual_audit_sha256": prefix._sha256_file(residual_path),
            "residual_replay": str(residual_replay_path).replace("\\", "/"),
            "residual_replay_sha256": prefix._sha256_file(
                residual_replay_path
            ),
            "residual_crosscheck": str(residual_crosscheck_path).replace(
                "\\",
                "/",
            ),
            "residual_crosscheck_sha256": prefix._sha256_file(
                residual_crosscheck_path
            ),
            "symbolic_map": str(symbolic_map_path).replace("\\", "/"),
            "symbolic_map_sha256": prefix._sha256_file(symbolic_map_path),
            "continuum_dependency_audit": str(continuum_path).replace(
                "\\",
                "/",
            ),
            "continuum_dependency_audit_sha256": prefix._sha256_file(
                continuum_path
            ),
        },
        "scope": (
            "This audit measures symbolic work, empirical storage, and "
            "dependency gates. It does not run or certify full finite "
            "inertia, a Ritz constant, continuum spectral capture, or "
            "Navier-Stokes regularity."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-iterations", type=int, default=100000)
    parser.add_argument("--skip-cpu-policy", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_audit(
        benchmark_iterations=args.benchmark_iterations,
        enforce_cpu_policy=not args.skip_cpu_policy,
    )
    prefix = _load_prefix_module()
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_current_stage_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
