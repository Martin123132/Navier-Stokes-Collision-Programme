#!/usr/bin/env python3
"""Map bounded symbolic-fill transitions under the frozen elimination order."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("work/ns_collision/results")
DEFAULT_COMPLETE_RESULT = RESULTS_DIR / (
    "neutral_strip_h006_positive_exponential_complete_assembly_v1.json"
)
DEFAULT_MATRICES = RESULTS_DIR / (
    "neutral_strip_h006_positive_exponential_assembly_matrices_v1.npz"
)
DEFAULT_GAUSSIAN_RESULT = RESULTS_DIR / (
    "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
)
DEFAULT_GAUSSIAN_CHECKPOINT = RESULTS_DIR / (
    "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_symbolic_transition_map32768_v1.json"
)
DEFAULT_MAXIMUM_PIVOTS = 32768
DEFAULT_CHECKPOINTS = (8192, 12288, 16384, 24576, 32768)
DAYTIME_BASELINE_CPU_LIMIT = 60.0


def _load_prefix_module():
    path = (
        SCRIPT_DIR
        / "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "symbolic_transition_prefix_base",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _block_boundaries(inventory: dict[str, Any], dimension: int):
    edge_count = int(inventory["edge_count"])
    triangle_count = int(inventory["triangle_count"])
    state_count = int(inventory["state_count"])
    return (
        ("edge_metric", 0, edge_count),
        (
            "triangle_constraint",
            edge_count,
            edge_count + triangle_count,
        ),
        (
            "state",
            edge_count + triangle_count,
            edge_count + triangle_count + state_count,
        ),
        (
            "source_triangle",
            edge_count + triangle_count + state_count,
            dimension,
        ),
    )


def _block_profile(
    problem,
    inventory: dict[str, Any],
    pivot_count: int,
) -> tuple[dict[str, int], dict[str, int | None]]:
    original = problem.order[:pivot_count]
    boundaries = _block_boundaries(inventory, problem.dimension)
    counts = {
        name: int(np.count_nonzero((original >= start) & (original < stop)))
        for name, start, stop in boundaries
    }
    first_pivots: dict[str, int | None] = {}
    for name, start, stop in boundaries:
        matches = np.flatnonzero((original >= start) & (original < stop))
        first_pivots[name] = int(matches[0]) if len(matches) else None
    return counts, first_pivots


def _new_transition_rows(
    profile: dict[str, Any],
    after_pivot: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mappings = (
        (
            "diagonal_term_count",
            profile["first_pivot_by_diagonal_term_count"],
        ),
        (
            "descendant_count",
            profile["first_pivot_by_descendant_count"],
        ),
        (
            "off_diagonal_common_term_count",
            profile["first_pivot_by_off_diagonal_common_term_count"],
        ),
    )
    for metric, mapping in mappings:
        for value, pivot in mapping.items():
            if int(pivot) > after_pivot:
                rows.append(
                    {
                        "pivot": int(pivot),
                        "metric": metric,
                        "new_value": int(value),
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["pivot"],
            row["metric"],
            row["new_value"],
        ),
    )


def summarize_map(
    problem,
    preparation: dict[str, Any],
    maximum_pivots: int,
    checkpoints: tuple[int, ...],
    prior_scan_pivots: int = 8192,
) -> dict[str, Any]:
    prefix = _load_prefix_module()
    profile = prefix._scan_symbolic_prefix(
        problem,
        maximum_pivots,
        checkpoints,
    )
    inventory = preparation["matrix_inventory"]
    checkpoint_profiles: dict[str, dict[str, Any]] = {}
    for checkpoint in checkpoints:
        counts, first_pivots = _block_profile(
            problem,
            inventory,
            checkpoint,
        )
        checkpoint_profiles[str(checkpoint)] = {
            **profile["checkpoints"][str(checkpoint)],
            "block_counts": counts,
            "first_block_pivots_within_prefix": first_pivots,
        }
    final_counts, first_block_pivots = _block_profile(
        problem,
        inventory,
        maximum_pivots,
    )
    transition_rows = _new_transition_rows(profile, prior_scan_pivots - 1)
    next_transition = transition_rows[0]["pivot"] if transition_rows else None
    recommended = (
        min(
            maximum_pivots,
            64 * ((int(next_transition) + 1 + 63) // 64),
        )
        if next_transition is not None
        else None
    )
    profile["checkpoints"] = checkpoint_profiles
    return {
        "profile": profile,
        "final_block_counts": final_counts,
        "first_block_pivots_within_scan": first_block_pivots,
        "new_transitions_at_or_after_prior_boundary": transition_rows,
        "next_transition_pivot": next_transition,
        "recommended_next_bounded_pivot_count": recommended,
    }


def run_map(
    complete_result_path: Path = DEFAULT_COMPLETE_RESULT,
    matrices_path: Path = DEFAULT_MATRICES,
    gaussian_result_path: Path = DEFAULT_GAUSSIAN_RESULT,
    gaussian_checkpoint_path: Path = DEFAULT_GAUSSIAN_CHECKPOINT,
    maximum_pivots: int = DEFAULT_MAXIMUM_PIVOTS,
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS,
    prior_scan_pivots: int = 8192,
    enforce_cpu_policy: bool = True,
) -> dict[str, Any]:
    if not checkpoints or checkpoints[-1] != maximum_pivots:
        raise ValueError("checkpoints must end at maximum pivots")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be unique and increasing")
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
    started = time.perf_counter()
    problem, preparation = prefix._prepare_production_problem(
        complete_result_path,
        matrices_path,
        gaussian_result_path,
        gaussian_checkpoint_path,
    )
    summary = summarize_map(
        problem,
        preparation,
        maximum_pivots,
        checkpoints,
        prior_scan_pivots,
    )
    return {
        "kind": "hypercircle-directed-ldl-symbolic-transition-map",
        "status": "pass",
        "scope": (
            "Exact potential-fill scan under the frozen input graph and "
            "elimination order. It performs no interval arithmetic and "
            "certifies no pivot sign or inertia."
        ),
        "contract": {
            "maximum_pivots": maximum_pivots,
            "checkpoints": list(checkpoints),
            "prior_scan_pivots": prior_scan_pivots,
            "scale_sha256": preparation["hashes"]["scale_sha256"],
            "raw_permutation_sha256": preparation["hashes"][
                "raw_permutation_sha256"
            ],
            "order_sha256": preparation["hashes"]["order_sha256"],
            "factor_pattern_sha256": preparation["hashes"][
                "factor_pattern_sha256"
            ],
        },
        **summary,
        "checks": {
            **preparation["preparation_checks"],
            "maximum_pivot_within_dimension": (
                maximum_pivots <= problem.dimension
            ),
            "arithmetic_signs_remain_uncertified": True,
            "full_inertia_claim_remains_false": True,
        },
        "certification_flags": {
            "symbolic_profile_validated": True,
            "any_new_pivot_sign_certified": False,
            "full_123816_pivot_inertia_certified": False,
            "continuum_spectrum_below_60_captured": False,
        },
        "preparation": preparation,
        "runtime": {
            "below_normal_priority_set": priority_set,
            "baseline_cpu_samples_percent": baseline,
            "baseline_cpu_mean_percent": baseline_mean,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated integers"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maximum-pivots",
        type=int,
        default=DEFAULT_MAXIMUM_PIVOTS,
    )
    parser.add_argument(
        "--checkpoints",
        type=_parse_ints,
        default=DEFAULT_CHECKPOINTS,
    )
    parser.add_argument("--prior-scan-pivots", type=int, default=8192)
    parser.add_argument("--skip-cpu-policy", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_map(
        maximum_pivots=args.maximum_pivots,
        checkpoints=args.checkpoints,
        prior_scan_pivots=args.prior_scan_pivots,
        enforce_cpu_policy=not args.skip_cpu_policy,
    )
    prefix = _load_prefix_module()
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
