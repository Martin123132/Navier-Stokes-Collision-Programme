"""Install the quantitative pressure-load realization-cost checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BOOKMARK = ROOT / "work/ns_collision/results/session_bookmark.json"
PRIOR_RESULT = (
    "work/ns_collision/results/"
    "fourier_pressure_load_surjectivity_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "7af42aaf85a6526bc914eee4ce90d7446d26befdb47e2b36d3ac024f94a8c0b4"
)
RESULT = (
    "work/ns_collision/results/"
    "pressure_load_realization_cost_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "pressure_load_realization_cost_audit.py",
    "work/ns_collision/tests/"
    "test_pressure_load_realization_cost.py",
    "work/ns_collision/notes/"
    "pressure_load_realization_cost.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_pressure_load_realization_cost_bookmark.py",
    "work/ns_collision/README.md",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _append_once(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-mode",
        choices=("complete", "inherited_cpu_parked_incremental"),
        default="inherited_cpu_parked_incremental",
    )
    parser.add_argument(
        "--resource-mode",
        choices=("daytime_policy", "user_authorized_late"),
        required=True,
    )
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, required=True)
    parser.add_argument("--regression-test-count", type=int, default=0)
    parser.add_argument("--regression-test-seconds", type=float, default=0.0)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    return parser.parse_args()


def _validate_result() -> dict[str, Any]:
    result = _load_json(RESULT)
    checks = result.get("positive_checks")
    flags = result.get("certification_flags")
    _require(
        result.get("kind") == "pressure_load_realization_cost_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "explicit_block_critical_carrier_noncoercivity_and_"
            "quadratic_Fisher_growth_certified"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(flags, dict),
        "realization-cost result is not the expected audit",
    )
    for key in (
        "load_realization_amplitude_scaling_derived",
        "load_realization_spatial_scaling_derived",
        "single_block_L2_minimum_derived",
        "single_block_H1_minimum_derived",
        "high_carrier_fixed_load_L2_bounded",
        "high_carrier_fixed_load_L3_upper_bounded",
        "explicit_block_H1_cost_grows_quadratically",
        "explicit_family_gradient_energy_partition_stencil_silent",
        "explicit_family_vertex_weighted_Fisher_is_one_eighth",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "critical_L3_cost_carrier_coercive",
        "global_H1_least_cost_coercivity_proved",
        "global_critical_L3_least_cost_computed",
        "pressure_L32_remainder_absorbed",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    scaling = result["exact_scaling"]
    block = result["single_block_optimization"]
    limits = result["normalized_coupling_limits"]
    support = result["uniform_lacunary_support"]
    family = result["high_carrier_realization_family"]
    seed81 = result["seed81_cost_benchmark"]
    _require(
        scaling.get("all_checks_pass") is True
        and scaling.get("load_scaling")
        == "b_m[u_(a,m)]=a^3 m b_1[u]",
        "amplitude/partition scaling changed",
    )
    _require(
        block.get("all_checks_pass") is True
        and block.get("H1_product_residual") == "0",
        "single-block optimizer changed",
    )
    _require(
        limits.get("all_checks_pass") is True
        and len(limits.get("rows", [])) == 7
        and limits.get("minimum_limit_normalized_coupling", 0.0) > 0.0,
        "normalized coupling limit certificate changed",
    )
    _require(
        support.get("all_checks_pass") is True
        and support.get("leading_signed_mode_count") == 42
        and support.get("leading_zero_triple_count") == 14
        and support.get("invalid_leading_zero_triple_count") == 0
        and support.get("leading_zero_pair_count") == 21
        and support.get("invalid_leading_zero_pair_count") == 0,
        "uniform lacunary support certificate changed",
    )
    _require(
        family.get("all_checks_pass") is True
        and family.get("bounded_L2_ratio_over_sample", 2.0) < 1.2
        and family.get("bounded_L3_upper_ratio_over_sample", 2.0) < 1.2
        and family.get("H1_over_M2_ratio_over_sample", 2.0) < 1.2
        and all(
            row.get("velocity_mode_count") == 42
            and row.get("maximum_load_residual", 1.0) < 1.0e-10
            and row.get(
                "maximum_relative_divergence_residual",
                1.0,
            )
            < 1.0e-14
            and row.get("quadratic_stencil_silence", {}).get(
                "all_checks_pass"
            )
            is True
            for row in family.get("direct_sparse_rows", [])
        ),
        "high-carrier realization family changed",
    )
    _require(
        seed81.get("all_checks_pass") is True
        and abs(seed81.get("velocity_L2_squared", 0.0) - 100.0)
        < 1.0e-10
        and seed81.get("sampled_velocity_L3_cubed", 0.0) > 1000.0,
        "seed-81 cost benchmark changed",
    )
    return result


def main() -> None:
    args = _parse_args()
    _require(
        args.targeted_test_count == 6,
        "this checkpoint requires exactly six focused tests",
    )
    _require(args.targeted_test_seconds >= 0.0, "invalid test runtime")
    _require(
        args.discovered_test_count == 195,
        "expected 189 inherited tests plus six new tests",
    )
    _require(
        0.0 <= args.baseline_average <= args.baseline_peak,
        "invalid CPU sample",
    )
    _require(
        1 <= args.worker_count <= 2,
        "this checkpoint permits at most two Python workers",
    )
    if args.resource_mode == "daytime_policy":
        _require(
            args.baseline_average <= 60.0,
            "daytime validation requires a baseline at most 60 percent",
        )
    if args.validation_mode == "complete":
        _require(
            args.regression_test_count == args.discovered_test_count,
            "complete mode requires all discovered tests",
        )
        _require(
            args.regression_test_seconds > 0.0,
            "complete mode requires the regression runtime",
        )
    else:
        _require(
            args.regression_test_count == 0,
            "incremental mode cannot claim a completed regression",
        )
        _require(
            args.regression_test_seconds == 0.0,
            "incremental mode cannot record a regression runtime",
        )

    for artifact in ARTIFACTS:
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")
    result = _validate_result()
    _require(
        _sha256(PRIOR_RESULT) == PRIOR_RESULT_SHA256,
        "the prerequisite pressure-load surjectivity result changed",
    )
    prior = _load_json(PRIOR_RESULT)
    _require(
        prior.get("kind") == "fourier_pressure_load_surjectivity_audit"
        and prior.get("all_positive_checks_pass") is True,
        "the prerequisite pressure-load audit is invalid",
    )

    bookmark = _load_json(BOOKMARK)
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    principal = bookmark.setdefault("principal_results", {})
    _require(
        bookmark.get("status") == "parked"
        and principal.get("pressure_load_targeted_test_count") == 5
        and principal.get("pressure_load_discovered_test_count") == 189
        and principal.get("pressure_load_monolithic_regression_passed")
        is False
        and principal.get(
            "fourier_pressure_load_surjectivity_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256,
        "the prerequisite surjectivity checkpoint changed",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The compatible pressure-load realization problem now has exact "
        "amplitude and partition co-scaling and exact single-block L2 and "
        "velocity-H1-seminorm minima. After unit polarization, every one "
        "of the seven geometric couplings has a finite nonzero high-carrier "
        "limit. The uniformly isolated 42-mode family therefore realizes "
        "the fixed Hamming load with bounded L2 and bounded critical-L3 "
        "upper cost as carrier rises, disproving carrier coercivity for "
        "those costs. Its block-optimal velocity Fisher cost grows like "
        "carrier squared. A separate exact 21-pair support certificate "
        "shows every compatible vertex weight sees precisely one eighth "
        "of that gradient cost. These are explicit-family results: global "
        "least H1 cost and a general pressure/Fisher absorption theorem "
        "remain open. "
        + (
            f"The complete {args.regression_test_count}-test regression "
            f"passed in {args.regression_test_seconds:.3f}s."
            if complete
            else (
                "Six focused tests pass. The audit and tests each used one "
                "Python worker under the recorded resource mode, while the "
                "inherited one-shot monolithic regression remains parked."
            )
        )
        + " No critical signed estimate or Navier-Stokes regularity "
        "conclusion is claimed."
    )

    principal.update(
        {
            "pressure_load_cost_coscaling_derived": True,
            "pressure_load_cost_single_block_L2_minimum_derived": True,
            "pressure_load_cost_single_block_H1_minimum_derived": True,
            "pressure_load_cost_normalized_coupling_limits_nonzero": True,
            "pressure_load_cost_uniform_cubic_support_certified": True,
            "pressure_load_cost_uniform_quadratic_silence_certified": True,
            "pressure_load_cost_high_carrier_L2_bounded": True,
            "pressure_load_cost_high_carrier_L3_upper_bounded": True,
            "pressure_load_cost_block_H1_quadratic_growth": True,
            "pressure_load_cost_global_H1_coercivity_proved": False,
            "pressure_load_cost_general_high_carrier_absorption_proved": (
                False
            ),
            "pressure_load_cost_critical_signed_bound_proved": False,
            "pressure_load_cost_Navier_Stokes_regularity_proved": False,
            "pressure_load_cost_targeted_test_count": (
                args.targeted_test_count
            ),
            "pressure_load_cost_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "pressure_load_cost_discovered_test_count": (
                args.discovered_test_count
            ),
            "pressure_load_cost_regression_test_count": (
                args.regression_test_count
            ),
            "pressure_load_cost_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "pressure_load_cost_monolithic_regression_passed": complete,
            "pressure_load_cost_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "pressure_load_cost_cpu_baseline_peak_percent": (
                args.baseline_peak
            ),
            "pressure_load_cost_resource_mode": args.resource_mode,
            "pressure_load_cost_worker_count": args.worker_count,
            "pressure_load_cost_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Derived exact pressure-load realization co-scaling and "
            "single-block L2/Fisher minima; proved an explicit uniformly "
            "isolated high-carrier family has bounded L2 and critical-L3 "
            "upper cost but quadratic block-Fisher growth visible through "
            "every compatible vertex weight, without promoting this to a "
            "global least-cost or regularity theorem."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is a general high-carrier absorption "
        "estimate for arbitrary smooth divergence-free velocity and "
        "compatible nonnegative weights, retaining velocity Fisher and "
        "allowing zero partition faces. It must give an explicit "
        "carrier-to-partition threshold or identify a counterexample. Only "
        "after that theorem may the unresolved search be reduced to a "
        "bounded carrier-ratio band. Critical signed control, low-regularity "
        "passage, and exceptional-set removal remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 195-test "
            "suite must pass when the daytime baseline CPU is at most 60%. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Derive a frequency-separated upper bound for the compatible "
        "high-pressure load in terms of the retained vertex-weighted "
        "velocity Fisher term and controlled low-frequency/coefficient "
        "remainders. Stress the estimate at zero faces and against the "
        "explicit lacunary family before attempting the comparable-carrier "
        "finite problem."
    )

    primary_artifacts = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary_artifacts, artifact)

    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary_artifacts),
                "status": bookmark["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
