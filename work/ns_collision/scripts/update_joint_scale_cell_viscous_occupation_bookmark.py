"""Install the joint scale-cell viscous occupation checkpoint."""

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
    "dyadic_three_shell_atlas_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "52f79c57cd3bb99d8b6048f8797ab75f4cc1640436bb8ed1cf3bdac5cfaef513"
)
RESULT = (
    "work/ns_collision/results/"
    "joint_scale_cell_viscous_occupation_audit_v1.json"
)
RESULT_SHA256 = (
    "ab47bc3bbf35a7296471cae8ec1514e475ad42c38b0d547991d76a3047873e0d"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "joint_scale_cell_viscous_occupation_audit.py",
    "work/ns_collision/tests/"
    "test_joint_scale_cell_viscous_occupation.py",
    "work/ns_collision/notes/"
    "joint_scale_cell_viscous_occupation.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_joint_scale_cell_viscous_occupation_bookmark.py",
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
    parser.add_argument("--baseline-average", type=float)
    parser.add_argument("--baseline-peak", type=float)
    return parser.parse_args()


def _validate_result() -> dict[str, Any]:
    result = _load_json(RESULT)
    checks = result.get("positive_checks")
    flags = result.get("certification_flags")
    _require(
        _sha256(RESULT) == RESULT_SHA256,
        "joint scale-cell result hash changed",
    )
    _require(
        result.get("kind")
        == "joint_scale_cell_viscous_occupation_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == "pointwise_joint_channel_no_go_Stokes_occupation_certified"
        and result.get("all_positive_checks_pass") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(flags, dict),
        "joint scale-cell result is not the expected audit",
    )
    for key in (
        "baseline_cumulative_stress_ell1_bound_proved",
        "common_low_Fourier_top_Walsh_channel_exhibited",
        "pointwise_high_shell_ell2_orthogonality_gain_falsified",
        "linear_Stokes_HHL_viscous_occupation_bound_proved",
        "conditional_forced_relaxation_bound_proved",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "all_pointwise_Carleson_estimates_falsified",
        "Navier_Stokes_nonlinear_regeneration_bound_proved",
        "Navier_Stokes_time_integrated_compensation_proved",
        "critical_signed_large_data_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    baseline = result["baseline_cumulative_stress_bound"]
    _require(
        baseline.get("all_checks_pass") is True
        and "L^(3/2)" in baseline.get("estimate", "")
        and "ell1" in baseline.get("scope", ""),
        "baseline cumulative stress theorem changed",
    )

    channel = result["pointwise_Fourier_Walsh_channel"]
    rows = channel.get("prefix_rows", [])
    individual = channel.get("individual_channels", [])
    _require(
        channel.get("all_checks_pass") is True
        and channel.get("common_low_Fourier_mode") == [1, 1, 0]
        and channel.get("common_cell_Walsh_mask") == 7
        and len(rows) == 8
        and len(individual) == 8
        and all(row.get("all_checks_pass") is True for row in rows)
        and all(row.get("all_checks_pass") is True for row in individual)
        and float(rows[-1]["stress_sum_over_square_function"]) > 2.0
        and float(rows[-1]["flux_sum_over_square_function"]) > 2.0
        and max(
            float(row["stress_cross_shell_residual"]) for row in rows
        )
        < 1.0e-12
        and max(
            float(row["flux_cross_shell_residual"]) for row in rows
        )
        < 1.0e-12,
        "pointwise common-channel certificate changed",
    )

    occupation = result["viscous_occupation_bound"]
    _require(
        occupation.get("all_checks_pass") is True
        and float(occupation["beta_in_mu_squared"]) == 5.5
        and float(occupation["minimum_lacunarity_ratio"]) > 1.9
        and float(
            occupation["maximum_exact_Stokes_damping_residual"]
        )
        < 1.0e-12
        and float(occupation["exact_L2_time_norm_squared"])
        <= float(occupation["Schur_upper"])
        and float(occupation["half_peak_occupation"])
        <= float(occupation["Chebyshev_occupation_upper"]),
        "viscous occupation certificate changed",
    )

    forced = result["forced_relaxation_bound"]
    _require(
        forced.get("all_checks_pass") is True
        and float(forced["exact_replay_response_L2_norm"])
        <= float(forced["replay_upper"])
        and "No bound" in forced.get("Navier_Stokes_gap", ""),
        "forced relaxation certificate changed",
    )
    return result


def main() -> None:
    args = _parse_args()
    _require(
        args.targeted_test_count == 6,
        "this checkpoint requires exactly six focused tests",
    )
    _require(args.targeted_test_seconds > 0.0, "invalid test runtime")
    _require(
        args.discovered_test_count == 237,
        "expected 231 inherited tests plus six new tests",
    )
    _require(
        1 <= args.worker_count <= 2,
        "this checkpoint permits at most two Python workers",
    )
    if args.resource_mode == "daytime_policy":
        _require(
            args.baseline_average is not None
            and args.baseline_peak is not None
            and 0.0
            <= args.baseline_average
            <= args.baseline_peak
            and args.baseline_average <= 60.0,
            "daytime validation requires a permitted CPU baseline",
        )
    elif (
        args.baseline_average is not None
        or args.baseline_peak is not None
    ):
        _require(
            args.baseline_average is not None
            and args.baseline_peak is not None
            and 0.0
            <= args.baseline_average
            <= args.baseline_peak,
            "partial or invalid optional CPU sample",
        )

    if args.validation_mode == "complete":
        _require(
            args.regression_test_count == args.discovered_test_count
            and args.regression_test_seconds > 0.0,
            "complete mode requires the full regression",
        )
    else:
        _require(
            args.regression_test_count == 0
            and args.regression_test_seconds == 0.0,
            "incremental mode cannot claim a full regression",
        )

    for artifact in ARTIFACTS:
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")
    result = _validate_result()
    _require(
        _sha256(PRIOR_RESULT) == PRIOR_RESULT_SHA256,
        "the prerequisite dyadic-atlas result changed",
    )
    prior = _load_json(PRIOR_RESULT)
    _require(
        prior.get("kind") == "dyadic_three_shell_atlas_audit"
        and prior.get("all_positive_checks_pass") is True,
        "the prerequisite dyadic-atlas audit is invalid",
    )

    bookmark = _load_json(BOOKMARK)
    _require(
        bookmark.get("kind")
        == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    principal = bookmark.setdefault("principal_results", {})
    _require(
        bookmark.get("status") == "parked"
        and principal.get("dyadic_atlas_targeted_test_count") == 6
        and principal.get("dyadic_atlas_discovered_test_count") == 231
        and principal.get("dyadic_atlas_monolithic_regression_passed")
        is False
        and principal.get(
            "dyadic_three_shell_atlas_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256,
        "the prerequisite dyadic-atlas checkpoint changed",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The cumulative low-output Reynolds stress obeys the baseline "
        "L1-to-L2 bound C L^(3/2) sum_H ||u_H||_2^2. The dyadic "
        "sideband family shows why pointwise shell orthogonality cannot "
        "improve this to an ell2 norm of shell stresses: every carrier "
        "lands in the same q=(1,1,0), top-Walsh channel, and the stress "
        "and complete HHL flux sum-over-square-function ratios grow as "
        "sqrt(N). Retaining time produces a positive theorem. Under full "
        "Stokes evolution b_H(t)=b_H(0)exp[-nu(2H^2+11)t], and a "
        "Gram-Schur argument bounds the L2-time norm by a weighted ell2 "
        "shell sum, yielding finite threshold occupation. A conditional "
        "forced-relaxation estimate is also proved. Control of the actual "
        "Navier-Stokes nonlinear regeneration, critical signed closure, "
        "and regularity remain open. Six focused tests pass with one "
        "Python worker."
    )

    principal.update(
        {
            "joint_scale_cell_baseline_stress_ell1_proved": True,
            "joint_scale_cell_common_Fourier_Walsh_channel_proved": True,
            "joint_scale_cell_pointwise_ell2_gain_falsified": True,
            "joint_scale_cell_all_pointwise_Carleson_falsified": False,
            "joint_scale_cell_linear_Stokes_occupation_proved": True,
            "joint_scale_cell_conditional_Duhamel_bound_proved": True,
            "joint_scale_cell_NS_regeneration_bound_proved": False,
            "joint_scale_cell_NS_time_compensation_proved": False,
            "joint_scale_cell_critical_signed_bound_proved": False,
            "joint_scale_cell_Navier_Stokes_regularity_proved": False,
            "joint_scale_cell_targeted_test_count": (
                args.targeted_test_count
            ),
            "joint_scale_cell_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "joint_scale_cell_discovered_test_count": (
                args.discovered_test_count
            ),
            "joint_scale_cell_regression_test_count": (
                args.regression_test_count
            ),
            "joint_scale_cell_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "joint_scale_cell_monolithic_regression_passed": complete,
            "joint_scale_cell_resource_mode": args.resource_mode,
            "joint_scale_cell_worker_count": args.worker_count,
            "joint_scale_cell_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "joint_scale_cell_cpu_baseline_peak_percent": (
                args.baseline_peak
            ),
            "joint_scale_cell_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Defined the cumulative low-output Reynolds stress, falsified "
            "pointwise high-shell ell2 orthogonality in its exact common "
            "Fourier-Walsh channel, and proved the surviving Stokes "
            "viscous-occupation and conditional Duhamel bounds."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is the nonlinear regeneration gate. "
        "Derive the exact shellwise evolution of the low-output "
        "Fourier-Walsh Reynolds-stress channel for smooth Navier-Stokes, "
        "decompose its Duhamel forcing into HHH, HHL, and transport "
        "pieces, and prove or falsify the weighted ell2_H L2_t forcing "
        "bound required by the viscous occupation theorem. Critical "
        "signed closure, low-regularity passage, and exceptional-set "
        "removal remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 237-test "
            "suite must pass in an admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Write the projected evolution identity "
        "dot c_LH+2nu mu_H^2 c_LH=f_LH before estimating it. Partition "
        "f_LH by the exact HHH/HHL atlas and test whether signed shell "
        "transfer controls its weighted ell2_H L2_t norm. Replay the "
        "first candidate on externally forced coherent sidebands; if it "
        "fails, identify the missing time, scale, or cell cancellation "
        "rather than returning to pointwise high-shell orthogonality."
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
