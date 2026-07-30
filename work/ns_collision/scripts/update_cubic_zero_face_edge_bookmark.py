"""Install the sharp cubic zero-face pressure-edge checkpoint."""

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
    "intrinsic_pressure_tail_gate_audit_v1.json"
)
RESULT = (
    "work/ns_collision/results/"
    "cubic_zero_face_edge_envelope_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/cubic_zero_face_edge_envelope_audit.py",
    "work/ns_collision/tests/test_cubic_zero_face_edge_envelope.py",
    "work/ns_collision/notes/cubic_zero_face_edge_envelope.md",
    RESULT,
    "work/ns_collision/scripts/update_cubic_zero_face_edge_bookmark.py",
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
    flags = result.get("certification_flags")
    checks = result.get("positive_checks")
    _require(
        result.get("kind") == "cubic_zero_face_edge_envelope_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "zero_face_singularity_removed_"
            "critical_L32_pressure_remainder_open"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict)
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        "cubic zero-face result is not the expected passing audit",
    )
    for key in (
        "sharp_zero_face_edge_supremum_derived",
        "zero_face_reciprocal_singularity_removed",
        "edge_remainder_reduced_to_pressure_L32",
        "cubic_edge_ratio_is_local_Reynolds_power_3_over_2",
        "full_nonnegative_directionwise_edge_supremum_preserved",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "globally_compatible_partition_supremum_evaluated",
        "pressure_L32_remainder_absorbed",
        "signed_inter_edge_Carleson_cancellation_proved",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    sharp = result["sharp_cubic_edge_envelope"]
    pressure = result["conditional_pressure_L32_reduction"]
    scaling = result["scale_homogeneity"]
    stress = result["taylor_green_edge_stress"]
    _require(
        sharp.get("all_checks_pass") is True
        and sharp.get("symbolic_optimum_residual") == "0"
        and sharp.get("sharp_pressure_constant")
        == "8*sqrt(3)/(9*m*sqrt(nu))"
        and sharp.get("maximum_random_residual", 1.0) < 1.0e-10,
        "sharp edge optimization lost an exact or numerical check",
    )
    _require(
        pressure.get("all_checks_pass") is True
        and pressure.get("partition_derivative_cube_mean")
        == "m**3/(6*pi)"
        and pressure.get("intrinsic_m_equals_U_over_nu")
        == "4*sqrt(2)*U**2/(3*sqrt(pi)*nu)",
        "conditional pressure L3/2 reduction changed",
    )
    _require(
        scaling.get("all_checks_pass") is True
        and scaling.get("symbolic_residual") == "0"
        and scaling.get("local_Reynolds_power")
        == "(a/(nu m))^(3/2)",
        "cubic envelope scaling check changed",
    )
    _require(
        stress.get("all_checks_pass") is True
        and stress["x_direction"]["maximum_pointwise_residual"]
        < 2.0e-16
        and stress["y_direction"]["maximum_pointwise_residual"]
        < 2.0e-16
        and stress.get("summed_sharp_envelope", 0.0) > 0.0,
        "Taylor-Green cubic edge stress failed",
    )
    return result


def main() -> None:
    args = _parse_args()
    _require(
        args.targeted_test_count == 5,
        "this checkpoint requires exactly five focused tests",
    )
    _require(args.targeted_test_seconds >= 0.0, "invalid test runtime")
    _require(
        args.discovered_test_count == 179,
        "expected 174 inherited tests plus five new tests",
    )
    _require(
        0.0 <= args.baseline_average <= args.baseline_peak,
        "invalid CPU sample",
    )
    _require(
        args.baseline_average <= 60.0,
        "the focused validation requires a permitted daytime baseline",
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

    prior_result = _load_json(PRIOR_RESULT)
    _require(
        prior_result.get("kind") == "intrinsic_pressure_tail_gate_audit"
        and prior_result.get("all_positive_checks_pass") is True
        and _sha256(PRIOR_RESULT)
        == "70bea91db5ed1bc4d43694706743cf3a7f6ffaf4f4a1a7a0b815c44f08a9cfa3",
        "the prerequisite intrinsic pressure-tail result changed",
    )

    bookmark = _load_json(BOOKMARK)
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark",
        "refusing to update a non-NS bookmark",
    )
    _require(
        bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "bookmark is outside the standalone workspace boundary",
    )
    principal = bookmark.setdefault("principal_results", {})
    _require(
        principal.get("rho_terminal_tax_monolithic_regression_passed")
        is True
        and principal.get("rho_terminal_tax_regression_test_count") == 169
        and principal.get(
            "rho_positive_finite_window_advantage_excluded_in_dual_class"
        )
        is True,
        "the prerequisite terminal-tax checkpoint is absent",
    )
    _require(
        principal.get("intrinsic_tail_targeted_test_count") == 5
        and principal.get("intrinsic_tail_discovered_test_count") == 174
        and principal.get("intrinsic_tail_monolithic_regression_passed")
        is False
        and principal.get("intrinsic_pressure_tail_gate_audit_v1_sha256")
        == _sha256(PRIOR_RESULT),
        "the prerequisite parked intrinsic-tail checkpoint changed",
    )
    _require(
        principal.get(
            "reversible_weighted_hypercircle_componentwise64128_certified"
        )
        is True
        and principal.get(
            "reversible_weighted_hypercircle_full_inertia_certified"
        )
        is False,
        "the independent finite-pencil checkpoint changed unexpectedly",
    )
    if args.validation_mode == "inherited_cpu_parked_incremental":
        _require(
            bookmark.get("status") == "parked"
            and principal.get(
                "intrinsic_tail_cpu_parked_test_runtime_seconds"
            )
            > 0.0
            and principal.get(
                "intrinsic_tail_cpu_first_high_average_percent"
            )
            > 75.0
            and principal.get(
                "intrinsic_tail_cpu_second_high_average_percent"
            )
            > 75.0,
            "the inherited CPU-parked validation state is absent",
        )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The conditional zero-face pressure edge is now optimized on its "
        "actual feasible cone H=A+B>=|A-B|. The sharp supremum of "
        "(A-B)e-(nu m^2/16)(A+B)(A-B)^2 is finite and equals "
        "8|e|^(3/2)/(3sqrt(3)m sqrt(nu)); equality occurs with one face "
        "exactly zero. Conditional Holder reduces the relaxed "
        "directionwise remainder to pressure L^(3/2), and exact scaling "
        "gives the ratio Re_cell^(3/2). Taylor-Green verifies the scalar "
        "optimizer, but the globally compatible eight-cell coefficient "
        "supremum and pressure absorption remain open. "
        + (
            f"The complete {args.regression_test_count}-test regression "
            f"passed in {args.regression_test_seconds:.3f}s."
            if complete
            else (
                "The prior 169-test monolithic regression, five "
                "intrinsic-tail focused tests, and five new cubic-envelope "
                "focused tests pass. The one-shot 179-test regression was "
                "not relaunched after the inherited 174-test attempt "
                f"parked at {principal['intrinsic_tail_cpu_parked_test_runtime_seconds']:.3f}s "
                "under two consecutive daytime CPU threshold breaches."
            )
        )
        + " No critical estimate or Navier-Stokes regularity conclusion "
        "is claimed."
    )

    principal.update(
        {
            "cubic_zero_face_sharp_supremum_derived": True,
            "cubic_zero_face_reciprocal_singularity_removed": True,
            "cubic_zero_face_pressure_L32_reduction_derived": True,
            "cubic_zero_face_local_Reynolds_power": "3/2",
            "cubic_zero_face_directionwise_supremum_preserved": True,
            "cubic_zero_face_Taylor_Green_stress_passed": True,
            "cubic_zero_face_global_partition_supremum_evaluated": False,
            "cubic_zero_face_pressure_L32_remainder_absorbed": False,
            "cubic_zero_face_signed_Carleson_cancellation_proved": False,
            "cubic_zero_face_critical_bound_proved": False,
            "cubic_zero_face_low_regularity_passage_proved": False,
            "cubic_zero_face_exceptional_set_upgrade_proved": False,
            "cubic_zero_face_Navier_Stokes_regularity_proved": False,
            "cubic_zero_face_targeted_test_count": args.targeted_test_count,
            "cubic_zero_face_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "cubic_zero_face_discovered_test_count": (
                args.discovered_test_count
            ),
            "cubic_zero_face_regression_test_count": (
                args.regression_test_count
            ),
            "cubic_zero_face_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "cubic_zero_face_monolithic_regression_passed": complete,
            "cubic_zero_face_incremental_prior_full_test_count": 169,
            "cubic_zero_face_incremental_prior_focused_test_count": 5,
            "cubic_zero_face_incremental_total_focused_test_count": 10,
            "cubic_zero_face_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "cubic_zero_face_cpu_baseline_peak_percent": args.baseline_peak,
            "cubic_zero_face_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Removed the apparent zero-face singularity by optimizing the "
            "conditional pressure edge on H>=|D|, derived its sharp cubic "
            "L^(3/2) envelope and Re_cell^(3/2) scaling, and verified the "
            "scalar extremizer on Taylor-Green without promoting the "
            "relaxed directionwise bound to a global partition theorem."
        ),
    )
    theorem_obligation = (
        "The live theorem obligation is the globally compatible "
        "nonnegative coefficient supremum on the complete eight-cell "
        "partition graph. Derive its homogeneous cubic graph reduction while "
        "retaining antisymmetric edge cancellation, compare it with the "
        "directionwise L^(3/2) envelope, impose 2:1 intrinsic-scale "
        "balance, and test whether compatibility supplies an absorbable "
        "gain. Low-regularity and exceptional-set gates remain."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
        bookmark["next_action"] = (
            "Write the eight-cell incidence formulation and solve the "
            "nonnegative cubic graph supremum first on symbolic symmetric "
            "fluxes, then on Taylor-Green and seed-81 pressure edges. Do "
            "not replace the signed graph objective by edgewise absolute "
            "values before testing cancellation."
        )
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation is parked: the one-shot 179-test suite "
            "must pass when the daytime baseline CPU is at most 60%. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
        bookmark["next_action"] = (
            "On a fresh turn, sample total CPU for at least five seconds. "
            "If the daytime average is at most 60%, run the parked "
            "179-test command below normal priority and install complete "
            "validation. Then derive the globally compatible eight-cell "
            "cubic graph supremum without edgewise absolute-value loss."
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
