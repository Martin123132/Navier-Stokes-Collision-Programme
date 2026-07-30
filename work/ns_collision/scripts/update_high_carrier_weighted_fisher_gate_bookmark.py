"""Install the high-carrier weighted-Fisher gate checkpoint."""

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
    "pressure_load_realization_cost_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "e0f0b99b30812a54e0c7090adbe357f7f42c2ef9bb9416829017c334896859be"
)
RESULT = (
    "work/ns_collision/results/"
    "high_carrier_weighted_fisher_gate_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "high_carrier_weighted_fisher_gate_audit.py",
    "work/ns_collision/tests/"
    "test_high_carrier_weighted_fisher_gate.py",
    "work/ns_collision/notes/"
    "high_carrier_weighted_fisher_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_high_carrier_weighted_fisher_gate_bookmark.py",
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


def _strictly_monotone(
    rows: list[dict[str, Any]],
    field: str,
    increasing: bool,
) -> bool:
    values = [float(row[field]) for row in rows]
    comparisons = zip(values, values[1:])
    if increasing:
        return all(first < second for first, second in comparisons)
    return all(first > second for first, second in comparisons)


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
        result.get("kind")
        == "high_carrier_weighted_fisher_gate_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "highpass_linear_unweighted_and_square_factor_"
            "bridges_certified"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(flags, dict),
        "high-carrier gate result is not the expected audit",
    )
    for key in (
        "pure_highpass_unweighted_H1_least_cost_linear_coercivity_proved",
        "vertex_square_factor_highpass_mass_bridge_proved",
        "vertex_zero_face_gradient_controlled_by_weighted_Fisher",
        "zero_face_uncertainty_mechanism_exactly_realized",
        "PDE_pressure_packet_mechanism_numerically_realized",
        "intrinsic_ratio_A_over_nuK_survives_scaling",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "global_quadratic_carrier_coercivity_proved",
        "weighted_Fisher_carrier_coercivity_from_support_alone",
        "PDE_pressure_packet_asymptotic_counterexample_proved",
        "general_intrinsic_high_carrier_absorption_proved",
        "mixed_low_high_velocity_remainder_controlled",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    theorem = result["unweighted_highpass_coercivity"]
    bridge = result["square_factor_highpass_bridge"]
    uncertainty = result["zero_face_uncertainty_packet"]
    scaling = result["zero_face_concentration_scaling"]
    pilot = result["PDE_zero_face_packet_pilot"]
    _require(
        theorem.get("all_checks_pass") is True
        and theorem.get("carrier_exponent_for_fixed_load") == 1
        and "K**(3/2)" in theorem.get("load_upper_bound", ""),
        "unweighted high-pass theorem changed",
    )
    _require(
        bridge.get("all_checks_pass") is True
        and bridge.get("validity_threshold") == "K>sqrt(3)m"
        and bridge.get("weighted_mass_upper_bound")
        == "E_v/(K*(K - sqrt(3)*m))"
        and bridge.get("gradient_factor_at_K_equals_2sqrt3m")
        == "1 + 3*sqrt(2)/4",
        "square-factor high-pass bridge changed",
    )
    uncertainty_rows = uncertainty.get("rows", [])
    _require(
        uncertainty.get("all_checks_pass") is True
        and len(uncertainty_rows) == 6
        and uncertainty.get("maximum_weighted_Dirichlet", 100.0) < 6.0
        and uncertainty.get("ratio_drop", 0.0) > 800.0
        and all(row.get("all_checks_pass") is True for row in uncertainty_rows),
        "zero-face uncertainty certificate changed",
    )
    _require(
        scaling.get("all_checks_pass") is True
        and [
            row.get("fixed_load_weighted_delta_exponent")
            for row in scaling.get("rows", [])
        ]
        == ["-1/3", "0", "1/3"],
        "zero-face scaling certificate changed",
    )
    pilot_rows = pilot.get("rows", [])
    _require(
        pilot.get("all_checks_pass") is True
        and [row.get("order") for row in pilot_rows] == [2, 3, 4, 5]
        and all(
            row.get("all_checks_pass") is True
            and row.get("minimum_mode_over_order", 0.0) >= 3.0
            and row.get("maximum_relative_divergence_residual", 1.0)
            < 1.0e-10
            and abs(abs(row.get("normalized_pressure_load", 0.0)) - 1.0)
            < 1.0e-10
            for row in pilot_rows
        )
        and _strictly_monotone(
            pilot_rows,
            "normalized_weighted_Fisher",
            increasing=False,
        )
        and _strictly_monotone(
            pilot_rows,
            "normalized_unweighted_Fisher",
            increasing=True,
        )
        and _strictly_monotone(
            pilot_rows,
            "intrinsic_Reynolds_proxy",
            increasing=True,
        ),
        "pressure-active finite-Fourier pilot changed",
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
        args.discovered_test_count == 201,
        "expected 195 inherited tests plus six new tests",
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
        "the prerequisite realization-cost result changed",
    )
    prior = _load_json(PRIOR_RESULT)
    _require(
        prior.get("kind") == "pressure_load_realization_cost_audit"
        and prior.get("all_positive_checks_pass") is True,
        "the prerequisite realization-cost audit is invalid",
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
        and principal.get("pressure_load_cost_targeted_test_count") == 6
        and principal.get("pressure_load_cost_discovered_test_count") == 195
        and principal.get("pressure_load_cost_monolithic_regression_passed")
        is False
        and principal.get(
            "pressure_load_realization_cost_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256,
        "the prerequisite realization-cost checkpoint changed",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "For every smooth pure high-pass divergence-free velocity with "
        "carrier at least K, the compatible pressure load obeys a rigorous "
        "K^(-3/2) trilinear upper bound, so fixed nonzero load requires "
        "unweighted enstrophy at least linear in K. This does not promote "
        "the explicit block family's quadratic power globally. Factoring "
        "each vertex basis as Phi_v=psi_v^2 gives an exact ground-state "
        "identity and, for K>sqrt(3)m, controls both psi_v u and "
        "u grad psi_v by vertex-weighted velocity Fisher. An exact "
        "sine-window packet at the vertex zero face has unit L2 and "
        "unweighted Dirichlet growth of order K^2 while its vertex-weighted "
        "Dirichlet cost stays uniformly bounded, disproving support-only "
        "weighted coercivity. A four-row alias-free divergence-free "
        "finite-Fourier pilot with its induced Poisson pressure realizes "
        "a nonzero vertex load and shows decreasing weighted versus "
        "increasing unweighted Fisher after load normalization. That pilot "
        "is numerical, not an asymptotic theorem. Exact concentration "
        "scaling preserves A/(nu K), leaving amplitude-relative intrinsic "
        "absorption open. Six focused tests pass with complete result "
        "replay equality. One Python worker was used throughout under the "
        "recorded resource mode. No regularity conclusion is claimed."
    )

    principal.update(
        {
            "high_carrier_unweighted_linear_coercivity_proved": True,
            "high_carrier_square_factor_mass_bridge_proved": True,
            "high_carrier_zero_face_gradient_bridge_proved": True,
            "high_carrier_global_quadratic_coercivity_proved": False,
            "high_carrier_weighted_support_only_coercivity_falsified": True,
            "high_carrier_zero_face_uncertainty_exact": True,
            "high_carrier_pressure_packet_pilot_passed": True,
            "high_carrier_pressure_packet_asymptotic_proved": False,
            "high_carrier_intrinsic_ratio_survives": True,
            "high_carrier_intrinsic_absorption_proved": False,
            "high_carrier_mixed_paraproduct_controlled": False,
            "high_carrier_critical_signed_bound_proved": False,
            "high_carrier_Navier_Stokes_regularity_proved": False,
            "high_carrier_targeted_test_count": args.targeted_test_count,
            "high_carrier_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "high_carrier_discovered_test_count": (
                args.discovered_test_count
            ),
            "high_carrier_regression_test_count": args.regression_test_count,
            "high_carrier_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "high_carrier_monolithic_regression_passed": complete,
            "high_carrier_resource_mode": args.resource_mode,
            "high_carrier_worker_count": args.worker_count,
            "high_carrier_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "high_carrier_cpu_baseline_peak_percent": args.baseline_peak,
            "high_carrier_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    legacy_completion = (
        "Proved linear unweighted enstrophy coercivity for fixed "
        "compatible loads of pure high-pass velocities; constructed an "
        "exact vertex-zero-face uncertainty packet falsifying support-only "
        "weighted coercivity; and cross-checked the pressure-active "
        "concentration mechanism in four alias-free finite-Fourier PDE "
        "pilots without promoting their asymptotics."
    )
    if legacy_completion in completed:
        completed.remove(legacy_completion)
    _append_once(
        completed,
        (
            "Proved linear unweighted enstrophy coercivity for fixed "
            "compatible loads of pure high-pass velocities and an exact "
            "square-factor bridge from vertex-weighted Fisher to the "
            "weighted mass and zero-face gradient factor; constructed "
            "an exact vertex-zero-face uncertainty packet falsifying "
            "support-only weighted coercivity; and cross-checked the "
            "pressure-active concentration mechanism in four alias-free "
            "finite-Fourier PDE pilots without promoting their asymptotics."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is the high-output pressure "
        "commutator companion to the certified square-factor bridge. For "
        "T=Q_H R_iR_j, prove or falsify control of psi_v T(u_i u_j) by "
        "||u||_infinity||psi_v u|| plus a K^(-1)||u||_infinity times "
        "||u grad psi_v|| remainder. This would give floor-free intrinsic "
        "absorption when K>=C||u||_infinity/nu. Only after that gate should "
        "mixed low/high pressure paraproducts be added. "
        "Critical signed control, low-regularity passage, and exceptional-"
        "set removal remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 201-test "
            "suite must pass in an admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Write the exact eight-shift Fourier formula for "
        "[psi_v,Q_H R_iR_j](u_i u_j). Factor each multiplier difference "
        "through its half-shift and test whether the resulting terms are "
        "bounded by ||u||_infinity/K times the already certified "
        "u grad psi_v norm. Stress the candidate on the sine and Fejer "
        "packets before treating mixed paraproducts."
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
