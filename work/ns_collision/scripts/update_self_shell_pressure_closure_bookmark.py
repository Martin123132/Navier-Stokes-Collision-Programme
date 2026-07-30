"""Install the full self-shell pressure closure checkpoint."""

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
    "annular_vertex_commutator_gate_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "7274c0084146e78de9f6ee97d24edab93f544d50e01bec45e0eab1c8f043ae7a"
)
RESULT = (
    "work/ns_collision/results/"
    "self_shell_pressure_closure_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/self_shell_pressure_closure_audit.py",
    "work/ns_collision/tests/test_self_shell_pressure_closure.py",
    "work/ns_collision/notes/self_shell_pressure_closure.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_self_shell_pressure_closure_bookmark.py",
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
        result.get("kind") == "self_shell_pressure_closure_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == "full_self_shell_pressure_closure_certified"
        and result.get("all_positive_checks_pass") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(flags, dict),
        "self-shell result is not the expected audit",
    )
    for key in (
        "exact_far_low_pressure_load_orthogonality_proved",
        "smooth_split_recovers_actual_uncut_pressure_load",
        "full_self_shell_pressure_load_bound_proved",
        "full_self_shell_intrinsic_absorption_proved",
        "gap_dependent_closure_for_K_gt_sqrt3m_proved",
        "uniform_fixed_split_for_K_ge_2sqrt3m_proved",
        "fixed_half_cutoff_can_fail_below_2sqrt3m",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "K_ge_2sqrt3m_uniform_threshold_proved_sharp",
        "cross_shell_high_high_to_low_controlled",
        "three_shell_paraproduct_summation_proved",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    support = result["support_exclusion"]
    support_rows = support.get("rows", [])
    below_support = support.get("below_threshold_probe", {})
    _require(
        support.get("all_checks_pass") is True
        and len(support_rows) == 4
        and all(
            row.get("all_checks_pass") is True
            and float(row["strict_triangle_margin"]) > 0.0
            and row.get("admissible_support_resonance_count") == 0
            for row in support_rows
        )
        and below_support.get("condition_K_gt_2sqrt3m_holds") is False
        and int(below_support.get("support_resonance_count", 0)) > 0,
        "far-low support certificate changed",
    )

    theorem = result["full_self_shell_theorem"]
    gap_rows = theorem.get("adaptive_gap_replay", [])
    _require(
        theorem.get("all_checks_pass") is True
        and "K>sqrt(3)m" in theorem.get("assumptions", "")
        and "(K-R)/2" in theorem.get("exact_low_output_identity", "")
        and "delta^(-3)" in theorem.get("annular_constant", "")
        and len(gap_rows) == 5
        and all(
            row.get("strict_support_exclusion_holds") is True
            and float(row["cutoff_plus_stencil_over_carrier"]) < 1.0
            for row in gap_rows
        ),
        "gap-dependent self-shell theorem changed",
    )

    stress = result["adversarial_sparse_shell_stress"]
    valid_rows = stress.get("valid_shell_rows", [])
    below_field = stress.get("below_threshold_nonzero_channel", {})
    _require(
        stress.get("all_checks_pass") is True
        and len(valid_rows) == 6
        and all(
            row.get("all_checks_pass") is True
            and float(row["smooth_low_pressure_L2"]) > 1.0e-8
            and float(row["maximum_full_pressure_load"]) > 1.0e-8
            and float(row["maximum_smooth_low_pressure_load"]) == 0.0
            and float(row["maximum_full_minus_smooth_high_load"]) == 0.0
            and row.get("maximum_low_load_resonance_count") == 0
            and float(row["maximum_divergence_residual"]) < 1.0e-12
            for row in valid_rows
        )
        and below_field.get("all_checks_pass") is True
        and below_field.get("condition_K_gt_2sqrt3m_holds") is False
        and float(below_field["maximum_smooth_low_pressure_load"])
        > 1.0e-8
        and int(below_field["maximum_low_load_resonance_count"]) > 0,
        "adversarial sparse-shell certificate changed",
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
        args.discovered_test_count == 219,
        "expected 213 inherited tests plus six new tests",
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
        "the prerequisite annular result changed",
    )
    prior = _load_json(PRIOR_RESULT)
    _require(
        prior.get("kind") == "annular_vertex_commutator_gate_audit"
        and prior.get("all_positive_checks_pass") is True,
        "the prerequisite annular audit is invalid",
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
        and principal.get("annular_vertex_targeted_test_count") == 6
        and principal.get("annular_vertex_discovered_test_count") == 213
        and principal.get("annular_vertex_monolithic_regression_passed")
        is False
        and principal.get(
            "annular_vertex_commutator_gate_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256,
        "the prerequisite annular checkpoint changed",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The actual uncut pressure generated and tested by one annular "
        "shell is now controlled. Fourier support gives exact far-low "
        "orthogonality because grad Phi_v has radius at most sqrt(3)m. "
        "For every K>sqrt(3)m, an adaptive smooth split at half the "
        "spectral gap makes the low load vanish and places the complement "
        "under the sharp residue-chain annular theorem. This yields a "
        "gap-dependent complete self-shell bound and intrinsic absorption "
        "condition. For K>=2sqrt(3)m a fixed K/2 split gives uniform "
        "constants. Six sparse divergence-free adversaries contain "
        "genuine low pressure and nonzero full work but exactly zero "
        "valid low-load triples; a below-threshold probe has a nonzero "
        "fixed-split low channel. Six focused replay tests pass. One "
        "Python worker was used. Cross-shell high-high-to-low pressure, "
        "paraproduct summation, the critical signed bound, and regularity "
        "remain open."
    )

    principal.update(
        {
            "self_shell_far_low_orthogonality_proved": True,
            "self_shell_actual_uncut_pressure_bound_proved": True,
            "self_shell_gap_dependent_K_gt_sqrt3m_proved": True,
            "self_shell_uniform_K_ge_2sqrt3m_proved": True,
            "self_shell_intrinsic_component_absorption_proved": True,
            "self_shell_cross_shell_high_high_low_controlled": False,
            "self_shell_paraproduct_sum_proved": False,
            "self_shell_critical_signed_bound_proved": False,
            "self_shell_Navier_Stokes_regularity_proved": False,
            "self_shell_targeted_test_count": args.targeted_test_count,
            "self_shell_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "self_shell_discovered_test_count": args.discovered_test_count,
            "self_shell_regression_test_count": args.regression_test_count,
            "self_shell_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "self_shell_monolithic_regression_passed": complete,
            "self_shell_resource_mode": args.resource_mode,
            "self_shell_worker_count": args.worker_count,
            "self_shell_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "self_shell_cpu_baseline_peak_percent": args.baseline_peak,
            "self_shell_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Proved exact far-low pressure-load orthogonality and combined "
            "an adaptive spectral-gap split with the annular Hamming "
            "commutator theorem to control the complete uncut pressure "
            "generated and tested by one velocity shell."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is the genuinely cross-shell "
        "high-high-to-low channel. For H>>L>=m, derive the low-frequency "
        "limit of P_L R_iR_j(u_H,i u_H,j) paired with "
        "u_L dot grad Phi_v using slowly modulated divergence-free waves. "
        "Determine whether pressure alone has any H-decay. If it does not, "
        "combine pressure and kinetic transport in the signed local-energy "
        "flux before taking absolute values. Only then attempt the exact "
        "three-shell atlas and dyadic summation. Critical closure, "
        "low-regularity passage, and exceptional-set removal remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 219-test "
            "suite must pass in an admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Build an exact two-scale WKB stress with a slowly varying "
        "divergence-free amplitude and carrier H. Derive the weak limit of "
        "u_H tensor u_H, its low pressure, and its vertex pressure load "
        "against a low velocity. Stress arbitrary realizable Reynolds "
        "tensors; if no H-gain survives, repeat for the combined pressure "
        "plus kinetic local-energy flux."
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
