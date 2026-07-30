"""Install the smooth Galerkin shell-response checkpoint."""

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
    "scalar_local_energy_regeneration_gate_audit_v2.json"
)
PRIOR_RESULT_SHA256 = (
    "316efc095ac8b03cb97c7902f41e1934bf32aa4958d1c123c9eb0edafbdd7755"
)
RESULT = (
    "work/ns_collision/results/"
    "smooth_galerkin_shell_response_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "a226f430ea1518c54780671abd3f17055333770737566e6badf4eb4a8931f0ad"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "smooth_galerkin_shell_response_gate_audit.py",
    "work/ns_collision/tests/"
    "test_smooth_galerkin_shell_response_gate.py",
    "work/ns_collision/notes/"
    "smooth_galerkin_shell_response_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_smooth_galerkin_shell_response_bookmark.py",
    "work/ns_collision/README.md",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_json(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
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
        choices=("complete", "focused_complete_full_regression_parked"),
        default="focused_complete_full_regression_parked",
    )
    parser.add_argument("--targeted-test-count", type=int, default=5)
    parser.add_argument("--dependency-test-count", type=int, default=23)
    parser.add_argument("--dependency-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=266)
    parser.add_argument("--regression-test-count", type=int, default=0)
    parser.add_argument("--regression-test-seconds", type=float, default=0.0)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    return parser.parse_args()


def _validate_result(result: dict[str, Any]) -> None:
    _require(_sha256(RESULT) == RESULT_SHA256, "result hash changed")
    prerequisites = result["prerequisites"]
    _require(
        _sha256(PRIOR_RESULT) == PRIOR_RESULT_SHA256
        and prerequisites["corrected_scalar_regeneration_gate_sha256"]
        == PRIOR_RESULT_SHA256,
        "corrected scalar prerequisite changed",
    )
    _require(
        result.get("all_positive_checks_pass") is True,
        "smooth Galerkin gate did not pass",
    )
    pairwise = result["exact_pairwise_evolution"]
    weighted_hhl = result["heat_weighted_HHL_commutator"]
    forcing = result["complete_weighted_forcing_square"]
    response = result["summed_response_and_initial_stress"]
    _require(
        pairwise["single_shell_scalar_rate_used"] is False
        and pairwise["maximum_relative_residual"] < 1.0e-12
        and weighted_hhl["maximum_bound_ratio"] <= 1.0
        and weighted_hhl["maximum_rate_identity_residual"] < 1.0e-12
        and forcing["derived_combined_constant"] < 104.0
        and forcing["retained_integer_constant"] == 104.0
        and response["all_checks_pass"] is True,
        "a shell-response quantitative gate changed",
    )
    flags = result["certification_flags"]
    required_true = (
        "smooth_Galerkin_exact_pair_rates_retained",
        "heat_weighted_HHL_commutator_proved",
        "smooth_shell_filter_leakage_paid",
        "sharp_Galerkin_cutoff_leakage_paid",
        "complete_HHH_HHL_weighted_forcing_square_proved",
        "initial_high_stress_heat_tail_controlled",
        "finite_low_channel_high_stress_tail_vanishes",
    )
    required_false = (
        "single_artificial_shell_rate_used",
        "scale_uniform_spatial_localization_proved",
        "suitable_weak_solution_passage_proved",
        "Navier_Stokes_global_regularity_proved",
    )
    _require(
        all(flags.get(name) is True for name in required_true)
        and all(flags.get(name) is False for name in required_false),
        "shell-response scope flags changed",
    )


def main() -> None:
    args = _parse_args()
    result = _load_json(RESULT)
    _validate_result(result)
    bookmark = _load_json(BOOKMARK)
    principal = bookmark.setdefault("principal_results", {})
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    predecessor = bool(
        bookmark.get("status") == "checkpointed"
        and len(bookmark.get("completed_obligations", [])) == 144
        and len(bookmark.get("primary_artifacts", [])) == 492
        and principal.get(
            "scalar_local_energy_regeneration_gate_audit_v2_sha256"
        )
        == PRIOR_RESULT_SHA256
        and principal.get("scalar_regeneration_regression_test_count") == 261
        and principal.get("scalar_regeneration_monolithic_regression_passed")
        is True
    )
    installed = bool(
        bookmark.get("status") in ("parked", "checkpointed")
        and len(bookmark.get("completed_obligations", [])) == 145
        and len(bookmark.get("primary_artifacts", [])) == 497
        and principal.get(
            "smooth_galerkin_shell_response_gate_audit_v1_sha256"
        )
        == RESULT_SHA256
        and principal.get("smooth_shell_exact_pair_rates_retained") is True
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed shell-response checkpoint "
        "matches",
    )

    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The complete smooth Galerkin low-output high-shell stress "
        "response is now controlled without replacing pairwise viscous "
        "rates by an artificial shell rate. Exact rate differences and "
        "smooth shell-selector leakage preserve the HHL low factor after "
        "heat weighting. A sharp orthogonal Galerkin top cutoff can remove "
        "one paired term; its worst unpaired O(H) contribution is paid "
        "directly in the same H^(-3) norm. Comparable-shell HHH, paired "
        "HHL, and sharp-cutoff leakage have derived combined constant "
        "102.77678571428571, below the retained 104. The forced tail is "
        "bounded by C E_*sqrt(D)/(nu sqrt(H0)); the initial heat tail is "
        "bounded by C E(0)/(sqrt(nu)H0). Thus every fixed finite family "
        "of low Fourier/tensor channels has a vanishing high-shell stress "
        "tail, uniformly across Galerkin truncations with common Leray "
        "bounds. Scale-uniform spatial localization, suitable-weak "
        "passage, and regularity remain open."
    )
    if complete:
        bookmark["validated_checkpoint"] += (
            f" All {args.regression_test_count} discovered tests pass in "
            f"{args.regression_test_seconds:.3f} seconds."
        )

    principal.update(
        {
            "smooth_shell_exact_pair_rates_retained": True,
            "smooth_shell_artificial_scalar_rate_used": False,
            "smooth_shell_heat_weighted_HHL_commutator_proved": True,
            "smooth_shell_filter_leakage_paid": True,
            "smooth_shell_sharp_Galerkin_cutoff_leakage_paid": True,
            "smooth_shell_complete_HHH_HHL_forcing_square_proved": True,
            "smooth_shell_forcing_square_constant": 104.0,
            "smooth_shell_initial_stress_tail_controlled": True,
            "smooth_shell_fixed_low_channel_tail_vanishes": True,
            "smooth_shell_scale_uniform_localization_proved": False,
            "smooth_shell_suitable_weak_passage_proved": False,
            "smooth_shell_Navier_Stokes_regularity_proved": False,
            "smooth_shell_targeted_test_count": args.targeted_test_count,
            "smooth_shell_dependency_test_count": (
                args.dependency_test_count
            ),
            "smooth_shell_dependency_test_runtime_seconds": (
                args.dependency_test_seconds
            ),
            "smooth_shell_discovered_test_count": (
                args.discovered_test_count
            ),
            "smooth_shell_regression_test_count": (
                args.regression_test_count
            ),
            "smooth_shell_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "smooth_shell_monolithic_regression_passed": complete,
            "smooth_shell_resource_mode": args.resource_mode,
            "smooth_shell_worker_count": args.worker_count,
            "smooth_shell_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "smooth_shell_cpu_baseline_peak_percent": args.baseline_peak,
            "smooth_shell_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Closed the smooth Galerkin shell-response gate: retained "
            "exact pairwise viscous rates, proved the heat-weighted HHL "
            "and filter commutators, paid sharp top-cutoff leakage, "
            "established the complete H^(-3) forcing square, and bounded "
            "both forced and initial high-shell stress tails at fixed "
            "finite low channels."
        ),
    )
    next_obligation = (
        "The next theorem obligation is scale-uniform low-output "
        "localization. Replace the fixed finite Fourier/tensor channel "
        "family by a low-output Littlewood-Paley, partition-space, or "
        "Carleson norm with constants uniform in its scale. Then prove "
        "that estimate is stable under Galerkin limits and controls every "
        "nonlinear defect needed for suitable weak solutions. Suitable-"
        "weak passage, exceptional-set removal, and global regularity "
        "remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = next_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot "
            f"{args.discovered_test_count}-test suite must pass in an "
            "admissible resource window. "
            + next_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Define the low-output stress tail in a scale-uniform "
        "Littlewood-Paley or partition-space Hilbert norm. Test the exact "
        "sideband family against candidate q/cell weights, then prove the "
        "strongest surviving vector-valued version of the smooth "
        "Galerkin response estimate before attempting any weak limit."
    )

    primary_artifacts = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary_artifacts, artifact)
    _require(len(completed) == 145, "unexpected completed count")
    _require(len(primary_artifacts) == 497, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary_artifacts),
                "result_sha256": _sha256(RESULT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
