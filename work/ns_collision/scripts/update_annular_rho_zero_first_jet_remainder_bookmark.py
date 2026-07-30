"""Install the annular rho-zero total first-jet limit checkpoint."""

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
RESULT = (
    "work/ns_collision/results/"
    "annular_rho_zero_first_jet_remainder_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "582a6a4997928b8cd7b67f1d9fd58b5fef6326ee6ffa42bede33a5d9854f36c9"
)
PREDECESSOR_RESULT_SHA256 = (
    "e07d6511f0ca52484065ba58674594bd9b0a828f4b0525e26caa136153ebcdaf"
)
PREDECESSOR_BOOKMARK_SHA256 = (
    "01ddf62d21f222c30b14abf0c0080b6106587672182a6901d6f3afa42929edbf"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "7f59bee80c38ab3da1071728d7ca3fe5f6d108e4af476d5e0663e1abaa4019df"
)
RUNNER_SHA256 = (
    "4fd9cca40a4133bfce8bba21161dd827dc98ccbeee9c13f70f7b3718167a4609"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "annular_rho_zero_first_jet_remainder_gate_audit.py",
    "work/ns_collision/tests/"
    "test_annular_rho_zero_first_jet_remainder_gate.py",
    "work/ns_collision/notes/"
    "annular_rho_zero_first_jet_remainder_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_annular_rho_zero_first_jet_remainder_bookmark.py",
    "work/ns_collision/README.md",
    "work/ns_collision/scripts/run_full_regression_checkpoint.py",
    FULL_REGRESSION,
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
    parser.add_argument("--targeted-test-count", type=int, default=9)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=416)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument("--periodic-average", type=float, required=True)
    parser.add_argument("--periodic-peak", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = _load_json(RESULT)
    _require(_sha256(RESULT) == RESULT_SHA256, "result hash changed")
    _require(
        result.get("status")
        == "annular_rho_zero_total_first_jet_N5_limit_certified"
        and result.get("all_positive_checks_pass") is True,
        "annular first-jet remainder result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["compatible_stencil_orders_proved"] is True
        and flags["odd_high_incidence_channels_excluded"] is True
        and flags["viscous_weighted_Fisher_o_N5_proved"] is True
        and flags["Euler_remainder_o_N5_proved"] is True
        and flags["weight_advection_remainder_o_N5_proved"] is True
        and flags["weight_antidiffusion_remainder_o_N5_proved"] is True
        and flags["total_first_jet_N5_limit_certified"] is True
        and flags["total_first_jet_eventually_negative_proved"] is True
        and flags["required_N2_amplification_excluded"] is False
        and flags["finite_parabolic_window_controlled"] is False
        and flags["second_time_jet_needed"] is True
        and flags["critical_L3_controlled"] is False
        and flags["finite_time_blowup_proved"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "annular total first-jet certification scope changed",
    )

    stencil = result["compatible_stencil_certificate"]
    Fisher = result["viscous_weighted_Fisher_theorem"]
    ledger = result["remainder_bound_ledger"]
    total = result["total_first_jet_limit_certificate"]
    mixed_rows = result["two_high_two_low_finite_rows"]
    _require(
        stencil["all_checks_pass"] is True
        and stencil["tensor_difference_orders"]["Phi"] == 6
        and stencil["tensor_difference_orders"]["gradient_Phi"] == 5
        and stencil["tensor_difference_orders"]["Laplacian_Phi"] == 4
        and Fisher["all_checks_pass"] is True
        and Fisher["maximum_FFT_replay_residual"] < 1.0e-13
        and len(Fisher["finite_rows"]) == 7
        and ledger["all_checks_pass"] is True
        and ledger["maximum_optimized_power_upper_bound"] == 4
        and all(
            branch["optimized_power_upper_bound"] < 5
            for branch in ledger["branch_rows"]
        )
        and len(mixed_rows) == 5
        and all(row["all_checks_pass"] for row in mixed_rows)
        and total["all_checks_pass"] is True
        and total["total_first_jet_over_N5_limit"] < 0.0
        and total["finite_negative_sizes"] == [25, 29, 33, 37, 41],
        "annular total first-jet replay changed",
    )

    regression = _load_json(FULL_REGRESSION)
    _require(
        _sha256(FULL_REGRESSION) == FULL_REGRESSION_SHA256,
        "full regression report hash changed",
    )
    _require(
        regression.get("schema_version") == 2
        and regression.get("configuration", {}).get("test_engine")
        == "pytest"
        and regression.get("discovered_test_count")
        == arguments.discovered_test_count
        and regression.get("tests_run") == arguments.discovered_test_count
        and regression.get("passed_count")
        == arguments.discovered_test_count
        and regression.get("successful") is True
        and regression.get("exit_code") == 0
        and not regression.get("failures")
        and not regression.get("errors"),
        "full pytest regression did not pass",
    )
    _require(
        _sha256(
            "work/ns_collision/scripts/run_full_regression_checkpoint.py"
        )
        == RUNNER_SHA256,
        "full regression runner hash changed",
    )

    bookmark = _load_json(BOOKMARK)
    principal = bookmark.setdefault("principal_results", {})
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    predecessor = bool(
        _sha256(BOOKMARK) == PREDECESSOR_BOOKMARK_SHA256
        and len(bookmark.get("completed_obligations", [])) == 159
        and len(bookmark.get("primary_artifacts", [])) == 569
        and principal.get(
            "annular_rho_zero_first_jet_audit_v1_sha256"
        )
        == PREDECESSOR_RESULT_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 160
        and len(bookmark.get("primary_artifacts", [])) == 574
        and principal.get(
            "annular_rho_zero_first_jet_remainder_gate_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed remainder checkpoint matches",
    )

    regression_seconds = float(regression["duration_seconds"])
    total_limit = float(total["total_first_jet_over_N5_limit"])
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "Every non-pressure component of the exact rho=0 first generator "
        "jet has now been bounded below the N^5 scale for the "
        "static-optimal annular +++ family. Parity-gauged tensor weights "
        "supply exact difference orders six for Phi, five for grad Phi, "
        "and four for Hessian/Laplacian terms; the x-support gap excludes "
        "all odd-high incidences. The viscous weighted-Fisher term has the "
        "exact mixed-difference form 2nu^2 t_N(P_N+a_N^2), with "
        "P_N=O(N^-1), and replays the FFT values within 8.83e-15. A "
        "finite-shell plus dyadic pressure-output argument controls the "
        "projector singularity, while the dangerous two-high/two-low "
        "branch is O(t_N a_N^2 N)=O(N^4). Every ledger branch is at most "
        "O(N^4), so the complete derivative has the certified limit "
        f"{total_limit:.16e}/nu times N^5 and is eventually negative. "
        "This closes positive initial N^5 amplification for this witness "
        "but does not control a later T/N^2 turnaround. All 9 focused "
        "tests and all 416 corpus tests pass."
    )
    principal.update(
        {
            "annular_first_jet_remainder_stencil_orders": {
                "Phi": 6,
                "gradient_Phi": 5,
                "Laplacian_Phi": 4,
            },
            "annular_first_jet_remainder_odd_high_channels_excluded": True,
            "annular_first_jet_remainder_maximum_parity_residual": result[
                "channel_parity_replay"
            ]["maximum_parity_residual"],
            "annular_first_jet_remainder_viscous_Fisher_bound": (
                "O(N^3)"
            ),
            "annular_first_jet_remainder_maximum_Fisher_replay_residual": (
                Fisher["maximum_FFT_replay_residual"]
            ),
            "annular_first_jet_remainder_maximum_branch_power": 4,
            "annular_first_jet_remainder_total_bound": "O(N^4)=o(N^5)",
            "annular_first_jet_total_N5_limit": total_limit,
            "annular_first_jet_total_eventually_negative_proved": True,
            "annular_first_jet_finite_negative_sizes": total[
                "finite_negative_sizes"
            ],
            "annular_first_jet_finite_window_controlled": False,
            "annular_first_jet_required_N2_amplification_excluded": False,
            "annular_first_jet_second_time_jet_needed": True,
            "annular_first_jet_critical_L3_controlled": False,
            "annular_first_jet_Navier_Stokes_regularity_proved": False,
            "annular_first_jet_remainder_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "annular_first_jet_remainder_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "annular_first_jet_remainder_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_first_jet_remainder_full_pytest_passed": True,
            "annular_first_jet_remainder_full_pytest_runtime_seconds": (
                regression_seconds
            ),
            "annular_first_jet_remainder_resource_mode": (
                arguments.resource_mode
            ),
            "annular_first_jet_remainder_worker_count": (
                arguments.worker_count
            ),
            "annular_first_jet_remainder_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "annular_first_jet_remainder_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "annular_first_jet_remainder_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "annular_first_jet_remainder_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "annular_first_jet_remainder_result_status": result["status"],
            "annular_rho_zero_first_jet_remainder_gate_audit_v1_sha256": (
                RESULT_SHA256
            ),
            "full_regression_checkpoint_v1_sha256": (
                FULL_REGRESSION_SHA256
            ),
        }
    )
    for artifact in ARTIFACTS:
        parent = Path(artifact).parent.name.replace("-", "_")
        stem = Path(artifact).stem.replace("-", "_")
        principal[f"{parent}_{stem}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Proved every non-pressure annular rho-zero first-jet branch is "
            "O(N4)=o(N5), including the mixed-difference viscous Fisher and "
            "low-output pressure shells, thereby promoting the negative "
            "viscous-pressure coefficient to the complete total first-jet "
            "N5 limit."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The complete restart-time first derivative of the static-optimal "
        "annular witness is now eventually negative at order N5. It remains "
        "to control the Taylor remainder uniformly on T/N^2 and decide "
        "whether the generator can turn around rapidly enough to overcome "
        "the order-N3 Legendre reset deficit. A second-time-jet calculation "
        "must retain Navier-Stokes acceleration, pressure Hessians, and the "
        "backward-weight second derivative. Optimization over dynamically "
        "relevant terminal weights, critical L3, exceptional-set removal, "
        "and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_rho_zero_first_jet_remainder_gate_audit.py"
    )
    bookmark["next_action"] = (
        "Derive the exact second time derivative of the normalized rho=0 "
        "generator along coupled Navier-Stokes and backward-weight flow. "
        "First produce a symbolic multilinear decomposition and carrier "
        "power ledger; do not launch a full dealiased second-jet grid until "
        "the ledger identifies which channels can survive at order N7. "
        "Seek a uniform Taylor bound on 0<=t<=T/N^2 strong enough to retain "
        "the negative first-jet contribution."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 160, "unexpected completed count")
    _require(len(primary) == 574, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary),
                "result_sha256": _sha256(RESULT),
                "full_regression_sha256": _sha256(FULL_REGRESSION),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
