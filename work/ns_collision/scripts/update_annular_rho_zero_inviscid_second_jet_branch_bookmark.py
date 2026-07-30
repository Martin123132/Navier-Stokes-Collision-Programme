"""Install the annular inviscid second-jet branch checkpoint."""

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
    "annular_rho_zero_inviscid_second_jet_branch_audit_v1.json"
)
RESULT_SHA256 = (
    "ef89d9b9f39ace8b886a4d40bdd7fed6aa908ffcd2ea2e41ca179a3bb82705c7"
)
CORRECTED_PREDECESSOR_RESULT = (
    "work/ns_collision/results/"
    "annular_rho_zero_second_jet_route_guard_audit_v1.json"
)
CORRECTED_PREDECESSOR_RESULT_SHA256 = (
    "7c985480afc51a084eefa0e2fb614fd3b900e9d2e347a6effb0c99b7259c693d"
)
SUPERSEDED_PREDECESSOR_RESULT_SHA256 = (
    "590490f67dd4d22989a6aae35dd9dcfb118bc521a0cb0b69f49ef8476d8a7cb3"
)
PREDECESSOR_BOOKMARK_SHA256 = (
    "d646489a53c4d16777ff6a8c5eae5106de2da7e9374a493d4351154bf8b917b7"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "89d4f382a4a270b83ed1ed8e3eec901255fea91735b86096a93b038fc303587a"
)
RUNNER_SHA256 = (
    "4fd9cca40a4133bfce8bba21161dd827dc98ccbeee9c13f70f7b3718167a4609"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "annular_rho_zero_second_jet_route_guard_audit.py",
    "work/ns_collision/tests/"
    "test_annular_rho_zero_second_jet_route_guard.py",
    "work/ns_collision/notes/"
    "annular_rho_zero_second_jet_route_guard.md",
    CORRECTED_PREDECESSOR_RESULT,
    "work/ns_collision/scripts/"
    "annular_rho_zero_inviscid_second_jet_branch_audit.py",
    "work/ns_collision/tests/"
    "test_annular_rho_zero_inviscid_second_jet_branch.py",
    "work/ns_collision/notes/"
    "annular_rho_zero_inviscid_second_jet_branch.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_annular_rho_zero_inviscid_second_jet_branch_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, default=20)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=436)
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
        == "annular_inviscid_second_jet_pressure_branches_isolated"
        and result.get("all_positive_checks_pass") is True,
        "annular inviscid second-jet branch result did not pass",
    )
    corrected = _load_json(CORRECTED_PREDECESSOR_RESULT)
    _require(
        _sha256(CORRECTED_PREDECESSOR_RESULT)
        == CORRECTED_PREDECESSOR_RESULT_SHA256,
        "corrected second-jet route-guard hash changed",
    )
    corrected_guard = corrected["second_jet_power_route_guard"]
    nonlinear_route = next(
        row
        for row in corrected_guard["rows"]
        if row["channel_group"] == "pure_nonlinear_velocity_pressure"
    )
    _require(
        corrected.get("all_positive_checks_pass") is True
        and nonlinear_route["route_power"] == 9
        and corrected_guard["all_channels_above_N7_excluded"] is False
        and corrected["certification_flags"][
            "all_second_jet_channels_above_N7_excluded"
        ]
        is False,
        "corrected predecessor route ledger did not match",
    )

    flags = result["certification_flags"]
    _require(
        flags["combined_inviscid_pressure_identity_proved"] is True
        and flags["low_shear_stationarity_proved"] is True
        and flags["amplitude_polynomial_reduced_to_a1_a3"] is True
        and flags["four_high_one_low_coefficient_isolated"] is True
        and flags["two_high_three_low_coefficient_isolated"] is True
        and flags["four_high_N9_limit_certified"] is False
        and flags["two_high_N7_limit_certified"] is False
        and flags["full_inviscid_pressure_N9_limit_certified"] is False
        and flags["full_second_jet_N7_coefficient_certified"] is False
        and flags["uniform_second_jet_Taylor_bound_proved"] is False
        and flags["finite_parabolic_window_controlled"] is False
        and flags["critical_L3_controlled"] is False
        and flags["finite_time_blowup_proved"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "annular inviscid branch certification scope changed",
    )

    symbolic = result["symbolic_compact_identity_certificate"]
    support = result["branch_support_certificate"]
    padding = result["padding_replay"]
    replays = result["predecessor_twenty_channel_replays"]
    rows = result["carrier_rows"]
    finite = result["finite_pressure_output_diagnostics"]
    route = result["route_decision"]
    largest = rows[-1]
    _require(
        symbolic["all_checks_pass"] is True
        and symbolic["chain_rule_residual"] == "0"
        and support["all_checks_pass"] is True
        and support["implemented_joint_factor"] == 8
        and padding["all_checks_pass"] is True
        and padding["maximum_residual"] < 1.0e-12
        and [row["size"] for row in replays] == [3, 5]
        and all(row["all_checks_pass"] for row in replays)
        and max(row["absolute_residual"] for row in replays) < 1.0e-12
        and [row["size"] for row in rows]
        == [5, 7, 9, 13, 17, 21, 25, 29]
        and all(row["all_checks_pass"] for row in rows)
        and all(row["a1_coefficient"] < 0.0 for row in rows)
        and all(row["a3_coefficient"] > 0.0 for row in rows)
        and finite["all_checks_pass"] is True
        and finite["bounded_output_definition"] == "|q|<4"
        and finite["largest_bounded_fraction_of_dominant"] > 0.999
        and abs(finite["largest_outside_bounded_output"]) < 0.1
        and largest["a1_over_N7"] < 0.0
        and route["all_checks_pass"] is True
        and route["four_high_N9_limit_certified"] is False
        and route["full_inviscid_pressure_N9_limit_certified"] is False
        and route["large_full_second_jet_FFT_authorized"] is False,
        "annular inviscid branch replay changed",
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
        and len(bookmark.get("completed_obligations", [])) == 161
        and len(bookmark.get("primary_artifacts", [])) == 579
        and principal.get(
            "annular_rho_zero_second_jet_route_guard_audit_v1_sha256"
        )
        == SUPERSEDED_PREDECESSOR_RESULT_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 162
        and len(bookmark.get("primary_artifacts", [])) == 584
        and principal.get(
            "annular_rho_zero_inviscid_second_jet_branch_audit_v1_sha256"
        )
        == RESULT_SHA256
        and principal.get(
            "annular_rho_zero_second_jet_route_guard_audit_v1_sha256"
        )
        == CORRECTED_PREDECESSOR_RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed inviscid checkpoint matches",
    )

    regression_seconds = float(regression["duration_seconds"])
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The coupled Euler/transport pressure part of the exact rho=0 "
        "second generator jet now has the compact four-form identity and "
        "the exact amplitude reduction t[c_1,N a+c_3,N a^3]. Eightfold "
        "branch padding replays tenfold padding below 2e-16, and the "
        "projected polynomial replays the stored 20-channel N=3 and N=5 "
        "rows below 6e-16. Rows through N=29 have c_1,N<0 and c_3,N>0. "
        "The four-high coefficient is permitted a fixed-output N7 scale, "
        "creating a candidate optimized N9 route that corrects the prior "
        "N7-only triage; no N9 limit is claimed. At N=29, finitely many "
        "outputs |q|<4 contribute -7659.153409 while all larger outputs "
        "contribute only 0.035795 to the two dominant forms. The remaining "
        "task is a combined fixed-q Riemann-sum certificate. All 20 "
        "focused tests and all 436 corpus tests pass."
    )
    principal.update(
        {
            "annular_second_jet_corrected_nonlinear_route_power": 9,
            "annular_second_jet_all_channels_above_N7_excluded": False,
            "annular_rho_zero_second_jet_route_guard_audit_v1_sha256": (
                CORRECTED_PREDECESSOR_RESULT_SHA256
            ),
            "annular_inviscid_second_jet_compact_identity_proved": True,
            "annular_inviscid_second_jet_amplitude_reduction": (
                "t[c_1,N a+c_3,N a^3]"
            ),
            "annular_inviscid_second_jet_branch_dealias_factor": 8,
            "annular_inviscid_second_jet_padding_maximum_residual": padding[
                "maximum_residual"
            ],
            "annular_inviscid_second_jet_replay_maximum_residual": max(
                row["absolute_residual"] for row in replays
            ),
            "annular_inviscid_second_jet_finite_sizes": [
                row["size"] for row in rows
            ],
            "annular_inviscid_second_jet_largest_c1": largest[
                "a1_coefficient"
            ],
            "annular_inviscid_second_jet_largest_c1_over_N7": largest[
                "a1_over_N7"
            ],
            "annular_inviscid_second_jet_largest_c3": largest[
                "a3_coefficient"
            ],
            "annular_inviscid_second_jet_bounded_output_definition": (
                "|q|<4"
            ),
            "annular_inviscid_second_jet_largest_bounded_output_sum": (
                finite["largest_bounded_output_sum"]
            ),
            "annular_inviscid_second_jet_largest_outside_bounded_output": (
                finite["largest_outside_bounded_output"]
            ),
            "annular_inviscid_second_jet_candidate_optimized_route": "N9",
            "annular_inviscid_second_jet_N9_limit_certified": False,
            "annular_inviscid_second_jet_uniform_Taylor_bound_proved": False,
            "annular_inviscid_second_jet_finite_window_controlled": False,
            "annular_inviscid_second_jet_critical_L3_controlled": False,
            "annular_inviscid_second_jet_Navier_Stokes_regularity_proved": (
                False
            ),
            "annular_inviscid_second_jet_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "annular_inviscid_second_jet_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "annular_inviscid_second_jet_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_inviscid_second_jet_full_pytest_passed": True,
            "annular_inviscid_second_jet_full_pytest_runtime_seconds": (
                regression_seconds
            ),
            "annular_inviscid_second_jet_resource_mode": (
                arguments.resource_mode
            ),
            "annular_inviscid_second_jet_worker_count": (
                arguments.worker_count
            ),
            "annular_inviscid_second_jet_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "annular_inviscid_second_jet_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "annular_inviscid_second_jet_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "annular_inviscid_second_jet_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "annular_inviscid_second_jet_result_status": result["status"],
            "annular_rho_zero_inviscid_second_jet_branch_audit_v1_sha256": (
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
            "Reduced the coupled inviscid pressure second jet exactly to "
            "four-high/one-low and two-high/three-low amplitude branches, "
            "corrected the nonlinear route guard to a possible optimized "
            "N9 scale, and localized the observed leading coefficient to "
            "the finite pressure-output set |q|<4."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The inviscid pressure second jet is exactly reduced to "
        "t[c_1,N a+c_3,N a^3], and its candidate leading four-high signal "
        "is localized to finitely many outputs |q|<4. It remains to derive "
        "the combined fixed-q Riemann-sum limits, prove a quantitative "
        "remainder, and decide whether c_1,N/N7 has a nonzero signed limit. "
        "No optimized N9 law, full second-jet leading coefficient, uniform "
        "Taylor window, critical L3 estimate, exceptional-set removal, or "
        "global regularity theorem is yet proved."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_rho_zero_inviscid_second_jet_branch_audit.py"
    )
    bookmark["next_action"] = (
        "For each bounded pressure output q with |q|<4 in the two dominant "
        "forms -6S(BHH,BHH,U;Phi) and "
        "-12S(B(H,BHH),H,U;Phi), eliminate the resonance constraints and "
        "write the normalized contribution as a fixed-domain Riemann sum. "
        "Combine all q before testing the sign because individual mode "
        "contributions cancel strongly. Prove convergence and an explicit "
        "remainder for c_1,N/N7; do not infer a nonzero limit from the "
        "finite N=29 row and do not launch a full second-jet grid."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 162, "unexpected completed count")
    _require(len(primary) == 584, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary),
                "corrected_predecessor_sha256": _sha256(
                    CORRECTED_PREDECESSOR_RESULT
                ),
                "result_sha256": _sha256(RESULT),
                "full_regression_sha256": _sha256(FULL_REGRESSION),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
