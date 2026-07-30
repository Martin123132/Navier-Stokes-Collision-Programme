"""Install the annular rho-zero second-jet route-guard checkpoint."""

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
    "annular_rho_zero_second_jet_route_guard_audit_v1.json"
)
RESULT_SHA256 = (
    "590490f67dd4d22989a6aae35dd9dcfb118bc521a0cb0b69f49ef8476d8a7cb3"
)
PREDECESSOR_RESULT_SHA256 = (
    "582a6a4997928b8cd7b67f1d9fd58b5fef6326ee6ffa42bede33a5d9854f36c9"
)
PREDECESSOR_BOOKMARK_SHA256 = (
    "347a48a03166d494ffff019c14c9abed9c9e3527a92b5576b150e585e37592c8"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "d1c37ee3bd7db5d138f156f8071b8bd7ccd15d9a3c28329d0ef400a2e92a9389"
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
    RESULT,
    "work/ns_collision/scripts/"
    "update_annular_rho_zero_second_jet_route_guard_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, default=10)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=426)
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
        == "annular_rho_zero_second_jet_route_guard_certified"
        and result.get("all_positive_checks_pass") is True,
        "annular second-jet route-guard result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["exact_second_variation_formula_proved"] is True
        and flags["Navier_Stokes_acceleration_retained"] is True
        and flags["pressure_Hessian_retained"] is True
        and flags["backward_weight_second_derivative_retained"] is True
        and flags["tenfold_second_jet_dealiasing_validated"] is True
        and flags["pure_heat_pressure_N7_limit_certified"] is True
        and flags["pure_heat_pressure_N7_coefficient_positive"] is True
        and flags["full_second_jet_N7_coefficient_certified"] is False
        and flags["uniform_second_jet_Taylor_bound_proved"] is False
        and flags["required_N2_amplification_excluded"] is False
        and flags["finite_parabolic_window_controlled"] is False
        and flags["critical_L3_controlled"] is False
        and flags["finite_time_blowup_proved"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "annular second-jet certification scope changed",
    )

    symbolic = result["symbolic_second_variation_certificate"]
    support = result["second_jet_support_ledger"]
    asymptotic = result["pure_heat_pressure_asymptotic_certificate"]
    validation = result["small_carrier_validation"]
    padding = result["padding_replay"]
    second_row = result["fixed_amplitude_second_small_carrier_row"]
    heat_rows = result["finite_second_heat_load_rows"]
    guard = result["second_jet_power_route_guard"]
    coefficient = float(
        asymptotic["pure_heat_pressure_second_jet_over_N7_limit"]
    )
    turnaround = float(
        asymptotic["pure_heat_quadratic_slope_turnaround_scale_N2t"]
    )
    _require(
        symbolic["all_checks_pass"] is True
        and support["all_checks_pass"] is True
        and support["implemented_dealias_factor"] == 10
        and validation["all_checks_pass"] is True
        and validation["second_variation"]["decomposition_residual"]
        < 2.0e-10
        and validation["finite_difference_validation"][
            "relative_residual"
        ]
        < 3.0e-10
        and validation["pure_heat_pressure_replay_residual"] < 1.0e-10
        and padding["all_checks_pass"] is True
        and padding["maximum_channel_residual"] < 1.0e-10
        and second_row["all_checks_pass"] is True
        and second_row["coefficient_scale"] > 0.0
        and second_row["pure_heat_pressure_replay_residual"] < 1.0e-10
        and asymptotic["all_checks_pass"] is True
        and coefficient > 0.0
        and 0.07 < turnaround < 0.09
        and [row["size"] for row in heat_rows] == [25, 33, 49, 65]
        and all(row["all_checks_pass"] for row in heat_rows)
        and guard["all_checks_pass"] is True
        and guard["full_N7_coefficient_certified"] is False
        and guard["large_carrier_FFT_authorized"] is False
        and guard["unresolved_possible_N7_channel_groups"]
        == [
            "pure_nonlinear_velocity_pressure",
            "pure_weight_transport_and_mixed_pressure",
        ],
        "annular second-jet replay changed",
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
        and len(bookmark.get("completed_obligations", [])) == 160
        and len(bookmark.get("primary_artifacts", [])) == 574
        and principal.get(
            "annular_rho_zero_first_jet_remainder_gate_audit_v1_sha256"
        )
        == PREDECESSOR_RESULT_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 161
        and len(bookmark.get("primary_artifacts", [])) == 579
        and principal.get(
            "annular_rho_zero_second_jet_route_guard_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed second-jet checkpoint matches",
    )

    regression_seconds = float(regression["duration_seconds"])
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The exact rho=0 second generator jet is now decomposed into 20 "
        "Hessian and acceleration channels, retaining Navier-Stokes "
        "acceleration, both pressure variations, and the backward-weight "
        "second derivative. Tenfold support padding, a twelvefold replay, "
        "and a Richardson second difference agree to relative residual "
        "2.23e-11 or better. The pure velocity-heat pressure channel has "
        "the exact identity -nu^2 a_N t_N B_(2),N and the certified "
        f"positive limit {coefficient:.16e} N^7. Its isolated quadratic "
        f"slope turns near N^2 t={turnaround:.16f}, but this is not a "
        "full-flow prediction. Pure nonlinear velocity-pressure and pure "
        "weight-transport/mixed-pressure groups still require N7 bounds, "
        "so no uniform Taylor window or regularity conclusion is claimed. "
        "All 10 focused tests and all 426 corpus tests pass."
    )
    principal.update(
        {
            "annular_second_jet_exact_formula_proved": True,
            "annular_second_jet_channel_count": 20,
            "annular_second_jet_maximum_support": "5K+O(L)",
            "annular_second_jet_dealias_factor": 10,
            "annular_second_jet_decomposition_residual": validation[
                "second_variation"
            ]["decomposition_residual"],
            "annular_second_jet_finite_difference_relative_residual": (
                validation["finite_difference_validation"][
                    "relative_residual"
                ]
            ),
            "annular_second_jet_padding_maximum_channel_residual": padding[
                "maximum_channel_residual"
            ],
            "annular_second_heat_pressure_load_N5_limit": result[
                "second_heat_pressure_limit_certificate"
            ]["pressure_load_limits"]["second_heat_B2_over_N5"],
            "annular_second_jet_pure_heat_pressure_N7_limit": coefficient,
            "annular_second_jet_pure_heat_quadratic_turnaround_N2t": (
                turnaround
            ),
            "annular_second_jet_finite_heat_sizes": [
                row["size"] for row in heat_rows
            ],
            "annular_second_jet_unresolved_N7_groups": guard[
                "unresolved_possible_N7_channel_groups"
            ],
            "annular_second_jet_full_N7_coefficient_certified": False,
            "annular_second_jet_uniform_Taylor_bound_proved": False,
            "annular_second_jet_finite_window_controlled": False,
            "annular_second_jet_critical_L3_controlled": False,
            "annular_second_jet_Navier_Stokes_regularity_proved": False,
            "annular_second_jet_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "annular_second_jet_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "annular_second_jet_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_second_jet_full_pytest_passed": True,
            "annular_second_jet_full_pytest_runtime_seconds": (
                regression_seconds
            ),
            "annular_second_jet_resource_mode": arguments.resource_mode,
            "annular_second_jet_worker_count": arguments.worker_count,
            "annular_second_jet_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "annular_second_jet_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "annular_second_jet_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "annular_second_jet_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "annular_second_jet_result_status": result["status"],
            "annular_rho_zero_second_jet_route_guard_audit_v1_sha256": (
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
            "Derived and dealiased the complete annular rho-zero second "
            "generator jet and certified the pure double-velocity-heat "
            "pressure channel has a strictly positive N7 limit, while "
            "isolating the two nonlinear channel groups still capable of "
            "altering the full N7 coefficient."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The exact second-time formula and positive pure-heat pressure N7 "
        "coefficient are certified, but the complete second-jet N7 "
        "coefficient is not. The pure nonlinear velocity-pressure group "
        "and the pure weight-transport/mixed-pressure group must be bounded "
        "below N7 or assigned certified N7 limits. Only after that can a "
        "uniform Taylor remainder on 0<=t<=T/N^2 be attempted. Optimization "
        "over dynamically relevant terminal weights, critical L3 control, "
        "exceptional-set removal, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_rho_zero_second_jet_route_guard_audit.py"
    )
    bookmark["next_action"] = (
        "Reduce the two unresolved N7 groups before any production full "
        "second-jet FFT. Start with the pure nonlinear velocity-pressure "
        "combination H_uu[u_E,u_E]+D_u[D E[u_E]]: extract its sparse "
        "carrier identity, separate zero/low pressure outputs from dyadic "
        "shells, and test whether the compatible five- and six-difference "
        "stencils force o(N7). Then perform the same incidence ledger for "
        "the u_E/lambda_A transport group. Keep all conclusions fail-closed "
        "until both groups are resolved."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 161, "unexpected completed count")
    _require(len(primary) == 579, "unexpected artifact count")
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
