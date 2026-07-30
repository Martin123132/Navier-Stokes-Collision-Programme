"""Install the annular rho-zero first-jet checkpoint."""

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
    "annular_rho_zero_first_jet_audit_v1.json"
)
RESULT_SHA256 = (
    "e07d6511f0ca52484065ba58674594bd9b0a828f4b0525e26caa136153ebcdaf"
)
PREDECESSOR_RESULT_SHA256 = (
    "2f32255887eb18ec0aa22dadfacf681b930434e73f0c457041d65a66e8c04e6d"
)
PREDECESSOR_BOOKMARK_SHA256 = (
    "edf2b007c013258a86536ad0b4d6ebcf336684d9b2ecd454db92c0821355e8a6"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "6396c5d1fbefe664de4f5e427b557e37d9bc4996214abcd3272e4c5f070c8649"
)
RUNNER_SHA256 = (
    "4fd9cca40a4133bfce8bba21161dd827dc98ccbeee9c13f70f7b3718167a4609"
)
ARTIFACTS = (
    "work/ns_collision/scripts/annular_rho_zero_first_jet_audit.py",
    "work/ns_collision/tests/test_annular_rho_zero_first_jet.py",
    "work/ns_collision/notes/annular_rho_zero_first_jet.md",
    RESULT,
    "work/ns_collision/scripts/update_annular_rho_zero_first_jet_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, default=8)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=407)
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
        == "annular_rho_zero_viscous_pressure_N5_limit_certified"
        and result.get("all_positive_checks_pass") is True,
        "annular rho-zero first-jet result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["exact_first_variation_formula_proved"] is True
        and flags["rectangular_dealiasing_validated"] is True
        and flags["finite_carrier_first_jet_computed"] is True
        and flags["finite_carrier_first_jet_negative"] is True
        and flags[
            "asymptotic_viscous_pressure_N5_coefficient_certified"
        ]
        is True
        and flags[
            "asymptotic_total_first_jet_N5_coefficient_certified"
        ]
        is False
        and flags["required_N2_amplification_excluded"] is False
        and flags["second_time_jet_needed"] is True
        and flags["critical_L3_controlled"] is False
        and flags["finite_time_blowup_proved"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "annular first-jet certification scope changed",
    )

    rows = result["carrier_rows"]
    scaling = result["scaling_diagnostics"]
    asymptotic = result["asymptotic_viscous_pressure_certificate"]
    _require(
        [row["size"] for row in rows] == [25, 29, 33, 37, 41]
        and all(row["all_checks_pass"] for row in rows)
        and all(row["first_derivative"] < 0.0 for row in rows)
        and max(
            row["viscous_pressure_replay_residual"] for row in rows
        )
        < 3.0e-13
        and scaling["all_heat_weighted_pressure_loads_negative"] is True
        and scaling["largest_carrier_absolute_remainder_fraction"] < 0.02
        and asymptotic["all_checks_pass"] is True
        and asymptotic[
            "viscous_pressure_first_jet_over_N5_limit"
        ]
        < 0.0,
        "annular first-jet finite or asymptotic replay changed",
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
        and len(bookmark.get("completed_obligations", [])) == 158
        and len(bookmark.get("primary_artifacts", [])) == 564
        and principal.get(
            "deficit_retaining_annular_restart_gate_audit_v1_sha256"
        )
        == PREDECESSOR_RESULT_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 159
        and len(bookmark.get("primary_artifacts", [])) == 569
        and principal.get(
            "annular_rho_zero_first_jet_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed first-jet checkpoint matches",
    )

    row41 = rows[-1]
    regression_seconds = float(regression["duration_seconds"])
    maximum_replay_residual = max(
        row["viscous_pressure_replay_residual"] for row in rows
    )
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The exact rho=0 generator has been differentiated at the "
        "static-optimal annular restart. Symbolic differentiation, "
        "central differences, and independent six/eight-padding replays "
        "validate the Euler, velocity-viscosity, weight-advection, and "
        "weight-antidiffusion directions. The complete first derivative is "
        "negative for N=25,29,33,37,41. More strongly, inserting the exact "
        "three-slot heat multiplier into every HHL pressure monomial gives "
        "D_u g_pressure[nu Delta u_N]=nu a_N t_N B_heat,N, which matches "
        "the dealiased FFT derivative within 1.96e-14. A pointwise-sign "
        "Riemann-limit proof certifies B_heat,N/N^3 -> "
        "-0.0174939570... and hence a strictly negative viscous-pressure "
        "N^5 coefficient -1.0442344590e-7/nu. At N=41 all other first-jet "
        "pieces total only 1.80 percent of the pressure term, but their "
        "o(N^5) bounds and the finite-window Taylor remainder remain open. "
        "All 8 focused tests and all 407 corpus tests pass."
    )
    principal.update(
        {
            "annular_first_jet_exact_variation_formula_proved": True,
            "annular_first_jet_rectangular_dealiasing_validated": True,
            "annular_first_jet_finite_sizes": [row["size"] for row in rows],
            "annular_first_jet_all_finite_derivatives_negative": True,
            "annular_first_jet_maximum_pressure_replay_residual": (
                maximum_replay_residual
            ),
            "annular_first_jet_static_pressure_limit": asymptotic[
                "static_pressure_load_limit"
            ],
            "annular_first_jet_heat_weighted_pressure_limit": asymptotic[
                "heat_weighted_pressure_load_limit"
            ],
            "annular_first_jet_viscous_pressure_N5_limit": asymptotic[
                "viscous_pressure_first_jet_over_N5_limit"
            ],
            "annular_first_jet_analytic_negative_upper_bound": asymptotic[
                "analytic_strict_negative_upper_bound"
            ],
            "annular_first_jet_N41_total_over_N5": row41[
                "first_derivative_over_N5"
            ],
            "annular_first_jet_N41_remainder_fraction": scaling[
                "largest_carrier_absolute_remainder_fraction"
            ],
            "annular_first_jet_total_N5_limit_certified": False,
            "annular_first_jet_required_N2_amplification_excluded": False,
            "annular_first_jet_second_time_jet_needed": True,
            "annular_first_jet_critical_L3_controlled": False,
            "annular_first_jet_Navier_Stokes_regularity_proved": False,
            "annular_first_jet_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "annular_first_jet_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "annular_first_jet_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_first_jet_full_pytest_regression_passed": True,
            "annular_first_jet_full_pytest_runtime_seconds": (
                regression_seconds
            ),
            "annular_first_jet_resource_mode": arguments.resource_mode,
            "annular_first_jet_worker_count": arguments.worker_count,
            "annular_first_jet_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "annular_first_jet_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "annular_first_jet_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "annular_first_jet_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "annular_first_jet_result_status": result["status"],
            "annular_rho_zero_first_jet_audit_v1_sha256": RESULT_SHA256,
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
            "Derived and dealiased the exact rho-zero annular generator "
            "first jet, proved the viscous-pressure component has a "
            "strictly negative N5 limit, and verified negative complete "
            "finite derivatives on five carriers while retaining the "
            "uncertified total-remainder and finite-window obligations."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The static-optimal annular witness now has a rigorously negative "
        "viscous-pressure first-jet coefficient at the exact N5 scale "
        "required by the reset deficit. It remains to prove that the "
        "viscous weighted-Fisher, Euler, weight-advection, and "
        "weight-antidiffusion remainders are o(N5), promote the component "
        "limit to the total first jet, and control the second-time Taylor "
        "remainder on T/N^2. Optimization over dynamically relevant "
        "terminal weights, critical L3, exceptional-set removal, and "
        "global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_rho_zero_first_jet_audit.py"
    )
    bookmark["next_action"] = (
        "Prove carrier-uniform bounds for the first-jet remainder. Start "
        "with the viscous weighted-Fisher term by applying the exact "
        "mixed-difference representation to |k|^2 F_N, then isolate the "
        "two-high/two-low Euler and weight-advection incidences and derive "
        "their continuum leading matrices. If every remainder is o(N5), "
        "install the total negative first-jet limit before computing a "
        "second jet or finite-window Taylor bound."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 159, "unexpected completed count")
    _require(len(primary) == 569, "unexpected artifact count")
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
