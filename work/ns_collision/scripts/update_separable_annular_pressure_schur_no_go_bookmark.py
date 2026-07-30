"""Install the separable annular pressure-Schur no-go checkpoint."""

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
    "separable_annular_pressure_schur_no_go_audit_v1.json"
)
RESULT_SHA256 = (
    "16579e713c5bacb7b19bb9e3d63f059b9f0915588013e40aa49fdb8bf0bfea0b"
)
PREDECESSOR_SHA256 = (
    "216e41e650e2421c4ef4a2c0100a656618f169a8d9dd758ae5a507a7e23837df"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "274d9a716841cf0c568c8e7e6d201bdb740fb647bb514e77f63d03da5dd783a1"
)
RUNNER_SHA256 = (
    "4fd9cca40a4133bfce8bba21161dd827dc98ccbeee9c13f70f7b3718167a4609"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "separable_annular_pressure_schur_no_go_audit.py",
    "work/ns_collision/tests/"
    "test_separable_annular_pressure_schur_no_go.py",
    "work/ns_collision/notes/"
    "separable_annular_pressure_schur_no_go.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_separable_annular_pressure_schur_no_go_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, default=19)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=375)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument(
        "--periodic-average", type=float, required=True
    )
    parser.add_argument("--periodic-peak", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = _load_json(RESULT)
    _require(_sha256(RESULT) == RESULT_SHA256, "result hash changed")
    _require(
        result.get("status")
        == (
            "analytic_separable_annular_complete_HHL_"
            "Schur_no_go_certified"
        )
        and result.get("all_positive_checks_pass") is True,
        "separable annular pressure-Schur result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["explicit_real_divergence_free_family_constructed"] is True
        and flags["single_uniformly_bounded_annulus_proved"] is True
        and flags["exact_mixed_difference_Fisher_identity_proved"] is True
        and flags["Fisher_energy_O_N_minus_3_proved"] is True
        and flags["strict_nonzero_pressure_limit_proved"] is True
        and flags["complete_HHL_over_Fisher_at_least_order_N4_proved"]
        is True
        and flags[
            "uniform_joint_complete_HHL_Fisher_Schur_bound_falsified"
        ]
        is True
        and flags["isolated_primitive_chain_Hardy_theorem_falsified"]
        is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "separable annular certification scope changed",
    )
    _require(
        len(result["annular_family_rows"]) == 10
        and all(
            row["all_checks_pass"]
            for row in result["annular_family_rows"]
        )
        and result["dictionary_replay"]["all_checks_pass"] is True,
        "separable annular finite replay changed",
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
        and regression.get("tests_run")
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
        len(bookmark.get("completed_obligations", [])) == 154
        and len(bookmark.get("primary_artifacts", [])) == 543
        and principal.get(
            "joint_primitive_hhl_incidence_schur_gate_audit_v1_sha256"
        )
        == PREDECESSOR_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 155
        and len(bookmark.get("primary_artifacts", [])) == 549
        and principal.get(
            "separable_annular_pressure_schur_no_go_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed annular checkpoint matches",
    )

    summary = result["numerical_summary"]
    exact = result["exact_algebra_certificates"]
    regression_seconds = float(regression["duration_seconds"])
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "An explicit real divergence-free tensor-Dirichlet family now "
        "certifies an analytic no-go for the proposed static joint "
        "complete-HHL Fisher-Schur estimate. Every h_N lies in one annulus "
        "of shell ratio below 2. The exact mixed-difference identity gives "
        "E_lambda=O(N^-3), while the complete HHL load divided by N tends "
        "to a strictly negative pressure limit bounded away from zero by "
        "51*sqrt(2)/438976. The kinetic leading matrix cancels exactly and "
        "cross pressure is lower order, so |B_complete|/E_lambda grows at "
        "least as c*N^4. At N=65 the replay ratio is 7046.37. The "
        "dictionary crosscheck, 19 focused tests, and corrected atomic "
        "375-test pytest regression all pass. This closes only the static "
        "one-vertex joint-Schur route; all-vertex cancellation, temporal "
        "payment, critical L3, and global regularity remain open."
    )
    principal.update(
        {
            "separable_annular_explicit_family_constructed": True,
            "separable_annular_uniform_shell_ratio_below_two_proved": True,
            "separable_annular_exact_mixed_difference_Fisher_identity": True,
            "separable_annular_Fisher_O_N_minus_3_proved": True,
            "separable_annular_pressure_limit_strictly_negative_proved": True,
            "separable_annular_pressure_limit_lower_bound": (
                exact["pressure_limit_absolute_lower_bound"]
            ),
            "separable_annular_continuum_pressure_limit_quadrature": (
                summary["continuum_pressure_limit"]
            ),
            "separable_annular_complete_HHL_over_Fisher_N4_no_go": True,
            "separable_annular_largest_finite_size": summary[
                "largest_size"
            ],
            "separable_annular_largest_complete_to_Fisher_ratio": summary[
                "largest_complete_to_Fisher_ratio"
            ],
            "separable_annular_largest_ratio_over_N4": summary[
                "largest_ratio_over_size_to_fourth"
            ],
            "separable_annular_fixed_transverse_control_final_ratio": (
                summary["fixed_transverse_final_ratio"]
            ),
            "separable_annular_uniform_joint_static_Schur_bound_falsified": (
                True
            ),
            "separable_annular_isolated_chain_Hardy_falsified": False,
            "separable_annular_all_pressure_Fisher_routes_falsified": False,
            "separable_annular_all_cross_shell_HHL_absorbed": False,
            "separable_annular_terminal_dual_supremum_controlled": False,
            "separable_annular_critical_L3_controlled": False,
            "separable_annular_Navier_Stokes_regularity_proved": False,
            "separable_annular_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "separable_annular_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "separable_annular_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "separable_annular_full_pytest_regression_passed": True,
            "separable_annular_full_pytest_regression_runtime_seconds": (
                regression_seconds
            ),
            "separable_annular_resource_mode": arguments.resource_mode,
            "separable_annular_worker_count": arguments.worker_count,
            "separable_annular_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "separable_annular_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "separable_annular_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "separable_annular_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "separable_annular_result_status": result["status"],
            "separable_annular_pressure_schur_no_go_audit_v1_sha256": (
                RESULT_SHA256
            ),
            "full_regression_checkpoint_v1_sha256": (
                FULL_REGRESSION_SHA256
            ),
            "full_regression_runner_schema_version": 2,
            "full_regression_runner_engine": "pytest",
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
            "Constructed and proved the explicit separable bounded-annulus "
            "pressure family: exact mixed-difference Fisher energy "
            "O(N^-3), strictly nonzero complete-HHL load of order N, and "
            "an N^4 no-go for the proposed static joint Fisher-Schur "
            "bound, with independent Fourier and full-regression replays."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The static one-vertex joint complete-HHL Fisher-Schur route is "
        "now rigorously closed as stated. It remains to determine whether "
        "the signed sum over all eight compatible tensor vertices cancels "
        "the explicit annular family, or whether an evolution-dependent "
        "parabolic/time-decorrelation payment controls it. Cross-shell HHL "
        "absorption, terminal dual control, critical L3, exceptional-set "
        "removal, and Navier-Stokes global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "separable_annular_pressure_schur_no_go_audit.py"
    )
    bookmark["next_action"] = (
        "Build the signed all-eight-vertex response of the explicit "
        "annular family. Derive the exact vertex-sign incidence matrix "
        "before any large sweep and test the true partition identity, not "
        "eight separately charged absolute values. If the leading "
        "pressure limit cancels, quantify the first surviving order and "
        "its Fisher payment. If it survives, propagate the same family "
        "through one parabolic time window to test whether viscosity or "
        "phase evolution supplies a scale-uniform dynamic bound."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 155, "unexpected completed count")
    _require(len(primary) == 549, "unexpected artifact count")
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
