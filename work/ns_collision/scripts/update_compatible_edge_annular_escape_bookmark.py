"""Install the compatible-edge annular escape checkpoint."""

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
    "compatible_edge_annular_escape_audit_v1.json"
)
RESULT_SHA256 = (
    "fffa314fc9fa516dc0c8f6ac010392d438845912f6d4bc2d16cc1f2dc02b83e0"
)
PREDECESSOR_SHA256 = (
    "5313001d5a136babf1be6d99b66767db4161e526cd08158631cde2a68c942789"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "41edd7ea93689e1423f4c0f303ce47fd8ca34c0f34b0732419441639a1f67bc2"
)
RUNNER_SHA256 = (
    "4fd9cca40a4133bfce8bba21161dd827dc98ccbeee9c13f70f7b3718167a4609"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "compatible_edge_annular_escape_audit.py",
    "work/ns_collision/tests/"
    "test_compatible_edge_annular_escape.py",
    "work/ns_collision/notes/"
    "compatible_edge_annular_escape.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_compatible_edge_annular_escape_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, default=21)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=391)
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
        result.get("status") == "complete_compatible_edge_escape_certified"
        and result.get("all_positive_checks_pass") is True,
        "compatible-edge annular escape result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["complete_full_field_flux_included"] is True
        and flags["full_weighted_velocity_Fisher_included"] is True
        and flags["low_field_Fisher_cost_included"] is True
        and flags["exact_twelve_edge_cubic_penalty_included"] is True
        and flags[
            "exact_joint_low_amplitude_and_ray_scale_optimization_proved"
        ]
        is True
        and flags["fixed_ray_asymptotic_dichotomy_proved"] is True
        and flags["delta_plus_optimized_escape_proved"] is True
        and flags["bounded_compatible_coefficient_escape_proved"] is True
        and flags["static_arbitrary_coefficient_coercivity_proved"] is False
        and flags["dynamic_adjoint_coefficient_escape_proved"] is False
        and flags["critical_L3_growth_proved"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "compatible-edge certification scope changed",
    )
    finite = result["finite_escape_summary"]
    _require(
        finite["finite_escape_checks_pass"] is True
        and finite["first_audited_positive_optimized_size"] == 25
        and finite["first_audited_positive_bounded_scale_one_size"] == 137
        and finite["largest_size"] == 137
        and len(result["finite_annular_rows"]) == 11
        and all(
            row["all_checks_pass"]
            for row in result["finite_annular_rows"]
        )
        and result["full_field_support_replay"]["all_checks_pass"] is True
        and result["exact_edge_penalty_certificate"][
            "common_exact_value"
        ]
        == "75/256",
        "compatible-edge finite replay changed",
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
        len(bookmark.get("completed_obligations", [])) == 156
        and len(bookmark.get("primary_artifacts", [])) == 554
        and principal.get(
            "annular_eight_vertex_heat_window_gate_audit_v1_sha256"
        )
        == PREDECESSOR_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 157
        and len(bookmark.get("primary_artifacts", [])) == 559
        and principal.get(
            "compatible_edge_annular_escape_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed compatible-edge checkpoint matches",
    )

    asymptotic = result["asymptotic_ray_certificate"]
    plus_limits = asymptotic["delta_plus_asymptotics"]
    row25 = next(
        row for row in result["finite_annular_rows"] if row["size"] == 25
    )
    row137 = next(
        row for row in result["finite_annular_rows"] if row["size"] == 137
    )
    regression_seconds = float(regression["duration_seconds"])
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The exact compatible twelve-edge coefficient objective has now "
        "been tested on the explicit annular family with every adverse "
        "term retained. The low plane wave has zero standalone flux load "
        "and exact vertex Fisher cost 1/2; HHH, HLL, and high-low Fisher "
        "cross terms miss the partition stencil; and every delta-vertex "
        "coefficient has exact cubic penalty Q=75/256. Joint optimization "
        "over low amplitude and coefficient scale gives an explicit +++ "
        "escape: the optimized objective is positive at audited N=25 and "
        "grows as a positive constant times N^3. Even coefficient scale "
        "t=1 is positive at N=137 and eventually grows as N^2. Fixed rays "
        "with positive --- mass are suppressed by its N^3 Fisher cost, "
        "while rays on z_---=0 with nonzero limiting load escape. The "
        "static arbitrary-coefficient coercivity route is closed. The 21 "
        "focused tests and all 391 corpus tests pass. Dynamic adjoint "
        "admissibility, critical endpoint control, and regularity remain "
        "open."
    )
    principal.update(
        {
            "compatible_edge_complete_full_field_flux_included": True,
            "compatible_edge_full_weighted_velocity_Fisher_included": True,
            "compatible_edge_low_vertex_Fisher_exact": 0.5,
            "compatible_edge_delta_vertex_Q_exact": "75/256",
            "compatible_edge_joint_ray_optimization_exact": True,
            "compatible_edge_fixed_ray_dichotomy_proved": True,
            "compatible_edge_beta_plus_signed": asymptotic[
                "beta_plus_signed"
            ],
            "compatible_edge_beta_star": asymptotic["beta_star"],
            "compatible_edge_optimal_low_amplitude_over_N_limit": (
                plus_limits["oriented_low_amplitude_over_N_limit"]
            ),
            "compatible_edge_margin_over_N2_limit": plus_limits[
                "linear_margin_over_N_squared_limit"
            ],
            "compatible_edge_coefficient_scale_over_N_limit": (
                plus_limits["coefficient_scale_over_N_limit"]
            ),
            "compatible_edge_objective_over_N3_limit": plus_limits[
                "optimized_objective_over_N_cubed_limit"
            ],
            "compatible_edge_first_audited_optimized_escape_size": 25,
            "compatible_edge_N25_optimized_objective": row25[
                "ray_optimization"
            ]["optimized_objective"],
            "compatible_edge_first_audited_bounded_escape_size": 137,
            "compatible_edge_N137_bounded_objective": row137[
                "ray_optimization"
            ]["bounded_coefficient_scale_one_objective"],
            "compatible_edge_static_arbitrary_coefficient_coercivity": False,
            "compatible_edge_dynamic_adjoint_escape_proved": False,
            "compatible_edge_critical_L3_controlled": False,
            "compatible_edge_Navier_Stokes_regularity_proved": False,
            "compatible_edge_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "compatible_edge_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "compatible_edge_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "compatible_edge_full_pytest_regression_passed": True,
            "compatible_edge_full_pytest_runtime_seconds": (
                regression_seconds
            ),
            "compatible_edge_resource_mode": arguments.resource_mode,
            "compatible_edge_worker_count": arguments.worker_count,
            "compatible_edge_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "compatible_edge_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "compatible_edge_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "compatible_edge_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "compatible_edge_result_status": result["status"],
            "compatible_edge_annular_escape_audit_v1_sha256": (
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
            "Solved the exact low-amplitude and coefficient-ray "
            "optimization for the complete compatible twelve-edge "
            "objective, proved the fixed-ray Fisher dichotomy, and "
            "certified an explicit annular +++ escape with both optimized "
            "and bounded coefficient finite replays."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "Static arbitrary compatible coefficients are now known not to "
        "close the complete instantaneous objective, even after retaining "
        "the low-field Fisher cost and exact cubic edge penalty. It remains "
        "to impose the actual backward-adjoint coefficient evolution or a "
        "state-coupled admissibility law and determine whether the annular "
        "escape pays a controlled critical endpoint tax over a restart "
        "window. Critical L3, delayed large-amplitude compensation, "
        "exceptional-set removal, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "compatible_edge_annular_escape_audit.py"
    )
    bookmark["next_action"] = (
        "Insert u_N=h_N-a_N U and terminal weight t_N Phi_+++ into the "
        "exact rho=0 backward-adjoint restart identity. Derive the "
        "coefficient evolution and endpoint terms at the scales "
        "a_N,t_N~N. First test whether the required terminal weight can be "
        "generated or dominated by the physical state without an order-N3 "
        "critical tax. If the tax pays the escape, state the sharp dynamic "
        "admissibility theorem; otherwise construct a finite restart-window "
        "positive contribution with all endpoint terms included."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 157, "unexpected completed count")
    _require(len(primary) == 559, "unexpected artifact count")
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
