"""Install the deficit-retaining annular restart checkpoint."""

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
    "deficit_retaining_annular_restart_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "2f32255887eb18ec0aa22dadfacf681b930434e73f0c457041d65a66e8c04e6d"
)
PREDECESSOR_SHA256 = (
    "fffa314fc9fa516dc0c8f6ac010392d438845912f6d4bc2d16cc1f2dc02b83e0"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "65a0a6937a33053a116555b1c5d10fc0313d0743aa7a2fd0c9d301f985853fcd"
)
RUNNER_SHA256 = (
    "4fd9cca40a4133bfce8bba21161dd827dc98ccbeee9c13f70f7b3718167a4609"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "deficit_retaining_annular_restart_gate_audit.py",
    "work/ns_collision/tests/"
    "test_deficit_retaining_annular_restart_gate.py",
    "work/ns_collision/notes/"
    "deficit_retaining_annular_restart_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_deficit_retaining_annular_restart_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, default=26)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=399)
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
        result.get("status") == "annular_static_escape_reset_tax_certified"
        and result.get("all_positive_checks_pass") is True,
        "deficit-retaining annular restart result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["rho_zero_pressure_only_correction_applied"] is True
        and flags["exact_deficit_retaining_restart_identity_proved"] is True
        and flags["reset_Legendre_deficit_nonnegative_proved"] is True
        and flags["backward_weight_L3_contraction_retained"] is True
        and flags["pressure_only_static_escape_replayed"] is True
        and flags["static_optimal_annular_reset_tax_order_N3_proved"] is True
        and flags[
            "parabolic_survival_requires_order_N2_amplification_proved"
        ]
        is True
        and flags["static_escape_is_direct_dynamic_counterexample"] is False
        and flags["required_nonlinear_amplification_excluded"] is False
        and flags["all_terminal_weights_dynamically_controlled"] is False
        and flags["critical_L3_controlled"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "deficit-retaining certification scope changed",
    )
    finite = result["finite_summary"]
    _require(
        finite["finite_gate_passes"] is True
        and finite[
            "first_audited_positive_pressure_only_static_size"
        ]
        == 25
        and finite["size_25_full_reset_deficit_lower_bound"] > 0.5
        and finite["size_25_required_average_amplification"] > 1.0e6
        and len(result["pressure_only_annular_rows"]) == 4
        and len(result["reset_tax_rows"]) == 4
        and all(
            row["all_checks_pass"]
            for row in result["pressure_only_annular_rows"]
        )
        and all(
            row["all_checks_pass"] for row in result["reset_tax_rows"]
        ),
        "deficit-retaining finite replay changed",
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
        len(bookmark.get("completed_obligations", [])) == 157
        and len(bookmark.get("primary_artifacts", [])) == 559
        and principal.get(
            "compatible_edge_annular_escape_audit_v1_sha256"
        )
        == PREDECESSOR_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 158
        and len(bookmark.get("primary_artifacts", [])) == 564
        and principal.get(
            "deficit_retaining_annular_restart_gate_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed reset-tax checkpoint matches",
    )

    asymptotic = result["asymptotic_reset_tax_certificate"]
    row25 = next(
        row for row in result["reset_tax_rows"] if row["size"] == 25
    )
    regression_seconds = float(regression["duration_seconds"])
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The static annular compatible-edge escape has now been inserted "
        "into the exact rho=0 backward-adjoint restart identity with its "
        "previously dropped reset-time Legendre deficit restored. At "
        "rho=0 the strain/kinetic cancellation means the true generator "
        "uses pressure only; direct finite replay shows that correction "
        "does not remove the static escape. However, the exact identity is "
        "||u(T)||_3^3=||u(s)||_3^3+sup[J_0-Delta_s], with "
        "Delta_s=integral(|u|-lambda_s)^2(|u|+lambda_s/2). Backward L3 "
        "contraction and Parseval give Delta_s=Omega(N^3) for the "
        "static-optimal t_N Phi_+++ weight. A T/N^2 window can overcome "
        "this tax only if the average pressure generator reaches "
        "Omega(N^5), an N^2 amplification over its initial value. At N=25 "
        "the rigorous deficit lower bound is 0.559805 and the required "
        "average amplification exceeds 2.91e7. Thus the static escape is "
        "not itself a dynamic restart counterexample. The 26 focused tests "
        "and all 399 corpus tests pass. The required nonlinear "
        "amplification, all-weight control, critical L3, and regularity "
        "remain open."
    )
    principal.update(
        {
            "restart_deficit_rho_zero_pressure_only_correction": True,
            "restart_deficit_exact_retaining_identity_proved": True,
            "restart_deficit_pointwise_factorization": (
                "(r-lambda)^2(r+lambda/2)"
            ),
            "restart_deficit_Phi_L3_norm_exact": "5/16",
            "restart_deficit_Phi_weight_Fisher_exact": "75/4096",
            "restart_deficit_unit_low_L2_squared_exact": "2",
            "restart_deficit_pressure_only_static_escape_size": 25,
            "restart_deficit_N25_lower_bound": row25[
                "full_L2_reset_deficit_lower_bound"
            ],
            "restart_deficit_N25_required_average_amplification": row25[
                "required_average_amplification_over_initial_generator"
            ],
            "restart_deficit_beta_star": asymptotic["beta_star"],
            "restart_deficit_over_N3_lower_limit": asymptotic[
                "reset_deficit_over_N_cubed_lower_limit"
            ],
            "restart_deficit_tax_to_three_generator_time_limit": asymptotic[
                "reset_tax_to_three_static_generator_time_limit"
            ],
            "restart_deficit_parabolic_required_amplification": (
                "[5/(288nu T)]N^2+o(N^2)"
            ),
            "restart_deficit_static_escape_direct_dynamic_counterexample": (
                False
            ),
            "restart_deficit_required_N2_amplification_excluded": False,
            "restart_deficit_all_terminal_weights_controlled": False,
            "restart_deficit_critical_L3_controlled": False,
            "restart_deficit_Navier_Stokes_regularity_proved": False,
            "restart_deficit_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "restart_deficit_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "restart_deficit_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "restart_deficit_full_pytest_regression_passed": True,
            "restart_deficit_full_pytest_runtime_seconds": (
                regression_seconds
            ),
            "restart_deficit_resource_mode": arguments.resource_mode,
            "restart_deficit_worker_count": arguments.worker_count,
            "restart_deficit_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "restart_deficit_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "restart_deficit_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "restart_deficit_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "restart_deficit_result_status": result["status"],
            "deficit_retaining_annular_restart_gate_audit_v1_sha256": (
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
            "Restored the exact reset-time Legendre deficit in the rho=0 "
            "backward-adjoint supremum, corrected the annular generator to "
            "pressure only, and proved that the static-optimal witness pays "
            "an order-N3 reset tax and requires order-N2 dynamic "
            "amplification on parabolic windows."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The unpenalized static annular escape is now known not to be a "
        "direct dynamic restart obstruction because its compatible weight "
        "pays an order-N3 Legendre mismatch at the reset. It remains to "
        "calculate the exact first time jet of the pressure-only generator "
        "under coupled Navier-Stokes and backward-weight evolution and "
        "decide whether the required N2 amplification can occur. "
        "Optimization over all near-Legendre terminal weights, critical L3, "
        "exceptional-set removal, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "deficit_retaining_annular_restart_gate_audit.py"
    )
    bookmark["next_action"] = (
        "Differentiate g_0(t)=integral[p u dot grad lambda"
        "-nu lambda|grad u|^2-nu lambda|grad lambda|^2] at the restart for "
        "u_N=h_N-a_N U and terminal lambda=t_N Phi_+++. Use the exact "
        "Navier-Stokes acceleration and backward-weight derivative. "
        "Separate viscous, low-high transport, pressure-response, and "
        "weight-advection contributions, determine every N5 leading "
        "coefficient and sign, and compare the integrated first jet with "
        "the exact reset deficit before computing any second jet."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 158, "unexpected completed count")
    _require(len(primary) == 564, "unexpected artifact count")
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
