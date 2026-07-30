"""Install the validated adjoint replica pressure-edge checkpoint."""

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
    "adjoint_replica_pressure_edge_gate_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "adjoint_replica_pressure_edge_gate_audit.py",
    "work/ns_collision/tests/"
    "test_adjoint_replica_pressure_edge_gate.py",
    "work/ns_collision/notes/"
    "adjoint_replica_pressure_edge_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_adjoint_replica_pressure_edge_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--regression-test-count", type=int, required=True)
    parser.add_argument(
        "--regression-test-seconds",
        type=float,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require(args.targeted_test_count > 0, "targeted count must be positive")
    _require(
        args.regression_test_count > 0,
        "regression count must be positive",
    )
    _require(args.targeted_test_seconds >= 0.0, "invalid targeted runtime")
    _require(
        args.regression_test_seconds >= 0.0,
        "invalid regression runtime",
    )
    for artifact in ARTIFACTS:
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")

    result = _load_json(RESULT)
    flags = result.get("certification_flags")
    checks = result.get("positive_checks")
    _require(
        result.get("kind") == "adjoint_replica_pressure_edge_gate_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "backward_restart_dual_derived_"
            "universal_flux_sign_falsified_edge_budget_open"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict)
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        "adjoint pressure-edge result is not the expected passing audit",
    )
    for key in (
        "backward_adjoint_restart_dual_inequality_derived",
        "backward_terminal_penalty_contraction_proved",
        "replica_reset_endpoint_used",
        "positive_rho_cross_gradient_lower_bound_retained",
        "rho_zero_physical_pressure_form_derived",
        "rho_zero_is_instantaneously_best_at_reset",
        "scalar_pressure_partition_edge_identity_derived",
        "conditional_partition_Fisher_identity_derived",
        "degenerate_edge_Young_budget_derived",
        "smooth_universal_flux_nonpositivity_falsified",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "universal_restart_dual_flux_is_nonpositive",
        "positive_rho_improves_the_instantaneous_reset_generator",
        "edge_Young_remainder_absorbed_by_replica_dissipation",
        "finite_partition_represents_all_terminal_dual_weights",
        "critical_signed_replica_bound_proved",
        "low_regularity_adjoint_replica_system_justified",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    pressure = result["high_amplitude_pressure_sign_falsifier"]
    partition = result["partition_pressure_edge_gate"]
    _require(
        pressure.get("all_checks_pass") is True
        and pressure["rho_zero_sign_change_amplitude_range"][0] > 168.16
        and pressure["rho_zero_sign_change_amplitude_range"][1] < 168.18,
        "critical amplitude sign stress lost its resolved margin",
    )
    _require(
        partition.get("all_checks_pass") is True
        and partition["direct_weighted_pressure_flux"] > 1.0
        and partition["smooth_partition_sign_change_amplitude"] < 700.0
        and partition[
            "smooth_partition_scaled_rate_over_3_scale_cubed_at_700"
        ]
        > 0.0
        and partition["edge_young_upper_at_nu_one"]
        > partition["exact_dual_flux_at_nu_one"],
        "partition pressure-edge gate did not retain its no-go margins",
    )

    bookmark = _load_json(BOOKMARK)
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark",
        "refusing to update a non-NS bookmark",
    )
    _require(
        bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "bookmark is outside the standalone workspace boundary",
    )
    principal = bookmark.setdefault("principal_results", {})
    _require(
        principal.get("signed_projected_replica_SPDE_derived") is True
        and principal.get(
            "signed_projected_replica_cross_gradient_lower_bound_proved"
        )
        is True
        and principal.get("signed_projected_replica_L3_bound_proved")
        is False,
        "the prerequisite signed projected replica checkpoint is absent",
    )
    _require(
        principal.get(
            "reversible_weighted_hypercircle_componentwise64128_certified"
        )
        is True
        and principal.get(
            "reversible_weighted_hypercircle_full_inertia_certified"
        )
        is False,
        "the independent finite-pencil checkpoint changed unexpectedly",
    )

    threshold = pressure["rho_zero_sign_change_amplitude_range"]
    smooth_threshold = partition[
        "smooth_partition_sign_change_amplitude"
    ]
    shear = result["exact_periodic_shear"]
    abc = result["periodic_ABC"]

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The adjoint replica stage derives an exact smooth backward restart "
        "dual. The terminal cubic penalty contributes the additional "
        "dissipation 3nu integral lambda|grad lambda|^2, while the replica "
        "term retains 3nu(1-rho)lambda G_rho with "
        "G_rho>=|grad u|^2. At a reset, rho=0 is instantaneously optimal; "
        "positive correlation can help only through later pathwise changes "
        "in pressure or strain. Universal flux nonpositivity is false. The "
        "critical seed-81 weight changes sign at amplitude "
        f"[{threshold[0]:.12g},{threshold[1]:.12g}], and a separate smooth "
        "strictly positive partition weight changes sign at "
        f"{smooth_threshold:.12g}. Pressure is nevertheless an exact "
        "antisymmetric cell-edge transfer, and the adjoint Fisher term has "
        "an exact cubic conditional edge form. The valid reciprocal-weight "
        "Young bound is far too loose on the stored adversary, so edge "
        "absorption, a full terminal-weight representation, low-regularity "
        "passage, exceptional-set upgrade, and global regularity remain "
        "open. The independent 64128-pivot checkpoint remains valid."
    )

    principal.update(
        {
            "adjoint_replica_restart_dual_inequality_derived": True,
            "adjoint_replica_terminal_penalty_contraction_proved": True,
            "adjoint_replica_cross_gradient_lower_bound_retained": True,
            "adjoint_replica_rho_zero_physical_form_derived": True,
            "adjoint_replica_rho_zero_reset_optimal_proved": True,
            "adjoint_replica_scalar_pressure_edge_identity_derived": True,
            "adjoint_replica_conditional_Fisher_edge_identity_derived": True,
            "adjoint_replica_degenerate_edge_Young_budget_derived": True,
            "adjoint_replica_smooth_universal_sign_falsified": True,
            "adjoint_replica_universal_flux_nonpositive": False,
            "adjoint_replica_positive_rho_reset_improvement": False,
            "adjoint_replica_edge_remainder_absorbed": False,
            "adjoint_replica_full_terminal_partition_proved": False,
            "adjoint_replica_critical_bound_proved": False,
            "adjoint_replica_low_regularity_system_proved": False,
            "adjoint_replica_exceptional_set_upgrade_proved": False,
            "adjoint_replica_Navier_Stokes_regularity_proved": False,
            "adjoint_replica_critical_sign_threshold_minimum": threshold[0],
            "adjoint_replica_critical_sign_threshold_maximum": threshold[1],
            "adjoint_replica_smooth_partition_sign_threshold": (
                smooth_threshold
            ),
            "adjoint_replica_partition_direct_pressure_flux": partition[
                "direct_weighted_pressure_flux"
            ],
            "adjoint_replica_partition_exact_flux": partition[
                "exact_dual_flux_at_nu_one"
            ],
            "adjoint_replica_partition_Young_upper": partition[
                "edge_young_upper_at_nu_one"
            ],
            "adjoint_replica_shear_identity_residual": shear[
                "identity_residual"
            ],
            "adjoint_replica_ABC_balance_residual": abc[
                "Delta_u_equals_minus_u_balance_residual"
            ],
            "adjoint_replica_targeted_test_count": args.targeted_test_count,
            "adjoint_replica_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "adjoint_replica_regression_test_count": (
                args.regression_test_count
            ),
            "adjoint_replica_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Derived the exact backward terminal-weight replica restart "
            "dual, retained its cubic Fisher dissipation, proved rho=0 is "
            "instantaneously optimal at a reset, and converted scalar "
            "pressure into an exact partition-edge transfer with a cubic "
            "conditional weight penalty. Two smooth amplitude stresses "
            "falsify universal flux nonpositivity, while the reciprocal "
            "edge Young bound is recorded as valid but nonclosing."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The restart dual is exact but its signed pressure-edge flux is not "
        "globally nonpositive. The live obligation is to couple each "
        "conditional edge density directly to "
        "nu^2(1-rho)lambda G_rho without the loose reciprocal-weight Young "
        "split, while allowing terminal weights to vanish and preserving "
        "the full Legendre supremum through a scale-adapted multilevel "
        "partition. Alternatively, a finite-time rho>0 expansion must show "
        "that pathwise changes in R_rho or F_rho repay the dissipation lost "
        "at reset. Any smooth estimate must then pass to Leray regularity "
        "and exceptional points. No such estimate is currently proved."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "Derive the scale-covariant conditional edge density and test a "
        "direct weighted estimate of sum e_j^2/(A_j+B_j) against "
        "nu^2(1-rho) integral lambda G_rho, including the zero-face limit. "
        "In parallel algebra, compute the first nonzero short-time rho "
        "correction to R_rho and F_rho after reset. Reject any candidate "
        "that fails the amplitude-scaled seed-81 datum or replaces the full "
        "terminal dual by a fixed finite partition."
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
