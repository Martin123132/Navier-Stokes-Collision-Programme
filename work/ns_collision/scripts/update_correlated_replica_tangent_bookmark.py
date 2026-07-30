"""Install the validated correlated-replica tangent Gramian checkpoint."""

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
    "correlated_replica_tangent_gramian_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "correlated_replica_tangent_gramian_audit.py",
    "work/ns_collision/tests/"
    "test_correlated_replica_tangent_gramian.py",
    "work/ns_collision/notes/"
    "correlated_replica_tangent_gramian_bridge.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_correlated_replica_tangent_bookmark.py",
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
    parser.add_argument("--legacy-test-count", type=int, required=True)
    parser.add_argument("--legacy-test-seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require(args.targeted_test_count > 0, "targeted test count must be positive")
    _require(args.legacy_test_count > 0, "legacy test count must be positive")
    _require(args.targeted_test_seconds >= 0.0, "invalid targeted runtime")
    _require(args.legacy_test_seconds >= 0.0, "invalid legacy runtime")
    for artifact in ARTIFACTS:
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")

    result = _load_json(RESULT)
    flags = result.get("certification_flags")
    _require(
        result.get("kind") == "correlated_replica_tangent_gramian_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "smooth_flow_correlation_to_deformation_identity_proved_"
            "critical_trace_estimate_open"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict),
        "correlated-replica result is not the expected passing audit",
    )
    for key in (
        "correlated_replica_Ito_homotopy_derived",
        "common_noise_tangent_limit_derived_for_smooth_drift",
        "conditional_forward_inverse_Gramian_congruence_proved",
        "conditional_cross_covariance_recovers_flow_Jacobian",
        "incompressible_Gramian_determinant_balance_proved",
        "Minkowski_determinant_floor_proved",
        "radial_trace_deformation_bound_proved",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "radial_noncollision_alone_controls_deformation",
        "Leray_energy_controls_critical_forward_inverse_traces",
        "low_regularity_inverse_time_probe_justified",
        "critical_L3_continuation_bridge_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

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
        principal.get(
            "reversible_weighted_hypercircle_componentwise64128_certified"
        )
        is True,
        "the prior 64128-pivot checkpoint is not present",
    )
    _require(
        principal.get(
            "reversible_weighted_hypercircle_full_inertia_certified"
        )
        is False,
        "unexpected promotion of the prior finite pencil",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The correlation-to-deformation stage proves an exact smooth-flow "
        "bridge. Symmetrically rho-correlated replicas converge at rho=1 "
        "to a tangent diffusion. Its conditioned forward Gramian F, "
        "inverse-time Gramian B, and shared-probe covariance H obey "
        "F=J B J^T and J=H B^(-1). Incompressibility gives det(F)=det(B) "
        ">=(4nuT)^3, hence the normalized radial traces f and b satisfy "
        "||J||^2<=f b^2/4 and ||J^(-1)||^2<=b f^2/4. Planar strain "
        "disproves deformation control from noncollision alone, while "
        "simple shear quantifies the loss in the trace-only reduction. "
        "No Leray-energy trace bound, low-regularity inverse-time "
        "construction, L3 continuation theorem, or global regularity "
        "claim is made. The independent 64128-pivot finite certificate "
        "remains valid and unchanged but is no longer the active research "
        "priority."
    )

    principal.update(
        {
            "correlated_replica_Ito_homotopy_derived": True,
            "correlated_replica_common_noise_tangent_limit_derived": True,
            "correlated_replica_forward_inverse_Gramian_congruence_proved": True,
            "correlated_replica_cross_covariance_Jacobian_recovery_proved": True,
            "correlated_replica_incompressible_determinant_balance_proved": True,
            "correlated_replica_Minkowski_determinant_floor_proved": True,
            "correlated_replica_radial_trace_deformation_bound_proved": True,
            "correlated_replica_radial_noncollision_alone_sufficient": False,
            "correlated_replica_Leray_critical_trace_bound_proved": False,
            "correlated_replica_low_regularity_inverse_probe_proved": False,
            "correlated_replica_critical_L3_bridge_proved": False,
            "correlated_replica_Navier_Stokes_regularity_proved": False,
            "correlated_replica_planar_stress_deformation_norm": result[
                "stress_tests"
            ]["one_sided_noncollision"]["maximum_deformation_norm"],
            "correlated_replica_shear_trace_bound_loss": result[
                "stress_tests"
            ]["simple_shear_nonnormality"][
                "trace_bound_over_actual_squared"
            ],
            "correlated_replica_targeted_test_count": (
                args.targeted_test_count
            ),
            "correlated_replica_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "correlated_replica_legacy_test_count": args.legacy_test_count,
            "correlated_replica_legacy_test_runtime_seconds": (
                args.legacy_test_seconds
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
            "Derived and independently stress-tested the rho-correlated "
            "replica limit. The smooth-flow forward/inverse-time Gramian "
            "congruence, shared-probe Jacobian recovery, incompressible "
            "determinant floor, and scale-invariant radial-trace "
            "deformation bounds pass symbolic, affine, shear, rotation, "
            "noncommuting, closed-form, and parabolic-scaling checks."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The smooth-flow correlation-to-deformation identity is exact, "
        "but the decisive a priori estimate is open: control suitable "
        "moments of both normalized forward and inverse-time Gramian "
        "traces on every parabolic window using only Navier-Stokes "
        "structure and critical data. The inverse-time tangent probe must "
        "also be justified below the classical regime, and the resulting "
        "functional must imply a standard continuation criterion and "
        "exclude exceptional singular points. The older 64128-pivot "
        "finite certificate remains valid; 64256 and the full pencil are "
        "secondary and should not be resumed unless this bridge creates "
        "a concrete need for those constants."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "Derive the local parabolic-window form of F, B, and H along a "
        "common stochastic trajectory. Determine the weakest moment of "
        "f=tr(F)/(4nu tau) and b=tr(B)/(4nu tau) that closes the "
        "Constantin-Iyer representation or an L-infinity-time L3-space "
        "criterion. Stress that candidate on exact planar strain, "
        "nonnormal shear, rigid rotation, Burgers-vortex-type profiles, "
        "and finite Fourier Navier-Stokes trajectories before attempting "
        "a Leray-energy estimate."
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
