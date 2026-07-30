"""Install the validated projected-Weber replica-gate checkpoint."""

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
    "projected_weber_replica_gate_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "projected_weber_replica_gate_audit.py",
    "work/ns_collision/tests/"
    "test_projected_weber_replica_gate.py",
    "work/ns_collision/notes/"
    "projected_weber_replica_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_projected_weber_replica_bookmark.py",
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
        result.get("kind") == "projected_weber_replica_gate_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "projection_loss_identified_"
            "signed_projected_replica_closure_open"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict)
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        "projected-Weber result is not the expected passing audit",
    )
    for key in (
        "joint_common_path_tangent_covector_generator_derived",
        "smooth_tensor_spectral_proxy_generator_derived",
        "mean_Weber_magnetization_equation_derived",
        "mean_magnetization_reset_strain_integral_cancels",
        "unprojected_positive_moment_retains_Leray_gradient_loss",
        "projected_positive_Jensen_moment_retains_noise_variance",
        "two_replica_signed_L3_identity_derived",
        "three_replica_signed_cubic_identity_derived",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "bare_directional_cubic_has_direct_collision_diffusion",
        "single_superharmonic_collision_weight_closes_pointwise",
        "Leray_energy_bounds_unprojected_directional_moment",
        "projected_positive_Jensen_moment_bound_proved",
        "signed_projected_replica_closure_bound_proved",
        "low_regularity_projected_replica_flow_justified",
        "exceptional_set_upgrade_proved",
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
        principal.get("parabolic_Gramian_restart_laws_proved") is True
        and principal.get(
            "parabolic_directional_cubic_L3_sufficiency_proved"
        )
        is True,
        "the prerequisite parabolic-Gramian checkpoint is not present",
    )
    _require(
        principal.get(
            "reversible_weighted_hypercircle_componentwise64128_certified"
        )
        is True,
        "the independent 64128-pivot checkpoint is not present",
    )
    _require(
        principal.get(
            "reversible_weighted_hypercircle_full_inertia_certified"
        )
        is False,
        "unexpected promotion of the prior finite pencil",
    )

    affine = result["affine_generator_stress"]
    shear = result["exact_periodic_shear"]
    inflation_rows = shear["unprojected_magnetization_inflation"]["rows"]
    harmonic = shear["projected_common_path_harmonic_variance"]
    tensor_rows = result["tensor_proxy"]["rows"]

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The projection-loss stage narrows the classical continuation "
        "target without claiming its missing estimate. The exact joint "
        "common-path tangent/covector generator gives no direct collision "
        "diffusion to the bare directional cubic or its smooth tensor "
        "proxy, and one superharmonic collision weight fails a pointwise "
        "affine sign test. Averaging before the norm creates a viscous "
        "magnetization balance and exact reset cancellation, but exact "
        "periodic shear makes the unprojected cubic inflate by "
        f"{inflation_rows[-1]['unprojected_inflation_ratio']:.12g} "
        "through a pure Leray gradient while the physical velocity norm "
        "does not rise. Projecting pathwise still leaves strictly positive "
        "common-noise harmonic variance whose signed mean is zero. Exact "
        "two- and three-replica identities retain both cancellations. No "
        "signed replica bound, low-regularity flow construction, "
        "exceptional-set upgrade, or global regularity claim is made. The "
        "Gramian hierarchy remains sufficient but secondary, and the "
        "independent 64128-pivot certificate remains valid."
    )

    principal.update(
        {
            "projected_Weber_joint_tangent_covector_generator_derived": True,
            "projected_Weber_bare_cubic_direct_collision_diffusion": False,
            "projected_Weber_smooth_tensor_proxy_generator_derived": True,
            "projected_Weber_single_weight_pointwise_closure": False,
            "projected_Weber_mean_magnetization_equation_derived": True,
            "projected_Weber_reset_strain_cancellation_proved": True,
            "projected_Weber_unprojected_gradient_loss_proved": True,
            "projected_Weber_positive_Jensen_noise_loss_proved": True,
            "projected_Weber_signed_two_replica_identity_proved": True,
            "projected_Weber_signed_three_replica_identity_proved": True,
            "projected_Weber_Leray_unprojected_bound_proved": False,
            "projected_Weber_positive_Jensen_bound_proved": False,
            "projected_Weber_signed_replica_bound_proved": False,
            "projected_Weber_low_regularity_flow_proved": False,
            "projected_Weber_exceptional_set_upgrade_proved": False,
            "projected_Weber_Navier_Stokes_regularity_proved": False,
            "projected_Weber_affine_minimum_superharmonic_rate": affine[
                "minimum_rate_over_0_le_q_le_1_with_expanding_tangent"
            ],
            "projected_Weber_shear_maximum_unprojected_inflation": max(
                row["unprojected_inflation_ratio"]
                for row in inflation_rows
            ),
            "projected_Weber_shear_harmonic_variance": harmonic[
                "closed_form_variance"
            ],
            "projected_Weber_shear_variance_formula_residual": harmonic[
                "relative_residual"
            ],
            "projected_Weber_tensor_proxy_maximum_residual": max(
                row["proxy_derivative_relative_residual"]
                for row in tensor_rows
            ),
            "projected_Weber_targeted_test_count": args.targeted_test_count,
            "projected_Weber_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "projected_Weber_regression_test_count": args.regression_test_count,
            "projected_Weber_regression_test_runtime_seconds": (
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
            "Derived the joint common-path tangent/inverse-covector "
            "generator and smooth tensor-spectral proxy; proved the lack "
            "of direct collision diffusion and a one-weight affine sign "
            "obstruction; then used exact periodic shear to separate "
            "unprojected Leray-gradient loss from pathwise-projected "
            "common-noise variance. Exact signed two- and three-replica "
            "identities identify the cancellation-preserving descendant "
            "without promoting an unproved bound."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The continuation hierarchy remains exact, but positive moments "
        "discard cancellations that are already decisive on smooth shear. "
        "The Clay-relevant obligation is now to bound a signed projected "
        "two- or three-replica functional at critical scaling. The "
        "derivation must retain each realization's Leray pressure transfer "
        "and cancellation across common-noise paths before absolute values. "
        "It must then justify projected stochastic flows at Leray "
        "regularity and exclude exceptional singular points. No estimate "
        "for that signed functional is currently proved. The older "
        "64128-pivot finite certificate and positive Gramian criteria "
        "remain valid but secondary."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "Derive the signed projected two-replica correlation and "
        "three-replica directional generators in the random-translation "
        "frame. Keep each Weber pressure as an exact Poisson or partition-"
        "flux term, and compute the rho-dependent cross-diffusion before "
        "taking any absolute value. Test the resulting signed identity on "
        "exact shear, ABC flow, the existing adversarial pressure field, "
        "and Burgers-vortex-type strain. First seek or falsify a global "
        "spatial cancellation; do not launch a large numerical search "
        "without an explicit signed analytic target."
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
