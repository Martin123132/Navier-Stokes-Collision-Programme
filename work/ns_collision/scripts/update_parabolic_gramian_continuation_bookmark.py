"""Install the validated parabolic-Gramian continuation checkpoint."""

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
    "parabolic_gramian_continuation_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "parabolic_gramian_continuation_audit.py",
    "work/ns_collision/tests/"
    "test_parabolic_gramian_continuation.py",
    "work/ns_collision/notes/"
    "parabolic_gramian_continuation_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_parabolic_gramian_continuation_bookmark.py",
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
        result.get("kind") == "parabolic_gramian_continuation_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "critical_continuation_hierarchy_proved_"
            "unconditional_Gramian_moment_bound_open"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict)
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        "parabolic-Gramian result is not the expected passing audit",
    )
    for key in (
        "parabolic_window_Gramian_definitions_proved",
        "Jacobian_and_Gramian_restart_laws_proved",
        "local_Constantin_Iyer_velocity_restart_formula_used",
        "exact_directional_cubic_moment_is_sufficient_for_L3_control",
        "tensor_spectral_cubic_moment_is_sufficient_for_L3_control",
        "scalar_radial_cubic_moment_is_sufficient_for_L3_control",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "scalar_radial_criterion_is_quantitatively_viable",
        "Leray_energy_bounds_tensor_spectral_moment",
        "Leray_energy_bounds_scalar_radial_moment",
        "low_regularity_inverse_time_probe_justified",
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
        principal.get(
            "reversible_weighted_hypercircle_componentwise64128_certified"
        )
        is True,
        "the prior 64128-pivot checkpoint is not present",
    )
    _require(
        principal.get(
            "correlated_replica_forward_inverse_Gramian_congruence_proved"
        )
        is True,
        "the prerequisite correlated-replica bridge is not present",
    )
    _require(
        principal.get(
            "reversible_weighted_hypercircle_full_inertia_certified"
        )
        is False,
        "unexpected promotion of the prior finite pencil",
    )

    burgers = result["Burgers_vortex_axis_stress"]
    shear = result["periodic_finite_Fourier_shear"]
    abc = result["periodic_finite_Fourier_ABC"]
    expansion = result["small_window_strain_expansion"]
    cocycle_residuals = abc["restart_cocycle"]["residuals"]

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The local parabolic-window stage closes the classical continuation "
        "bridge, but not its a priori estimate. Conditioned forward, "
        "inverse-time, and cross Gramians obey exact restart laws. The "
        "restarted Constantin-Iyer representation shows that a bounded "
        "directional cubic pullback moment controls critical L3; tensor-"
        "spectral and scalar-radial Gramian moments are successively weaker "
        "sufficient conditions. Burgers-axis strain exposes a scalar-radial "
        f"loss of {burgers['largest_radial_loss_over_tensor']:.12g} against "
        "the tensor bound, so radial traces alone are rejected as the "
        "primary closure target. Exact periodic shear and ABC solutions "
        "pass the hierarchy and all restart checks. No Leray-energy moment "
        "bound, low-regularity inverse probe, exceptional-set upgrade, or "
        "global regularity claim is made. The independent 64128-pivot "
        "finite certificate remains valid and secondary."
    )

    principal.update(
        {
            "parabolic_Gramian_window_definitions_proved": True,
            "parabolic_Gramian_restart_laws_proved": True,
            "parabolic_directional_cubic_L3_sufficiency_proved": True,
            "parabolic_tensor_spectral_L3_sufficiency_proved": True,
            "parabolic_scalar_radial_L3_sufficiency_proved": True,
            "parabolic_scalar_radial_primary_closure_viable": False,
            "parabolic_Leray_tensor_moment_bound_proved": False,
            "parabolic_Leray_scalar_moment_bound_proved": False,
            "parabolic_low_regularity_inverse_probe_proved": False,
            "parabolic_exceptional_set_upgrade_proved": False,
            "parabolic_Navier_Stokes_regularity_proved": False,
            "parabolic_Burgers_radial_loss_over_tensor": burgers[
                "largest_radial_loss_over_tensor"
            ],
            "parabolic_shear_exact_J_residual": shear["window"][
                "exact_J_residual"
            ],
            "parabolic_ABC_maximum_tensor_inverse_loss": abc["summary"][
                "maximum_tensor_inverse_loss"
            ],
            "parabolic_ABC_maximum_radial_inverse_loss": abc["summary"][
                "maximum_radial_inverse_loss"
            ],
            "parabolic_ABC_maximum_restart_residual": max(
                cocycle_residuals.values()
            ),
            "parabolic_small_window_forward_relative_error": expansion[
                "forward_relative_error"
            ],
            "parabolic_small_window_inverse_relative_error": expansion[
                "inverse_relative_error"
            ],
            "parabolic_targeted_test_count": args.targeted_test_count,
            "parabolic_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "parabolic_regression_test_count": args.regression_test_count,
            "parabolic_regression_test_runtime_seconds": (
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
            "Derived the local parabolic-window forward, inverse-time, and "
            "cross Gramian restart laws; identified exact directional, "
            "tensor-spectral, and scalar-radial sufficient critical L3 "
            "moments; and stress-tested the hierarchy on Burgers-axis "
            "strain plus exact periodic shear and ABC Navier-Stokes flows. "
            "The scalar-radial route was quantitatively falsified as a "
            "credible primary closure target without overstating the "
            "surviving tensorial bridge."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The classical continuation bridge is exact, but the Clay-relevant "
        "estimate remains open: bound the common-path expectation of the "
        "exact directional cubic pullback, or the tensor-spectral Gramian "
        "weight, on shrinking parabolic windows using only energy-class "
        "Navier-Stokes information. The proof must preserve directional "
        "pairing and exploit pressure, vorticity, Leray, or two-point "
        "generator cancellation before taking an operator norm. It must "
        "then justify the inverse-time probe below the classical regime and "
        "exclude exceptional singular points. Scalar radial traces are only "
        "a diagnostic fallback. The older 64128-pivot certificate remains "
        "valid and secondary."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "Derive the coupled two-point/tangent generator for the exact "
        "directional cubic pullback and a smooth tensor-spectral proxy. "
        "Compute every drift term, including pressure and Leray structure, "
        "before applying absolute values. Seek or falsify a scale-critical "
        "integrated inequality from the Leray energy class, and stress any "
        "candidate first on exact shear, ABC, Burgers-vortex-type, and "
        "concentrated self-similar profiles. Do not launch a large numerical "
        "search until an analytic sign or cancellation target is explicit."
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
