"""Install the validated signed projected replica-generator checkpoint."""

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
    "signed_projected_replica_generator_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "signed_projected_replica_generator_audit.py",
    "work/ns_collision/tests/"
    "test_signed_projected_replica_generator.py",
    "work/ns_collision/notes/"
    "signed_projected_replica_generator.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_signed_projected_replica_bookmark.py",
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
        result.get("kind") == "signed_projected_replica_generator_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "signed_projected_replica_generator_derived_"
            "weighted_pressure_flux_open"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict)
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        "signed projected replica result is not the expected passing audit",
    )
    for key in (
        "projected_Weber_SPDE_derived",
        "rho_two_replica_local_balance_derived",
        "rho_global_cross_energy_balance_derived",
        "independent_endpoint_recovers_energy_equality",
        "Gaussian_chaos_correlation_homotopy_derived",
        "positive_rho_cross_gradient_dissipation_proved",
        "weighted_two_replica_identity_derived",
        "critical_L3_Legendre_representation_derived",
        "three_replica_tensor_generator_derived",
        "critical_weight_pressure_flux_can_be_nonzero",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "rho_cross_energy_has_universal_strain_sign",
        "unweighted_global_pressure_cancellation_closes_L3",
        "weighted_pressure_flux_bound_proved",
        "signed_replica_L3_bound_proved",
        "low_regularity_projected_replica_flow_justified",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    pressure = result["adversarial_pressure_stress"]
    _require(
        pressure.get("all_checks_pass") is True
        and pressure["critical_pressure_work_range"][0] > 40.5
        and pressure["critical_pressure_relative_spread"] < 1.0e-5
        and pressure["maximum_absolute_critical_convective_work"] < 3.1e-4,
        "critical pressure stress did not retain its resolved margin",
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
        principal.get(
            "projected_Weber_joint_tangent_covector_generator_derived"
        )
        is True
        and principal.get(
            "projected_Weber_signed_two_replica_identity_proved"
        )
        is True
        and principal.get(
            "projected_Weber_signed_replica_bound_proved"
        )
        is False,
        "the prerequisite projected-Weber checkpoint is not present",
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

    pressure_range = pressure["critical_pressure_work_range"]
    reset = result["rho_reset_stresses"]
    chaos = result["Gaussian_chaos_homotopy"]
    triple = result["three_replica_tensor_generator"]

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The signed projected replica stage now has an exact smooth "
        "rho-dependent two-replica generator. Independent replicas recover "
        "the Navier-Stokes energy equality, while reset decorrelation "
        "produces exactly 2nu(1-rho)||grad u||_2^2. A Wiener-chaos "
        "correlation homotopy connects |u|^2 to the pathwise projected "
        "second moment and proves G_rho>=|grad u|^2 for 0<=rho<=1, so the "
        "decorrelation term is genuinely dissipative. Exact weighted and "
        "three-replica formulations reach critical L3 without taking a "
        "positive Jensen moment. The critical lift remains open: the stored "
        "smooth periodic pressure adversary has pressure work in "
        f"[{pressure_range[0]:.12g},{pressure_range[1]:.12g}] while its "
        "resolved convective residual tends to zero, so unweighted global "
        "pressure cancellation does not close the nonconstant critical "
        "weight. No weighted pressure bound, signed L3 estimate, "
        "low-regularity passage, exceptional-set upgrade, or global "
        "regularity claim is made. The independent 64128-pivot checkpoint "
        "remains valid."
    )

    principal.update(
        {
            "signed_projected_replica_SPDE_derived": True,
            "signed_projected_replica_rho_local_balance_derived": True,
            "signed_projected_replica_rho_global_balance_derived": True,
            "signed_projected_replica_energy_endpoint_proved": True,
            "signed_projected_replica_chaos_homotopy_derived": True,
            "signed_projected_replica_cross_gradient_lower_bound_proved": (
                True
            ),
            "signed_projected_replica_weighted_identity_derived": True,
            "signed_projected_replica_critical_L3_dual_derived": True,
            "signed_projected_replica_triple_tensor_generator_derived": True,
            "signed_projected_replica_pressure_nonzero_stress_passed": True,
            "signed_projected_replica_universal_strain_sign": False,
            "signed_projected_replica_unweighted_pressure_closure": False,
            "signed_projected_replica_weighted_pressure_bound_proved": False,
            "signed_projected_replica_L3_bound_proved": False,
            "signed_projected_replica_low_regularity_flow_proved": False,
            "signed_projected_replica_exceptional_set_upgrade_proved": False,
            "signed_projected_replica_Navier_Stokes_regularity_proved": False,
            "signed_projected_replica_pressure_work_minimum": pressure_range[
                0
            ],
            "signed_projected_replica_pressure_work_maximum": pressure_range[
                1
            ],
            "signed_projected_replica_pressure_relative_spread": pressure[
                "critical_pressure_relative_spread"
            ],
            "signed_projected_replica_convective_residual_maximum": pressure[
                "maximum_absolute_critical_convective_work"
            ],
            "signed_projected_replica_shear_gradient_energy": reset[
                "periodic_shear"
            ]["normalized_gradient_energy"],
            "signed_projected_replica_abc_gradient_energy": reset[
                "abc_flow"
            ]["normalized_gradient_energy"],
            "signed_projected_replica_chaos_variance": (
                sum(chaos["chaos_energies"][1:])
            ),
            "signed_projected_replica_sample_correlation_minimum_eigenvalue": (
                min(triple["sample_correlation_eigenvalues"])
            ),
            "signed_projected_replica_targeted_test_count": (
                args.targeted_test_count
            ),
            "signed_projected_replica_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "signed_projected_replica_regression_test_count": (
                args.regression_test_count
            ),
            "signed_projected_replica_regression_test_runtime_seconds": (
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
            "Derived and audited the smooth signed projected Weber "
            "two-replica generator with correlated Brownian drivers, its "
            "exact energy endpoint, reset decorrelation law, Wiener-chaos "
            "homotopy, weighted critical dual, and three-replica tensor "
            "generator. A four-grid smooth-weight pressure stress falsifies "
            "naive promotion of global energy-level pressure cancellation "
            "to the critical weight without asserting a closure theorem."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The exact signed replica generator reaches critical scaling but "
        "does not yet control its weighted pressure and strain flux. A "
        "successful estimate must choose a backward, adapted, or partitioned "
        "lambda that keeps pressure and strain together before absolute "
        "values, quantitatively uses the proved lower bound "
        "G_rho>=|grad u|^2, and survives the stored periodic pressure "
        "adversary. The available cross-gradient coercivity has not yet been "
        "shown strong enough to pay the weighted pressure-strain flux. Any "
        "smooth estimate must then be justified for projected replicas at "
        "Leray regularity and upgraded across exceptional singular points. "
        "The older 64128-pivot finite certificate remains valid but "
        "independent."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "Derive the backward Euler-Lagrange or adjoint equation for lambda "
        "in the weighted rho-replica identity. Express its pressure term "
        "through the existing partition-flux edge antisymmetry and determine "
        "whether a joint pressure-strain cancellation can be exact or "
        "bounded at critical scaling using "
        "G_rho>=|grad u|^2. Stress every candidate on exact shear, ABC flow, "
        "Burgers strain, and the seed-81 periodic pressure field. Do not "
        "launch a large search until the signed analytic target and terminal "
        "cost are explicit."
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
