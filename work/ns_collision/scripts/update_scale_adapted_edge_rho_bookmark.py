"""Install the validated scale-adapted edge and rho-expansion checkpoint."""

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
    "scale_adapted_edge_rho_expansion_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "scale_adapted_edge_rho_expansion_audit.py",
    "work/ns_collision/tests/"
    "test_scale_adapted_edge_rho_expansion.py",
    "work/ns_collision/notes/"
    "scale_adapted_edge_rho_expansion.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_scale_adapted_edge_rho_bookmark.py",
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
    parser.add_argument(
        "--validation-mode",
        choices=("complete", "cpu_parked_incremental"),
        default="complete",
    )
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--regression-test-count", type=int, default=0)
    parser.add_argument(
        "--regression-test-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument("--discovered-test-count", type=int, default=0)
    parser.add_argument("--parked-test-seconds", type=float, default=0.0)
    parser.add_argument("--peak-sampled-cpu", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require(args.targeted_test_count > 0, "targeted count must be positive")
    _require(args.targeted_test_seconds >= 0.0, "invalid targeted runtime")
    if args.validation_mode == "complete":
        _require(
            args.regression_test_count > 0,
            "complete regression count must be positive",
        )
        _require(
            args.regression_test_seconds >= 0.0,
            "invalid regression runtime",
        )
    else:
        _require(
            args.regression_test_count == 0,
            "parked mode cannot claim a completed regression",
        )
        _require(
            args.discovered_test_count
            == args.targeted_test_count + 159,
            "parked mode requires the inherited 159 plus new targeted tests",
        )
        _require(
            args.parked_test_seconds > 0.0,
            "parked mode requires the interrupted runtime",
        )
        _require(
            args.peak_sampled_cpu > 75.0,
            "parked mode requires the measured CPU threshold breach",
        )
    for artifact in ARTIFACTS:
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")

    result = _load_json(RESULT)
    flags = result.get("certification_flags")
    checks = result.get("positive_checks")
    _require(
        result.get("kind") == "scale_adapted_edge_rho_expansion_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "fixed_scale_edge_absorption_falsified_"
            "short_time_positive_rho_advantage_excluded"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict)
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        "scale-adapted edge/rho result is not the expected passing audit",
    )
    for key in (
        "scale_adapted_edge_identity_derived",
        "edge_Young_ratio_equals_local_Reynolds_squared",
        "fixed_scale_universal_edge_absorption_falsified_by_scaling",
        "partition_frequency_must_track_local_amplitude",
        "first_short_time_rho_correction_derived",
        "positive_rho_is_worse_on_sufficiently_short_restart_windows",
        "replica_pressure_linearization_implemented",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "scale_adapted_edge_remainder_absorbed",
        "finite_time_positive_rho_advantage_proved",
        "Taylor_crossover_is_a_sign_certificate",
        "critical_signed_replica_bound_proved",
        "low_regularity_scale_adapted_partition_justified",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    sweep = result["partition_frequency_sweep"]
    expansion = result["short_time_rho_expansion"]
    _require(
        sweep.get("all_checks_pass") is True
        and sweep["positive_pressure_frequencies"] == [1, 2, 3, 4, 5, 6]
        and sweep["spectrally_silent_frequencies"]
        == [7, 8, 9, 10, 11, 12],
        "partition-frequency sweep lost its resolved structure",
    )
    _require(
        expansion.get("all_checks_pass") is True
        and expansion["leading_reset_loss_range"][0] > 2361.35
        and expansion["leading_reset_loss_range"][1] < 2361.36
        and expansion["first_time_coefficient_range"][0] > -62459.7
        and expansion["first_time_coefficient_range"][1] < -62459.6,
        "short-time rho expansion lost its resolved margins",
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
            "adjoint_replica_restart_dual_inequality_derived"
        )
        is True
        and principal.get(
            "adjoint_replica_conditional_Fisher_edge_identity_derived"
        )
        is True
        and principal.get("adjoint_replica_critical_bound_proved")
        is False,
        "the prerequisite adjoint replica checkpoint is absent",
    )
    if args.validation_mode == "cpu_parked_incremental":
        _require(
            principal.get("adjoint_replica_regression_test_count") == 159,
            "the inherited 159-test full regression is not recorded",
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

    leading_range = expansion["leading_reset_loss_range"]
    coefficient_range = expansion["first_time_coefficient_range"]
    crossover_values = [
        row["formal_integrated_crossover_time"]
        for row in expansion["rows"]
    ]
    frequency_rows = sweep["rows"]

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = (
        "checkpointed"
        if args.validation_mode == "complete"
        else "parked"
    )
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The scale-adapted edge stage derives the exact local-Reynolds "
        "homogeneity of the adjoint pressure gate. Exact pressure/Fisher "
        "balance scales as Re_cell=a/(nu m), while the separated Young "
        "remainder scales as Re_cell^2; fixed-scale universal absorption is "
        "therefore impossible and partition frequency must track amplitude. "
        "On the finite-mode seed-81 stress, frequencies 1 through 6 couple "
        "to pressure and 7 through 12 are spectrally silent, which is useful "
        "diagnostic evidence but not a pressure-tail theorem. The complete "
        "first short-time rho expansion has reset loss in "
        f"[{leading_range[0]:.12g},{leading_range[1]:.12g}] and first time "
        "coefficient in "
        f"[{coefficient_range[0]:.12g},{coefficient_range[1]:.12g}]. "
        "Thus rho>0 is strictly worse on every sufficiently short restart "
        "window. Its quadratic truncation crosses near "
        f"{crossover_values[0]:.12g}, but this is not a sign certificate. "
        "Intrinsic-scale pressure tails, a nonperturbative finite-window "
        "rho advantage, the critical signed bound, low-regularity passage, "
        "exceptional-set upgrade, and global regularity all remain open. "
        + (
            "The complete 164-test regression passed."
            if args.validation_mode == "complete"
            else (
                "The prior 159-test full regression and all 5 new targeted "
                "tests pass, but the one-shot 164-test run was parked after "
                f"{args.parked_test_seconds:.3f}s when sampled CPU reached "
                f"{args.peak_sampled_cpu:.2f}%; it remains a required "
                "operational validation."
            )
        )
        + " The independent 64128-pivot checkpoint remains valid."
    )

    principal.update(
        {
            "scale_edge_frequency_identity_derived": True,
            "scale_edge_exact_ratio_is_local_Reynolds": True,
            "scale_edge_Young_ratio_is_local_Reynolds_squared": True,
            "scale_edge_fixed_scale_absorption_falsified": True,
            "scale_edge_frequency_tracks_amplitude_required": True,
            "scale_edge_seed81_frequency_sweep_passed": True,
            "scale_edge_pressure_tail_bound_proved": False,
            "scale_edge_remainder_absorbed": False,
            "rho_short_time_first_correction_derived": True,
            "rho_short_time_positive_correlation_worse_proved": True,
            "rho_replica_pressure_linearization_implemented": True,
            "rho_finite_window_advantage_proved": False,
            "rho_Taylor_crossover_certified": False,
            "scale_rho_critical_signed_bound_proved": False,
            "scale_rho_low_regularity_partition_proved": False,
            "scale_rho_exceptional_set_upgrade_proved": False,
            "scale_rho_Navier_Stokes_regularity_proved": False,
            "scale_edge_frequency_one_Young_remainder": frequency_rows[0][
                "scale_adapted_young_remainder"
            ],
            "scale_edge_frequency_six_Young_remainder": frequency_rows[5][
                "scale_adapted_young_remainder"
            ],
            "scale_edge_frequency_one_absorption_amplitude": (
                frequency_rows[0]["edge_absorption_amplitude_threshold"]
            ),
            "scale_edge_frequency_six_absorption_amplitude": (
                frequency_rows[5]["edge_absorption_amplitude_threshold"]
            ),
            "rho_short_time_reset_loss": sum(leading_range) / 2.0,
            "rho_short_time_first_time_coefficient": (
                sum(coefficient_range) / 2.0
            ),
            "rho_short_time_formal_integrated_crossover": (
                sum(crossover_values) / len(crossover_values)
            ),
            "scale_rho_targeted_test_count": args.targeted_test_count,
            "scale_rho_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "scale_rho_regression_test_count": args.regression_test_count,
            "scale_rho_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "scale_rho_discovered_test_count": (
                args.discovered_test_count
                if args.discovered_test_count > 0
                else args.regression_test_count
            ),
            "scale_rho_monolithic_regression_passed": (
                args.validation_mode == "complete"
            ),
            "scale_rho_incremental_prior_full_test_count": (
                159
                if args.validation_mode == "cpu_parked_incremental"
                else 0
            ),
            "scale_rho_cpu_parked_test_runtime_seconds": (
                args.parked_test_seconds
            ),
            "scale_rho_cpu_parked_peak_sample_percent": (
                args.peak_sampled_cpu
            ),
            "scale_rho_validation_mode": args.validation_mode,
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Derived the exact frequency-dependent pressure-edge/Fisher "
            "identity and proved that its Young ratio is the square of the "
            "local Reynolds number, ruling out fixed-scale universal "
            "absorption. A twelve-frequency seed-81 sweep records resolved "
            "scale improvement. The complete first short-time rho "
            "coefficient, including linearized replica pressure, proves "
            "positive correlation remains worse on sufficiently short "
            "restart windows and locates, without certifying, the next "
            "finite-window diagnostic."
        ),
    )
    theorem_obligation = (
        "Fixed-scale edge absorption and infinitesimal positive-correlation "
        "improvement are both excluded. The live edge obligation is a "
        "pressure-tail estimate uniform when partition frequency tracks "
        "local amplitude through Re_cell=a/(nu m), including adaptation, "
        "overlap, zero-face degeneracy, and preservation of the terminal "
        "dual supremum. The live replica obligation is a nonperturbative "
        "finite-window comparison near h=0.0756 that includes the terminal "
        "variance majorization and proves, rather than Taylor-extrapolates, "
        "any rho advantage. Low-regularity and exceptional-set gates remain."
    )
    if args.validation_mode == "complete":
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
        bookmark["next_action"] = (
            "Build a deterministic finite-window solver for the two-replica "
            "correlation tensors on the seed-81 smooth Navier-Stokes "
            "trajectory over h around 0.0756, with rho=0 as the exact "
            "baseline and explicit terminal variance tax. Separately "
            "decompose pressure above the intrinsic frequency "
            "m approximately amplitude/nu and derive or falsify a uniform "
            "tail payment. Do not infer a rho sign change from the "
            "quadratic Taylor diagnostic."
        )
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation is parked: the one-shot 164-test suite "
            "must pass when the daytime baseline CPU is at most 60%. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
        bookmark["next_action"] = (
            "Sample total CPU for at least five seconds. If the daytime "
            "average is at most 60%, rerun the parked 164-test command below "
            "normal priority, then reinstall this checkpoint in complete "
            "mode and run the split validator. After that, build the "
            "finite-window rho solver near h=0.0756 and the intrinsic-scale "
            "pressure-tail audit."
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
