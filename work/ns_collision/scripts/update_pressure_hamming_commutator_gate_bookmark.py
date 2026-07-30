"""Install the pressure-Hamming commutator gate checkpoint."""

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
PRIOR_RESULT = (
    "work/ns_collision/results/"
    "high_carrier_weighted_fisher_gate_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "a533faec71e4941e6a1dc5458199e5684cf750db61df2be2823c04a6a3a7c5be"
)
RESULT = (
    "work/ns_collision/results/"
    "pressure_hamming_commutator_gate_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "pressure_hamming_commutator_gate_audit.py",
    "work/ns_collision/tests/"
    "test_pressure_hamming_commutator_gate.py",
    "work/ns_collision/notes/"
    "pressure_hamming_commutator_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_pressure_hamming_commutator_gate_bookmark.py",
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


def _strictly_increasing(
    rows: list[dict[str, Any]],
    field: str,
) -> bool:
    values = [float(row[field]) for row in rows]
    return all(
        first < second for first, second in zip(values, values[1:])
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-mode",
        choices=("complete", "inherited_cpu_parked_incremental"),
        default="inherited_cpu_parked_incremental",
    )
    parser.add_argument(
        "--resource-mode",
        choices=("daytime_policy", "user_authorized_late"),
        required=True,
    )
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, required=True)
    parser.add_argument("--regression-test-count", type=int, default=0)
    parser.add_argument("--regression-test-seconds", type=float, default=0.0)
    parser.add_argument("--baseline-average", type=float)
    parser.add_argument("--baseline-peak", type=float)
    return parser.parse_args()


def _validate_result() -> dict[str, Any]:
    result = _load_json(RESULT)
    checks = result.get("positive_checks")
    flags = result.get("certification_flags")
    _require(
        result.get("kind")
        == "pressure_hamming_commutator_gate_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "lower_carrier_diagonal_commutator_falsified_"
            "hamming_leakage_certified"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(flags, dict),
        "pressure-Hamming result is not the expected audit",
    )
    for key in (
        "exact_eight_shift_Walsh_identity_proved",
        "distance_two_and_three_pressure_leakage_genuine",
        "coupled_eight_cell_multiplier_bound_proved",
        "lower_carrier_only_diagonal_commutator_bound_falsified",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "intrinsic_amplitude_condition_repairs_diagonal_bound",
        "annular_shell_diagonal_commutator_bound_proved",
        "mixed_low_high_paraproduct_controlled",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    walsh = result["exact_vertex_Walsh_calculus"]
    leakage = result["Hamming_leakage_bound"]
    theorem = result["two_scale_counterexample_theorem"]
    pilot = result["finite_Fourier_counterexample_pilot"]
    _require(
        walsh.get("all_checks_pass") is True
        and walsh.get("distance_two_terms_are_genuine") is True
        and walsh.get("distance_three_term_is_genuine") is True
        and walsh.get(
            "phase_corrected_Fourier_identity_residual",
            1.0,
        )
        < 1.0e-15
        and walsh.get("nonzero_masks_by_hamming_order", {}).get("2")
        == [3, 5, 6]
        and walsh.get("nonzero_masks_by_hamming_order", {}).get("3")
        == [7],
        "exact vertex Walsh certificate changed",
    )
    _require(
        leakage.get("all_checks_pass") is True
        and abs(
            float(leakage.get("bounded_multiplier_matrix_norm", 0.0))
            - 8.0
        )
        < 1.0e-12,
        "Hamming leakage certificate changed",
    )
    _require(
        theorem.get("all_checks_pass") is True
        and theorem.get("fixed_minimum_velocity_mode")
        == "sqrt(3)*(2+1)=3*sqrt(3)"
        and theorem.get("proved_asymptotic_bounds", {}).get(
            "diagonal_ratio_lower"
        )
        == "ratio>=c N^(1/2)",
        "two-scale counterexample theorem changed",
    )
    rows = pilot.get("rows", [])
    _require(
        pilot.get("all_checks_pass") is True
        and [row.get("order") for row in rows]
        == [4, 6, 8, 10, 12, 14]
        and all(
            row.get("all_checks_pass") is True
            and abs(
                row.get("minimum_velocity_mode", 0.0)
                - row.get("expected_minimum_velocity_mode", 1.0)
            )
            < 1.0e-10
            and row.get("maximum_product_coordinate_mode_upper", 1.0)
            < row.get("nyquist", 0.0)
            and row.get("maximum_relative_divergence_residual", 1.0)
            < 1.0e-10
            for row in rows
        )
        and _strictly_increasing(rows, "diagonal_commutator_ratio")
        and _strictly_increasing(rows, "N_cubed_weighted_pressure")
        and pilot.get("observed_ratio_growth_factor", 0.0) > 50.0,
        "finite-Fourier counterexample pilot changed",
    )
    return result


def main() -> None:
    args = _parse_args()
    _require(
        args.targeted_test_count == 6,
        "this checkpoint requires exactly six focused tests",
    )
    _require(args.targeted_test_seconds > 0.0, "invalid test runtime")
    _require(
        args.discovered_test_count == 207,
        "expected 201 inherited tests plus six new tests",
    )
    _require(
        1 <= args.worker_count <= 2,
        "this checkpoint permits at most two Python workers",
    )
    if args.resource_mode == "daytime_policy":
        _require(
            args.baseline_average is not None
            and args.baseline_peak is not None
            and 0.0
            <= args.baseline_average
            <= args.baseline_peak
            and args.baseline_average <= 60.0,
            "daytime validation requires a permitted CPU baseline",
        )
    elif (
        args.baseline_average is not None
        or args.baseline_peak is not None
    ):
        _require(
            args.baseline_average is not None
            and args.baseline_peak is not None
            and 0.0
            <= args.baseline_average
            <= args.baseline_peak,
            "partial or invalid optional CPU sample",
        )

    if args.validation_mode == "complete":
        _require(
            args.regression_test_count == args.discovered_test_count
            and args.regression_test_seconds > 0.0,
            "complete mode requires the full regression",
        )
    else:
        _require(
            args.regression_test_count == 0
            and args.regression_test_seconds == 0.0,
            "incremental mode cannot claim a full regression",
        )

    for artifact in ARTIFACTS:
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")
    result = _validate_result()
    _require(
        _sha256(PRIOR_RESULT) == PRIOR_RESULT_SHA256,
        "the prerequisite high-carrier result changed",
    )
    prior = _load_json(PRIOR_RESULT)
    _require(
        prior.get("kind") == "high_carrier_weighted_fisher_gate_audit"
        and prior.get("all_positive_checks_pass") is True,
        "the prerequisite high-carrier audit is invalid",
    )

    bookmark = _load_json(BOOKMARK)
    _require(
        bookmark.get("kind")
        == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    principal = bookmark.setdefault("principal_results", {})
    _require(
        bookmark.get("status") == "parked"
        and principal.get("high_carrier_targeted_test_count") == 6
        and principal.get("high_carrier_discovered_test_count") == 201
        and principal.get("high_carrier_monolithic_regression_passed")
        is False
        and principal.get(
            "high_carrier_weighted_fisher_gate_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256,
        "the prerequisite high-carrier checkpoint changed",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The vertex pressure commutator has an exact eight-shift Walsh "
        "formula whose derivative order is Hamming distance on the "
        "partition cube. Concrete double-Riesz arithmetic has genuine "
        "distance-two and distance-three coefficients, and the resulting "
        "coupled eight-cell multiplier bound is certified. The proposed "
        "bandwidth-independent single-vertex estimate is false under a "
        "lower carrier cutoff alone. An exact divergence-free curl-Fejer "
        "family keeps minimum mode 3sqrt(3) and bounded amplitude while "
        "its bandwidth concentrates at a triple partition zero; its fixed "
        "smooth high-output pressure norm is bounded below by cN^(-3), "
        "while the proposed weighted denominator is O(N^(-7/2)), so the "
        "ratio grows at least as N^(1/2). Six alias-free finite-Fourier "
        "rows independently show a 69-fold ratio increase with divergence "
        "residual below 3e-15. Six focused replay tests pass. One Python "
        "worker was used throughout under the recorded resource mode. "
        "Annular shell control, paraproduct summation, the critical signed "
        "bound, and Navier-Stokes regularity remain open."
    )

    principal.update(
        {
            "pressure_Hamming_exact_Walsh_identity_proved": True,
            "pressure_Hamming_distance_two_three_genuine": True,
            "pressure_Hamming_coupled_multiplier_bound_proved": True,
            "pressure_Hamming_lower_carrier_diagonal_bound_falsified": True,
            "pressure_Hamming_intrinsic_amplitude_repair_falsified": True,
            "pressure_Hamming_annular_shell_bound_proved": False,
            "pressure_Hamming_mixed_paraproduct_controlled": False,
            "pressure_Hamming_critical_signed_bound_proved": False,
            "pressure_Hamming_Navier_Stokes_regularity_proved": False,
            "pressure_Hamming_targeted_test_count": (
                args.targeted_test_count
            ),
            "pressure_Hamming_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "pressure_Hamming_discovered_test_count": (
                args.discovered_test_count
            ),
            "pressure_Hamming_regression_test_count": (
                args.regression_test_count
            ),
            "pressure_Hamming_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "pressure_Hamming_monolithic_regression_passed": complete,
            "pressure_Hamming_resource_mode": args.resource_mode,
            "pressure_Hamming_worker_count": args.worker_count,
            "pressure_Hamming_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "pressure_Hamming_cpu_baseline_peak_percent": (
                args.baseline_peak
            ),
            "pressure_Hamming_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Derived the exact eight-shift pressure multiplier calculus and "
            "its Hamming-cube leakage bound, then falsified the "
            "lower-carrier-only diagonal commutator estimate with an exact "
            "divergence-free two-scale curl-Fejer family while preserving "
            "annular and signed coupled-cell routes."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is the annular finite-type "
        "commutator gate. For K<=|k|<=Lambda K, prove or falsify uniform "
        "control of the distance-two and distance-three vertex masses by "
        "||psi_v u|| plus K^(-1)||u grad psi_v||, with explicit dependence "
        "on Lambda. Insert that result into the exact Walsh formula and "
        "only then treat comparable-shell and separated-shell pressure "
        "paraproducts. Critical signed control, low-regularity passage, "
        "and exceptional-set removal remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 207-test "
            "suite must pass in an admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Formulate the annular vertex uncertainty inequality on the "
        "half-lattice, first for one tensor sine/cosine factor and then for "
        "the triple-zero product. Seek a uniform constant for shell ratio "
        "Lambda=2, stress it against optimized shell packets, and combine "
        "it with the exact distance-two/distance-three Walsh coefficients."
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
