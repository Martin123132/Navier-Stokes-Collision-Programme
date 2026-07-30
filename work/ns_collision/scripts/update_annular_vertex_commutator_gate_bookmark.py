"""Install the annular vertex commutator gate checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BOOKMARK = ROOT / "work/ns_collision/results/session_bookmark.json"
PRIOR_RESULT = (
    "work/ns_collision/results/"
    "pressure_hamming_commutator_gate_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "9bffa20e16b9f1831df682cb601545508a492952317a102b082823a7006bf9da"
)
RESULT = (
    "work/ns_collision/results/"
    "annular_vertex_commutator_gate_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "annular_vertex_commutator_gate_audit.py",
    "work/ns_collision/tests/"
    "test_annular_vertex_commutator_gate.py",
    "work/ns_collision/notes/"
    "annular_vertex_commutator_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_annular_vertex_commutator_gate_bookmark.py",
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
        result.get("kind") == "annular_vertex_commutator_gate_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == "annular_diagonal_pressure_commutator_certified"
        and result.get("all_positive_checks_pass") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(flags, dict),
        "annular result is not the expected audit",
    )
    for key in (
        "sharp_residue_chain_toggle_inequality_proved",
        "annular_Hamming_leakage_collapse_proved",
        "single_vertex_annular_high_output_pressure_bound_proved",
        "single_vertex_annular_intrinsic_absorption_proved",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "sharp_high_output_cutoff_supported",
        "low_output_high_high_beat_controlled",
        "cross_shell_paraproduct_summation_proved",
        "mixed_low_high_paraproduct_controlled",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    toggle = result["residue_chain_toggle_theorem"]
    rows = toggle.get("rows", [])
    _require(
        toggle.get("all_checks_pass") is True
        and len(rows) == 7
        and all(
            row.get("all_checks_pass") is True
            and row.get("maximum_chain_length")
            == row.get("expected_maximum_chain_length")
            and abs(
                float(row["numerical_sum_over_difference_norm"])
                - float(row["sharp_toggle_constant"])
            )
            < 1.0e-11
            and abs(
                float(row["numerical_difference_over_sum_norm"])
                - float(row["sharp_toggle_constant"])
            )
            < 1.0e-11
            and float(row["sharp_toggle_constant"])
            < float(row["cotangent_upper_bound"])
            for row in rows
        ),
        "residue-chain toggle certificate changed",
    )

    hamming = result["tensor_Hamming_collapse"]
    constant = float(hamming.get("example_toggle_constant", 0.0))
    ratios = hamming.get("extremizing_tensor_ratios", {})
    _require(
        hamming.get("all_checks_pass") is True
        and all(
            abs(float(ratios[str(distance)]) - constant**distance)
            < 1.0e-11
            for distance in range(4)
        ),
        "tensor Hamming certificate changed",
    )

    theorem = result["annular_pressure_commutator_theorem"]
    _require(
        theorem.get("all_checks_pass") is True
        and abs(
            float(theorem.get("Lambda_equals_2_theta_upper", 0.0))
            - 6.0 / math.pi
        )
        < 1.0e-14
        and float(theorem["Lambda_equals_2_theta_upper"]) < 2.0
        and theorem.get("validity_threshold") == "K>sqrt(3)m"
        and "low-output" in theorem.get("scope", ""),
        "annular pressure theorem certificate changed",
    )

    stress = result["shellized_counterexample_stress"]
    stress_rows = stress.get("rows", [])
    stress_ratios = [
        float(row["diagonal_commutator_ratio"]) for row in stress_rows
    ]
    _require(
        stress.get("all_checks_pass") is True
        and [row.get("order") for row in stress_rows] == [3, 4, 5, 6, 7]
        and all(
            row.get("all_checks_pass") is True
            and float(row["radial_shell_ratio"]) < 2.0
            and float(row["maximum_product_coordinate_mode_upper"])
            < float(row["nyquist"])
            and float(row["maximum_relative_divergence_residual"])
            < 1.0e-10
            for row in stress_rows
        )
        and all(
            first > second
            for first, second in zip(stress_ratios, stress_ratios[1:])
        )
        and float(stress["ratio_variation_factor"]) < 1.2,
        "shellized counterexample stress changed",
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
        args.discovered_test_count == 213,
        "expected 207 inherited tests plus six new tests",
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
        "the prerequisite pressure-Hamming result changed",
    )
    prior = _load_json(PRIOR_RESULT)
    _require(
        prior.get("kind") == "pressure_hamming_commutator_gate_audit"
        and prior.get("all_positive_checks_pass") is True,
        "the prerequisite pressure-Hamming audit is invalid",
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
        and principal.get("pressure_Hamming_targeted_test_count") == 6
        and principal.get("pressure_Hamming_discovered_test_count") == 207
        and principal.get(
            "pressure_Hamming_monolithic_regression_passed"
        )
        is False
        and principal.get(
            "pressure_hamming_commutator_gate_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256,
        "the prerequisite pressure-Hamming checkpoint changed",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "A sharp residue-chain theorem now compares sine and cosine "
        "half-frequency weights on coordinate degree L with constant "
        "cot(pi/[2(ceil((2L+1)/m)+1)]). Tensorization collapses all "
        "Hamming-distance vertex leakage. On K<=|k|<=Lambda K, the "
        "multiplier factor m/K cancels the chain length O(K/m), giving "
        "theta<=2(Lambda+1)/pi. Inserted into the exact eight-shift "
        "Walsh identity, this proves a floor-free single-vertex bound and "
        "an explicit intrinsic absorption condition for the smooth "
        "high-output pressure of one shell. Five alias-free shellized "
        "curl-Fejer rows remain in one radial annulus, have divergence "
        "residual below 2.4e-15, and show a decreasing diagonal ratio. "
        "Six focused replay tests pass. One Python worker was used. A "
        "sharp cutoff, the low-output high-high beat, shell interaction "
        "sums, the critical signed bound, and regularity remain open."
    )

    principal.update(
        {
            "annular_vertex_sharp_residue_chain_toggle_proved": True,
            "annular_vertex_Hamming_collapse_proved": True,
            "annular_vertex_smooth_high_output_bound_proved": True,
            "annular_vertex_intrinsic_component_absorption_proved": True,
            "annular_vertex_sharp_cutoff_supported": False,
            "annular_vertex_low_output_beat_controlled": False,
            "annular_vertex_cross_shell_sum_proved": False,
            "annular_vertex_critical_signed_bound_proved": False,
            "annular_vertex_Navier_Stokes_regularity_proved": False,
            "annular_vertex_targeted_test_count": args.targeted_test_count,
            "annular_vertex_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "annular_vertex_discovered_test_count": (
                args.discovered_test_count
            ),
            "annular_vertex_regression_test_count": (
                args.regression_test_count
            ),
            "annular_vertex_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "annular_vertex_monolithic_regression_passed": complete,
            "annular_vertex_resource_mode": args.resource_mode,
            "annular_vertex_worker_count": args.worker_count,
            "annular_vertex_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "annular_vertex_cpu_baseline_peak_percent": args.baseline_peak,
            "annular_vertex_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Proved the sharp residue-chain sine/cosine toggle theorem, "
            "tensorized it across Hamming vertices, and used the annular "
            "m/K cancellation to close the smooth high-output pressure "
            "commutator of one shell at a single zero-face vertex."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is the low-output high-high beat of "
        "one annular shell. Retain the signed eight-cell pressure load "
        "before absolute values and determine whether incompressibility, "
        "near-opposite carrier geometry, and pressure-load conservation "
        "give a summable gain. Only after that gate should comparable, "
        "separated, and mixed shell paraproducts be assembled. The sharp "
        "cutoff, critical signed bound, low-regularity passage, and "
        "exceptional-set removal remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 213-test "
            "suite must pass in an admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Write the one-shell low-output pressure as a near-opposite-pair "
        "Fourier bilinear form. First derive the exact divergence-free "
        "symbol at output q<<K and its q/K cancellation order. Then test "
        "whether the signed eight-cell load retains that gain after the "
        "partition shifts; seek a counterexample before attempting shell "
        "summation."
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
