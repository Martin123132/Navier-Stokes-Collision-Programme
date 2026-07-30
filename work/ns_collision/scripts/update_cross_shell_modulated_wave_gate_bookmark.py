"""Install the cross-shell modulated-wave no-go checkpoint."""

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
    "self_shell_pressure_closure_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "2077e3d082e0b0b2096eb6c0c19583b8c5915b05d5e35b40aa1119ebc75ef086"
)
RESULT = (
    "work/ns_collision/results/"
    "cross_shell_modulated_wave_gate_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "cross_shell_modulated_wave_gate_audit.py",
    "work/ns_collision/tests/"
    "test_cross_shell_modulated_wave_gate.py",
    "work/ns_collision/notes/"
    "cross_shell_modulated_wave_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_cross_shell_modulated_wave_gate_bookmark.py",
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
        result.get("kind") == "cross_shell_modulated_wave_gate_audit"
        and result.get("schema_version") == 1
        and result.get("status") == "cross_shell_carrier_decay_falsified"
        and result.get("all_positive_checks_pass") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(flags, dict),
        "cross-shell result is not the expected audit",
    )
    for key in (
        "pressure_only_cross_shell_H_decay_falsified",
        "complete_signed_HHL_flux_H_decay_falsified",
        "cross_pressure_leading_order_cancellation_proved",
        "anisotropic_Reynolds_stress_survives_in_flux",
        "self_shell_pressure_closure_preserved",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "dyadic_amplitude_summation_proved",
        "inter_shell_telescoping_proved",
        "time_integrated_compensation_proved",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    theorem = result["analytic_no_go"]
    exact = theorem.get("exact_rational_checks", {})
    _require(
        theorem.get("all_checks_pass") is True
        and exact.get("all_checks_pass") is True
        and exact.get("q_plus_k_plus_r") == [0, 0, 0]
        and exact.get("k_dot_C") == "0"
        and exact.get("r_dot_C") == "2/9"
        and exact.get("limiting_kinetic_vertex_pairing") == "0"
        and exact.get("pressure_load_limit") == "1/144"
        and exact.get("anisotropic_flux_load_limit") == "1/144"
        and "alpha>0" in theorem.get("falsified_statement", ""),
        "analytic modulated-wave certificate changed",
    )

    replay = result["finite_mode_asymptotic_replay"]
    rows = replay.get("rows", [])
    _require(
        replay.get("all_checks_pass") is True
        and replay.get("carriers")
        == [8, 16, 32, 64, 128, 256, 512, 1024]
        and len(rows) == 8
        and all(
            row.get("all_checks_pass") is True
            and float(row["high_shell_ratio"]) < 1.02
            and abs(
                float(row["low_pressure_coefficient_at_q"])
                + float(row["exact_q_dot_first_polarization"])
            )
            < 1.0e-13
            and abs(
                float(row["combined_HHL_load"])
                - float(row["direct_polynomial_linear_load"])
            )
            < 1.0e-12
            and float(row["component_vs_direct_flux_residual"])
            < 1.0e-12
            and float(row["maximum_divergence_residual"]) < 1.0e-12
            for row in rows
        )
        and float(replay["pressure_last_over_limit"]) > 0.999
        and float(replay["combined_last_over_limit"]) > 0.999
        and float(replay["maximum_H_times_kinetic_load"]) < 0.02
        and float(
            replay["maximum_H_squared_times_cross_pressure_load"]
        )
        < 0.25,
        "finite-mode cross-shell replay changed",
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
        args.discovered_test_count == 225,
        "expected 219 inherited tests plus six new tests",
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
        "the prerequisite self-shell result changed",
    )
    prior = _load_json(PRIOR_RESULT)
    _require(
        prior.get("kind") == "self_shell_pressure_closure_audit"
        and prior.get("all_positive_checks_pass") is True,
        "the prerequisite self-shell audit is invalid",
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
        and principal.get("self_shell_targeted_test_count") == 6
        and principal.get("self_shell_discovered_test_count") == 219
        and principal.get("self_shell_monolithic_regression_passed")
        is False
        and principal.get(
            "self_shell_pressure_closure_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256,
        "the prerequisite self-shell checkpoint changed",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "An exact divergence-free two-sideband family now settles the "
        "first cross-shell gate negatively. High modes (1,1,H) and "
        "(0,0,H) generate a fixed low Reynolds stress converging to "
        "2e_1 tensor e_1. Against a fixed low velocity and the all-cosine "
        "partition vertex, both the pressure-only HHL load and the "
        "complete signed cubic HHL local-energy flux converge to 1/144. "
        "The isotropic kinetic term cancels the pressure scalar, but the "
        "anisotropic Reynolds stress survives; cross pressure decays. "
        "An independent cubic-polynomial reconstruction agrees with the "
        "assembled components below 2.3e-16 through H=1024. Exact rational "
        "checks certify the resonance, incompressibility, cancellation, "
        "and surviving limit. Six focused tests pass with one Python "
        "worker. Carrier-separation decay is falsified, while dyadic "
        "amplitude summation, conservative telescoping, time integration, "
        "critical closure, and regularity remain open."
    )

    principal.update(
        {
            "cross_shell_pressure_H_decay_falsified": True,
            "cross_shell_complete_HHL_flux_H_decay_falsified": True,
            "cross_shell_anisotropic_Reynolds_stress_survives": True,
            "cross_shell_dyadic_amplitude_sum_proved": False,
            "cross_shell_telescoping_proved": False,
            "cross_shell_time_compensation_proved": False,
            "cross_shell_critical_signed_bound_proved": False,
            "cross_shell_Navier_Stokes_regularity_proved": False,
            "cross_shell_targeted_test_count": args.targeted_test_count,
            "cross_shell_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "cross_shell_discovered_test_count": (
                args.discovered_test_count
            ),
            "cross_shell_regression_test_count": (
                args.regression_test_count
            ),
            "cross_shell_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "cross_shell_monolithic_regression_passed": complete,
            "cross_shell_resource_mode": args.resource_mode,
            "cross_shell_worker_count": args.worker_count,
            "cross_shell_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "cross_shell_cpu_baseline_peak_percent": args.baseline_peak,
            "cross_shell_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Constructed an exact modulated two-sideband family and "
            "proved that neither cross-shell pressure alone nor the "
            "complete instantaneous signed high-high-low local-energy "
            "flux has a universal positive carrier-separation gain."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is the exact dyadic three-shell "
        "interaction atlas. Retain shell amplitudes and use the fact that "
        "the two largest occupied frequencies are comparable. Remove the "
        "self-shell pressure terms already closed, then express the "
        "surviving O(1) HHL Reynolds-stress channel as signed transfer "
        "between scale boundaries. Test conservative telescoping in shell "
        "index, eight-cell cancellation, and time-integrated viscous "
        "payment before taking absolute values. Critical closure, "
        "low-regularity passage, and exceptional-set removal remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 225-test "
            "suite must pass in an admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Derive the exact Littlewood-Paley trilinear local-energy identity "
        "indexed by pressure-input shells and testing shell. Prove the "
        "largest-two-comparable support rule, identify all permutations "
        "of the HHL channel, and compute their signed transfer symmetry. "
        "Stress telescoping on the exact two-sideband family before "
        "proposing any absolute dyadic estimate."
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
