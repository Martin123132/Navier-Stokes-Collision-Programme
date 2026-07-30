"""Install the nonlinear stress-regeneration gate checkpoint."""

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
    "joint_scale_cell_viscous_occupation_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "ab47bc3bbf35a7296471cae8ec1514e475ad42c38b0d547991d76a3047873e0d"
)
RESULT = (
    "work/ns_collision/results/"
    "nonlinear_stress_regeneration_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "a78a6064db7e8c94e6fbbd9cc85469ec505fe4070e5b88d21f968120e20858e1"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "nonlinear_stress_regeneration_gate_audit.py",
    "work/ns_collision/tests/"
    "test_nonlinear_stress_regeneration_gate.py",
    "work/ns_collision/notes/"
    "nonlinear_stress_regeneration_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_nonlinear_stress_regeneration_gate_bookmark.py",
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
    _require(_sha256(RESULT) == RESULT_SHA256, "result hash changed")
    _require(
        result.get("kind")
        == "nonlinear_stress_regeneration_gate_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "HHL_regeneration_commutator_certified_"
            "HHH_pressure_strain_obstruction_exhibited"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(flags, dict),
        "result is not the expected regeneration audit",
    )
    for key in (
        "exact_projected_stress_evolution_derived",
        "HHL_leading_carrier_terms_cancel_proved",
        "HHL_regeneration_low_factor_bound_proved",
        "HHH_anisotropic_pressure_strain_carrier_witness_proved",
        "pointwise_energy_only_forcing_bound_falsified_sparse",
        "sparse_parabolic_forcing_summability_proved",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "all_regeneration_low_factor_bound_proved",
        "dense_packet_multiplicity_control_proved",
        "full_Navier_Stokes_regeneration_norm_from_Leray_proved",
        "critical_signed_large_data_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    evolution = result["projected_stress_evolution"]
    _require(
        evolution.get("all_checks_pass") is True
        and evolution.get("test_output_wave") == [1, 1, 0]
        and float(evolution["paired_e11_formula_residual"]) < 1.0e-13
        and float(evolution["unpaired_to_paired_e11_ratio"]) > 1.0e6,
        "projected evolution certificate changed",
    )
    hhl = result["HHL_sweeping_commutator"]
    _require(
        hhl.get("all_checks_pass") is True
        and "18 L" in hhl.get("theorem", "")
        and float(hhl["maximum_random_forcing_over_low_scale"]) < 18.0
        and float(hhl["maximum_divergence_residual"]) < 1.0e-11,
        "HHL commutator certificate changed",
    )
    hhh = result["HHH_pressure_strain_obstruction"]
    _require(
        hhh.get("all_checks_pass") is True
        and float(hhh["limiting_projected_Frobenius_norm"]) > 0.1
        and float(hhh["limiting_projected_trace"]) < 1.0e-12
        and float(hhh["limiting_raw_transport_norm"]) < 1.0e-12
        and float(hhh["rows"][-1]["normalized_residual_from_limit"])
        < 0.02,
        "HHH pressure-strain certificate changed",
    )
    pulses = result["sparse_parabolic_pulse_test"]
    _require(
        pulses.get("all_checks_pass") is True
        and float(
            pulses["rows"][-1][
                "coherent_sum_over_shell_square_function"
            ]
        )
        > 2.5
        and float(pulses["energy_increment_ratio_max"]) < 0.5
        and float(pulses["enstrophy_increment_ratio_max"]) < 0.5
        and float(pulses["forcing_increment_ratio_max"]) < 0.1,
        "parabolic pulse certificate changed",
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
        args.discovered_test_count == 243,
        "expected 237 inherited tests plus six new tests",
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
        "the prerequisite occupation result changed",
    )
    prior = _load_json(PRIOR_RESULT)
    _require(
        prior.get("kind")
        == "joint_scale_cell_viscous_occupation_audit"
        and prior.get("all_positive_checks_pass") is True,
        "the prerequisite occupation audit is invalid",
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
        and principal.get("joint_scale_cell_targeted_test_count") == 6
        and principal.get("joint_scale_cell_discovered_test_count") == 237
        and principal.get("joint_scale_cell_monolithic_regression_passed")
        is False
        and principal.get(
            "joint_scale_cell_viscous_occupation_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256,
        "the prerequisite occupation checkpoint changed",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The exact low-output high-shell stress evolution has been "
        "derived pair by pair. Every HHL regeneration triad has a "
        "carrier-independent bound: after both high legs and Leray "
        "projections are paired, the O(H) sweeping terms reduce to an "
        "O(L) commutator, with explicit constant 18. An exact coherent "
        "pump has unpaired terms more than 1e6 times its paired e11 "
        "remainder. This gain is not universal: a complete HHH family "
        "retains a traceless anisotropic pressure-strain term with "
        "||G_H/H|| tending to 2.27653869. A seven-shell parabolic pulse "
        "test shows pointwise sqrt(N) coherence but finite energy, "
        "enstrophy-time, and forcing L2-time sums in the sparse model. "
        "Dense annular mode multiplicity, the full Leray-to-regeneration "
        "bound, critical closure, and regularity remain open. Six focused "
        "tests pass with one Python worker."
    )

    principal.update(
        {
            "regeneration_exact_stress_evolution_proved": True,
            "regeneration_HHL_carrier_cancellation_proved": True,
            "regeneration_HHL_low_factor_bound_proved": True,
            "regeneration_all_low_factor_bound_proved": False,
            "regeneration_HHH_pressure_strain_witness_proved": True,
            "regeneration_sparse_pointwise_energy_bound_falsified": True,
            "regeneration_sparse_parabolic_summability_proved": True,
            "regeneration_dense_packet_multiplicity_controlled": False,
            "regeneration_full_Leray_bound_proved": False,
            "regeneration_critical_signed_bound_proved": False,
            "regeneration_Navier_Stokes_regularity_proved": False,
            "regeneration_targeted_test_count": args.targeted_test_count,
            "regeneration_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "regeneration_discovered_test_count": (
                args.discovered_test_count
            ),
            "regeneration_regression_test_count": (
                args.regression_test_count
            ),
            "regeneration_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "regeneration_monolithic_regression_passed": complete,
            "regeneration_resource_mode": args.resource_mode,
            "regeneration_worker_count": args.worker_count,
            "regeneration_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "regeneration_cpu_baseline_peak_percent": args.baseline_peak,
            "regeneration_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Derived the exact nonlinear high-stress evolution, proved "
            "the HHL sweeping/Leray low-factor commutator, exhibited the "
            "HHH anisotropic pressure-strain carrier obstruction, and "
            "certified sparse parabolic forcing summability."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is the dense HHH multiplicity gate. "
        "Construct a divergence-free annular packet normalized to fixed "
        "shell energy whose HHH triads feed one low Fourier/tensor/Walsh "
        "channel. Measure nonlinear stress forcing versus carrier and "
        "mode count, compare its parabolic L2-time cost with integrated "
        "enstrophy, and prove the trilinear square-function bound or "
        "certify its sharp Bernstein loss. Critical signed closure, "
        "low-regularity passage, and exceptional-set removal remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 243-test "
            "suite must pass in an admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Build the dense annular HHH packet audit before attempting a "
        "general forcing estimate. Normalize each packet to unit shell "
        "L2 energy, project the complete three-leg stress regeneration "
        "onto one fixed low traceless tensor channel, and determine the "
        "power of H and the mode-count gain. Then impose a parabolic "
        "lifetime H^(-2) and compare forcing L2-time cost directly with "
        "the packet's enstrophy-time cost."
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
