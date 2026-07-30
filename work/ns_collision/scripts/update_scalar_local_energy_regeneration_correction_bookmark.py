"""Install the corrected scalar local-energy regeneration checkpoint."""

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
OLD_RESULT = (
    "work/ns_collision/results/"
    "scalar_local_energy_regeneration_gate_audit_v1.json"
)
OLD_RESULT_SHA256 = (
    "361834f006cfe5db7db61315de6da94c517cc985c7bad2842a3d3231ca0d11a8"
)
CONTINUUM_RESULT = (
    "work/ns_collision/results/"
    "dense_spaced_continuum_positivity_audit_v1.json"
)
CONTINUUM_RESULT_SHA256 = (
    "36022e7657409dd29ce54ef21a3bb5d6e3c3e2768b900b1b5433869b17837d2d"
)
RESULT = (
    "work/ns_collision/results/"
    "scalar_local_energy_regeneration_gate_audit_v2.json"
)
RESULT_SHA256 = (
    "316efc095ac8b03cb97c7902f41e1934bf32aa4958d1c123c9eb0edafbdd7755"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "dense_spaced_continuum_positivity_audit.py",
    "work/ns_collision/tests/"
    "test_dense_spaced_continuum_positivity.py",
    CONTINUUM_RESULT,
    "work/ns_collision/scripts/"
    "scalar_local_energy_regeneration_gate_audit.py",
    "work/ns_collision/tests/"
    "test_scalar_local_energy_regeneration_gate.py",
    "work/ns_collision/notes/"
    "scalar_local_energy_regeneration_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_scalar_local_energy_regeneration_correction_bookmark.py",
    "work/ns_collision/README.md",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_json(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
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
    parser.add_argument("--targeted-test-count", type=int, default=12)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=261)
    parser.add_argument("--regression-test-count", type=int, default=0)
    parser.add_argument("--regression-test-seconds", type=float, default=0.0)
    parser.add_argument(
        "--resource-mode",
        default="daytime_one_worker",
    )
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    return parser.parse_args()


def _validate_continuum(result: dict[str, Any]) -> float:
    _require(
        _sha256(CONTINUUM_RESULT) == CONTINUUM_RESULT_SHA256,
        "continuum result hash changed",
    )
    _require(
        result.get("all_positive_checks_pass") is True,
        "continuum certificate did not pass",
    )
    _require(
        result.get("prior_scalar_gate_sha256") == OLD_RESULT_SHA256,
        "continuum certificate does not bind the preserved v1 result",
    )
    certificate = result["certificate"]
    leading = certificate["leading_stress_certificate"]
    derivative = certificate["complete_low_frequency_correction"]
    center = certificate["central_symbol_self_audit"]
    tau_zero = certificate["tau_zero_identity_self_audit"]
    lower = certificate[
        "complete_actual_positive_quartet_coefficient_lower"
    ]
    _require(
        certificate["physical_carrier"] == "R=16384M"
        and certificate[
            "carrier_multiple_relative_to_offset_box"
        ]
        == 4096
        and leading["scaled_coefficient_interval"][0] > 0.14
        and derivative["mean_value_correction_upper"] < 5.0e-5
        and center["all_checks_pass"] is True
        and center["exact_scaled_center_coefficient"] == "3/16"
        and tau_zero["all_checks_pass"] is True
        and lower > 0.10,
        "continuum positivity margin or identity audit changed",
    )
    return float(lower)


def _validate_result(result: dict[str, Any], lower: float) -> None:
    _require(_sha256(RESULT) == RESULT_SHA256, "v2 result hash changed")
    _require(
        result.get("superseded_result_sha256") == OLD_RESULT_SHA256,
        "v2 does not bind the preserved v1 result",
    )
    _require(
        result.get("all_positive_checks_pass") is True,
        "corrected scalar gate did not pass",
    )
    trace = result["scalar_trace_identity"]
    reconstruction = result["independent_quartic_reconstruction"]
    center = result["central_complete_quartic_symbol"]
    packet = result["dense_spaced_packet"]
    rows = packet["rows"]
    _require(
        trace["maximum_identity_residual"] < 1.0e-9
        and trace["zero_output_trace_residual"] < 1.0e-10
        and reconstruction["maximum_vector_residual"] < 1.0e-9
        and center["last_complete_relative_error"] < 2.0e-5,
        "an algebraic scalar-gate residual changed",
    )
    _require(
        len(rows) == 3
        and packet["carrier_multiple_relative_to_box_width"] == 4096
        and packet["continuum_positive_coefficient_lower"] == lower
        and packet[
            "all_coherent_quartets_have_positive_selected_load"
        ]
        is True
        and rows[-1]["exact_coherent_triad_count"] == 50653
        and rows[-1]["real_high_mode_count"] == 2058
        and all(
            row["coherent_count_normalized_complete_coefficient"]
            >= lower
            for row in rows
        ),
        "corrected dense packet changed",
    )
    flags = result["certification_flags"]
    required_true = (
        "ordinary_scalar_local_energy_trace_removes_H_five_halves",
        "complete_HHL_transfer_time_derivative_derived",
        "linearized_low_velocity_evolution_included",
        "all_kinetic_and_pressure_quartic_terms_included",
        "complete_differentiated_HHL_H_five_halves_survives",
        "pure_top_Walsh_frequency_isolated",
        "fixed_width_center_limit_claim_withdrawn",
        "continuous_offset_domain_uniform_positivity_proved",
        "sharp_shell_negative_three_halves_forcing_norm_proved",
        "Leray_control_of_weighted_HHH_forcing_proved",
    )
    required_false = (
        "full_nonlinear_shell_response_closed",
        "suitable_weak_solution_passage_proved",
        "Navier_Stokes_global_regularity_proved",
    )
    _require(
        all(flags.get(name) is True for name in required_true)
        and all(flags.get(name) is False for name in required_false),
        "corrected theorem scope flags changed",
    )


def main() -> None:
    args = _parse_args()
    _require(_sha256(OLD_RESULT) == OLD_RESULT_SHA256, "preserved v1 changed")
    continuum = _load_json(CONTINUUM_RESULT)
    lower = _validate_continuum(continuum)
    result = _load_json(RESULT)
    _validate_result(result, lower)
    bookmark = _load_json(BOOKMARK)
    principal = bookmark.setdefault("principal_results", {})
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    predecessor_checkpoint = bool(
        bookmark.get("status") == "parked"
        and len(bookmark.get("completed_obligations", [])) == 143
        and len(bookmark.get("primary_artifacts", [])) == 487
        and principal.get(
            "scalar_local_energy_regeneration_gate_audit_v1_sha256"
        )
        == OLD_RESULT_SHA256
        and principal.get("scalar_regeneration_targeted_test_count") == 6
        and principal.get("scalar_regeneration_discovered_test_count") == 255
        and principal.get("scalar_regeneration_monolithic_regression_passed")
        is False,
    )
    installed_checkpoint = bool(
        bookmark.get("status") in ("parked", "checkpointed")
        and len(bookmark.get("completed_obligations", [])) == 144
        and len(bookmark.get("primary_artifacts", [])) == 492
        and principal.get(
            "scalar_local_energy_regeneration_gate_audit_v1_sha256"
        )
        == OLD_RESULT_SHA256
        and principal.get(
            "dense_spaced_continuum_positivity_audit_v1_sha256"
        )
        == CONTINUUM_RESULT_SHA256
        and principal.get(
            "scalar_local_energy_regeneration_gate_audit_v2_sha256"
        )
        == RESULT_SHA256
        and principal.get(
            "scalar_regeneration_v1_fixed_width_center_limit_claim_"
            "superseded"
        )
        is True
    )
    _require(
        predecessor_checkpoint or installed_checkpoint,
        "neither the predecessor nor installed correction checkpoint "
        "matches",
    )

    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "Independent review correctly exposed a theorem-level error in "
        "the scalar gate v1: with R=128M, normalized offsets retain fixed "
        "width, so finite packet coefficients need not converge to the "
        "central 3sqrt(2)/32 value. The v1 result is preserved but that "
        "limit statement is superseded. The corrected construction uses "
        "R=16384M and proves positivity on a relaxed continuous offset "
        "box by outward-rounded interval arithmetic. The leading scaled "
        "coefficient is at least 0.14667827693662766; exact tau=0 "
        "identities and interval automatic differentiation pay the full "
        "finite-low-wave correction, leaving normalized complete "
        f"coefficient at least {lower:.16g}. This holds for every M>=1, "
        "rather than only the three replayed packets, and preserves the "
        "H^(5/2) lower exponent. The zero-output pressure gauge and the "
        "distinction between quartic flux output -r and final paired "
        "scalar output zero are explicit. The original trace identity, "
        "complete HHHL differentiation, central coefficient, sharp "
        "H^(-3/2) norm, and viscous summation remain unchanged."
    )
    if complete:
        bookmark["validated_checkpoint"] += (
            f" All {args.regression_test_count} discovered regression "
            f"tests pass in {args.regression_test_seconds:.3f} seconds."
        )

    principal.update(
        {
            "scalar_regeneration_v1_fixed_width_center_limit_claim_"
            "superseded": True,
            "scalar_regeneration_v1_preserved_sha256": OLD_RESULT_SHA256,
            "scalar_regeneration_continuous_offset_positivity_proved": True,
            "scalar_regeneration_continuum_carrier_law": "R=16384M",
            "scalar_regeneration_continuum_positive_lower": lower,
            "scalar_regeneration_zero_pressure_gauge_explicit": True,
            "scalar_regeneration_frequency_bookkeeping_explicit": True,
            "scalar_regeneration_targeted_test_count": (
                args.targeted_test_count
            ),
            "scalar_regeneration_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "scalar_regeneration_discovered_test_count": (
                args.discovered_test_count
            ),
            "scalar_regeneration_regression_test_count": (
                args.regression_test_count
            ),
            "scalar_regeneration_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "scalar_regeneration_monolithic_regression_passed": complete,
            "scalar_regeneration_resource_mode": args.resource_mode,
            "scalar_regeneration_worker_count": args.worker_count,
            "scalar_regeneration_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "scalar_regeneration_cpu_baseline_peak_percent": (
                args.baseline_peak
            ),
            "scalar_regeneration_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Corrected the dense HHHL packet asymptotics after independent "
            "review: withdrew the false fixed-width centre-limit claim, "
            "proved a uniform continuous-domain positive coefficient by "
            "directed intervals, and revalidated the complete H^(5/2) "
            "lower exponent without relying on finite-M extrapolation."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation remains the complete weighted dyadic "
        "shell-response gate. Lift the fixed finite low-channel estimate "
        "sum_H H^(-3)||G_H||_L2_t^2 <= C E_*^2 D to all comparable-shell "
        "HHH interactions, include the carrier-independent HHL sweeping "
        "commutator, retain exact pairwise viscous rates, and bound all "
        "filter commutators and initial-stress heat terms for smooth "
        "Galerkin solutions. Suitable-weak passage, exceptional-set "
        "removal, and global regularity remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot "
            f"{args.discovered_test_count}-test suite must pass in an "
            "admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Write the exact smooth Galerkin evolution for every low-output "
        "high-shell stress c_(H,q), retaining pairwise viscous rates. "
        "Decompose its forcing into comparable-shell HHH, the proved HHL "
        "sweeping commutator, and filter leakage. Prove the H^(-3)-weighted "
        "L2_t forcing square bound with finite shell overlap, apply the "
        "weighted zero-initial Duhamel estimate, and add the initial heat "
        "term before attempting weak-solution passage."
    )

    primary_artifacts = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary_artifacts, artifact)
    _require(len(completed) == 144, "unexpected completed-obligation count")
    _require(
        len(primary_artifacts) == 492,
        "unexpected primary-artifact count",
    )
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary_artifacts),
                "continuum_result_sha256": _sha256(CONTINUUM_RESULT),
                "result_sha256": _sha256(RESULT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
