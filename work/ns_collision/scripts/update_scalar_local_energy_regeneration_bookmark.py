"""Install the scalar local-energy regeneration gate checkpoint."""

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
    "dense_annular_hhh_packet_gate_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "9123904c9199f11f6064081d4e2d5de983b768e4d5cc265b5de6735825e7ecee"
)
RESULT = (
    "work/ns_collision/results/"
    "scalar_local_energy_regeneration_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "361834f006cfe5db7db61315de6da94c517cc985c7bad2842a3d3231ca0d11a8"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "scalar_local_energy_regeneration_gate_audit.py",
    "work/ns_collision/tests/"
    "test_scalar_local_energy_regeneration_gate.py",
    "work/ns_collision/notes/"
    "scalar_local_energy_regeneration_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_scalar_local_energy_regeneration_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, default=6)
    parser.add_argument(
        "--targeted-test-seconds",
        type=float,
        default=1.1295728,
    )
    parser.add_argument("--discovered-test-count", type=int, default=255)
    parser.add_argument("--regression-test-count", type=int, default=0)
    parser.add_argument("--regression-test-seconds", type=float, default=0.0)
    parser.add_argument(
        "--resource-mode",
        default="daytime_one_worker",
    )
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, default=50.09813694)
    parser.add_argument("--baseline-peak", type=float, default=70.55864298)
    return parser.parse_args()


def _validate_result(result: dict[str, Any]) -> None:
    _require(_sha256(RESULT) == RESULT_SHA256, "result hash changed")
    _require(
        result.get("prerequisite_result_sha256")
        == PRIOR_RESULT_SHA256,
        "dense-packet prerequisite hash changed",
    )
    _require(
        result.get("all_positive_checks_pass") is True,
        "the scalar gate did not pass",
    )
    trace = result["scalar_trace_identity"]
    _require(
        trace["maximum_identity_residual"] < 1.0e-9
        and trace["zero_output_trace_residual"] < 1.0e-10,
        "scalar trace identity changed",
    )
    reconstruction = result["independent_quartic_reconstruction"]
    _require(
        reconstruction["maximum_vector_residual"] < 1.0e-9,
        "independent quartic reconstruction changed",
    )
    center = result["central_complete_quartic_symbol"]
    _require(
        center["stress_prediction_is_exactly_at_limit"] is True
        and center["last_complete_relative_error"] < 2.0e-5,
        "central complete coefficient changed",
    )
    packet = result["dense_spaced_packet"]
    _require(
        packet["all_coherent_quartets_have_positive_selected_load"] is True
        and packet["rows"][-1]["exact_coherent_triad_count"] == 50653
        and packet["rows"][-1]["real_high_mode_count"] == 2058,
        "dense spaced packet changed",
    )
    negative = result["sharp_negative_shell_norm"]
    _require(
        negative["sharp_weight_exponent"] == "3/2"
        and negative["squared_norm_weight"] == "H^(-3)"
        and negative["finite_sequence_replay"]["all_checks_pass"] is True,
        "sharp negative shell norm changed",
    )
    flags = result["certification_flags"]
    required_true = (
        "ordinary_scalar_local_energy_trace_removes_H_five_halves",
        "complete_HHL_transfer_time_derivative_derived",
        "linearized_low_velocity_evolution_included",
        "all_kinetic_and_pressure_quartic_terms_included",
        "complete_differentiated_HHL_H_five_halves_survives",
        "pure_top_Walsh_frequency_isolated",
        "sharp_shell_negative_three_halves_forcing_norm_proved",
        "Leray_control_of_weighted_HHH_forcing_proved",
    )
    required_false = (
        "full_nonlinear_shell_response_closed",
        "suitable_weak_solution_passage_proved",
        "Navier_Stokes_global_regularity_proved",
    )
    _require(
        all(flags.get(name) is True for name in required_true),
        "a positive scope flag changed",
    )
    _require(
        all(flags.get(name) is False for name in required_false),
        "a negative scope flag changed",
    )


def main() -> None:
    args = _parse_args()
    _require(_sha256(PRIOR_RESULT) == PRIOR_RESULT_SHA256, "prior changed")
    result = _load_json(RESULT)
    _validate_result(result)
    bookmark = _load_json(BOOKMARK)
    principal = bookmark.setdefault("principal_results", {})
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    _require(
        bookmark.get("status") == "parked"
        and principal.get(
            "dense_annular_hhh_packet_gate_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256
        and principal.get("dense_HHH_targeted_test_count") == 6
        and principal.get("dense_HHH_discovered_test_count") == 249
        and principal.get("dense_HHH_monolithic_regression_passed")
        is False,
        "the prerequisite dense checkpoint changed",
    )

    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The exact Fourier identity tr G(q)=-2i q dot F(q) proves that "
        "the sharp dense H^(5/2) tensor disappears from the ordinary "
        "q=0 scalar energy trace. The complete time derivative of the "
        "HHL local-energy transfer was then derived with the linearized "
        "low velocity and every kinetic, high-high-pressure, "
        "cross-pressure, and Leray-projection term retained. Independent "
        "16-sign polarization agrees below 7.3e-13. The complete real "
        "quartet coefficient converges to 3sqrt(2)/16, so the anisotropic "
        "H^(5/2) regeneration survives this differentiated scalar "
        "transfer. A 4-spaced unit-energy packet isolates the pure "
        "top-Walsh channel; 2058 high modes and 50653 coherent quartets "
        "all retain the selected sign. The fixed-low-channel forcing "
        "obeys the sharp Leray-controlled norm "
        "sum_H H^(-3)||G_H||_L2_t^2 <= C E_*^2 D. Viscous response leaves "
        "a dyadically summable half derivative. Six focused tests pass "
        "with one worker. Full filtered-shell closure and weak passage "
        "remain open."
    )

    principal.update(
        {
            "scalar_trace_H_five_halves_cancellation_proved": True,
            "complete_differentiated_HHL_identity_proved": True,
            "complete_differentiated_HHL_H_five_halves_survival_proved": (
                True
            ),
            "scalar_regeneration_low_evolution_included": True,
            "scalar_regeneration_all_pressure_terms_included": True,
            "scalar_regeneration_independent_polarization_passed": True,
            "scalar_regeneration_top_Walsh_isolation_proved": True,
            "shell_negative_three_halves_norm_proved": True,
            "shell_negative_three_halves_norm_sharp_proved": True,
            "weighted_HHH_forcing_Leray_control_proved": True,
            "weighted_viscous_forced_response_proved": True,
            "complete_filtered_shell_response_closed": False,
            "scalar_regeneration_suitable_weak_passage_proved": False,
            "scalar_regeneration_Navier_Stokes_regularity_proved": False,
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
            "Separated the ordinary scalar trace from the differentiated "
            "HHL transfer, derived and independently reconstructed the "
            "complete HHHL coefficient, proved its sharp dense H^(5/2) "
            "top-Walsh survival, and established the sharp "
            "Leray-controlled H^(-3/2) shell forcing norm with weighted "
            "viscous response."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is the complete weighted dyadic "
        "shell-response gate. Lift the fixed finite low-channel estimate "
        "sum_H H^(-3)||G_H||_L2_t^2 <= C E_*^2 D to all comparable-shell "
        "HHH interactions, include the carrier-independent HHL sweeping "
        "commutator, retain exact pairwise viscous rates, and bound all "
        "filter commutators and initial-stress heat terms. First prove "
        "the result for smooth Galerkin solutions. Suitable-weak passage, "
        "exceptional-set removal, and global regularity remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 255-test "
            "suite must pass in an admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Write the exact smooth Galerkin evolution for every low-output "
        "high-shell stress c_(H,q), keeping individual pairwise viscous "
        "rates. Decompose its forcing into comparable-shell HHH, the "
        "proved HHL sweeping commutator, and filter leakage. Prove the "
        "H^(-3)-weighted L2_t forcing square bound with finite shell "
        "overlap, then apply the weighted zero-initial Duhamel estimate "
        "and add the initial heat term. Do not begin weak-solution passage "
        "until this complete smooth-shell identity closes."
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
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary_artifacts),
                "result_sha256": _sha256(RESULT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
