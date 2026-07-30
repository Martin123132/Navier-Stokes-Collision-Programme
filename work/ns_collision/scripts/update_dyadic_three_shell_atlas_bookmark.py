"""Install the dyadic three-shell atlas checkpoint."""

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
    "cross_shell_modulated_wave_gate_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "d6c330cba935e2bc8bcac55e462adfb97d91f04b42ed92faf209cea598d35597"
)
RESULT = (
    "work/ns_collision/results/"
    "dyadic_three_shell_atlas_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/dyadic_three_shell_atlas_audit.py",
    "work/ns_collision/tests/test_dyadic_three_shell_atlas.py",
    "work/ns_collision/notes/dyadic_three_shell_atlas.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_dyadic_three_shell_atlas_bookmark.py",
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
        result.get("kind") == "dyadic_three_shell_atlas_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == "dyadic_atlas_certified_naive_telescoping_falsified"
        and result.get("all_positive_checks_pass") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(flags, dict),
        "dyadic-atlas result is not the expected audit",
    )
    for key in (
        "largest_two_velocity_scales_comparable_proved",
        "global_shell_transfer_antisymmetry_proved",
        "localized_shell_skew_defect_identity_proved",
        "fixed_vertex_pure_shell_telescoping_falsified",
        "equal_weight_eight_vertex_cancellation_proved",
        "pure_top_Walsh_HHL_channel_exhibited",
        "nonconstant_nonnegative_vertex_selector_retains_half_L1",
        "coherent_multishell_HHL_accumulation_proved",
        "HHL_amplitude_square_sum_bound_proved",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "large_data_viscous_absorption_from_Leray_energy_proved",
        "joint_scale_cell_Carleson_bound_proved",
        "time_integrated_viscous_compensation_proved",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    atlas = result["support_atlas"]
    occupied = result["occupied_triple_support_stress"]
    _require(
        atlas.get("all_checks_pass") is True
        and "M_2>=(M_1-R)/2" in atlas.get("sorted_support_rule", "")
        and "HHH" in atlas.get("atlas", "")
        and "HHL" in atlas.get("atlas", "")
        and all(
            row.get(
                "at_most_four_when_largest_at_least_twice_stencil"
            )
            is True
            for row in atlas.get("rows", [])
        )
        and occupied.get("all_checks_pass") is True
        and int(occupied.get("occupied_ordered_triple_count", 0)) > 0
        and float(occupied["maximum_largest_over_second_ratio"]) < 4.0,
        "dyadic support atlas changed",
    )

    skew = result["localized_shell_skew_identity"]
    _require(
        skew.get("all_checks_pass") is True
        and float(skew["maximum_localized_skew_residual"]) < 1.0e-12
        and float(
            skew["maximum_HHL_kinetic_reconstruction_residual"]
        )
        < 1.0e-12
        and float(skew["maximum_global_antisymmetry_residual"])
        < 1.0e-12,
        "localized shell-skew certificate changed",
    )

    vertices = result["eight_vertex_flux_structure"]
    _require(
        vertices.get("all_checks_pass") is True
        and float(vertices["all_cosine_vertex_load"]) > 1.0e-4
        and float(vertices["equal_weight_eight_vertex_sum"]) == 0.0
        and float(vertices["maximum_off_top_Walsh_coefficient"]) == 0.0
        and abs(float(vertices["selector_sum_over_L1"]) - 0.5)
        < 1.0e-13,
        "eight-vertex Walsh certificate changed",
    )

    multishell = result["multishell_coherence_stress"]
    multishell_rows = multishell.get("rows", [])
    _require(
        multishell.get("all_checks_pass") is True
        and multishell.get("individual_carriers")
        == [16, 32, 64, 128, 256]
        and len(multishell_rows) == 5
        and all(
            row.get("all_checks_pass") is True
            and float(row["combined_vertex_load"]) > 0.0
            and float(row["cross_shell_coherence_residual"]) < 1.0e-12
            and abs(
                float(row["high_Fourier_L2_energy_proxy"])
                - 4.0 * int(row["high_shell_count"])
            )
            < 1.0e-12
            for row in multishell_rows
        ),
        "multishell coherence certificate changed",
    )

    envelope = result["HHL_amplitude_envelope"]
    envelope_rows = envelope.get("rows", [])
    _require(
        envelope.get("all_checks_pass") is True
        and "sqrt(2m)" in envelope.get("dyadic_sequence_bound", "")
        and len(envelope_rows) == 4
        and all(
            row.get("all_checks_pass") is True
            and float(row["HHL_amplitude_sum"])
            <= float(row["universal_sqrt2_upper"])
            and float(row["cubic_scaling_residual"]) < 1.0e-13
            for row in envelope_rows
        ),
        "HHL amplitude envelope changed",
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
        args.discovered_test_count == 231,
        "expected 225 inherited tests plus six new tests",
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
        "the prerequisite cross-shell result changed",
    )
    prior = _load_json(PRIOR_RESULT)
    _require(
        prior.get("kind") == "cross_shell_modulated_wave_gate_audit"
        and prior.get("all_positive_checks_pass") is True,
        "the prerequisite cross-shell audit is invalid",
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
        and principal.get("cross_shell_targeted_test_count") == 6
        and principal.get("cross_shell_discovered_test_count") == 225
        and principal.get("cross_shell_monolithic_regression_passed")
        is False
        and principal.get(
            "cross_shell_modulated_wave_gate_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256,
        "the prerequisite cross-shell checkpoint changed",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The exact dyadic cubic atlas now contains only HHH and HHL above "
        "the partition scale: the two largest velocity frequencies differ "
        "by at most a factor four. Global shell transfer is antisymmetric, "
        "while localization converts its defect exactly into spatial "
        "boundary flux. The modulated HHL family is a pure top Walsh "
        "character across eight cells; equal weights cancel, but a "
        "nonnegative selector retains half the L1 load. Five separated "
        "high shells accumulate coherently with load exactly linear in "
        "their Fourier L2 energy, falsifying automatic fixed-vertex shell "
        "telescoping. A positive amplitude theorem survives: "
        "|B_(v,L;HHL)|<=C m L^(3/2)a_L sum_(H>=4L)a_H^2, and dyadic "
        "summation gives C sqrt(2m)||u||_2||grad u||_2^2. This is "
        "perturbative only at small global Reynolds size. Six focused "
        "tests pass with one Python worker. A joint scale-cell Carleson "
        "gain, time-integrated compensation, critical closure, and "
        "regularity remain open."
    )

    principal.update(
        {
            "dyadic_atlas_largest_two_comparable_proved": True,
            "dyadic_atlas_global_transfer_antisymmetry_proved": True,
            "dyadic_atlas_localized_skew_defect_proved": True,
            "dyadic_atlas_fixed_vertex_telescoping_falsified": True,
            "dyadic_atlas_top_Walsh_HHL_channel_proved": True,
            "dyadic_atlas_nonnegative_selector_retains_half_L1": True,
            "dyadic_atlas_multishell_coherence_proved": True,
            "dyadic_atlas_HHL_amplitude_square_sum_proved": True,
            "dyadic_atlas_large_data_absorption_proved": False,
            "dyadic_atlas_joint_Carleson_bound_proved": False,
            "dyadic_atlas_time_compensation_proved": False,
            "dyadic_atlas_critical_signed_bound_proved": False,
            "dyadic_atlas_Navier_Stokes_regularity_proved": False,
            "dyadic_atlas_targeted_test_count": args.targeted_test_count,
            "dyadic_atlas_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "dyadic_atlas_discovered_test_count": (
                args.discovered_test_count
            ),
            "dyadic_atlas_regression_test_count": (
                args.regression_test_count
            ),
            "dyadic_atlas_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "dyadic_atlas_monolithic_regression_passed": complete,
            "dyadic_atlas_resource_mode": args.resource_mode,
            "dyadic_atlas_worker_count": args.worker_count,
            "dyadic_atlas_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "dyadic_atlas_cpu_baseline_peak_percent": args.baseline_peak,
            "dyadic_atlas_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Derived the exact HHH/HHL dyadic interaction atlas, proved "
            "the localized shell-transfer defect identity, falsified "
            "automatic fixed-vertex shell telescoping, identified the "
            "pure top-Walsh cell channel, and proved the surviving HHL "
            "amplitude square-sum bound."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is to beat the large-data coefficient "
        "C sqrt(m)||u||_2/nu in the HHL amplitude envelope. Form the "
        "cumulative high-shell Reynolds stress above each low scale and "
        "test a joint scale-cell Carleson estimate with coefficient "
        "variation measured by Walsh or cube-edge differences. If no "
        "pointwise gain exists, retain time and test whether large joint "
        "flux has summable viscous occupation. Critical signed closure, "
        "low-regularity passage, and exceptional-set removal remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 231-test "
            "suite must pass in an admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Define the cumulative low-output Reynolds stress R_>=L and its "
        "Walsh-resolved vertex flux. Prove the baseline "
        "L1-to-L2/Bernstein amplitude estimate exactly, then seek or "
        "falsify a Carleson improvement using orthogonality between high "
        "shells. Stress the candidate on coherent sidebands before adding "
        "time; if coherence saturates it, formulate a viscous occupation "
        "functional instead."
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
