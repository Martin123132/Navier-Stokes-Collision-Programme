"""Install the dense annular HHH packet gate checkpoint."""

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
    "nonlinear_stress_regeneration_gate_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "a78a6064db7e8c94e6fbbd9cc85469ec505fe4070e5b88d21f968120e20858e1"
)
RESULT = (
    "work/ns_collision/results/"
    "dense_annular_hhh_packet_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "9123904c9199f11f6064081d4e2d5de983b768e4d5cc265b5de6735825e7ecee"
)
ARTIFACTS = (
    "work/ns_collision/scripts/dense_annular_hhh_packet_gate_audit.py",
    "work/ns_collision/tests/test_dense_annular_hhh_packet_gate.py",
    "work/ns_collision/notes/dense_annular_hhh_packet_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_dense_annular_hhh_packet_gate_bookmark.py",
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
        result.get("kind") == "dense_annular_hhh_packet_gate_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "sharp_dense_HHH_Bernstein_loss_certified_"
            "Leray_input_only_forcing_bound_falsified"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(flags, dict),
        "result is not the expected dense-packet audit",
    )
    for key in (
        "dense_packet_divergence_free_annular_unit_energy_proved",
        "fixed_traceless_tensor_channel_nonzero_proved",
        "coherent_H_six_triad_count_proved",
        "sharp_H_five_halves_tensor_forcing_growth_proved",
        "top_Walsh_cell_channel_survives_proved",
        "raw_tensor_forcing_bound_from_Leray_inputs_alone_falsified",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "unforced_Navier_Stokes_dynamic_counterexample_proved",
        "trace_local_energy_channel_obstructed",
        "complete_signed_flux_occupation_bound_proved",
        "equation_specific_temporal_correlation_bound_proved",
        "critical_signed_large_data_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    center = result["exact_center_witness"]
    _require(
        center.get("all_checks_pass") is True
        and center.get("exact_Frobenius_norm") == "12*sqrt(43)"
        and float(center["exact_matrix_residual"]) < 1.0e-13
        and float(center["central_channel_pairing"]) > 70.0,
        "exact center witness changed",
    )
    dense = result["dense_annular_packet"]
    rows = dense.get("rows", [])
    _require(
        dense.get("all_checks_pass") is True
        and len(rows) == 4
        and rows[-1].get("real_field_mode_count") == 4374
        and rows[-1].get("exact_coherent_triad_count") == 226981
        and float(rows[-1]["channel_over_full_tensor_norm"]) > 0.99999
        and float(dense["minimum_channel_over_count_scale"]) > 70.0
        and all(row.get("all_checks_pass") is True for row in rows),
        "dense lattice certificate changed",
    )
    theorem = result["sharp_scaling_theorem"]
    _require(
        theorem.get("all_checks_pass") is True
        and theorem.get("sharp_forcing_exponent") == "5/2"
        and theorem.get("parabolic_forcing_L2_cost_exponent") == "3"
        and theorem.get("parabolic_enstrophy_cost_exponent") == "0",
        "sharp scaling theorem changed",
    )
    walsh = result["fixed_top_Walsh_coupling"]
    _require(
        walsh.get("all_checks_pass") is True
        and walsh.get("exact_pairing_magnitude") == "1/sqrt(86)"
        and float(walsh["pairing_magnitude_residual"]) < 1.0e-13,
        "Walsh coupling certificate changed",
    )
    no_go = result["parabolic_Leray_input_no_go"]
    _require(
        no_go.get("all_checks_pass") is True
        and no_go.get("forcing_cost_growth_exponent") == "3"
        and no_go.get("enstrophy_cost_growth_exponent") == "0"
        and "not claimed to solve unforced" in no_go.get("scope", ""),
        "parabolic no-go scope changed",
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
        args.discovered_test_count == 249,
        "expected 243 inherited tests plus six new tests",
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
        "the prerequisite regeneration result changed",
    )
    prior = _load_json(PRIOR_RESULT)
    _require(
        prior.get("kind") == "nonlinear_stress_regeneration_gate_audit"
        and prior.get("all_positive_checks_pass") is True,
        "the prerequisite regeneration audit is invalid",
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
        and principal.get("regeneration_targeted_test_count") == 6
        and principal.get("regeneration_discovered_test_count") == 243
        and principal.get("regeneration_monolithic_regression_passed")
        is False
        and principal.get(
            "nonlinear_stress_regeneration_gate_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256,
        "the prerequisite regeneration checkpoint changed",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "An exact rational HHH center symbol has been thickened into a "
        "real smooth divergence-free unit-energy annular packet. Its "
        "O(H^3) Fourier modes generate O(H^6) same-sign triples in one "
        "fixed low traceless tensor channel. Coherent counting and a "
        "matching Bernstein upper bound prove sharp H^(5/2) tensor "
        "forcing. An exhaustive K=32 replay reaches 4374 modes and "
        "226981 triples, with over 0.999998 of the forcing in the selected "
        "channel. The tensor couples nontrivially to the pure top-Walsh "
        "cell character. Under a parabolic H^(-2) pulse, raw forcing "
        "L2-time cost grows as H^3 while energy and enstrophy-time cost "
        "remain bounded. Thus the raw tensor Duhamel norm cannot follow "
        "from Leray inputs alone for arbitrary smooth divergence-free "
        "paths. No unforced Navier-Stokes counterexample is claimed; the "
        "trace and complete signed local-energy routes remain open. Six "
        "focused tests pass with one Python worker."
    )

    principal.update(
        {
            "dense_HHH_unit_energy_annular_packet_proved": True,
            "dense_HHH_exact_traceless_channel_proved": True,
            "dense_HHH_coherent_six_power_count_proved": True,
            "dense_HHH_sharp_five_halves_growth_proved": True,
            "dense_HHH_top_Walsh_coupling_proved": True,
            "dense_HHH_raw_Leray_input_bound_falsified": True,
            "dense_HHH_unforced_NS_counterexample_proved": False,
            "dense_HHH_trace_channel_obstructed": False,
            "dense_HHH_complete_signed_flux_bound_proved": False,
            "dense_HHH_critical_signed_bound_proved": False,
            "dense_HHH_Navier_Stokes_regularity_proved": False,
            "dense_HHH_targeted_test_count": args.targeted_test_count,
            "dense_HHH_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "dense_HHH_discovered_test_count": (
                args.discovered_test_count
            ),
            "dense_HHH_regression_test_count": (
                args.regression_test_count
            ),
            "dense_HHH_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "dense_HHH_monolithic_regression_passed": complete,
            "dense_HHH_resource_mode": args.resource_mode,
            "dense_HHH_worker_count": args.worker_count,
            "dense_HHH_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "dense_HHH_cpu_baseline_peak_percent": args.baseline_peak,
            "dense_HHH_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Constructed a dense unit-energy annular HHH packet, proved "
            "sharp H^(5/2) coherent tensor forcing and top-Walsh coupling, "
            "and falsified control of its raw parabolic forcing norm from "
            "Leray energy and enstrophy inputs alone."
        ),
    )
    theorem_obligation = (
        "The next theorem obligation is the complete signed scalar "
        "local-energy gate. Project the dense packet through low-velocity "
        "evolution, kinetic transport, high-high pressure, cross pressure, "
        "and all eight cell vertices. Determine whether the H^(5/2) "
        "traceless pressure-strain channel cancels in the scalar equation "
        "or survives. In parallel, identify the weakest shell-weighted "
        "negative forcing norm controlled by viscosity and Leray "
        "enstrophy. Critical closure, low-regularity passage, and "
        "exceptional-set removal remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot 249-test "
            "suite must pass in an admissible resource window. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Write the complete scalar local-energy time derivative before "
        "estimating the dense packet. Couple its q=0 tensor output to the "
        "fixed low mode k=-(1,1,1) and top-Walsh partition frequency, "
        "include the evolution of that low mode and every kinetic and "
        "pressure term, and test whether the H^(5/2) coefficient cancels. "
        "Do not reuse the now-falsified raw tensor forcing norm."
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
