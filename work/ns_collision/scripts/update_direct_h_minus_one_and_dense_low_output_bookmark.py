"""Install the direct H^-1 and dense low-output theorem checkpoints."""

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
BOUNDARY_REPORT = ROOT / "migration/LIVE_BOUNDARY_VALIDATION_REPORT.json"
SCALE_RESULT = (
    "work/ns_collision/results/"
    "scale_uniform_low_output_tail_gate_audit_v1.json"
)
SCALE_RESULT_SHA256 = (
    "3622a76234d31dcf298e0326b1be75f888fb925926df69fd65e45a9c80c6b657"
)
DIRECT_RESULT = (
    "work/ns_collision/results/"
    "direct_h_minus_one_stress_tail_gate_audit_v1.json"
)
DIRECT_RESULT_SHA256 = (
    "709d06bcd8528bed257001bdf543ae72fe8032af0d3af9bb63ad3e2a46a75ece"
)
DENSE_RESULT = (
    "work/ns_collision/results/"
    "dense_low_output_block_gate_audit_v1.json"
)
DENSE_RESULT_SHA256 = (
    "1c847cf7ded3f5246f15f34359fec1d80581fbc2a978d07817c600df02c7ba59"
)
CONTINUUM_RESULT = (
    "work/ns_collision/results/"
    "dense_spaced_continuum_positivity_audit_v1.json"
)
CONTINUUM_RESULT_SHA256 = (
    "36022e7657409dd29ce54ef21a3bb5d6e3c3e2768b900b1b5433869b17837d2d"
)
ACKNOWLEDGED_RH_SHA256 = (
    "6667ff494320e9159f7ad3a2639896b122e245280d94d61ed4a56175ca528eba"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "direct_h_minus_one_stress_tail_gate_audit.py",
    "work/ns_collision/tests/"
    "test_direct_h_minus_one_stress_tail_gate.py",
    "work/ns_collision/notes/"
    "direct_h_minus_one_stress_tail_gate.md",
    DIRECT_RESULT,
    "work/ns_collision/scripts/"
    "dense_low_output_block_gate_audit.py",
    "work/ns_collision/tests/"
    "test_dense_low_output_block_gate.py",
    "work/ns_collision/notes/"
    "dense_low_output_block_gate.md",
    DENSE_RESULT,
    "work/ns_collision/scripts/run_full_regression_checkpoint.py",
    "work/ns_collision/scripts/"
    "update_direct_h_minus_one_and_dense_low_output_bookmark.py",
    "work/ns_collision/notes/scale_uniform_low_output_tail_gate.md",
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
    parser.add_argument("--direct-chain-test-seconds", type=float, required=True)
    parser.add_argument("--dense-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=282)
    parser.add_argument("--monolithic-attempt-count", type=int, default=2)
    parser.add_argument("--resource-mode", default="explicit_capacity_override")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument("--boundary-test-seconds", type=float, required=True)
    return parser.parse_args()


def _validate_direct(result: dict[str, Any]) -> None:
    _require(_sha256(DIRECT_RESULT) == DIRECT_RESULT_SHA256, "direct result hash changed")
    _require(_sha256(SCALE_RESULT) == SCALE_RESULT_SHA256, "scale result hash changed")
    _require(
        result.get("status")
        == "direct_H_minus_one_high_high_stress_tail_certified"
        and result.get("all_positive_checks_pass") is True,
        "direct H^-1 result did not pass",
    )
    _require(
        result["prerequisites"]["scale_uniform_low_output_tail_sha256"]
        == SCALE_RESULT_SHA256
        and result["dyadic_overlap"]["integrated_squared_tail_constant"][
            "exact"
        ]
        == "155"
        and result["theorem"]["Galerkin_uniform"] is True,
        "direct H^-1 quantitative theorem changed",
    )
    flags = result["certification_flags"]
    _require(
        flags["actual_high_high_stress_H_minus_1_tail_vanishes"] is True
        and flags["H_minus_1_high_high_stress_Galerkin_passage_proved"]
        is True
        and flags["prior_pulse_admissible_as_actual_Reynolds_stress"]
        is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "direct H^-1 scope flags changed",
    )


def _validate_dense(result: dict[str, Any]) -> None:
    _require(_sha256(DENSE_RESULT) == DENSE_RESULT_SHA256, "dense result hash changed")
    _require(
        _sha256(CONTINUUM_RESULT) == CONTINUUM_RESULT_SHA256,
        "continuum positivity result hash changed",
    )
    _require(
        result.get("status")
        == "positive_volume_low_output_HHH_block_certified"
        and result.get("all_positive_checks_pass") is True,
        "dense low-output result did not pass",
    )
    _require(
        result["prerequisites"]["dense_continuum_positivity_sha256"]
        == CONTINUUM_RESULT_SHA256
        and result["prerequisites"]["scale_uniform_tail_sha256"]
        == SCALE_RESULT_SHA256
        and result["directed_interval_certificate"][
            "unit_channel_interval_per_carrier"
        ][0]
        > 0.0
        and result["exact_lattice_multiplicity"][
            "maximum_inclusion_exclusion_residual"
        ]
        == 0,
        "dense low-output quantitative theorem changed",
    )
    flags = result["certification_flags"]
    _require(
        flags["positive_volume_low_output_block_realized"] is True
        and flags["simultaneous_spatial_H_5_2_output_scaling_proved"]
        is True
        and flags["parabolic_time_persistence_proved"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "dense low-output scope flags changed",
    )


def _validate_boundary(report: dict[str, Any]) -> None:
    checks = report["checks"]
    checkpoint = checks["checkpoint"]
    _require(
        report.get("status") == "pass"
        and report.get("errors") == []
        and checkpoint["rh_bookmark_hash_state"]
        == "acknowledged_hash_matched"
        and checkpoint["rh_active_ns_references"] == []
        and checks["cross_workspace_runtime_references"] == []
        and checks["reparse_points"] == [],
        "workspace boundary validation did not pass",
    )


def main() -> None:
    arguments = _parse_args()
    direct = _load_json(DIRECT_RESULT)
    dense = _load_json(DENSE_RESULT)
    boundary = _load_json(BOUNDARY_REPORT)
    _validate_direct(direct)
    _validate_dense(dense)
    _validate_boundary(boundary)

    bookmark = _load_json(BOOKMARK)
    principal = bookmark.setdefault("principal_results", {})
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    predecessor = bool(
        len(bookmark.get("completed_obligations", [])) == 146
        and len(bookmark.get("primary_artifacts", [])) == 502
        and principal.get(
            "scale_uniform_low_output_tail_gate_audit_v1_sha256"
        )
        == SCALE_RESULT_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 148
        and len(bookmark.get("primary_artifacts", [])) == 512
        and principal.get(
            "direct_h_minus_one_stress_tail_gate_audit_v1_sha256"
        )
        == DIRECT_RESULT_SHA256
        and principal.get(
            "dense_low_output_block_gate_audit_v1_sha256"
        )
        == DENSE_RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed direct/dense checkpoint matches",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The actual comparable high-high Reynolds-stress tail now vanishes "
        "at the endpoint: ||sum_(H>=K) C_H||_(L2_t H_x^(-1))^2 <= "
        "155 C_selector^2 E_*D/K, uniformly over smooth Galerkin cutoffs. "
        "This corrects the prior s>1-only inference while preserving the "
        "narrow scalar-envelope no-go. Independently, an outward-rounded "
        "interval certificate proves one fixed tensor channel is positive "
        "on a positive-volume low-output block and realizes simultaneous "
        "instantaneous H^(5/2) forcing on O(H^3) outputs. The direct theorem "
        "prevents interpreting that spatial derivative as an endpoint "
        "failure. Temporal signed-triad/quartic rigidity and global "
        "regularity remain open."
    )

    dense_interval = dense["directed_interval_certificate"][
        "unit_channel_interval_per_carrier"
    ]
    principal.update(
        {
            "direct_H_minus_one_actual_high_high_stress_tail_vanishes": True,
            "direct_H_minus_one_Galerkin_passage_proved": True,
            "direct_H_minus_one_integrated_squared_tail_constant": 155,
            "direct_H_minus_one_prior_envelope_no_go_remains_valid": True,
            "direct_H_minus_one_prior_pulse_actual_stress_admissible": False,
            "direct_H_minus_one_complete_cubic_passage_proved": False,
            "direct_H_minus_one_Navier_Stokes_regularity_proved": False,
            "direct_H_minus_one_chain_test_count": 15,
            "direct_H_minus_one_chain_test_runtime_seconds": (
                arguments.direct_chain_test_seconds
            ),
            "dense_low_output_positive_volume_block_proved": True,
            "dense_low_output_simultaneous_H_five_halves_proved": True,
            "dense_low_output_interval_lower": dense_interval[0],
            "dense_low_output_interval_upper": dense_interval[1],
            "dense_low_output_parabolic_persistence_proved": False,
            "dense_low_output_Navier_Stokes_regularity_proved": False,
            "dense_low_output_targeted_test_count": 6,
            "dense_low_output_targeted_test_runtime_seconds": (
                arguments.dense_test_seconds
            ),
            "direct_dense_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "direct_dense_monolithic_attempt_count": (
                arguments.monolithic_attempt_count
            ),
            "direct_dense_monolithic_exit_status": (
                "indeterminate_desktop_wrapper_timeouts"
            ),
            "direct_dense_monolithic_regression_passed": False,
            "direct_dense_resource_mode": arguments.resource_mode,
            "direct_dense_worker_count": arguments.worker_count,
            "direct_dense_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "direct_dense_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "workspace_boundary_validation_passed": True,
            "workspace_boundary_validation_runtime_seconds": (
                arguments.boundary_test_seconds
            ),
            "workspace_boundary_acknowledged_RH_sha256": (
                ACKNOWLEDGED_RH_SHA256
            ),
            "workspace_boundary_active_RH_NS_reference_count": 0,
            "direct_h_minus_one_stress_tail_gate_audit_v1_sha256": (
                DIRECT_RESULT_SHA256
            ),
            "dense_low_output_block_gate_audit_v1_sha256": (
                DENSE_RESULT_SHA256
            ),
        }
    )
    for artifact in ARTIFACTS:
        parent = Path(artifact).parent.name.replace("-", "_")
        stem = Path(artifact).stem.replace("-", "_")
        principal[f"{parent}_{stem}_sha256"] = _sha256(artifact)

    direct_deferred = bookmark.setdefault(
        "direct_h_minus_one_stress_tail_deferred_calculation",
        {},
    )
    direct_deferred.update(
        {
            "status": "resolved",
            "resolved_at": now,
            "resolution": (
                "The direct audit passed, its 15-test dependency chain "
                "passed, the endpoint theorem was independently rechecked, "
                "and the dense spatial diagnostic passed six focused tests."
            ),
            "result_sha256": DIRECT_RESULT_SHA256,
            "dense_result_sha256": DENSE_RESULT_SHA256,
            "unfinished_obligation": None,
        }
    )
    rh_deferred = bookmark.setdefault(
        "rh_boundary_validation_deferred_calculation",
        {},
    )
    rh_deferred.update(
        {
            "status": "resolved",
            "resolved_at": now,
            "resolution": (
                "Post-acknowledgement split validation passed with the "
                "acknowledged RH hash matched and zero active cross-workspace "
                "runtime references."
            ),
            "unfinished_obligation": None,
        }
    )

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Closed the actual quadratic high-high stress endpoint: restored "
            "the physical-space product structure, proved the direct "
            "L2_t H_x^(-1) tail bound with explicit constant 155, and "
            "classified the old channel-saturated pulse as inadmissible for "
            "a unit-energy Reynolds-stress trajectory."
        ),
    )
    _append_once(
        completed,
        (
            "Closed the dense low-output spatial gate: certified one fixed "
            "tensor channel positive on a positive-volume output block, "
            "proved exact O(H^3) output multiplicity and simultaneous "
            "instantaneous H^(5/2) forcing, without claiming temporal "
            "persistence or a Navier-Stokes endpoint obstruction."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "Operationally, run the atomic one-process 282-test regression "
        "checkpoint in an admissible resource window; two completed attempts "
        "in this cycle lost their exit codes when the desktop wrapper timed "
        "out, so no monolithic pass is claimed. Mathematically, return to "
        "collision_defect_dynamics.md: construct a time-integrated signed "
        "triad measure and control its quartic Navier-Stokes transfer against "
        "the palinstrophy denominator. Complete cubic local-energy passage, "
        "exceptional-set removal, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/run_full_regression_checkpoint.py "
        "--expected-count 282"
    )
    bookmark["next_action"] = (
        "Use the direct H^(-1) theorem to close the quadratic-stress detour. "
        "Next derive the exact evolution of a signed time-integrated triad "
        "measure, expose every quartic transfer and viscous term, and test "
        "whether any normalization by palinstrophy yields a coercive or "
        "monotone rigidity quantity. Keep the dense packet as a falsification "
        "family for candidate sign or persistence claims."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 148, "unexpected completed count")
    _require(len(primary) == 512, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary),
                "direct_result_sha256": _sha256(DIRECT_RESULT),
                "dense_result_sha256": _sha256(DENSE_RESULT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
