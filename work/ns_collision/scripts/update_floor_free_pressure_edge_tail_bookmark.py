"""Install the floor-free far-carrier pressure-edge tail checkpoint."""

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
DIRECT_RESULT = (
    "work/ns_collision/results/"
    "direct_h_minus_one_stress_tail_gate_audit_v1.json"
)
DIRECT_RESULT_SHA256 = (
    "709d06bcd8528bed257001bdf543ae72fe8032af0d3af9bb63ad3e2a46a75ece"
)
RESULT = (
    "work/ns_collision/results/"
    "floor_free_pressure_edge_tail_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "3cffc4a951b4b9806a505093ca3fff2a5475341427117bb0339d76b4acfc6f44"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "floor_free_pressure_edge_tail_gate_audit.py",
    "work/ns_collision/tests/"
    "test_floor_free_pressure_edge_tail_gate.py",
    "work/ns_collision/notes/"
    "floor_free_pressure_edge_tail_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_floor_free_pressure_edge_tail_bookmark.py",
    "work/ns_collision/notes/direct_h_minus_one_stress_tail_gate.md",
    "work/ns_collision/notes/collision_defect_dynamics.md",
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
    parser.add_argument("--targeted-test-count", type=int, default=5)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=287)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    return parser.parse_args()


def _validate_result(result: dict[str, Any]) -> None:
    _require(_sha256(RESULT) == RESULT_SHA256, "result hash changed")
    _require(
        _sha256(DIRECT_RESULT) == DIRECT_RESULT_SHA256,
        "direct H^-1 prerequisite hash changed",
    )
    _require(
        result.get("status")
        == "floor_free_far_carrier_pressure_edge_tail_certified"
        and result.get("all_positive_checks_pass") is True,
        "floor-free pressure-edge result did not pass",
    )
    _require(
        result["prerequisite"]["actual_sha256"] == DIRECT_RESULT_SHA256
        and result["prerequisite"]["tail_constant"] == 155
        and result["theorem"]["uniform_in_terminal_time"] is True
        and result["theorem"]["requires_positive_weight_floor"] is False
        and result["scale_adapted_partition_family"]["all_checks_pass"]
        is True,
        "floor-free pressure-edge quantitative theorem changed",
    )
    flags = result["certification_flags"]
    _require(
        flags["floor_free_far_carrier_pressure_edge_tail_vanishes"]
        is True
        and flags["fixed_smooth_partition_pressure_tail_uniform_in_time"]
        is True
        and flags["near_carrier_signed_pressure_edge_absorbed"] is False
        and flags["terminal_dual_supremum_controlled"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "floor-free pressure-edge scope flags changed",
    )


def main() -> None:
    arguments = _parse_args()
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
    predecessor = bool(
        len(bookmark.get("completed_obligations", [])) == 148
        and len(bookmark.get("primary_artifacts", [])) == 512
        and principal.get(
            "direct_h_minus_one_stress_tail_gate_audit_v1_sha256"
        )
        == DIRECT_RESULT_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 149
        and len(bookmark.get("primary_artifacts", [])) == 518
        and principal.get(
            "floor_free_pressure_edge_tail_gate_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed pressure-edge checkpoint matches",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The direct H^(-1) stress endpoint now feeds a floor-free replica "
        "pressure-edge theorem. For every smooth partition weight with "
        "uniform W^(2,infinity) cost L_lambda, the low-output comparable "
        "high-high pressure tail satisfies "
        "|integral <p_K,u dot grad lambda>dt| <=sqrt(155) C_selector "
        "L_lambda sqrt(E_*)D/sqrt(K), hence an energy-only bound independent "
        "of terminal time. No positive weight floor is used. For a scale-m "
        "partition with L_lambda<=C m^2, choosing "
        "K_m=m^(4+2epsilon) removes far carriers at rate m^(-epsilon). "
        "This corrects the stale signed-triad roadmap: that hierarchy was "
        "already constructed and found sign-indefinite/perturbative. The "
        "near-carrier signed pressure edge, terminal dual supremum, "
        "exceptional-set removal, and global regularity remain open."
    )
    principal.update(
        {
            "floor_free_pressure_edge_far_carrier_tail_vanishes": True,
            "floor_free_pressure_edge_uniform_in_terminal_time": True,
            "floor_free_pressure_edge_positive_weight_floor_required": False,
            "floor_free_pressure_edge_scale_diagonalization_proved": True,
            "floor_free_pressure_edge_near_carrier_absorbed": False,
            "floor_free_pressure_edge_terminal_dual_supremum_controlled": False,
            "floor_free_pressure_edge_critical_L3_bound_proved": False,
            "floor_free_pressure_edge_Navier_Stokes_regularity_proved": False,
            "floor_free_pressure_edge_tail_constant": 155,
            "floor_free_pressure_edge_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "floor_free_pressure_edge_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "floor_free_pressure_edge_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "floor_free_pressure_edge_monolithic_regression_passed": False,
            "floor_free_pressure_edge_resource_mode": arguments.resource_mode,
            "floor_free_pressure_edge_worker_count": arguments.worker_count,
            "floor_free_pressure_edge_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "floor_free_pressure_edge_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "floor_free_pressure_edge_result_status": result["status"],
            "floor_free_pressure_edge_tail_gate_audit_v1_sha256": (
                RESULT_SHA256
            ),
            "signed_triad_roadmap_construction_already_completed": True,
            "finite_heat_normal_form_large_Reynolds_route_retired": True,
        }
    )
    for artifact in ARTIFACTS:
        parent = Path(artifact).parent.name.replace("-", "_")
        stem = Path(artifact).stem.replace("-", "_")
        principal[f"{parent}_{stem}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Closed the floor-free far-carrier replica pressure-edge gate: "
            "combined the direct H^(-1) high-high stress tail with "
            "H^(-1)-H^1 time duality and the Leray energy inequality to "
            "remove arbitrarily remote low-output pressure beats without a "
            "positive partition-weight floor, while isolating the unresolved "
            "near-carrier signed edge."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "Operationally, run the atomic one-process 287-test regression "
        "checkpoint in an admissible resource window; the prior monolithic "
        "exit statuses remain indeterminate. Mathematically, prove or sharply "
        "falsify a signed near-carrier pressure-edge estimate on the balanced "
        "annular range. It must retain the Hamming/vertex commutator before "
        "absolute values, couple it to replica Fisher dissipation, survive "
        "the seed-81 and Taylor-Green co-scaling adversaries, and avoid a "
        "positive weight floor. Terminal dual control, critical L3, "
        "exceptional-set removal, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/run_full_regression_checkpoint.py "
        "--expected-count 287"
    )
    bookmark["next_action"] = (
        "Build the near-carrier signed pressure-edge decision gate. Start "
        "from the exact annular vertex/Hamming multiplier and the cubic graph "
        "Fisher form, retain neighboring edge signs, and test whether a "
        "finite-band Carleson or Schur complement estimate is coercive. Use "
        "the existing amplitude-scaled pressure adversaries as mandatory "
        "falsifiers before any PDE-level claim."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 149, "unexpected completed count")
    _require(len(primary) == 518, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary),
                "result_sha256": _sha256(RESULT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
