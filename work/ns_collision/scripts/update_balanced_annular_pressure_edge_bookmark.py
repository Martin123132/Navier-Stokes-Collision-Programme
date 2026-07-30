"""Install the balanced-annular pressure-edge checkpoint."""

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
RESULT = (
    "work/ns_collision/results/"
    "balanced_annular_pressure_edge_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "9a024a23381d62e7842d7d26406fcea2a5343a168f386d3bad85e5308cef99dd"
)
FAR_RESULT_SHA256 = (
    "3cffc4a951b4b9806a505093ca3fff2a5475341427117bb0339d76b4acfc6f44"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "balanced_annular_pressure_edge_gate_audit.py",
    "work/ns_collision/tests/"
    "test_balanced_annular_pressure_edge_gate.py",
    "work/ns_collision/notes/"
    "balanced_annular_pressure_edge_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_balanced_annular_pressure_edge_bookmark.py",
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
    parser.add_argument("--discovered-test-count", type=int, default=292)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = _load_json(RESULT)
    _require(_sha256(RESULT) == RESULT_SHA256, "result hash changed")
    _require(
        result.get("status")
        == "balanced_annular_self_pressure_edge_intrinsic_absorption_certified"
        and result.get("all_positive_checks_pass") is True,
        "balanced-annular result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["floor_free_balanced_annular_self_edge_absorbed"] is True
        and flags["complete_single_band_pressure_included"] is True
        and flags["full_multiband_pressure_edge_absorbed"] is False
        and flags["cross_shell_HHL_pressure_absorbed"] is False
        and flags["terminal_dual_supremum_controlled"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "balanced-annular scope flags changed",
    )

    bookmark = _load_json(BOOKMARK)
    principal = bookmark.setdefault("principal_results", {})
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    predecessor = bool(
        len(bookmark.get("completed_obligations", [])) == 149
        and len(bookmark.get("primary_artifacts", [])) == 518
        and principal.get(
            "floor_free_pressure_edge_tail_gate_audit_v1_sha256"
        )
        == FAR_RESULT_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 150
        and len(bookmark.get("primary_artifacts", [])) == 523
        and principal.get(
            "balanced_annular_pressure_edge_gate_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed annular checkpoint matches",
    )

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The complete pressure edge generated and transported by one "
        "bounded annular velocity band is now absorbed floor-free at an "
        "explicit intrinsic scale. The exact Hamming shift identity gives "
        "|P_v|<=2sqrt(2)(1+C^2)^3 m||u||_infinity K^(-2)E_v; nonnegative "
        "compatible coefficients sum directly and the cubic terminal "
        "weight Fisher remains unspent. Taylor-Green, seed-81, and exact "
        "co-scaling replays pass. This is a single-band theorem: weighted "
        "Fisher recombination across bands, HHL pressure, the terminal "
        "supremum, exceptional-set removal, and regularity remain open."
    )
    principal.update(
        {
            "balanced_annular_self_pressure_edge_absorbed": True,
            "balanced_annular_complete_single_band_pressure_included": True,
            "balanced_annular_positive_weight_floor_required": False,
            "balanced_annular_cubic_weight_Fisher_retained": True,
            "balanced_annular_full_multiband_pressure_absorbed": False,
            "balanced_annular_cross_shell_HHL_absorbed": False,
            "balanced_annular_terminal_dual_supremum_controlled": False,
            "balanced_annular_critical_L3_bound_proved": False,
            "balanced_annular_Navier_Stokes_regularity_proved": False,
            "balanced_annular_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "balanced_annular_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "balanced_annular_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "balanced_annular_monolithic_regression_passed": False,
            "balanced_annular_resource_mode": arguments.resource_mode,
            "balanced_annular_worker_count": arguments.worker_count,
            "balanced_annular_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "balanced_annular_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "balanced_annular_result_status": result["status"],
            "balanced_annular_pressure_edge_gate_audit_v1_sha256": (
                RESULT_SHA256
            ),
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
            "Closed the floor-free balanced-annular self-pressure edge: "
            "combined Hamming energy comparison, the exact complete-output "
            "pressure shift identity, and high-pass Poincare to obtain an "
            "explicit intrinsic absorption theorem for every compatible "
            "nonnegative vertex weight."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "Operationally, run the atomic one-process 292-test regression in "
        "an admissible window. Mathematically, decide the finite-overlap "
        "multiband weighted-Fisher graph: multiplication by the partition "
        "stencil couples neighboring spectral pieces, so prove a signed "
        "frame lower bound that charges their component Fisher costs to "
        "the physical weighted Fisher form, or construct an exact "
        "cancellation counterexample. Cross-shell HHL pressure, terminal "
        "dual control, critical L3, exceptional-set removal, and global "
        "regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/run_full_regression_checkpoint.py "
        "--expected-count 292"
    )
    bookmark["next_action"] = (
        "Build the finite-overlap multiband Fisher decision gate. Write the "
        "weighted gradient form in Fourier blocks coupled by the "
        "frequency-m partition stencil, compute its exact block symbol and "
        "null directions, and test whether nonnegative compatible cell "
        "weights give a uniform frame lower bound after retaining edge "
        "signs. Use seed-81, Taylor-Green, and amplitude co-scaling as "
        "mandatory adversaries."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 150, "unexpected completed count")
    _require(len(primary) == 523, "unexpected artifact count")
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
