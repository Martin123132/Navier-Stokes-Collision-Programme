"""Install the pressure-active Fisher-null compatibility checkpoint."""

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
    "pressure_active_fisher_null_compatibility_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "e39e238dcd78aabfb8c089b20f29b8ecab5bf1d8b9f505932317ef1700eff5da"
)
PREDECESSOR_SHA256 = (
    "47b8704985671f0dac66ae38ff87a186acd6b938928828d3299a571337a7f087"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "pressure_active_fisher_null_compatibility_gate_audit.py",
    "work/ns_collision/tests/"
    "test_pressure_active_fisher_null_compatibility_gate.py",
    "work/ns_collision/notes/"
    "pressure_active_fisher_null_compatibility_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_pressure_active_fisher_compatibility_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, default=22)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=303)
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
        == "canonical_pressure_active_chain_Fisher_compatibility_proved"
        and result.get("all_positive_checks_pass") is True,
        "pressure-active Fisher compatibility result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags[
            "canonical_two_polarization_complete_HHL_Fisher_bound_proved"
        ]
        is True
        and flags["canonical_bound_constant"] == 0.5
        and flags["pressure_active_phase_tilts_controlled"] is True
        and flags["full_signed_Fisher_interfaces_retained"] is True
        and flags["arbitrary_residue_chain_bound_proved"] is False
        and flags["all_cross_shell_HHL_absorbed"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "pressure-active Fisher compatibility scope flags changed",
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
        len(bookmark.get("completed_obligations", [])) == 151
        and len(bookmark.get("primary_artifacts", [])) == 528
        and principal.get(
            "multiband_weighted_fisher_recombination_no_go_"
            "audit_v1_sha256"
        )
        == PREDECESSOR_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 152
        and len(bookmark.get("primary_artifacts", [])) == 533
        and principal.get(
            "pressure_active_fisher_null_compatibility_gate_"
            "audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed compatibility checkpoint matches",
    )

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The pressure-active Fisher-null compatibility gate is positive "
        "for the canonical chain k_n=(n,1,0), both divergence-free "
        "polarizations, low wave (0,-1,0), and compatible weight "
        "phi_-(x_1)phi_+(x_2). The complete HHL load is an exact skew "
        "nearest-neighbor form. High-high pressure cancels its anisotropic "
        "kinetic partner, and a positive-polynomial coefficient certificate "
        "proves |B_HHL|<=E_lambda/2 against the unsplit physical Fisher "
        "graph. Real constant and first-Dirichlet rows vanish; phase tilts "
        "activate pressure and remain controlled. This is not yet an "
        "arbitrary-residue or full multiband theorem."
    )
    principal.update(
        {
            (
                "pressure_active_canonical_two_polarization_complete_"
                "HHL_Fisher_bound_proved"
            ): True,
            "pressure_active_canonical_bound_constant": 0.5,
            "pressure_active_constant_chain_null_compatible": True,
            "pressure_active_first_Dirichlet_null_compatible": True,
            "pressure_active_phase_tilts_controlled": True,
            "pressure_active_full_signed_Fisher_interfaces_retained": True,
            "pressure_active_high_high_pressure_kinetic_cancellation": True,
            "pressure_active_arbitrary_residue_chain_bound_proved": False,
            "pressure_active_arbitrary_low_vertex_block_proved": False,
            "pressure_active_cross_shell_HHL_absorbed": False,
            "pressure_active_terminal_dual_supremum_controlled": False,
            "pressure_active_critical_L3_controlled": False,
            "pressure_active_Navier_Stokes_regularity_proved": False,
            "pressure_active_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "pressure_active_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "pressure_active_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "pressure_active_monolithic_regression_passed": False,
            "pressure_active_resource_mode": arguments.resource_mode,
            "pressure_active_worker_count": arguments.worker_count,
            "pressure_active_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "pressure_active_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "pressure_active_result_status": result["status"],
            (
                "pressure_active_fisher_null_compatibility_gate_"
                "audit_v1_sha256"
            ): RESULT_SHA256,
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
            "Proved the canonical pressure-active Fisher-null "
            "compatibility theorem: the complete two-polarization HHL "
            "chain load is bounded by one half of the unsplit weighted "
            "Fisher graph, with exact pressure/kinetic cancellation and "
            "a positive-polynomial in-plane coefficient certificate."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "Operationally, run the atomic one-process 303-test regression in "
        "an admissible resource window. Mathematically, generalize the "
        "canonical pressure-active edge theorem to arbitrary primitive "
        "partition steps, transverse residues, low waves, and compatible "
        "vertex phases; then prove or falsify the resulting finite "
        "low-wave/vertex block Schur bound against the unsplit physical "
        "Fisher graph. The modulated-wave HHL adversary must remain an "
        "explicit block. Terminal dual control, critical L3, exceptional-"
        "set removal, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/run_full_regression_checkpoint.py "
        "--expected-count 303"
    )
    bookmark["next_action"] = (
        "Build the arbitrary-residue pressure-Fisher block gate. Derive "
        "the complete HHL edge matrices for a general primitive partition "
        "step r, transverse residue eta, low wave ell, and compatible "
        "vertex phase. First certify the scalar edge coefficient envelopes "
        "uniformly after rescaling; then assemble the finite polarization "
        "and low-wave matrix and test its Schur complement against the "
        "full signed Fisher graph. Replay the canonical chain, "
        "Taylor-Green, seed-81, and the modulated-wave HHL no-go."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 152, "unexpected completed count")
    _require(len(primary) == 533, "unexpected artifact count")
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
