"""Install the primitive HHL chain Hardy-envelope checkpoint."""

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
    "primitive_hhl_chain_hardy_envelope_audit_v1.json"
)
RESULT_SHA256 = (
    "89d5cee5520acead1deba0231bed2cc7e4e740a673223ff5ca733c4a8375d18a"
)
PREDECESSOR_SHA256 = (
    "e39e238dcd78aabfb8c089b20f29b8ecab5bf1d8b9f505932317ef1700eff5da"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "primitive_hhl_chain_hardy_envelope_audit.py",
    "work/ns_collision/tests/"
    "test_primitive_hhl_chain_hardy_envelope.py",
    "work/ns_collision/notes/"
    "primitive_hhl_chain_hardy_envelope.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_primitive_hhl_chain_hardy_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, default=28)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=309)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument(
        "--periodic-average", type=float, required=True
    )
    parser.add_argument("--periodic-peak", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = _load_json(RESULT)
    _require(_sha256(RESULT) == RESULT_SHA256, "result hash changed")
    _require(
        result.get("status")
        == "uniform_primitive_HHL_chain_Hardy_envelope_proved"
        and result.get("all_positive_checks_pass") is True,
        "primitive HHL chain Hardy result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["uniform_isolated_primitive_chain_envelope_proved"]
        is True
        and flags["uniform_primitive_chain_constant"] == 108.0
        and flags[
            "arbitrary_low_polarization_and_phase_within_chain_controlled"
        ]
        is True
        and flags["multiple_residue_chains_jointly_assembled"] is False
        and flags["finite_low_wave_vertex_Schur_bound_proved"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "primitive HHL chain Hardy scope flags changed",
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
        len(bookmark.get("completed_obligations", [])) == 152
        and len(bookmark.get("primary_artifacts", [])) == 533
        and principal.get(
            "pressure_active_fisher_null_compatibility_gate_"
            "audit_v1_sha256"
        )
        == PREDECESSOR_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 153
        and len(bookmark.get("primary_artifacts", [])) == 538
        and principal.get(
            "primitive_hhl_chain_hardy_envelope_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed Hardy checkpoint matches",
    )

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The scalar primitive HHL chain envelope is now uniform. For one "
        "isolated one-sided chain, every primitive cube step, arbitrary "
        "transverse residue, arbitrary complex low polarization and phase, "
        "and every translated compatible tensor vertex are controlled by "
        "the complete physical Fisher chain. The exact ordered HHL budget "
        "is 9/2, the exhaustive resonant partner degree is six, and "
        "matrix-valued discrete Hardy gives |B_chain|<=108(|Uhat|/m)E for "
        "axial steps. Two-coordinate and three-coordinate steps improve "
        "to 27/2 and 6. The orthogonal sine-phase block tends to norm 2/3. "
        "Different chains and primitive steps have not been jointly "
        "assembled, so their Fisher charges must not be summed."
    )
    principal.update(
        {
            "primitive_HHL_isolated_chain_envelope_proved": True,
            "primitive_HHL_uniform_chain_constant": 108.0,
            "primitive_HHL_two_coordinate_constant": 13.5,
            "primitive_HHL_three_coordinate_constant": 6.0,
            "primitive_HHL_complete_ordered_symbol_budget": "9/2",
            "primitive_HHL_maximum_resonant_partner_degree": 6,
            "primitive_HHL_discrete_Hardy_constant": 4.0,
            "primitive_HHL_orthogonal_phase_limit": "2/3",
            "primitive_HHL_arbitrary_transverse_residue_controlled": True,
            "primitive_HHL_arbitrary_low_phase_controlled": True,
            "primitive_HHL_translated_vertex_controlled": True,
            "primitive_HHL_multiple_residue_chains_assembled": False,
            "primitive_HHL_multiple_steps_unsplit_Fisher_bound": False,
            "primitive_HHL_finite_low_vertex_Schur_bound": False,
            "primitive_HHL_cross_shell_HHL_absorbed": False,
            "primitive_HHL_terminal_dual_supremum_controlled": False,
            "primitive_HHL_critical_L3_controlled": False,
            "primitive_HHL_Navier_Stokes_regularity_proved": False,
            "primitive_HHL_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "primitive_HHL_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "primitive_HHL_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "primitive_HHL_monolithic_regression_passed": False,
            "primitive_HHL_resource_mode": arguments.resource_mode,
            "primitive_HHL_worker_count": arguments.worker_count,
            "primitive_HHL_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "primitive_HHL_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "primitive_HHL_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "primitive_HHL_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "primitive_HHL_result_status": result["status"],
            "primitive_hhl_chain_hardy_envelope_audit_v1_sha256": (
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
            "Proved the uniform primitive HHL isolated-chain Hardy "
            "envelope for arbitrary transverse residue, low polarization, "
            "phase, and translated compatible vertex, with constants 108, "
            "27/2, and 6 and an exhaustive six-partner resonance count."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "Operationally, run the atomic one-process 309-test regression in "
        "an admissible resource window. Mathematically, assemble the "
        "primitive-step, residue-chain, low-wave, polarization, and vertex "
        "incidence blocks against one unsplit physical Fisher matrix. "
        "Determine whether the six-partner structure gives a finite joint "
        "Schur constant without repeating isolated-chain Fisher charges. "
        "The canonical half-bound and modulated-wave HHL adversary must "
        "remain explicit blocks. Terminal dual control, critical L3, "
        "exceptional-set removal, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/run_full_regression_checkpoint.py "
        "--expected-count 309"
    )
    bookmark["next_action"] = (
        "Build the joint primitive HHL incidence-Schur gate. Start with one "
        "translated tensor vertex and enumerate all resonant primitive "
        "steps q, low waves ell, low signs, and chain offsets. Assemble the "
        "complete polarization blocks and the shared Fourier Fisher matrix "
        "once, quotient its exact nullspace, and compute the normalized "
        "Schur spectrum. Test increasing residue windows before attempting "
        "an analytic row-sum or graph-coloring certificate. Replay the "
        "canonical chain, orthogonal sine phase, Taylor-Green, seed-81, and "
        "the modulated-wave HHL no-go."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 153, "unexpected completed count")
    _require(len(primary) == 538, "unexpected artifact count")
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
