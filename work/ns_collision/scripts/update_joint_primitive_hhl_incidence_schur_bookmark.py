"""Install the joint primitive HHL incidence-Schur checkpoint."""

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
    "joint_primitive_hhl_incidence_schur_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "216e41e650e2421c4ef4a2c0100a656618f169a8d9dd758ae5a507a7e23837df"
)
PREDECESSOR_SHA256 = (
    "89d5cee5520acead1deba0231bed2cc7e4e740a673223ff5ca733c4a8375d18a"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "joint_primitive_hhl_incidence_schur_gate_audit.py",
    "work/ns_collision/tests/"
    "test_joint_primitive_hhl_incidence_schur_gate.py",
    "work/ns_collision/notes/"
    "joint_primitive_hhl_incidence_schur_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_joint_primitive_hhl_incidence_schur_bookmark.py",
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
    parser.add_argument("--targeted-test-count", type=int, default=34)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=368)
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
        == "finite_window_joint_pressure_growth_witnesses_validated"
        and result.get("all_positive_checks_pass") is True,
        "joint primitive HHL incidence-Schur result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["complete_HHL_blocks_assembled_jointly"] is True
        and flags["shared_physical_Fisher_charged_once"] is True
        and flags["all_52_real_low_coordinates_included"] is True
        and flags[
            "finite_window_pressure_growth_witnesses_validated"
        ]
        is True
        and flags["analytic_unbounded_pressure_family_proved"] is False
        and flags["window_uniform_joint_Schur_bound_proved"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "joint incidence-Schur scope flags changed",
    )
    rows = result["window_rows"]
    _require(
        len(rows) == 8
        and all(row["all_checks_pass"] for row in rows),
        "joint incidence-Schur finite rows changed",
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
        len(bookmark.get("completed_obligations", [])) == 153
        and len(bookmark.get("primary_artifacts", [])) == 538
        and principal.get(
            "primitive_hhl_chain_hardy_envelope_audit_v1_sha256"
        )
        == PREDECESSOR_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 154
        and len(bookmark.get("primary_artifacts", [])) == 543
        and principal.get(
            "joint_primitive_hhl_incidence_schur_gate_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed joint checkpoint matches",
    )

    growth = result["growth_summary"]["directional_comparisons"]
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The finite joint primitive HHL incidence-Schur gate now retains "
        "one physical Fisher matrix across every tested residue, "
        "polarization, and all 52 real cube-low coordinates. Eight windows "
        "through 144 complex high coordinates pass direct reconstruction. "
        "Axial length 4 to 8 changes the physical lower witness by only "
        "factor 1.20524, while strip width 3 to 5 changes it by 2.03575 "
        "and slab width by 2.22634. Slab lengths 4,6,8 give normalized "
        "direct witnesses 0.389681, 0.607972, 0.840068. The nonaxial "
        "witnesses are 92 to 99 percent high-high pressure. These are "
        "validated finite growth witnesses, not an analytic divergence "
        "theorem or a global-regularity result."
    )
    principal.update(
        {
            "joint_primitive_HHL_blocks_assembled": True,
            "joint_primitive_HHL_shared_Fisher_charged_once": True,
            "joint_primitive_HHL_real_low_coordinate_count": 52,
            "joint_primitive_HHL_window_count": 8,
            "joint_primitive_HHL_maximum_complex_dimension": 144,
            "joint_primitive_HHL_axial_8_over_4_lower_ratio": growth[
                "axial_length_8_over_4_joint_lower_ratio"
            ],
            "joint_primitive_HHL_strip_5_over_3_lower_ratio": growth[
                "strip_width_5_over_3_joint_lower_ratio"
            ],
            "joint_primitive_HHL_slab_5_over_3_lower_ratio": growth[
                "slab_width_5_over_3_joint_lower_ratio"
            ],
            "joint_primitive_HHL_slab_length_lower_values": [
                row["joint_lower"] for row in growth["slab_length_rows"]
            ],
            "joint_primitive_HHL_slab_linear_fit_slope": growth[
                "slab_length_linear_fit_slope"
            ],
            "joint_primitive_HHL_slab_linear_fit_R_squared": growth[
                "slab_length_linear_fit_R_squared"
            ],
            "joint_primitive_HHL_pressure_growth_witnesses_validated": True,
            "joint_primitive_HHL_analytic_unbounded_family_proved": False,
            "joint_primitive_HHL_uniform_Schur_bound_proved": False,
            "joint_primitive_HHL_cross_shell_HHL_absorbed": False,
            "joint_primitive_HHL_terminal_dual_supremum_controlled": False,
            "joint_primitive_HHL_critical_L3_controlled": False,
            "joint_primitive_HHL_Navier_Stokes_regularity_proved": False,
            "joint_primitive_HHL_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "joint_primitive_HHL_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "joint_primitive_HHL_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "joint_primitive_HHL_monolithic_regression_passed": False,
            "joint_primitive_HHL_resource_mode": arguments.resource_mode,
            "joint_primitive_HHL_worker_count": arguments.worker_count,
            "joint_primitive_HHL_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "joint_primitive_HHL_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "joint_primitive_HHL_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "joint_primitive_HHL_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "joint_primitive_HHL_result_status": result["status"],
            "joint_primitive_hhl_incidence_schur_gate_audit_v1_sha256": (
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
            "Assembled the finite joint primitive HHL incidence blocks "
            "against one unsplit physical Fisher matrix for all 52 real "
            "cube-low coordinates, validated eight direct witnesses, and "
            "isolated pressure-dominated transverse growth without "
            "claiming an asymptotic theorem."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "Operationally, run the atomic one-process 368-test regression in "
        "an admissible resource window. Mathematically, extract an explicit "
        "separable divergence-free high-carrier family from the slab "
        "witnesses and compute its weighted Fisher energy and high-high "
        "pressure HHL load analytically. Prove a divergent lower ratio or "
        "find the cancellation which restores a uniform joint bound. "
        "Cross-shell absorption, terminal dual control, critical L3, "
        "exceptional-set removal, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/run_full_regression_checkpoint.py "
        "--expected-count 368"
    )
    bookmark["next_action"] = (
        "Build the explicit separable pressure-growth family suggested by "
        "the slab witnesses. Use tensor Dirichlet coefficient profiles on "
        "K(H,L,R_y,R_z), retain two divergence-free polarizations, and "
        "derive exact or interval-certified formulas for E_lambda and the "
        "p[h,h]U vertex load. Check longitudinal lengths 4,6,8 against the "
        "stored witnesses before taking an asymptotic limit. If the ratio "
        "diverges, install a rigorous no-go for the joint-Schur route; if "
        "it does not, identify and prove the missing cancellation. Replay "
        "the isolated-chain Hardy theorem and all mandatory adversaries."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 154, "unexpected completed count")
    _require(len(primary) == 543, "unexpected artifact count")
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
