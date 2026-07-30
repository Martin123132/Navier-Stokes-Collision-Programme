"""Install the multiband weighted-Fisher recombination no-go checkpoint."""

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
    "multiband_weighted_fisher_recombination_no_go_audit_v1.json"
)
RESULT_SHA256 = (
    "47b8704985671f0dac66ae38ff87a186acd6b938928828d3299a571337a7f087"
)
BALANCED_RESULT_SHA256 = (
    "9a024a23381d62e7842d7d26406fcea2a5343a168f386d3bad85e5308cef99dd"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "multiband_weighted_fisher_recombination_no_go_audit.py",
    "work/ns_collision/tests/"
    "test_multiband_weighted_fisher_recombination_no_go.py",
    "work/ns_collision/notes/"
    "multiband_weighted_fisher_recombination_no_go.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_multiband_weighted_fisher_no_go_bookmark.py",
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
    parser.add_argument("--discovered-test-count", type=int, default=297)
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
        == "uniform_multiband_weighted_Fisher_recombination_falsified"
        and result.get("all_positive_checks_pass") is True,
        "multiband Fisher no-go result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["uniform_floor_free_multiband_Fisher_recombination"]
        is False
        and flags["finite_overlap_degree_implies_coercivity"] is False
        and flags["signed_neighboring_Fisher_edges_must_be_retained"]
        is True
        and flags["balanced_single_band_pressure_theorem_invalidated"]
        is False
        and flags["joint_signed_pressure_Fisher_block_bound_proved"]
        is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "multiband Fisher no-go scope flags changed",
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
        len(bookmark.get("completed_obligations", [])) == 150
        and len(bookmark.get("primary_artifacts", [])) == 523
        and principal.get(
            "balanced_annular_pressure_edge_gate_audit_v1_sha256"
        )
        == BALANCED_RESULT_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 151
        and len(bookmark.get("primary_artifacts", [])) == 528
        and principal.get(
            "multiband_weighted_fisher_recombination_no_go_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed Fisher checkpoint matches",
    )

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "Uniform floor-free recombination of isolated annular weighted "
        "Fisher costs is now falsified exactly. For the compatible "
        "zero-face weight sin^2(x_1/2), a smooth pressure-free "
        "divergence-free shear split into J dyadic annuli has physical "
        "weighted Fisher 1/4 and component sum J/4; each neighboring "
        "interface restores exactly -1/4. Positive floors tending to zero, "
        "terminal weight Fisher as a fixed additive allowance, and exact "
        "amplitude/partition co-scaling do not repair the uniform step. "
        "The certified single-band and far-tail theorems remain valid. A "
        "joint signed pressure-Fisher block estimate remains open."
    )
    principal.update(
        {
            "multiband_weighted_Fisher_uniform_recombination_proved": False,
            "multiband_finite_overlap_degree_coercive": False,
            "multiband_component_absolute_summation_route_retired": True,
            "multiband_signed_Fisher_interfaces_must_be_retained": True,
            "multiband_exact_component_to_physical_ratio": "J",
            "multiband_exact_interface_correction": "-1/4",
            "multiband_single_band_pressure_theorem_invalidated": False,
            "multiband_far_tail_theorem_invalidated": False,
            "multiband_joint_pressure_Fisher_block_bound_proved": False,
            "multiband_cross_shell_HHL_absorbed": False,
            "multiband_terminal_dual_supremum_controlled": False,
            "multiband_Navier_Stokes_regularity_proved": False,
            "multiband_targeted_test_count": arguments.targeted_test_count,
            "multiband_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "multiband_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "multiband_monolithic_regression_passed": False,
            "multiband_resource_mode": arguments.resource_mode,
            "multiband_worker_count": arguments.worker_count,
            "multiband_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "multiband_cpu_baseline_peak_percent": arguments.baseline_peak,
            "multiband_result_status": result["status"],
            (
                "multiband_weighted_fisher_recombination_no_go_"
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
            "Falsified uniform floor-free multiband weighted-Fisher "
            "recombination with an exact smooth divergence-free dyadic "
            "shear whose component-to-physical ratio is J and whose "
            "neighboring signed interfaces restore the missing energy."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "Operationally, run the atomic one-process 297-test regression in "
        "an admissible resource window. Mathematically, decide pressure "
        "compatibility with the Fisher near-null chains: construct a "
        "pressure-active residue-chain family and determine whether the "
        "complete HHL pressure symbol vanishes on the constant-chain null "
        "direction or admits a transverse bound by the full signed Fisher "
        "graph. Do not split or absolutely sum component Fisher costs. "
        "Terminal dual control, critical L3, exceptional-set removal, and "
        "global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/run_full_regression_checkpoint.py "
        "--expected-count 297"
    )
    bookmark["next_action"] = (
        "Build the pressure-active Fisher-null compatibility gate. Start "
        "from two divergence-free polarizations on a long frequency "
        "residue chain, preserve the exact neighboring Fisher interfaces, "
        "and compute the complete kinetic plus pressure HHL symbol on the "
        "constant and first Dirichlet chain modes. A viable estimate must "
        "vanish on the exact Fisher null direction or pay only the "
        "transverse discrete derivative. Replay Taylor-Green, seed-81, and "
        "the modulated-wave HHL adversary."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 151, "unexpected completed count")
    _require(len(primary) == 528, "unexpected artifact count")
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
