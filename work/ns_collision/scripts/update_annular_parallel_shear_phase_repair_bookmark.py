"""Install the annular parallel-shear phase-repair checkpoint."""

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
PREDECESSOR_BOOKMARK_SHA256 = (
    "5ee0e936ea361ec8ce7a60ff27eef41a415f62a93f1fb90a24f0d67ccffe15d8"
)
AUDIT_SCRIPT = (
    "work/ns_collision/scripts/"
    "annular_parallel_shear_phase_repair_audit.py"
)
AUDIT_SCRIPT_SHA256 = (
    "e357a352e1679e20ba5e7f1613d56743bf037525440893f121d53e8bcc28e8e8"
)
AUDIT_RESULT = (
    "work/ns_collision/results/"
    "annular_parallel_shear_phase_repair_audit_v1.json"
)
AUDIT_RESULT_SHA256 = (
    "ab0d58bb824520167a90083795f0913da1cc9ca7b50e5e785ae7192f9f14efbd"
)
AUDIT_NOTE = (
    "work/ns_collision/notes/"
    "annular_parallel_shear_phase_repair.md"
)
AUDIT_NOTE_SHA256 = (
    "8890a9ee9786c24bc0006723f83740c9bf77998886ed2753021447fd84ffca2e"
)
AUDIT_TEST = (
    "work/ns_collision/tests/"
    "test_annular_parallel_shear_phase_repair.py"
)
AUDIT_TEST_SHA256 = (
    "564808340ef35818d6d94dee3b3b7cd31f6b77ddf7bc1a33874fd048c2399d1f"
)
README = "work/ns_collision/README.md"
README_SHA256 = (
    "7d6cd4bd28bbfbdcb3824eea32f2362f9387d9c960ab7424c2e5f65cd871abf0"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "751ff7663abe66c7580205c674bfe09733fdfc539fbaaefef061b696e983209d"
)
UPDATER = (
    "work/ns_collision/scripts/"
    "update_annular_parallel_shear_phase_repair_bookmark.py"
)
NEW_ARTIFACTS = (
    AUDIT_SCRIPT,
    AUDIT_RESULT,
    AUDIT_NOTE,
    AUDIT_TEST,
    UPDATER,
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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
    parser.add_argument("--focused-test-count", type=int, default=8)
    parser.add_argument("--focused-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=488)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument("--periodic-average", type=float, required=True)
    parser.add_argument("--periodic-peak", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    expected_hashes = {
        AUDIT_SCRIPT: AUDIT_SCRIPT_SHA256,
        AUDIT_RESULT: AUDIT_RESULT_SHA256,
        AUDIT_NOTE: AUDIT_NOTE_SHA256,
        AUDIT_TEST: AUDIT_TEST_SHA256,
        README: README_SHA256,
        FULL_REGRESSION: FULL_REGRESSION_SHA256,
    }
    for path, expected in expected_hashes.items():
        _require(_sha256(path) == expected, f"{path} changed")

    audit = _load_json(AUDIT_RESULT)
    regression = _load_json(FULL_REGRESSION)
    _require(
        audit.get("algorithm_revision")
        == "annular-parallel-shear-phase-repair-v1"
        and audit.get("all_positive_checks_pass") is True,
        "parallel-shear phase repair audit did not pass",
    )
    phase = audit["scalar_phase_family"]
    polarization = audit["exact_square_polarization_family"]
    parallel = audit["parallel_shear_repair"]
    stencil = audit["stencil_and_curvature_symmetry"]
    optimizer = audit["optimizer_and_restart_certificate"]
    tail = audit["parallel_full_c1_tail_port"]
    flags = audit["certification_flags"]
    _require(
        phase["all_phase_checks_pass"]
        and polarization["all_polarization_checks_pass"]
        and polarization["common_zero_branch"] == "a=d+2b"
        and audit["diagonal_cosine_no_go"][
            "all_diagonal_cosine_checks_pass"
        ],
        "phase or polarization classification changed",
    )
    _require(
        parallel["all_parallel_shear_checks_pass"]
        and parallel["weighted_Fisher"] == "9/8"
        and parallel["L2_mass"] == "4"
        and parallel["pressure_self_flux_load"] == "0"
        and parallel["complete_self_flux_load"] == "0",
        "parallel low-field certificate changed",
    )
    _require(
        stencil["all_stencil_symmetry_checks_pass"]
        and stencil["four_high_reduction"]["reduced_formula"]
        == "-sqrt(3)*Cyy/10"
        and all(
            row["all_curvature_replay_checks_pass"]
            for row in audit["curvature_matrix_replays"]
        ),
        "stencil or curvature symmetry certificate changed",
    )
    _require(
        optimizer["all_optimizer_restart_checks_pass"]
        and optimizer["reset_deficit_port"]["ratio_formula"]
        == "5/(36nu)"
        and tail["all_tail_port_checks_pass"]
        and flags["parallel_finite_static_optimizer_restored"]
        and flags["parallel_reset_N5_gate_restored"]
        and flags["parallel_full_c1_limit_negative"]
        and not flags["parallel_complete_finite_first_jet_ported"]
        and not flags["parallel_complete_finite_second_jet_ported"]
        and not flags["Navier_Stokes_global_regularity_proved"],
        "optimizer, tail, or fail-closed scope changed",
    )

    _require(
        regression.get("schema_version") == 2
        and regression.get("configuration", {}).get("test_engine")
        == "pytest"
        and regression.get("configuration", {}).get("expected_count")
        == arguments.discovered_test_count
        and regression.get("discovered_test_count")
        == arguments.discovered_test_count
        and regression.get("tests_run") == arguments.discovered_test_count
        and regression.get("passed_count")
        == arguments.discovered_test_count
        and regression.get("skipped_count") == 0
        and regression.get("successful") is True
        and regression.get("exit_code") == 0
        and not regression.get("failures")
        and not regression.get("errors"),
        "full pytest regression did not pass",
    )
    _require(
        arguments.focused_test_count == 8
        and arguments.worker_count == 1
        and arguments.baseline_average <= 60.0
        and arguments.periodic_average <= 75.0,
        "resource-policy measurements do not permit installation",
    )

    bookmark = _load_json(BOOKMARK)
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    principal = bookmark.setdefault("principal_results", {})
    predecessor = bool(
        _sha256(BOOKMARK) == PREDECESSOR_BOOKMARK_SHA256
        and len(bookmark.get("completed_obligations", [])) == 169
        and len(bookmark.get("primary_artifacts", [])) == 619
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 170
        and len(bookmark.get("primary_artifacts", [])) == 624
        and principal.get(
            "annular_parallel_shear_phase_repair_audit_v1_sha256"
        )
        == AUDIT_RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed phase-repair checkpoint matches",
    )

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The two-shear low self-flux obstruction now has an exact "
        "polarization repair. Scalar relative phases cannot cancel the "
        "self-flux in the strict-square quadrant. In the full "
        "divergence-free polarization family, pressure and complete "
        "self-fluxes share the zero branch a=d+2b. The canonical point "
        "uses r=(1,1,1)/sqrt(3), with yz polarization +r and xy "
        "polarization -r. Then U_*=r f with r.grad f=0: pressure and "
        "self-advection vanish pointwise, and the complete energy-flux "
        "divergence vanishes. The exact +++ Fisher mass is 9/8 and L2 "
        "mass is 4. Reflection symmetries force the modified high energy "
        "and curvature tensors diagonal, so the added off-diagonal stencil "
        "entries vanish in both signs. The static limit is "
        "-||b||2^2/(10sqrt(3)); the complete c1 limit is "
        "-(sqrt(3)/10)||v_y||2^2. The finite static optimizer is restored "
        "from N=25 in audited rows, and the reset ratio is 5/(36nu), "
        "restoring the Omega(N^5) heat-window gate. Complete finite jets, "
        "uniform Taylor control, critical L3 control, blowup, and global "
        "regularity remain open. All 8 focused tests and all 488 standalone "
        "tests pass."
    )
    principal.update(
        {
            "annular_canonical_low_field": (
                "U_*=2r[sin(ell_yz.x)-sin(ell_xy.x)], "
                "r=(1,1,1)/sqrt(3)"
            ),
            "annular_parallel_shear_scalar_phase_repair_exists": False,
            "annular_parallel_shear_full_polarization_repair_exists": True,
            "annular_parallel_shear_common_zero_branch": "a=d+2b",
            "annular_parallel_shear_pressure_zero": True,
            "annular_parallel_shear_self_advection_zero": True,
            "annular_parallel_shear_energy_flux_divergence_zero": True,
            "annular_parallel_shear_weighted_Fisher_mass": "9/8",
            "annular_parallel_shear_L2_mass": "4",
            "annular_parallel_shear_static_limit": (
                "-||b||_L2(D)^2/(10sqrt(3))<0"
            ),
            "annular_parallel_shear_four_high_limit": (
                "-(sqrt(3)/10)||v_y||_2^2<0"
            ),
            "annular_parallel_shear_full_c1_limit": (
                "c1_parallel,N/N^7 -> "
                "-(sqrt(3)/10)||v_y||_2^2<0"
            ),
            "annular_parallel_shear_tail_constant": 70_657_920,
            "annular_parallel_shear_static_optimizer_restored": True,
            "annular_parallel_shear_reset_ratio": "5/(36nu)",
            "annular_parallel_shear_N5_gate_restored": True,
            "annular_parallel_shear_complete_finite_jets_ported": False,
            "annular_parallel_shear_uniform_Taylor_remainder_proved": False,
            "annular_parallel_shear_parabolic_window_closed": False,
            "annular_parallel_shear_phase_repair_audit_v1_sha256": (
                AUDIT_RESULT_SHA256
            ),
            "annular_parallel_shear_focused_test_count": (
                arguments.focused_test_count
            ),
            "annular_parallel_shear_focused_test_runtime_seconds": (
                arguments.focused_test_seconds
            ),
            "annular_parallel_shear_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_parallel_shear_full_pytest_passed": True,
            "annular_parallel_shear_full_pytest_runtime_seconds": float(
                regression["duration_seconds"]
            ),
            "annular_parallel_shear_resource_mode": (
                arguments.resource_mode
            ),
            "annular_parallel_shear_worker_count": arguments.worker_count,
            "annular_parallel_shear_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "annular_parallel_shear_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "annular_parallel_shear_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "annular_parallel_shear_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "full_regression_checkpoint_v1_sha256": (
                FULL_REGRESSION_SHA256
            ),
        }
    )
    for artifact in (*NEW_ARTIFACTS, README, FULL_REGRESSION):
        parent = Path(artifact).parent.name.replace("-", "_")
        stem = Path(artifact).stem.replace("-", "_")
        principal[f"{parent}_{stem}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Classified the exact two-mode phase and polarization family "
            "and found a canonical common-polarization parallel-shear "
            "repair: proved pointwise low stationarity, zero pressure and "
            "energy-flux divergence, Fisher mass 9/8, reflection-protected "
            "strict static and full-c1 signs, a restored finite optimizer, "
            "and the reset ratio 5/(36nu)."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The common-polarization parallel shear repairs the low self-flux "
        "without losing the strict annular signs and is now the canonical "
        "low field. Its complete finite first and second rho=0 generator "
        "jets have not yet been ported. Enumerate every mixed "
        "polarization channel and its carrier power before asserting that "
        "the negative amplitude-one c1 coefficient controls a heat window. "
        "After the jet port, a uniform second-jet/Taylor remainder, dynamic "
        "adjoint evolution, critical L3 control, blowup, and global "
        "regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_parallel_shear_phase_repair_audit.py"
    )
    bookmark["next_action"] = (
        "Port the complete finite first and second rho=0 generator jets to "
        "the common-polarization low field. Use its exact stationarity and "
        "common Laplacian eigenvalue, but explicitly enumerate all HHHH, "
        "HHHL, HHLL, HLLL, viscous-Fisher, and pressure-cross channels. "
        "Derive the new N-power ledger and only then attempt a uniform "
        "heat-window Taylor remainder."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in NEW_ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 170, "unexpected completed count")
    _require(len(primary) == 624, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary),
                "audit_result_sha256": _sha256(AUDIT_RESULT),
                "full_regression_sha256": _sha256(FULL_REGRESSION),
                "updater_sha256": _sha256(UPDATER),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
