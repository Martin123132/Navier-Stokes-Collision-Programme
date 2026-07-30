"""Install the two-shear static-optimizer route-guard checkpoint."""

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
    "d8660838a42d26dce6e05cfc4c4bca2b7a59dd4235a6fb0d5070e98729751adf"
)
AUDIT_SCRIPT = (
    "work/ns_collision/scripts/"
    "annular_two_shear_static_optimizer_audit.py"
)
AUDIT_SCRIPT_SHA256 = (
    "f23838c97db1e40cf77bca31079c9ce85fe5c6df57b2fc7f063e4ff21a12b16c"
)
AUDIT_RESULT = (
    "work/ns_collision/results/"
    "annular_two_shear_static_optimizer_audit_v1.json"
)
AUDIT_RESULT_SHA256 = (
    "de11a2c2db530b81ee5c01c12a1270ebb8af8c957e28ed6676fd692f5d7a131e"
)
AUDIT_NOTE = (
    "work/ns_collision/notes/annular_two_shear_static_optimizer.md"
)
AUDIT_NOTE_SHA256 = (
    "f934911b00b999e45ca0e3f9508101fda5b9cbec5ea584d0a7aa8fc87014bf98"
)
AUDIT_TEST = (
    "work/ns_collision/tests/"
    "test_annular_two_shear_static_optimizer.py"
)
AUDIT_TEST_SHA256 = (
    "3e1dde8c6f2f02f82a8c3dfa3d8d988c55f19ce2a101aec43c31e62ddee1be84"
)
README = "work/ns_collision/README.md"
README_SHA256 = (
    "dbdb26ffad3dcd6c3ce2c3a9bbc4662a8f22d65bde1961b5f3cda07a476c636d"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "45203e35d4caad35ee7c7b50dac5d4985b9cb9951f99019237c13557625d4857"
)
UPDATER = (
    "work/ns_collision/scripts/"
    "update_annular_two_shear_static_optimizer_bookmark.py"
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
    parser.add_argument("--focused-test-count", type=int, default=6)
    parser.add_argument("--focused-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=480)
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
        == "annular-two-shear-static-optimizer-v1"
        and audit.get("all_positive_checks_pass") is True,
        "two-shear static optimizer audit did not pass",
    )
    symbolic = audit["exact_symbolic_low_field_certificate"]
    plus = symbolic["plus_vertex_exact"]
    _require(
        symbolic["all_symbolic_checks_pass"]
        and plus["weighted_Fisher_mass"] == "17/16"
        and plus["complete_flux_load"] == "-sqrt(2)/12"
        and plus["pressure_flux_load"] == "-sqrt(2)/48"
        and symbolic["combined_L2_mass"] == "4",
        "exact low-field certificate changed",
    )
    support = audit["full_field_support_replay"]
    _require(
        support["all_support_replay_checks_pass"]
        and support["maximum_complete_polynomial_residual"] < 3.0e-12
        and support["maximum_pressure_polynomial_residual"] < 3.0e-12
        and support["maximum_Fisher_polynomial_residual"] < 3.0e-12,
        "full-field support replay changed",
    )
    optimizer = audit["optimizer_and_scaling_certificate"]
    restart = optimizer["restart_scaling_decision"]
    flags = audit["certification_flags"]
    _require(
        optimizer["finite_N_joint_optimization"]["joint_supremum"]
        == "+infinity"
        and not optimizer["finite_N_joint_optimization"][
            "finite_stationary_optimizer_exists"
        ]
        and not restart["old_a_and_t_Theta_N_scaling_ports_unchanged"]
        and not restart[
            "old_Omega_N5_average_generator_gate_ports_unchanged"
        ]
        and flags["joint_static_supremum_is_infinite"]
        and not flags["phase_cancellation_gate_proved"]
        and not flags["complete_finite_first_jet_ported"]
        and not flags["complete_finite_second_jet_ported"]
        and not flags["Navier_Stokes_global_regularity_proved"],
        "optimizer conclusion or fail-closed scope changed",
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
        arguments.focused_test_count == 6
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
        and len(bookmark.get("completed_obligations", [])) == 168
        and len(bookmark.get("primary_artifacts", [])) == 614
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 169
        and len(bookmark.get("primary_artifacts", [])) == 619
        and principal.get(
            "annular_two_shear_static_optimizer_audit_v1_sha256"
        )
        == AUDIT_RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed static checkpoint matches",
    )

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The modified two-shear static optimizer has been ported without "
        "reusing one-shear constants. Exact symbolic enumeration gives "
        "mean(Phi_+++|grad U_*|^2)=17/16, ||U_*||_2^2=4, complete "
        "low-only flux -sqrt(2)/12, and pressure-only rho=0 low flux "
        "-sqrt(2)/48. Full finite Fourier replay proves the only visible "
        "terms are HHL, high Fisher, low Fisher, and this low cubic; HLL "
        "and high-low Fisher are excluded by support. For "
        "u_N=h_N-aU_*, both the complete and pressure-only static "
        "objectives are unbounded above as a->infinity for every fixed "
        "N>=3 and t>0. Thus the old finite a,t=Theta(N) optimizer and its "
        "Omega(N^3)/Omega(N^5) restart scaling do not port. At a=Theta(N), "
        "the new coefficient optimum is Theta(N^(3/2)) and the objective "
        "Theta(N^(9/2)). The phase/polarization cancellation gate, finite "
        "jets, dynamic restart, critical L3 control, blowup, and global "
        "regularity remain open. All 6 focused tests and all 480 standalone "
        "tests pass."
    )
    principal.update(
        {
            "annular_two_shear_static_optimizer_ported": True,
            "annular_two_shear_low_weighted_Fisher_mass": "17/16",
            "annular_two_shear_low_L2_mass": "4",
            "annular_two_shear_complete_self_flux": "-sqrt(2)/12",
            "annular_two_shear_pressure_self_flux": "-sqrt(2)/48",
            "annular_two_shear_finite_static_optimizer_exists": False,
            "annular_two_shear_joint_static_supremum": "+infinity",
            "annular_two_shear_old_restart_scaling_ports": False,
            "annular_two_shear_amplitude_N_coefficient_scale": (
                "Theta(N^(3/2))"
            ),
            "annular_two_shear_amplitude_N_optimized_objective": (
                "Theta(N^(9/2))"
            ),
            "annular_two_shear_phase_cancellation_gate_proved": False,
            "annular_two_shear_complete_finite_jets_ported": False,
            "annular_two_shear_uniform_Taylor_remainder_proved": False,
            "annular_two_shear_parabolic_window_closed": False,
            "annular_two_shear_static_optimizer_audit_v1_sha256": (
                AUDIT_RESULT_SHA256
            ),
            "annular_two_shear_static_optimizer_focused_test_count": (
                arguments.focused_test_count
            ),
            (
                "annular_two_shear_static_optimizer_"
                "focused_test_runtime_seconds"
            ): arguments.focused_test_seconds,
            "annular_two_shear_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_two_shear_full_pytest_passed": True,
            "annular_two_shear_full_pytest_runtime_seconds": float(
                regression["duration_seconds"]
            ),
            "annular_two_shear_static_optimizer_resource_mode": (
                arguments.resource_mode
            ),
            "annular_two_shear_static_optimizer_worker_count": (
                arguments.worker_count
            ),
            "annular_two_shear_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "annular_two_shear_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "annular_two_shear_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "annular_two_shear_cpu_periodic_peak_percent": (
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
            "Ported the modified two-shear static optimizer exactly: "
            "enumerated the 17/16 low Fisher mass and the complete and "
            "pressure-only self-fluxes -sqrt(2)/12 and -sqrt(2)/48, "
            "replayed the complete finite support polynomial, and proved "
            "that the joint static objective is unbounded rather than "
            "having the old finite a,t=Theta(N) optimizer."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The two-shear sign repair creates a favorable low-only cubic, so "
        "the old static optimizer and reset-tax scaling are invalid. The "
        "next gate must classify relative low-mode phase, amplitude, and "
        "polarization jointly with the strict HHL and four-high signs. "
        "Either find an exact self-flux-zero point inside the strict "
        "negative-square region or prove none exists in the admissible "
        "two-mode family. Only after that decision should the complete "
        "finite jets and backward-adjoint restart be ported. Uniform Taylor "
        "control, critical L3 control, blowup, and global regularity remain "
        "open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_two_shear_static_optimizer_audit.py"
    )
    bookmark["next_action"] = (
        "Build an exact symbolic phase/polarization audit for "
        "U_yz+c U_xy with the reality and divergence constraints retained. "
        "Compute the complete and pressure-only low self-flux polynomials, "
        "the weighted Fisher Gram matrix, the static HHL matrix, and the "
        "four-high curvature functional. Decide whether a self-flux-zero "
        "parameter lies in a quantitatively strict negative-square region; "
        "do not port the finite jets until this gate is resolved."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in NEW_ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 169, "unexpected completed count")
    _require(len(primary) == 619, "unexpected artifact count")
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
