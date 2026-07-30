"""Install the parallel-shear third internal-shell checkpoint."""

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
    "e6903092247c0d7e74ead2cf6db05c87b911c7c8566ea860d1b8b215e8a232b4"
)
AUDIT = (
    ROOT
    / "work/ns_collision/results/"
    "annular_parallel_shear_third_internal_shell_lemma_audit_v1.json"
)
FULL_REGRESSION = (
    ROOT / "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
UPDATER = (
    "work/ns_collision/scripts/"
    "update_annular_parallel_shear_third_internal_shell_bookmark.py"
)
ARTIFACT_HASHES = {
    (
        "work/ns_collision/scripts/"
        "annular_parallel_shear_third_internal_shell_lemma_audit.py"
    ): "8dd5dee3a8fb5276ae7a22e3ceb3e77332f49ada53c6905687bd0d7479bd2c38",
    (
        "work/ns_collision/results/"
        "annular_parallel_shear_third_internal_shell_lemma_audit_v1.json"
    ): "7430d7890cb3df97deccc84e1f98d7bbdb21bb0654b4273ef88c4732fa0144d2",
    (
        "work/ns_collision/notes/"
        "annular_parallel_shear_third_internal_shell_lemma.md"
    ): "f633214880a20eccf4e0415c3cb6d4cdae237b4a0dd322bb758b2a1a2d1eee54",
    (
        "work/ns_collision/tests/"
        "test_annular_parallel_shear_third_internal_shell_lemma.py"
    ): "39f5643db5bbc594b5ccaa72fbf0efd2134d39410a8cdd689c715bf9fdcc5ea6",
    "work/ns_collision/README.md": (
        "3dabeab4152dd36b873300bc2db11c470b65a3315bb41baf30b3d6e8b808afec"
    ),
    (
        "work/ns_collision/results/full_regression_checkpoint_v1.json"
    ): "a3fb86cb4500c27a2099bf6f038ef8e8aa496992cbb24d73a106070faf73063b",
}
NEW_ARTIFACTS = (
    "work/ns_collision/scripts/"
    "annular_parallel_shear_third_internal_shell_lemma_audit.py",
    "work/ns_collision/results/"
    "annular_parallel_shear_third_internal_shell_lemma_audit_v1.json",
    "work/ns_collision/notes/"
    "annular_parallel_shear_third_internal_shell_lemma.md",
    "work/ns_collision/tests/"
    "test_annular_parallel_shear_third_internal_shell_lemma.py",
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
    parser.add_argument("--focused-test-count", type=int, default=7)
    parser.add_argument("--focused-test-seconds", type=float, required=True)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument("--closeout-average", type=float, required=True)
    parser.add_argument("--closeout-peak", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    for path, expected in ARTIFACT_HASHES.items():
        _require(_sha256(path) == expected, f"{path} changed")

    audit = _load_json(AUDIT)
    topology = audit["dangerous_topology_ledger"]
    allocation = audit["difference_allocation_certificate"]
    closure = audit["power_closure_certificate"]
    explicit = audit["explicit_constant_certificate"]
    flags = audit["certification_flags"]
    _require(
        audit.get("status") == "passed"
        and audit.get("all_positive_checks_pass") is True
        and audit.get("algorithm_revision")
        == (
            "annular-parallel-shear-third-internal-shell-lemma-v2-"
            "fixed-output-correction"
        )
        and audit["tree_expansion_certificate"][
            "total_functional_absolute_coefficient_mass"
        ]
        == 1412
        and topology["protected_pressure_assignment_count"] == 579
        and topology["expanded_pressure_exception_count"] == 30
        and topology[
            "protected_velocity_Fisher_assignment_count"
        ]
        == 81
        and topology[
            "protected_four_high_rows_with_fixed_bounded_B_node"
        ]
        == 4
        and topology[
            "protected_four_high_maximum_fixed_bounded_B_nodes"
        ]
        == 1
        and topology["six_high_fixed_bounded_B_node_mass"] == 0
        and topology["post_resonance_topology_failures"] == []
        and allocation["protected_four_high_minimum_gain"] == 1
        and allocation["all_high_pressure_minimum_gain"] == 4
        and closure["maximum_final_power"] == 11
        and explicit["C0_decimal_digits"] == 422
        and explicit["finite_shell_count_charged_per_factor"] is True
        and flags["complete_restart_time_third_O_N11_proved"] is True
        and flags["uniform_parabolic_window_third_O_N11_proved"]
        is False
        and flags["finite_time_blowup_proved"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "third internal-shell theorem or its fail-closed scope changed",
    )

    regression = _load_json(FULL_REGRESSION)
    _require(
        regression.get("successful") is True
        and regression.get("discovered_test_count") == 519
        and regression.get("tests_run") == 519
        and regression.get("passed_count") == 519
        and regression.get("skipped_count") == 0
        and regression.get("failures") == []
        and regression.get("errors") == []
        and regression.get("expected_count_matched") is True
        and regression.get("exit_code") == 0
        and regression.get("runtime", {}).get("active_worker_count") == 1
        and regression.get("runtime", {}).get(
            "below_normal_priority_set"
        )
        is True,
        "the 519-test full regression is not complete",
    )
    _require(
        arguments.focused_test_count == 7
        and arguments.baseline_average <= 40.0
        and arguments.closeout_average <= 75.0,
        "resource-policy measurements do not match the run",
    )

    bookmark = _load_json(BOOKMARK)
    _require(
        _sha256(BOOKMARK) == PREDECESSOR_BOOKMARK_SHA256
        and bookmark.get("kind")
        == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision"
        and len(bookmark.get("completed_obligations", [])) == 175
        and len(bookmark.get("primary_artifacts", [])) == 648,
        "predecessor bookmark does not match the first shell checkpoint",
    )

    obligation = (
        "Expanded the complete third state and scalar trees with total "
        "absolute coefficient mass 1412; enumerated 579 protected "
        "pressure assignments, 30 expanded pressure exceptions, and 81 "
        "protected Fisher assignments. Proved a one-power single-factor "
        "bound for every four-high row without assuming independent "
        "shells, and proved the four-power all-high bound from the "
        "strictly nested post-resonance shell topology. This closes "
        "|g'''_N(0)|<=C0 max(nu,nu^-1)^13 N11 with an explicit 422-digit "
        "C0. Uniform propagation on the parabolic window remains open."
    )
    completed = bookmark.setdefault("completed_obligations", [])
    obsolete_obligation = obligation.replace("422-digit", "386-digit")
    if obsolete_obligation in completed:
        completed[completed.index(obsolete_obligation)] = obligation
    else:
        _append_once(completed, obligation)
    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in NEW_ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 175, "unexpected completed count")
    _require(len(primary) == 648, "unexpected artifact count")

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["parallel_shear_third_internal_shell_checkpoint"] = {
        "state": "completed",
        "focused_test_count": arguments.focused_test_count,
        "focused_test_seconds": arguments.focused_test_seconds,
        "full_regression_expected_test_count": 519,
        "full_regression_passed_test_count": 519,
        "full_regression_duration_seconds": regression[
            "duration_seconds"
        ],
        "active_worker_count": 1,
        "below_normal_priority_set": regression["runtime"][
            "below_normal_priority_set"
        ],
        "baseline_average_cpu_percent": arguments.baseline_average,
        "baseline_peak_cpu_percent": arguments.baseline_peak,
        "closeout_average_cpu_percent": arguments.closeout_average,
        "closeout_peak_cpu_percent": arguments.closeout_peak,
        "periodic_cpu_samples_recorded": False,
        "worker_stopped": True,
        "audit_sha256": _sha256(AUDIT),
        "full_regression_sha256": _sha256(FULL_REGRESSION),
        "resume_command": (
            "python work/ns_collision/scripts/"
            "annular_parallel_shear_third_internal_shell_lemma_audit.py"
        ),
    }
    bookmark["parallel_shear_full_regression_deferred_calculation"] = {
        "state": "completed",
        "expected_test_count": 519,
        "focused_test_count": arguments.focused_test_count,
        "attempted_progress_percent": 100,
        "active_worker_count": 1,
        "below_normal_priority_set": True,
        "owned_process_stopped": True,
        "current_checkpoint_sha256": _sha256(FULL_REGRESSION),
        "duration_seconds": regression["duration_seconds"],
        "resume_command": (
            "python work/ns_collision/scripts/"
            "run_full_regression_checkpoint.py --expected-count 519 "
            "--verbosity 0"
        ),
    }
    bookmark["validated_checkpoint"] = (
        "The exact third-tree expansion has coefficient mass 1412. The "
        "dangerous ledger contains 579 protected pressure assignments, "
        "30 expanded bounded-output pressure exceptions, and 81 "
        "protected Fisher assignments. Exactly four four-high paths "
        "contain one bounded internal Euler output; none of the 20 "
        "six-high paths does, and their free complement shells are "
        "strictly nested. Boundary-safe packet differences give one "
        "power on every four-high route and four raw powers on the "
        "all-high route. Therefore |g'''_N(0)|<=C0 "
        "max(nu,nu^-1)^13 N11 for the explicit 422-digit C0. All 7 "
        "focused and 519 standalone tests pass. The uniform parabolic-"
        "window third bound remains open."
    )
    principal = bookmark.setdefault("principal_results", {})
    principal.update(
        {
            "annular_parallel_shear_third_tree_coefficient_mass": 1412,
            (
                "annular_parallel_shear_third_"
                "protected_pressure_assignment_count"
            ): 579,
            (
                "annular_parallel_shear_third_"
                "expanded_pressure_exception_count"
            ): 30,
            (
                "annular_parallel_shear_third_"
                "protected_Fisher_assignment_count"
            ): 81,
            (
                "annular_parallel_shear_third_"
                "four_high_fixed_B_assignment_count"
            ): 4,
            (
                "annular_parallel_shear_third_"
                "six_high_fixed_B_assignment_count"
            ): 0,
            "annular_parallel_shear_third_four_high_minimum_gain": 1,
            "annular_parallel_shear_third_six_high_minimum_gain": 4,
            "annular_parallel_shear_restart_third_O_N11_proved": True,
            "annular_parallel_shear_restart_third_C0_digits": 422,
            (
                "annular_parallel_shear_restart_"
                "third_C0_leading_16_digits"
            ): explicit["C0_leading_16_digits"],
            "annular_parallel_shear_uniform_third_O_N11_proved": False,
            "annular_parallel_shear_uniform_Taylor_remainder_proved": False,
            "annular_parallel_shear_parabolic_window_closed": False,
            "annular_parallel_shear_finite_time_blowup_proved": False,
            "annular_parallel_shear_global_regularity_proved": False,
            (
                "annular_parallel_shear_"
                "third_internal_shell_lemma_audit_v1_sha256"
            ): _sha256(AUDIT),
            (
                "annular_parallel_shear_"
                "third_internal_shell_focused_test_count"
            ): arguments.focused_test_count,
            "annular_parallel_shear_full_regression_519_passed": True,
            "annular_parallel_shear_full_regression_519_pending": False,
            "full_regression_checkpoint_v1_sha256": _sha256(
                FULL_REGRESSION
            ),
            (
                "annular_parallel_shear_"
                "third_internal_shell_updater_sha256"
            ): _sha256(UPDATER),
        }
    )
    bookmark["unfinished_obligation"] = (
        "The restart-time third derivative is now closed at O(N11), but "
        "that constant has not been propagated along the evolving "
        "coupled Navier-Stokes/adjoint trajectory. Prove a uniform "
        "sup_(0<=s<=T/N2)|g'''_N(s)|<=C3 N11 with C3 independent of N, "
        "then compare T with c2/(2C3). Critical L3 control, finite-time "
        "blowup, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_parallel_shear_third_internal_shell_lemma_audit.py"
    )
    bookmark["next_action"] = (
        "Build the first dynamic parabolic-window bootstrap ledger. "
        "Express the evolved high packet, low optimizer, and adjoint "
        "weight by Duhamel formulas on 0<=s<=T/N2; identify which "
        "restart-time packet-difference and nested-shell constants remain "
        "uniform, and isolate every commutator or support-spreading term "
        "that could exceed the N11 budget before attempting a sign "
        "turnaround theorem."
    )

    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary),
                "audit_sha256": _sha256(AUDIT),
                "full_regression_sha256": _sha256(FULL_REGRESSION),
                "updater_sha256": _sha256(UPDATER),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
