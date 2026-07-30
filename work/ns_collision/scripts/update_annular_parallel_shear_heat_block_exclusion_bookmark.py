"""Install or finalize the parallel-shear heat-block checkpoint."""

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
    "4443201d46e6867250638c1367126346325aea1190c43176b8ea00381b0cd26e"
)
FULL_REGRESSION = (
    ROOT / "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
DEFERRED_FULL_REGRESSION_SHA256 = (
    "6befef38446b8950432e8c3e77cfe6d74ce3c70fe81705042ff6c7b77e5e6f09"
)
UPDATER = (
    "work/ns_collision/scripts/"
    "update_annular_parallel_shear_heat_block_exclusion_bookmark.py"
)
ARTIFACT_HASHES = {
    (
        "work/ns_collision/scripts/"
        "annular_parallel_shear_euler_transport_fisher_exclusion_audit.py"
    ): "42eb5d4601efce92e90187d6ef942bbf30b03c0ec9f1fc2bd91b28a7b3fb8272",
    (
        "work/ns_collision/results/"
        "annular_parallel_shear_euler_transport_fisher_exclusion_"
        "audit_v1.json"
    ): "74722ffabf83612a51fdd0f3ab71e90c7b6fd68c5b4eb15b6b5ed040876e5046",
    (
        "work/ns_collision/notes/"
        "annular_parallel_shear_euler_transport_fisher_exclusion.md"
    ): "d75878f8d1664f79c8c75b4550e827c9e9834425bfa10bbded3ce1a3b87f9df1",
    (
        "work/ns_collision/tests/"
        "test_annular_parallel_shear_euler_transport_fisher_exclusion.py"
    ): "3aa069db9ab9d9c11c78f74df7f0769e22e8bd13e6aec72a25a92c2574858673",
    (
        "work/ns_collision/scripts/"
        "annular_parallel_shear_heat_block_exclusion_audit.py"
    ): "8a35f324bf4526fee54fe6c0561143b99f846f0c0381745ef3fee7391f3df587",
    (
        "work/ns_collision/results/"
        "annular_parallel_shear_heat_block_exclusion_audit_v1.json"
    ): "e3ccdc9a380edf818943450203b0659d0a45b6a5b8255658c5a03681e8213c95",
    (
        "work/ns_collision/notes/"
        "annular_parallel_shear_heat_block_exclusion.md"
    ): "6a4e690ac1c04e1bcceb88e8fb105701136c0c1919383d2967a06c0d552d15c3",
    (
        "work/ns_collision/tests/"
        "test_annular_parallel_shear_heat_block_exclusion.py"
    ): "aa53542a1b969d7c0d8ea4c2bbdc4f15b2e0f3770bb588a7d7e05c70192cde01",
    "work/ns_collision/README.md": (
        "97a9f935202d381c02d945f7ed93b7db89057d1920cda1e7e041f221ff1b6b5e"
    ),
}
NEW_ARTIFACTS = (*ARTIFACT_HASHES.keys(), UPDATER)


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
    parser.add_argument("--focused-test-count", type=int, default=12)
    parser.add_argument("--focused-test-seconds", type=float, required=True)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument("--periodic-first-average", type=float, required=True)
    parser.add_argument("--periodic-first-peak", type=float, required=True)
    parser.add_argument("--periodic-second-average", type=float, required=True)
    parser.add_argument("--periodic-second-peak", type=float, required=True)
    parser.add_argument("--attempted-progress-percent", type=int, default=14)
    parser.add_argument("--owned-process-id", type=int, default=4164)
    parser.add_argument("--finalize-full-regression", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    for path, expected in ARTIFACT_HASHES.items():
        _require(_sha256(path) == expected, f"{path} changed")

    ea_audit = _load_json(
        "work/ns_collision/results/"
        "annular_parallel_shear_euler_transport_fisher_exclusion_"
        "audit_v1.json"
    )
    heat_audit = _load_json(
        "work/ns_collision/results/"
        "annular_parallel_shear_heat_block_exclusion_audit_v1.json"
    )
    _require(
        ea_audit.get("all_positive_checks_pass") is True
        and ea_audit["certification_flags"][
            "all_pure_EA_viscosity_bearing_Fisher_rows_o_N9_proved"
        ]
        is True,
        "Euler/transport Fisher exclusion did not pass",
    )
    _require(
        heat_audit.get("all_positive_checks_pass") is True
        and heat_audit["exhaustive_subterm_partition"][
            "second_subterm_count"
        ]
        == 69
        and heat_audit["certification_flags"][
            "complete_second_N9_limit_certified"
        ]
        is True
        and heat_audit["certification_flags"][
            "uniform_second_jet_Taylor_remainder_proved"
        ]
        is False,
        "heat-block exclusion or fail-closed scope changed",
    )
    resource_measurements_valid = bool(
        arguments.focused_test_count == 12
        and arguments.baseline_average <= 60.0
        and (
            (
                arguments.finalize_full_regression
                and arguments.periodic_first_average <= 75.0
                and arguments.periodic_second_average <= 75.0
            )
            or (
                not arguments.finalize_full_regression
                and arguments.periodic_first_average > 75.0
                and arguments.periodic_second_average > 75.0
            )
        )
    )
    _require(
        resource_measurements_valid,
        "resource-policy measurements do not match the run state",
    )

    full_regression = _load_json(FULL_REGRESSION)
    if arguments.finalize_full_regression:
        _require(
            full_regression.get("successful") is True
            and full_regression.get("discovered_test_count") == 506
            and full_regression.get("tests_run") == 506
            and full_regression.get("passed_count") == 506
            and full_regression.get("skipped_count") == 0
            and full_regression.get("exit_code") == 0
            and full_regression.get("runtime", {}).get(
                "active_worker_count"
            )
            == 1,
            "the 506-test full regression is not complete",
        )
    else:
        _require(
            _sha256(FULL_REGRESSION)
            == DEFERRED_FULL_REGRESSION_SHA256
            and full_regression.get("successful") is True
            and full_regression.get("passed_count") == 494,
            "the preserved predecessor regression checkpoint changed",
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
        and len(bookmark.get("completed_obligations", [])) == 171
        and len(bookmark.get("primary_artifacts", [])) == 629
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 173
        and len(bookmark.get("primary_artifacts", [])) == 638
        and principal.get(
            "annular_parallel_shear_heat_block_exclusion_audit_v1_sha256"
        )
        == ARTIFACT_HASHES[
            "work/ns_collision/results/"
            "annular_parallel_shear_heat_block_exclusion_audit_v1.json"
        ]
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed heat-block checkpoint matches",
    )

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Combined the five pure Euler/transport velocity-Fisher rows "
            "as the second material derivative of the weighted Dirichlet "
            "functional and the seven weight-self subterms as its scalar "
            "companion; proved O(N8) and O(N6), respectively, using the "
            "two-difference vertex lemma and a boundary-safe internal "
            "output split."
        ),
    )
    _append_once(
        completed,
        (
            "Partitioned all 69 atomic second-jet subterms into 21 pure "
            "E/A, 31 one-heat, and 17 two-heat rows; proved every V/D row "
            "O(N8) or lower and certified that the complete restart-time "
            "second jet has the strict negative inviscid-pressure N9 "
            "limit."
        ),
    )
    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in NEW_ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 173, "unexpected completed count")
    _require(len(primary) == 638, "unexpected artifact count")

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["parallel_shear_full_regression_deferred_calculation"] = {
        "state": (
            "completed"
            if arguments.finalize_full_regression
            else "deferred_resource_policy"
        ),
        "expected_test_count": 506,
        "focused_test_count": arguments.focused_test_count,
        "focused_test_seconds": arguments.focused_test_seconds,
        "attempted_progress_percent": (
            100
            if arguments.finalize_full_regression
            else arguments.attempted_progress_percent
        ),
        "baseline_average_cpu_percent": arguments.baseline_average,
        "baseline_peak_cpu_percent": arguments.baseline_peak,
        "periodic_first_average_cpu_percent": (
            arguments.periodic_first_average
        ),
        "periodic_first_peak_cpu_percent": arguments.periodic_first_peak,
        "periodic_second_average_cpu_percent": (
            arguments.periodic_second_average
        ),
        "periodic_second_peak_cpu_percent": arguments.periodic_second_peak,
        "daytime_threshold_percent": 75.0,
        "owned_process_id": arguments.owned_process_id,
        "owned_process_stopped": True,
        "prior_checkpoint_sha256": DEFERRED_FULL_REGRESSION_SHA256,
        "current_checkpoint_sha256": _sha256(FULL_REGRESSION),
        "resume_command": (
            "python work/ns_collision/scripts/"
            "run_full_regression_checkpoint.py --expected-count 506 "
            "--verbosity 0"
        ),
        "finalize_command": (
            "python work/ns_collision/scripts/"
            "update_annular_parallel_shear_heat_block_exclusion_"
            "bookmark.py --focused-test-seconds "
            f"{arguments.focused_test_seconds} --baseline-average "
            f"{arguments.baseline_average} --baseline-peak "
            f"{arguments.baseline_peak} --periodic-first-average "
            f"{arguments.periodic_first_average} --periodic-first-peak "
            f"{arguments.periodic_first_peak} --periodic-second-average "
            f"{arguments.periodic_second_average} --periodic-second-peak "
            f"{arguments.periodic_second_peak} --finalize-full-regression"
        ),
    }
    bookmark["validated_checkpoint"] = (
        "The five pure Euler/transport velocity-Fisher rows and seven "
        "weight-self subterms now have exact material-derivative "
        "reductions and are O(N8) or lower. The 69 atomic second-jet "
        "subterms are exhaustively partitioned as 21 pure E/A, 31 "
        "one-heat, and 17 two-heat rows. The one-heat HHHH pressure "
        "branch retains two compatible Phi differences and all other "
        "heat rows are subcritical by incidence and derivative count. "
        "Thus the complete restart-time second jet has the strict "
        "negative N9 inviscid-pressure limit. All 12 focused tests pass. "
        + (
            "All 506 standalone tests also pass."
            if arguments.finalize_full_regression
            else (
                "The 506-test full regression is deferred: its one-worker "
                "run was stopped at 14 percent after two sustained daytime "
                "CPU samples above 75 percent; the prior 494-test "
                "checkpoint remains unchanged."
            )
        )
    )
    principal.update(
        {
            "annular_parallel_shear_pure_EA_Fisher_o_N9_certified": True,
            "annular_parallel_shear_all_69_second_subterms_partitioned": (
                True
            ),
            "annular_parallel_shear_one_heat_o_N9_certified": True,
            "annular_parallel_shear_two_heat_o_N9_certified": True,
            "annular_parallel_shear_complete_second_N9_certified": True,
            "annular_parallel_shear_complete_second_N9_limit": (
                heat_audit["complete_second_jet_asymptotic"][
                    "inviscid_N9_limit"
                ]
            ),
            "annular_parallel_shear_uniform_Taylor_remainder_proved": (
                False
            ),
            "annular_parallel_shear_parabolic_window_closed": False,
            "annular_parallel_shear_finite_time_blowup_proved": False,
            "annular_parallel_shear_global_regularity_proved": False,
            (
                "annular_parallel_shear_"
                "euler_transport_fisher_exclusion_audit_v1_sha256"
            ): ARTIFACT_HASHES[
                "work/ns_collision/results/"
                "annular_parallel_shear_euler_transport_fisher_"
                "exclusion_audit_v1.json"
            ],
            (
                "annular_parallel_shear_"
                "heat_block_exclusion_audit_v1_sha256"
            ): ARTIFACT_HASHES[
                "work/ns_collision/results/"
                "annular_parallel_shear_heat_block_exclusion_audit_v1.json"
            ],
            "annular_parallel_shear_heat_block_focused_test_count": (
                arguments.focused_test_count
            ),
            "annular_parallel_shear_heat_block_focused_test_seconds": (
                arguments.focused_test_seconds
            ),
            "annular_parallel_shear_full_regression_506_passed": (
                arguments.finalize_full_regression
            ),
            "annular_parallel_shear_full_regression_pending": (
                not arguments.finalize_full_regression
            ),
            "annular_parallel_shear_heat_block_resource_mode": (
                (
                    "daytime_one_worker_throttled_four_cpu_affinity"
                    if arguments.finalize_full_regression
                    else "daytime_one_worker_parked_on_sustained_cpu"
                )
            ),
            "annular_parallel_shear_heat_block_cpu_baseline_average": (
                arguments.baseline_average
            ),
            "annular_parallel_shear_heat_block_cpu_baseline_peak": (
                arguments.baseline_peak
            ),
            "annular_parallel_shear_heat_block_cpu_periodic_first_average": (
                arguments.periodic_first_average
            ),
            "annular_parallel_shear_heat_block_cpu_periodic_second_average": (
                arguments.periodic_second_average
            ),
            "annular_parallel_shear_heat_block_updater_sha256": _sha256(
                UPDATER
            ),
        }
    )
    if arguments.finalize_full_regression:
        principal["full_regression_checkpoint_v1_sha256"] = _sha256(
            FULL_REGRESSION
        )

    if arguments.finalize_full_regression:
        bookmark["unfinished_obligation"] = (
            "The complete restart-time second N9 asymptotic is certified "
            "and all 506 tests pass. The next open theorem is a uniform "
            "third-order/integral Taylor remainder on 0<=s<=T/N^2; after "
            "that, dynamic adjoint evolution, critical L3 control, "
            "finite-time blowup, and global regularity remain open."
        )
        bookmark["resume_command"] = (
            "python work/ns_collision/scripts/"
            "annular_parallel_shear_heat_block_exclusion_audit.py"
        )
        bookmark["next_action"] = (
            "Build the exact third-jet degree/channel ledger and a uniform "
            "integral-remainder bound on the parabolic window. Do not "
            "claim turnaround from restart-time derivatives alone."
        )
    else:
        bookmark["unfinished_obligation"] = (
            "The complete restart-time second N9 asymptotic is certified "
            "by the focused audits, but the expanded 506-test standalone "
            "regression remains pending after a resource-policy park. "
            "After that validation, build the uniform third-order/integral "
            "Taylor remainder on 0<=s<=T/N^2. Dynamic adjoint evolution, "
            "critical L3 control, finite-time blowup, and global "
            "regularity remain open."
        )
        bookmark["resume_command"] = (
            "python work/ns_collision/scripts/"
            "run_full_regression_checkpoint.py --expected-count 506 "
            "--verbosity 0"
        )
        bookmark["next_action"] = (
            "When the daytime CPU baseline is at most 60 percent, rerun "
            "the one-worker 506-test full regression below normal "
            "priority. If it passes, rerun this updater with "
            "--finalize-full-regression; only then begin the third-jet "
            "Taylor-remainder stage."
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
                "full_regression_finalized": (
                    arguments.finalize_full_regression
                ),
                "ea_audit_sha256": _sha256(
                    "work/ns_collision/results/"
                    "annular_parallel_shear_euler_transport_fisher_"
                    "exclusion_audit_v1.json"
                ),
                "heat_audit_sha256": _sha256(
                    "work/ns_collision/results/"
                    "annular_parallel_shear_heat_block_exclusion_"
                    "audit_v1.json"
                ),
                "updater_sha256": _sha256(UPDATER),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
