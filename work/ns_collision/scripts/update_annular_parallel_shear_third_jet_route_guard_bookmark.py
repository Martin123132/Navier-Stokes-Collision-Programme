"""Install the parallel-shear third-jet route-guard checkpoint."""

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
    "69ca148e8435b3db8b553f7a286da3e6d61114f2209d03ea6f2429943bbe5411"
)
FULL_REGRESSION = (
    ROOT / "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
UPDATER = (
    "work/ns_collision/scripts/"
    "update_annular_parallel_shear_third_jet_route_guard_bookmark.py"
)
ARTIFACT_HASHES = {
    (
        "work/ns_collision/scripts/"
        "annular_parallel_shear_third_jet_route_guard_audit.py"
    ): "5a02575625266cb9bd4c65caa90f2b38b4ce3b7235b06ea7c8b80d278356b3dd",
    (
        "work/ns_collision/results/"
        "annular_parallel_shear_third_jet_route_guard_audit_v1.json"
    ): "ab1a95c7d4892122725a7dc0918eb3f1362fb998e9c580bf9b3fce4ea61bd2f2",
    (
        "work/ns_collision/notes/"
        "annular_parallel_shear_third_jet_route_guard.md"
    ): "d3cd34806ba0e93f8ef5f134271d3c45f81c06491631c1d4e15963beaea70d5e",
    (
        "work/ns_collision/tests/"
        "test_annular_parallel_shear_third_jet_route_guard.py"
    ): "24f29b2414036d1276d6b1b01c89dd053755442b95cd67e09cb6e72945efedb0",
    "work/ns_collision/README.md": (
        "02b373531aa870145c18984126269c193c529e2fe9c67fec57697b1d96f4b8a2"
    ),
    (
        "work/ns_collision/results/full_regression_checkpoint_v1.json"
    ): "e8bbd2697997c431a3a68a44ed3341e50ab98d9a63c51382d5d24175e57a807b",
}
NEW_ARTIFACTS = (
    "work/ns_collision/scripts/"
    "annular_parallel_shear_third_jet_route_guard_audit.py",
    "work/ns_collision/results/"
    "annular_parallel_shear_third_jet_route_guard_audit_v1.json",
    "work/ns_collision/notes/"
    "annular_parallel_shear_third_jet_route_guard.md",
    "work/ns_collision/tests/"
    "test_annular_parallel_shear_third_jet_route_guard.py",
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
    parser.add_argument("--focused-test-count", type=int, default=18)
    parser.add_argument("--focused-test-seconds", type=float, required=True)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument("--periodic-first-average", type=float, required=True)
    parser.add_argument("--periodic-first-peak", type=float, required=True)
    parser.add_argument("--periodic-second-average", type=float, required=True)
    parser.add_argument("--periodic-second-peak", type=float, required=True)
    parser.add_argument("--owned-process-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    for path, expected in ARTIFACT_HASHES.items():
        _require(_sha256(path) == expected, f"{path} changed")

    audit = _load_json(
        "work/ns_collision/results/"
        "annular_parallel_shear_third_jet_route_guard_audit_v1.json"
    )
    flags = audit["certification_flags"]
    _require(
        audit.get("all_route_guard_checks_pass") is True
        and audit.get("status") == "passed_route_guard"
        and audit["carrier_degree_ledger"]["row_count"] == 28
        and audit["carrier_degree_ledger"]["automatic_row_count"] == 22
        and audit["carrier_degree_ledger"]["dangerous_row_count"] == 6
        and audit["bounded_output_pressure_exceptions"][
            "family_count"
        ]
        == 13
        and audit["bounded_output_pressure_exceptions"][
            "N11_capable_family_count"
        ]
        == 5
        and flags["complete_restart_time_third_O_N11_proved"] is False
        and flags["uniform_parabolic_window_third_O_N11_proved"] is False,
        "third-jet route guard or its fail-closed scope changed",
    )
    _require(
        audit["finite_spectral_replay"]["size"] == 5
        and audit["finite_replay_checks"]["all_checks_pass"] is True
        and audit["padding_replay"]["relative_total_residual"] < 1.0e-9,
        "finite N=5 replay did not pass",
    )

    regression = _load_json(FULL_REGRESSION)
    _require(
        regression.get("successful") is True
        and regression.get("discovered_test_count") == 512
        and regression.get("tests_run") == 512
        and regression.get("passed_count") == 512
        and regression.get("skipped_count") == 0
        and regression.get("exit_code") == 0
        and regression.get("runtime", {}).get("active_worker_count") == 1
        and regression.get("runtime", {}).get(
            "below_normal_priority_set"
        )
        is True,
        "the 512-test full regression is not complete",
    )
    _require(
        arguments.focused_test_count == 18
        and arguments.baseline_average <= 40.0
        and arguments.periodic_first_average <= 75.0
        and arguments.periodic_second_average <= 75.0,
        "resource-policy measurements do not match the run",
    )

    bookmark = _load_json(BOOKMARK)
    _require(
        _sha256(BOOKMARK) == PREDECESSOR_BOOKMARK_SHA256
        and bookmark.get("kind")
        == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision"
        and len(bookmark.get("completed_obligations", [])) == 173
        and len(bookmark.get("primary_artifacts", [])) == 638,
        "predecessor bookmark does not match the finalized heat checkpoint",
    )

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Derived the exact four-block third-flow heat split and "
            "partitioned all 28 sector/heat/incidence rows: 22 are "
            "automatically O(N10) or lower, six require compatible "
            "differences, and the bounded-output pressure obstruction is "
            "an exact 13-family list with five O(N11)-capable families. "
            "Replayed the complete third chain rule at N=5 on 14K and "
            "16K grids. The depth-three shell lemma and uniform dynamic "
            "O(N11) bound remain explicitly open."
        ),
    )
    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in NEW_ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 174, "unexpected completed count")
    _require(len(primary) == 643, "unexpected artifact count")

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["parallel_shear_full_regression_deferred_calculation"] = {
        "state": "completed",
        "expected_test_count": 512,
        "focused_test_count": arguments.focused_test_count,
        "focused_test_seconds": arguments.focused_test_seconds,
        "attempted_progress_percent": 100,
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
        "current_checkpoint_sha256": _sha256(FULL_REGRESSION),
        "duration_seconds": regression["duration_seconds"],
        "resume_command": (
            "python work/ns_collision/scripts/"
            "run_full_regression_checkpoint.py --expected-count 512 "
            "--verbosity 0"
        ),
    }
    bookmark["validated_checkpoint"] = (
        "The exact third-flow chain rule is split into heat counts zero "
        "through three. The exhaustive carrier ledger has 28 rows: 22 "
        "are O(N10) or lower by direct incidence/counting and six need "
        "compatible-stencil closure. The bounded-output pressure branch "
        "is reduced to 13 structural families, only five of which can "
        "reach O(N11), and no heat-count-two exception exists. The N=5 "
        "multilinear third derivative agrees with Richardson and with "
        "independent 14K/16K padding. All 18 focused and 512 standalone "
        "tests pass. No restart-time or uniform O(N11) theorem is "
        "claimed."
    )
    principal = bookmark.setdefault("principal_results", {})
    principal.update(
        {
            "annular_parallel_shear_third_flow_heat_split_certified": True,
            "annular_parallel_shear_third_carrier_row_count": 28,
            "annular_parallel_shear_third_automatic_O_N10_row_count": 22,
            "annular_parallel_shear_third_dangerous_row_count": 6,
            "annular_parallel_shear_third_bounded_exception_count": 13,
            "annular_parallel_shear_third_N11_exception_count": 5,
            "annular_parallel_shear_restart_third_O_N11_proved": False,
            "annular_parallel_shear_uniform_third_O_N11_proved": False,
            "annular_parallel_shear_uniform_Taylor_remainder_proved": False,
            "annular_parallel_shear_parabolic_window_closed": False,
            "annular_parallel_shear_finite_time_blowup_proved": False,
            "annular_parallel_shear_global_regularity_proved": False,
            (
                "annular_parallel_shear_"
                "third_jet_route_guard_audit_v1_sha256"
            ): ARTIFACT_HASHES[
                "work/ns_collision/results/"
                "annular_parallel_shear_third_jet_route_guard_"
                "audit_v1.json"
            ],
            "annular_parallel_shear_third_jet_focused_test_count": (
                arguments.focused_test_count
            ),
            "annular_parallel_shear_third_jet_focused_test_seconds": (
                arguments.focused_test_seconds
            ),
            "annular_parallel_shear_full_regression_512_passed": True,
            "annular_parallel_shear_full_regression_512_pending": False,
            "full_regression_checkpoint_v1_sha256": _sha256(
                FULL_REGRESSION
            ),
            "annular_parallel_shear_third_jet_resource_mode": (
                "daytime_one_worker_below_normal_four_cpu_affinity"
            ),
            "annular_parallel_shear_third_jet_cpu_baseline_average": (
                arguments.baseline_average
            ),
            "annular_parallel_shear_third_jet_cpu_baseline_peak": (
                arguments.baseline_peak
            ),
            (
                "annular_parallel_shear_third_jet_"
                "cpu_periodic_first_average"
            ): arguments.periodic_first_average,
            (
                "annular_parallel_shear_third_jet_"
                "cpu_periodic_second_average"
            ): arguments.periodic_second_average,
            "annular_parallel_shear_third_jet_updater_sha256": _sha256(
                UPDATER
            ),
        }
    )
    bookmark["unfinished_obligation"] = (
        "The third-jet route is reduced but not closed. Prove the "
        "depth-three compatible-difference/dyadic-shell lemma for the "
        "protected four- and six-high rows and retain an explicit C3 in "
        "|g'''_N(0)|<=C3 N11. Then propagate the same bound uniformly "
        "along 0<=s<=T/N2 and compare T with c2/(2C3). Critical L3 "
        "control, finite-time blowup, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_parallel_shear_third_jet_route_guard_audit.py --size 5"
    )
    bookmark["next_action"] = (
        "Build the depth-three internal-output topology ledger. Close "
        "the protected four-high rows first with two compatible gains, "
        "then the all-high pressure row with four gains. Do not begin "
        "the dynamic Taylor bootstrap until a restart-time O(N11) "
        "constant is certified."
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
                "route_guard_sha256": ARTIFACT_HASHES[
                    "work/ns_collision/results/"
                    "annular_parallel_shear_third_jet_route_guard_"
                    "audit_v1.json"
                ],
                "full_regression_sha256": _sha256(FULL_REGRESSION),
                "updater_sha256": _sha256(UPDATER),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
