"""Install the complete annular c1 tail checkpoint."""

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
    "bb28a938f05f44fbaff2ce7d18bbfbb477f411433a3069222de90d58d120d643"
)
TAIL_RESULT = (
    "work/ns_collision/results/"
    "annular_rho_zero_full_c1_tail_ledger_audit_v1.json"
)
TAIL_RESULT_SHA256 = (
    "e8917ea3b781f72806bd2b560ccf65058027bb34bb15d997375a9d237020d773"
)
TAIL_SCRIPT = (
    "work/ns_collision/scripts/"
    "annular_rho_zero_full_c1_tail_ledger_audit.py"
)
TAIL_SCRIPT_SHA256 = (
    "4c72770f8b811a24e48fc8ce4403476d232694fa6c2e720e56b504c48bde9264"
)
TAIL_TEST = (
    "work/ns_collision/tests/"
    "test_annular_rho_zero_full_c1_tail_ledger.py"
)
TAIL_TEST_SHA256 = (
    "acd8cc9a875a2c5bc8aa69261d682a44ebb20b17e44a77705d0243b7f5f51cb2"
)
TAIL_NOTE = (
    "work/ns_collision/notes/"
    "annular_rho_zero_full_c1_tail_ledger.md"
)
TAIL_NOTE_SHA256 = (
    "0477ed8271bffea0757b72d6390ea52303b9deb8c4039c625c7328ab61d413e0"
)
README = "work/ns_collision/README.md"
README_SHA256 = (
    "1321984b9e51d93e8b100a88d32a596c64688e5da507d8343cba606174dc03ce"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "aca7d78735311c5b520ef716439dfa3a20a404b8cb14eeaae383d78a3c1f5318"
)
RUNNER = "work/ns_collision/scripts/run_full_regression_checkpoint.py"
RUNNER_SHA256 = (
    "4fd9cca40a4133bfce8bba21161dd827dc98ccbeee9c13f70f7b3718167a4609"
)
ARTIFACTS = (
    TAIL_SCRIPT,
    TAIL_TEST,
    TAIL_NOTE,
    TAIL_RESULT,
    (
        "work/ns_collision/scripts/"
        "update_annular_rho_zero_full_c1_tail_checkpoint_bookmark.py"
    ),
    README,
    FULL_REGRESSION,
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
    parser.add_argument("--targeted-test-count", type=int, default=31)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=457)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument("--periodic-average", type=float, required=True)
    parser.add_argument("--periodic-peak", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    tail = _load_json(TAIL_RESULT)
    regression = _load_json(FULL_REGRESSION)
    expected_hashes = {
        TAIL_RESULT: TAIL_RESULT_SHA256,
        TAIL_SCRIPT: TAIL_SCRIPT_SHA256,
        TAIL_TEST: TAIL_TEST_SHA256,
        TAIL_NOTE: TAIL_NOTE_SHA256,
        README: README_SHA256,
        FULL_REGRESSION: FULL_REGRESSION_SHA256,
        RUNNER: RUNNER_SHA256,
    }
    for path, expected in expected_hashes.items():
        _require(_sha256(path) == expected, f"{path} changed")

    _require(
        tail.get("status")
        == "annular_full_c1_over_N7_convergence_certified_sign_pending"
        and tail.get("all_positive_checks_pass") is True,
        "complete c1 tail audit did not pass",
    )
    flags = tail["certification_flags"]
    _require(
        flags["full_c1_remainder_ledger_complete"] is True
        and flags["full_c1_over_N7_convergence_proved"] is True
        and flags["single_zero_extended_packet_difference_used"] is True
        and flags["zero_extension_C2_or_higher_used"] is False
        and flags["continuum_limit_nonzero_certified"] is False
        and flags["continuum_limit_negative_certified"] is False
        and flags["four_high_N9_coefficient_certified"] is False
        and flags["uniform_second_jet_Taylor_bound_proved"] is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "complete c1 tail certification scope changed",
    )
    ledger = tail["termwise_tail_ledger"]
    limit = tail["full_limit_certificate"]
    _require(
        ledger["row_count"] == 14
        and ledger["absolute_atomic_coefficient_mass"] == 94
        and ledger["maximum_actual_coordinate_monomial_mass"] == 48
        and ledger["full_tail_constant"] == 35_328_960
        and limit["combined_constant"] == 35_578_960
        and limit["conclusion"] == "c_1,N/N^7 -> L_EE"
        and limit["continuum_sign_certified"] is False,
        "complete c1 tail theorem changed",
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
        arguments.worker_count == 1
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
        and len(bookmark.get("completed_obligations", [])) == 164
        and len(bookmark.get("primary_artifacts", [])) == 593
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 165
        and len(bookmark.get("primary_artifacts", [])) == 598
        and principal.get(
            "annular_rho_zero_full_c1_tail_ledger_audit_v1_sha256"
        )
        == TAIL_RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed c1-tail checkpoint matches",
    )

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The complete four-high amplitude-one tail is now bounded "
        "term by term. Exact polynomial expansion gives three omitted "
        "dominant permutations and eleven atomic rows from the seven "
        "remaining forms, with total absolute coefficient mass 94. "
        "Every tail contraction has a high leaf on the test side, so "
        "one exact parity-gauged vertex difference is absorbed there "
        "while the outer pressure projector is held fixed. Only the "
        "Lipschitz zero extension |Delta h_N|<=4/N^2 is used. The full "
        "Euler symbol is globally Lipschitz of degree one, eliminating "
        "the proposed shell logarithm. This proves "
        "|c_1,N-D_N|<=35328960 N6 for odd N>=5 and, with the predecessor "
        "bound, |c_1,N/N7-L_EE|<=35578960/N for odd N>=128. Hence "
        "c_1,N/N7 -> L_EE. The continuum sign remains uncertified. "
        "All 31 focused tests and all 457 corpus tests pass."
    )
    principal.update(
        {
            "annular_full_c1_tail_atomic_row_count": 14,
            "annular_full_c1_tail_absolute_coefficient_mass": 94,
            "annular_full_c1_tail_maximum_actual_coordinate_monomial_mass": (
                48
            ),
            "annular_full_c1_tail_constant": 35_328_960,
            "annular_full_c1_combined_normalized_remainder_constant": (
                35_578_960
            ),
            "annular_full_c1_over_N7_convergence_proved": True,
            "annular_full_c1_continuum_limit": "L_EE",
            "annular_full_c1_continuum_sign_certified": False,
            "annular_full_c1_N9_coefficient_certified": False,
            "annular_full_c1_result_status": tail["status"],
            "annular_rho_zero_full_c1_tail_ledger_audit_v1_sha256": (
                TAIL_RESULT_SHA256
            ),
            "annular_full_c1_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "annular_full_c1_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "annular_full_c1_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_full_c1_full_pytest_passed": True,
            "annular_full_c1_full_pytest_runtime_seconds": float(
                regression["duration_seconds"]
            ),
            "annular_full_c1_resource_mode": arguments.resource_mode,
            "annular_full_c1_worker_count": arguments.worker_count,
            "annular_full_c1_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "annular_full_c1_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "annular_full_c1_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "annular_full_c1_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "full_regression_checkpoint_v1_sha256": (
                FULL_REGRESSION_SHA256
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
            "Expanded all ten omitted four-high amplitude-one groups, "
            "certified fourteen atomic one-difference bounds with an "
            "explicit O(N^6) constant, and proved the complete "
            "c_1,N/N7 convergence to L_EE without assuming higher "
            "zero-extension smoothness."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The full four-high coefficient now has the proved limit "
        "c_1,N/N7 -> L_EE, and deterministic quadrature places L_EE "
        "near -2.99386e-7. The remaining immediate proof gate is a "
        "joint interval enclosure for the cancelling L_VV and L_GH "
        "integrals narrow enough to exclude zero. Until that enclosure "
        "passes, the limit sign, optimized N9 coefficient, complete "
        "viscous second jet, parabolic Taylor window, critical L3 "
        "control, blowup, and global regularity all remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_rho_zero_full_c1_tail_ledger_audit.py"
    )
    bookmark["next_action"] = (
        "Build a deterministic interval enclosure for L_EE=L_VV+L_GH "
        "on the fixed continuum domains. Bound truncation, interpolation, "
        "and quadrature errors jointly so cancellation is enclosed; do "
        "not certify the sign from inverse-N fits or floating-point "
        "agreement alone. Resume N9 and Taylor-window work only if the "
        "joint interval excludes zero."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 165, "unexpected completed count")
    _require(len(primary) == 598, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary),
                "tail_result_sha256": _sha256(TAIL_RESULT),
                "full_regression_sha256": _sha256(FULL_REGRESSION),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
