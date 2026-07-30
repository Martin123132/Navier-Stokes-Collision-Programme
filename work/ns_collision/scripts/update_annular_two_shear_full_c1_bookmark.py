"""Install the two-shear square and full c1-port checkpoint."""

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
    "c71a9b368d5e0c46045e4eb1fba82e1351a37b353191fb2c255ba19d66fb0974"
)
SQUARE_SCRIPT = (
    "work/ns_collision/scripts/annular_two_shear_square_gate_audit.py"
)
SQUARE_SCRIPT_SHA256 = (
    "936beb223c06042b27055aa3dad3abbb6ffb600c724f6f40bece44f107dcbc87"
)
SQUARE_RESULT = (
    "work/ns_collision/results/"
    "annular_two_shear_square_gate_audit_v1.json"
)
SQUARE_RESULT_SHA256 = (
    "7aca1d4c57ce970db5872979449368e65fb6add48c4ccd39b8f875cc1669f923"
)
SQUARE_NOTE = (
    "work/ns_collision/notes/annular_two_shear_square_gate.md"
)
SQUARE_NOTE_SHA256 = (
    "993590ba426ce279bbc045e2a2b6ffedbbb9b50e0577af377fb4ae47b9786163"
)
SQUARE_TEST = (
    "work/ns_collision/tests/test_annular_two_shear_square_gate.py"
)
SQUARE_TEST_SHA256 = (
    "87257edbc2c3945eb88406870c0b4dfe3010d4ee813c32794f83c650953dca25"
)
PORT_SCRIPT = (
    "work/ns_collision/scripts/annular_two_shear_full_c1_port_audit.py"
)
PORT_SCRIPT_SHA256 = (
    "68330a77789ada6b1ac1f3e226460e3cae8b56a133c6e8a4f6d3f39a4996986f"
)
PORT_RESULT = (
    "work/ns_collision/results/"
    "annular_two_shear_full_c1_port_audit_v1.json"
)
PORT_RESULT_SHA256 = (
    "af0039698cdbd5442be629b23ea259e97556c978a0f0d91e94e9e0d658b1f32f"
)
PORT_NOTE = (
    "work/ns_collision/notes/annular_two_shear_full_c1_port.md"
)
PORT_NOTE_SHA256 = (
    "2cafc8f402244028507dd6686c1a781cbeb98c0bf45ed9c13a6061028acb2635"
)
PORT_TEST = (
    "work/ns_collision/tests/test_annular_two_shear_full_c1_port.py"
)
PORT_TEST_SHA256 = (
    "b61c8f0afba26f13bb6e749edd0af14e4c035f114e0caa7134f91d7a05310a81"
)
README = "work/ns_collision/README.md"
README_SHA256 = (
    "2d9abfabc3ee0c55b7014519e30c8377f6fdaa96566a0ad2e2a2f7df42dc0415"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "4f4913f02801bbf24df5051908f6912071580d70ff4b050d09e6c0c378cf8db4"
)
UPDATER = (
    "work/ns_collision/scripts/"
    "update_annular_two_shear_full_c1_bookmark.py"
)
NEW_ARTIFACTS = (
    SQUARE_SCRIPT,
    SQUARE_RESULT,
    SQUARE_NOTE,
    SQUARE_TEST,
    PORT_SCRIPT,
    PORT_RESULT,
    PORT_NOTE,
    PORT_TEST,
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
    parser.add_argument("--focused-test-count", type=int, default=17)
    parser.add_argument("--focused-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=474)
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
        SQUARE_SCRIPT: SQUARE_SCRIPT_SHA256,
        SQUARE_RESULT: SQUARE_RESULT_SHA256,
        SQUARE_NOTE: SQUARE_NOTE_SHA256,
        SQUARE_TEST: SQUARE_TEST_SHA256,
        PORT_SCRIPT: PORT_SCRIPT_SHA256,
        PORT_RESULT: PORT_RESULT_SHA256,
        PORT_NOTE: PORT_NOTE_SHA256,
        PORT_TEST: PORT_TEST_SHA256,
        README: README_SHA256,
        FULL_REGRESSION: FULL_REGRESSION_SHA256,
    }
    for path, expected in expected_hashes.items():
        _require(_sha256(path) == expected, f"{path} changed")

    square = _load_json(SQUARE_RESULT)
    port = _load_json(PORT_RESULT)
    regression = _load_json(FULL_REGRESSION)
    _require(
        square.get("algorithm_revision")
        == "annular-two-shear-square-gate-v1"
        and square.get("all_route_guard_checks_pass") is True,
        "two-shear square audit did not pass",
    )
    _require(
        [row.get("size") for row in square.get("rows", [])]
        == [8, 16, 32]
        and all(
            row.get("all_numerical_checks_pass")
            for row in square.get("rows", [])
        ),
        "two-shear exact-box rows changed",
    )
    _require(
        square["exact_low_stencil"]["checks"]["combined_matrix_exact"]
        and square["certification"][
            "modified_four_high_continuum_sign_analytic"
        ]
        and square["certification"]["strict_nonzero_analytic"]
        and not square["certification"]["navier_stokes_clay_problem_solved"],
        "two-shear analytic sign scope changed",
    )
    static_replay = square["finite_static_packet_replay"]
    _require(
        [row["size"] for row in static_replay["rows"]]
        == [5, 9, 13, 17, 25]
        and static_replay["all_finite_static_rows_pass"]
        and abs(
            static_replay[
                "positive_packet_continuum_reference_gauss80"
            ]
            - (-0.0014140889924061505)
        )
        < 2.0e-17,
        "finite static packet replay changed",
    )

    _require(
        port.get("algorithm_revision")
        == "annular-two-shear-full-c1-port-v1"
        and port.get("all_port_checks_pass") is True,
        "two-shear full c1 port did not pass",
    )
    _require(
        port["two_shear_tail_port_certificate"]["new_tail_constant"]
        == 70_657_920
        and port["full_limit_certificate"]["conclusion"]
        == "c1_*,N/N^7 -> L_*<0"
        and port["certification"][
            "modified_full_c1_over_N7_convergence_proved"
        ]
        and port["certification"][
            "modified_full_c1_limit_negative_certified"
        ]
        and not port["certification"]["finite_time_blowup_proved"],
        "two-shear full limit or certification scope changed",
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
        and regression.get("passed_count") == arguments.discovered_test_count
        and regression.get("skipped_count") == 0
        and regression.get("successful") is True
        and regression.get("exit_code") == 0
        and not regression.get("failures")
        and not regression.get("errors"),
        "full pytest regression did not pass",
    )
    _require(
        arguments.focused_test_count == 17
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
        and len(bookmark.get("completed_obligations", [])) == 166
        and len(bookmark.get("primary_artifacts", [])) == 605
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 168
        and len(bookmark.get("primary_artifacts", [])) == 614
        and principal.get(
            "annular_two_shear_full_c1_port_audit_v1_sha256"
        )
        == PORT_RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed two-shear checkpoint matches",
    )

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "A modified annular witness now bypasses the old one-shear "
        "interval sign gate. The divergence-free profile "
        "b=S*(x^2+y^2)/(x*r^3)*(-z,0,x) has b_y=0. The original yz low "
        "shear plus a sign-flipped xy shear give the exact matrix "
        "Q_*=(sqrt(2)/40)diag(1,-2,1), so the static positive-packet "
        "load is -(sqrt(2)/20)||b||_L2(D)^2 and the leading four-high "
        "continuum coefficient is the strict square "
        "L_*=-(3sqrt(2)/20)||v_y||2^2<0. Strictness follows from the "
        "small-output covariance expansion, not numerics. The exact odd "
        "packet replays negative static rows through N=25. Its multiplier "
        "retains |hhat|<=1/(2N) and one-difference <4/N2. All fourteen "
        "tail profiles are linear in the low field, so the complete tail "
        "ports with |c1_*-D_*|<=70,657,920 N6 and proves "
        "c1_*,N/N7->L_*<0. The original L_EE sign remains unproved; the "
        "modified optimizer, complete finite jets, uniform Taylor "
        "remainder, parabolic window, and critical L3 gate remain open. "
        "All 17 focused tests and all 474 standalone tests pass."
    )
    principal.update(
        {
            "annular_two_shear_profile_formula": (
                "S*(x^2+y^2)/(x*r^3)*(-z,0,x)"
            ),
            "annular_two_shear_profile_y_component_zero": True,
            "annular_two_shear_combined_projector_matrix": (
                "(sqrt(2)/40)diag(1,-2,1)"
            ),
            "annular_two_shear_static_limit": (
                "-(sqrt(2)/20)||b||_L2(D)^2<0"
            ),
            "annular_two_shear_four_high_limit": (
                "-(3sqrt(2)/20)||v_y||_2^2<0"
            ),
            "annular_two_shear_strict_nonzero_analytic": True,
            "annular_two_shear_finite_static_sizes": [5, 9, 13, 17, 25],
            "annular_two_shear_static_gauss80_reference": (
                static_replay[
                    "positive_packet_continuum_reference_gauss80"
                ]
            ),
            "annular_two_shear_packet_first_difference_bound": "4/N^2",
            "annular_two_shear_tail_constant": 70_657_920,
            "annular_two_shear_full_c1_limit": "c1_*,N/N^7 -> L_*<0",
            "annular_two_shear_full_c1_convergence_proved": True,
            "annular_two_shear_full_c1_negative_certified": True,
            "annular_original_single_shear_L_EE_sign_certified": False,
            "annular_two_shear_static_optimizer_ported": False,
            "annular_two_shear_complete_finite_jets_ported": False,
            "annular_two_shear_uniform_Taylor_remainder_proved": False,
            "annular_two_shear_parabolic_window_closed": False,
            "annular_two_shear_square_gate_audit_v1_sha256": (
                SQUARE_RESULT_SHA256
            ),
            "annular_two_shear_full_c1_port_audit_v1_sha256": (
                PORT_RESULT_SHA256
            ),
            "annular_two_shear_focused_test_count": (
                arguments.focused_test_count
            ),
            "annular_two_shear_focused_test_runtime_seconds": (
                arguments.focused_test_seconds
            ),
            "annular_two_shear_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_two_shear_full_pytest_passed": True,
            "annular_two_shear_full_pytest_runtime_seconds": float(
                regression["duration_seconds"]
            ),
            "annular_two_shear_resource_mode": arguments.resource_mode,
            "annular_two_shear_worker_count": arguments.worker_count,
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
            "Constructed a legal divergence-free annular profile with one "
            "missing velocity component and two compatible low shears; "
            "proved by exact rational stencil algebra, Euler energy trace, "
            "and a covariance nonvanishing argument that its static load "
            "and leading four-high continuum coefficient are strict "
            "negative norms, without interval quadrature."
        ),
    )
    _append_once(
        completed,
        (
            "Ported the complete fourteen-profile amplitude-one c1 tail "
            "theorem to the two-shear witness: retained the 1/(2N) "
            "coefficient and 4/N^2 first-difference bounds, doubled the "
            "tail constant to 70,657,920, proved fixed-output convergence, "
            "and concluded c1_*,N/N^7 -> L_*<0."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The modified two-shear branch now has a strict negative complete "
        "amplitude-one N7 limit. It has not yet been shown that the old "
        "static joint optimizer and restart scaling survive the two-mode "
        "low field. Recompute every static flux, Fisher, and coefficient "
        "penalty term, then port the complete first and second finite jets. "
        "After that, the uniform Taylor remainder, parabolic-window "
        "amplification, critical L3 control, blowup, and global regularity "
        "remain open. The original one-shear L_EE interval branch remains "
        "a valid but currently secondary unresolved route."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_two_shear_full_c1_port_audit.py"
    )
    bookmark["next_action"] = (
        "Port the static optimizer to the modified witness. Build the exact "
        "finite two-shear low field and modified h_N in the static generator "
        "audit; enumerate all HHL, low-low, Fisher, and coefficient-penalty "
        "terms; derive the new joint optimizer and restart scaling without "
        "reusing single-shear constants. Only if that passes, port the "
        "complete first- and second-jet formulas and their uniform time "
        "remainder."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in NEW_ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 168, "unexpected completed count")
    _require(len(primary) == 614, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary),
                "square_result_sha256": _sha256(SQUARE_RESULT),
                "port_result_sha256": _sha256(PORT_RESULT),
                "full_regression_sha256": _sha256(FULL_REGRESSION),
                "updater_sha256": _sha256(UPDATER),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
