"""Install the annular Euler-Maclaurin and Leray-cusp checkpoint."""

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
    "21cc4c267ea018bd23ed6f76eff88801dc3acb0392fc9bf01334f7c764eab659"
)
DIRECT_SCRIPT = (
    "work/ns_collision/scripts/"
    "annular_rho_zero_direct_continuum_quadrature.py"
)
DIRECT_SCRIPT_SHA256 = (
    "380d7e499170810453ff2c8bf0f8931967a06cf5d6357bbb73b1a3972e484cb9"
)
DIRECT_RESULT = (
    "work/ns_collision/results/"
    "annular_rho_zero_direct_continuum_quadrature_v1.json"
)
DIRECT_RESULT_SHA256 = (
    "3aae498f94a1b63351368f7fb3c35b3a24242d58c16c462bd0e422d1af76686b"
)
BOUNDARY_SCRIPT = (
    "work/ns_collision/scripts/"
    "annular_rho_zero_euler_maclaurin_boundary_pilot.py"
)
BOUNDARY_SCRIPT_SHA256 = (
    "2470e538c9048b1e9dc0edd99bc47d923e6c6003accfa70c2d45b2aa488c6499"
)
BOUNDARY_RESULT = (
    "work/ns_collision/results/"
    "annular_rho_zero_euler_maclaurin_boundary_pilot_v1.json"
)
BOUNDARY_RESULT_SHA256 = (
    "ad9a2e8f8fd982947fe3cf3261c4933eac93941dce7fd959b9110f3ae7f9d25d"
)
NOTE = (
    "work/ns_collision/notes/"
    "annular_rho_zero_euler_maclaurin_cusp_gate.md"
)
NOTE_SHA256 = (
    "ede6bfefd78767c86b71188971b8fb5810b00f361bd18066dd83d892f257537f"
)
TEST = (
    "work/ns_collision/tests/"
    "test_annular_rho_zero_euler_maclaurin_cusp_gate.py"
)
TEST_SHA256 = (
    "e32271cd8d4f3de29bc1c24c1ca7f6733896922a61978f805ea005627cc15780"
)
README = "work/ns_collision/README.md"
README_SHA256 = (
    "60fc82fac90037954b35562d8f6d041a61348c689f12be401a0029e08187ce14"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "f375b073ffa30383139b31e8d26ce6803fbdccbc9d0464d5d0a31ea8c8f98c27"
)
UPDATER = (
    "work/ns_collision/scripts/"
    "update_annular_rho_zero_euler_maclaurin_cusp_checkpoint_bookmark.py"
)
NEW_ARTIFACTS = (
    DIRECT_SCRIPT,
    DIRECT_RESULT,
    BOUNDARY_SCRIPT,
    BOUNDARY_RESULT,
    NOTE,
    TEST,
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
    parser.add_argument("--focused-test-count", type=int, default=21)
    parser.add_argument("--focused-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=464)
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
        DIRECT_SCRIPT: DIRECT_SCRIPT_SHA256,
        DIRECT_RESULT: DIRECT_RESULT_SHA256,
        BOUNDARY_SCRIPT: BOUNDARY_SCRIPT_SHA256,
        BOUNDARY_RESULT: BOUNDARY_RESULT_SHA256,
        NOTE: NOTE_SHA256,
        TEST: TEST_SHA256,
        README: README_SHA256,
        FULL_REGRESSION: FULL_REGRESSION_SHA256,
    }
    for path, expected in expected_hashes.items():
        _require(_sha256(path) == expected, f"{path} changed")

    direct = _load_json(DIRECT_RESULT)
    boundary = _load_json(BOUNDARY_RESULT)
    regression = _load_json(FULL_REGRESSION)
    _require(
        direct.get("algorithm_revision")
        == "annular-rho-zero-direct-continuum-quadrature-v1"
        and direct.get("all_numerical_checks_pass") is True,
        "direct exact-box audit did not pass",
    )
    direct_rows = direct.get("rows", [])
    _require(
        [row.get("size") for row in direct_rows] == [8, 16, 32, 64]
        and all(row.get("all_numerical_checks_pass") for row in direct_rows),
        "direct exact-box production rows changed",
    )
    _require(
        abs(
            direct_rows[-1]["combined_continuum_quadrature"]
            - (-2.9883445926209503e-7)
        )
        < 2.0e-20
        and direct["certification"]["continuum_sign_interval_certified"]
        is False,
        "direct quadrature value or certification scope changed",
    )
    cusp = direct_rows[-1]["origin_leray_cusp_replay"]
    covariance = cusp["covariance_matrix_trapezoid"]
    _require(
        cusp["maximum_covariance_off_diagonal"] < 1.0e-18
        and cusp["maximum_residual_over_rho_cubed"] < 0.11
        and 0.039 < covariance[2][2] < 0.040,
        "origin Leray-cusp replay changed",
    )

    _require(
        boundary.get("algorithm_revision")
        == "annular-rho-zero-euler-maclaurin-boundary-pilot-v1"
        and boundary.get("all_pilot_checks_pass") is True,
        "Euler-Maclaurin boundary pilot did not pass",
    )
    boundary_rows = boundary.get("rows", [])
    _require(
        [row.get("size") for row in boundary_rows] == [8, 16, 32, 64]
        and all(row.get("all_pilot_checks_pass") for row in boundary_rows),
        "Euler-Maclaurin production rows changed",
    )
    final_boundary = boundary_rows[-1]
    _require(
        abs(
            final_boundary["face_correction_c2"]
            - 2.2565322021635291e-6
        )
        < 2.0e-19
        and abs(
            final_boundary["face_corrected_value"]
            - (-2.9938537044426388e-7)
        )
        < 2.0e-20
        and abs(
            final_boundary["sixth_order_corrected_quartic_value"]
            - (-2.993859498977249e-7)
        )
        < 2.0e-20,
        "Euler-Maclaurin correction values changed",
    )
    flags = boundary["certification"]
    _require(
        flags["face_correction_derived"] is True
        and flags["h4_remainder_interval_certified"] is False
        and flags["continuum_sign_interval_certified"] is False,
        "Euler-Maclaurin certification scope changed",
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
        and len(bookmark.get("completed_obligations", [])) == 165
        and len(bookmark.get("primary_artifacts", [])) == 598
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 166
        and len(bookmark.get("primary_artifacts", [])) == 605
        and principal.get(
            "annular_rho_zero_euler_maclaurin_boundary_pilot_v1_sha256"
        )
        == BOUNDARY_RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed cusp checkpoint matches",
    )

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The remaining continuum sign gate now has a direct exact-box "
        "tensor-trapezoid rule with h11 normalization. Its N=8,16,32,64 "
        "rows converge at second order and independently close the Euler "
        "energy trace. The two cancelling pieces occupy disjoint output "
        "bands. The explicit face measure "
        "mu_2=(1/12) sum orientation partial_normal(a) reproduces the "
        "complete h2 coefficient; corrected measures converge at fourth "
        "order. The residual is localized to the internal Leray cusp "
        "v(rho)=P_rho M rho+O(|rho|3), with diagonal covariance M. The "
        "elementary norm bound ||a||2^2<=13/288 gives a complete absolute "
        "small-cube budget 7.800e-8 for |rho_j|<=1/20. The regular "
        "complement and floating FFT roundoff are not yet interval "
        "enclosed, so L_EE<0 remains unproved. All 21 focused tests and "
        "all 464 standalone tests pass."
    )
    principal.update(
        {
            "annular_direct_continuum_mesh_sizes": [8, 16, 32, 64],
            "annular_direct_continuum_N64_value": (
                direct_rows[-1]["combined_continuum_quadrature"]
            ),
            "annular_direct_continuum_h11_normalization_proved": True,
            "annular_direct_continuum_output_bands_disjoint": True,
            "annular_euler_maclaurin_face_correction_derived": True,
            "annular_euler_maclaurin_N64_c2": (
                final_boundary["face_correction_c2"]
            ),
            "annular_euler_maclaurin_N64_face_corrected_value": (
                final_boundary["face_corrected_value"]
            ),
            "annular_euler_maclaurin_N64_sixth_packet_value": (
                final_boundary["sixth_order_corrected_quartic_value"]
            ),
            "annular_leray_cusp_covariance_diagonal": True,
            "annular_leray_cusp_M_xx_N64": covariance[0][0],
            "annular_leray_cusp_M_yy_N64": covariance[1][1],
            "annular_leray_cusp_M_zz_N64": covariance[2][2],
            "annular_leray_cusp_small_cube_delta": "1/20",
            "annular_leray_cusp_L2_squared_bound": "13/288",
            "annular_leray_cusp_small_cube_absolute_budget": 7.8e-8,
            "annular_continuum_regular_complement_interval_certified": False,
            "annular_continuum_fft_roundoff_interval_certified": False,
            "annular_full_c1_continuum_sign_certified": False,
            "annular_rho_zero_direct_continuum_quadrature_v1_sha256": (
                DIRECT_RESULT_SHA256
            ),
            "annular_rho_zero_euler_maclaurin_boundary_pilot_v1_sha256": (
                BOUNDARY_RESULT_SHA256
            ),
            "annular_euler_maclaurin_focused_test_count": (
                arguments.focused_test_count
            ),
            "annular_euler_maclaurin_focused_test_runtime_seconds": (
                arguments.focused_test_seconds
            ),
            "annular_euler_maclaurin_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_euler_maclaurin_full_pytest_passed": True,
            "annular_euler_maclaurin_full_pytest_runtime_seconds": float(
                regression["duration_seconds"]
            ),
            "annular_euler_maclaurin_resource_mode": arguments.resource_mode,
            "annular_euler_maclaurin_worker_count": arguments.worker_count,
            "annular_euler_maclaurin_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "annular_euler_maclaurin_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "annular_euler_maclaurin_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "annular_euler_maclaurin_cpu_periodic_peak_percent": (
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
            "Replaced inverse-N-only evidence for the L_EE sign gate by "
            "an exact-box h11 cubature, derived its full Euler-Maclaurin "
            "h2 face coefficient, isolated the internal Leray cusp "
            "v=P M rho+O(|rho|3), and closed an explicit 7.800e-8 "
            "small-cube contribution budget without claiming the "
            "remaining regular-complement interval."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The complete c_1,N/N7 limit and the exact-box continuum "
        "quadrature are proved, and the only nonsmooth internal region "
        "has a 7.800e-8 absolute budget. The immediate open gate is a "
        "directed interval enclosure of the regular complement "
        "|rho|_infinity>=1/20, plus a floating FFT roundoff ledger, tight "
        "enough that their combined error and the cusp budget keep the "
        "upper endpoint of L_EE below zero. Until then the continuum "
        "sign, optimized N9 coefficient, viscous second jet, parabolic "
        "window, critical L3 control, blowup, and global regularity remain "
        "open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_rho_zero_euler_maclaurin_boundary_pilot.py "
        "--sizes 8,16,32,64"
    )
    bookmark["next_action"] = (
        "Implement the regular-complement certificate. Split every "
        "occurrence of the intermediate mixed-sign velocity at "
        "|rho_j|<=1/20, retain the proved 7.800e-8 absolute budget there, "
        "and expand a deterministic Euler-Maclaurin derivative ledger on "
        "the complement where the Leray denominator is separated from "
        "zero. Add a directed FFT roundoff bound. Certify L_EE<0 only if "
        "the final joint interval has negative upper endpoint; do not use "
        "Richardson agreement as the missing bound."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in NEW_ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 166, "unexpected completed count")
    _require(len(primary) == 605, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary),
                "direct_result_sha256": _sha256(DIRECT_RESULT),
                "boundary_result_sha256": _sha256(BOUNDARY_RESULT),
                "full_regression_sha256": _sha256(FULL_REGRESSION),
                "updater_sha256": _sha256(UPDATER),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
