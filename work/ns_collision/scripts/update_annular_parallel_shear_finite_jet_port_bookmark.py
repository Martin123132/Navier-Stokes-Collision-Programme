"""Install the annular parallel-shear finite-jet checkpoint."""

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
    "451eea04954833e66e760a1e7247c4180043fe50100fc143f9a3d4277a0313ae"
)
AUDIT_SCRIPT = (
    "work/ns_collision/scripts/"
    "annular_parallel_shear_finite_jet_port_audit.py"
)
AUDIT_SCRIPT_SHA256 = (
    "cd9565237e7e6ebc5f5a35c5c145ce4afe91604574b4a8d6b0942644869afeea"
)
AUDIT_RESULT = (
    "work/ns_collision/results/"
    "annular_parallel_shear_finite_jet_port_audit_v1.json"
)
AUDIT_RESULT_SHA256 = (
    "1e7753a5280c136bbe34770a17d929f73cc7398579906f51e316c738ee660da0"
)
AUDIT_NOTE = (
    "work/ns_collision/notes/"
    "annular_parallel_shear_finite_jet_port.md"
)
AUDIT_NOTE_SHA256 = (
    "1b7b1e5327c0945efb40ac9a5198c8eae64957fab9a4b4921a2b009b755689d6"
)
AUDIT_TEST = (
    "work/ns_collision/tests/"
    "test_annular_parallel_shear_finite_jet_port.py"
)
AUDIT_TEST_SHA256 = (
    "7671d61ebe9b09101947b9a5261d1812d011ee63b53a52c8c8acad5180530d14"
)
README = "work/ns_collision/README.md"
README_SHA256 = (
    "08463dc2a26559a730c6e438f990ea3eeedd72b640aa034b75345d1f9b251471"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "6befef38446b8950432e8c3e77cfe6d74ce3c70fe81705042ff6c7b77e5e6f09"
)
UPDATER = (
    "work/ns_collision/scripts/"
    "update_annular_parallel_shear_finite_jet_port_bookmark.py"
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
    parser.add_argument("--discovered-test-count", type=int, default=494)
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
        == "annular-parallel-shear-finite-jet-port-v1"
        and audit.get("all_positive_checks_pass") is True,
        "parallel-shear finite-jet audit did not pass",
    )
    projection = audit["two_low_amplitude_polynomial_projection"]
    ledger = audit["carrier_power_ledger"]
    flags = audit["certification_flags"]
    _require(
        projection["all_projection_checks_pass"]
        and projection["finite_c1_equal_amplitude_coefficient"] < 0.0
        and projection["finite_c3_equal_amplitude_coefficient"] > 0.0
        and projection["first_inviscid_low_only_stationarity_residual"]
        < 2.0e-7
        and projection["second_inviscid_low_only_stationarity_residual"]
        < 2.0e-7,
        "two-amplitude polynomial projection changed",
    )
    _require(
        audit["padding_replay"]["all_padding_checks_pass"]
        and audit["weight_scale_homogeneity_replay"][
            "all_weight_homogeneity_checks_pass"
        ]
        and all(
            row["all_heat_load_checks_pass"]
            for row in audit["finite_heat_weighted_HHL_rows"]
        ),
        "finite jet, padding, or heat replay changed",
    )
    _require(
        ledger["first_total_N5_limit_certified"]
        and ledger["second_inviscid_pressure_N9_limit_certified"]
        and not ledger["all_noninviscid_second_channels_o_N9_certified"]
        and not ledger["total_second_N9_limit_certified"]
        and flags["parallel_complete_finite_first_jet_ported"]
        and flags["parallel_complete_finite_second_jet_ported"]
        and flags["parallel_mixed_polarization_channels_enumerated"]
        and not flags["parallel_complete_second_N9_limit_certified"]
        and not flags["uniform_second_jet_Taylor_remainder_proved"]
        and not flags["Navier_Stokes_global_regularity_proved"],
        "carrier ledger or fail-closed scope changed",
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
        and len(bookmark.get("completed_obligations", [])) == 170
        and len(bookmark.get("primary_artifacts", [])) == 624
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 171
        and len(bookmark.get("primary_artifacts", [])) == 629
        and principal.get(
            "annular_parallel_shear_finite_jet_port_audit_v1_sha256"
        )
        == AUDIT_RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed finite-jet checkpoint matches",
    )

    continuum = audit["continuum_heat_constants"]["beta"]
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The common-polarization two-mode low field now has complete finite "
        "first and second rho=0 generator jets. The N=3 chain rule matches "
        "independent first/second Richardson differences, tenfold and "
        "twelvefold padding agree, and all subterms have the required t or "
        "t^3 homogeneity. A 21-point bivariate low-amplitude projection at "
        "N=5 enumerates every yz/xy mixed channel: quartic HHHL and HLLL "
        "vanish by support, LLLL vanishes by stationarity, while HHLL has a "
        "genuine mixed term. The inviscid pressure second jet retains only "
        "HHHHL and HHLLL; its equal-ray finite coefficients are c1<0 and "
        "c3>0. Heat-weighted HHL identities give beta0, beta1, beta2 "
        "strictly positive, the total first jet has a negative N5 limit, "
        "and the complete inviscid-pressure second jet has a negative N9 "
        "limit. The complete second-jet N9 limit remains open until every "
        "viscosity-bearing quartic Fisher and mixed projector row is proved "
        "o(N9). All 6 focused tests and all 494 standalone tests pass."
    )
    principal.update(
        {
            "annular_parallel_shear_complete_finite_jets_ported": True,
            "annular_parallel_shear_mixed_channels_enumerated": True,
            "annular_parallel_shear_quartic_HHHL_zero": True,
            "annular_parallel_shear_quartic_HLLL_zero": True,
            "annular_parallel_shear_quartic_HHLL_mixed_nonzero": True,
            "annular_parallel_shear_quintic_surviving_branches": [
                "HHHHL",
                "HHLLL",
            ],
            "annular_parallel_shear_finite_c1_N5": projection[
                "finite_c1_equal_amplitude_coefficient"
            ],
            "annular_parallel_shear_finite_c3_N5": projection[
                "finite_c3_equal_amplitude_coefficient"
            ],
            "annular_parallel_shear_beta0": continuum["0"],
            "annular_parallel_shear_beta1": continuum["1"],
            "annular_parallel_shear_beta2": continuum["2"],
            "annular_parallel_shear_first_total_N5_limit": ledger[
                "first_total_limit"
            ],
            "annular_parallel_shear_second_inviscid_N9_limit": ledger[
                "second_inviscid_pressure_limit"
            ],
            "annular_parallel_shear_second_double_heat_N7_limit": ledger[
                "second_double_heat_pressure_limit"
            ],
            "annular_parallel_shear_second_inviscid_N9_certified": True,
            "annular_parallel_shear_complete_second_N9_certified": False,
            "annular_parallel_shear_uniform_Taylor_remainder_proved": False,
            "annular_parallel_shear_parabolic_window_closed": False,
            "annular_parallel_shear_finite_jet_port_audit_v1_sha256": (
                AUDIT_RESULT_SHA256
            ),
            "annular_parallel_shear_finite_jet_focused_test_count": (
                arguments.focused_test_count
            ),
            "annular_parallel_shear_finite_jet_focused_test_seconds": (
                arguments.focused_test_seconds
            ),
            "annular_parallel_shear_finite_jet_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_parallel_shear_finite_jet_full_pytest_passed": True,
            "annular_parallel_shear_finite_jet_full_pytest_seconds": float(
                regression["duration_seconds"]
            ),
            "annular_parallel_shear_finite_jet_resource_mode": (
                arguments.resource_mode
            ),
            "annular_parallel_shear_finite_jet_worker_count": (
                arguments.worker_count
            ),
            "annular_parallel_shear_finite_jet_cpu_baseline_average": (
                arguments.baseline_average
            ),
            "annular_parallel_shear_finite_jet_cpu_baseline_peak": (
                arguments.baseline_peak
            ),
            "annular_parallel_shear_finite_jet_cpu_periodic_average": (
                arguments.periodic_average
            ),
            "annular_parallel_shear_finite_jet_cpu_periodic_peak": (
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
            "Ported the complete finite first and second rho=0 generator "
            "jets to the common-polarization two-mode shear, enumerated "
            "every mixed yz/xy amplitude channel, proved the negative total "
            "first-jet N5 limit and negative inviscid-pressure second-jet "
            "N9 limit, and isolated the remaining viscosity-bearing "
            "quartic o(N9) exclusion gate."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The finite parallel-shear first and second jets and all mixed "
        "polarization amplitude channels are now ported. The total first "
        "jet has a strict negative N5 limit, and the complete inviscid "
        "pressure second jet has a strict negative N9 limit. The complete "
        "second-jet asymptotic is not yet certified: every viscosity-bearing "
        "quartic weighted-Fisher and mixed pressure-projector channel must "
        "be bounded o(N9). After that exclusion, a uniform heat-window "
        "Taylor remainder, dynamic adjoint evolution, critical L3 control, "
        "blowup, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_parallel_shear_finite_jet_port_audit.py"
    )
    bookmark["next_action"] = (
        "Prove the o(N9) exclusion for the viscosity-bearing quartic "
        "second-jet channels. Start with the H_uu[E,E] and D_u[u2_EE] "
        "weighted-Fisher pair, then the E-A transported-weight and mixed "
        "pressure-projector rows. Use parity-gauged six/five/four Phi "
        "differences and a finite-plus-dyadic internal-output split. Do not "
        "start the Taylor remainder until the complete second-jet leading "
        "coefficient is certified."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in NEW_ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 171, "unexpected completed count")
    _require(len(primary) == 629, "unexpected artifact count")
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
