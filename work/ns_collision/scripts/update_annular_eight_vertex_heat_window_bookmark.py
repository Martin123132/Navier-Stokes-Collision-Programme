"""Install the annular eight-vertex heat-window checkpoint."""

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
    "annular_eight_vertex_heat_window_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "5313001d5a136babf1be6d99b66767db4161e526cd08158631cde2a68c942789"
)
PREDECESSOR_SHA256 = (
    "16579e713c5bacb7b19bb9e3d63f059b9f0915588013e40aa49fdb8bf0bfea0b"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "ce8f3460e121ee0be3496051084216337d8ed59c12bbf8993279beb137d4e986"
)
RUNNER_SHA256 = (
    "4fd9cca40a4133bfce8bba21161dd827dc98ccbeee9c13f70f7b3718167a4609"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "annular_eight_vertex_heat_window_gate_audit.py",
    "work/ns_collision/tests/"
    "test_annular_eight_vertex_heat_window_gate.py",
    "work/ns_collision/notes/"
    "annular_eight_vertex_heat_window_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_annular_eight_vertex_heat_window_bookmark.py",
    "work/ns_collision/README.md",
    "work/ns_collision/scripts/run_full_regression_checkpoint.py",
    FULL_REGRESSION,
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
    parser.add_argument("--targeted-test-count", type=int, default=26)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=383)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument(
        "--periodic-average", type=float, required=True
    )
    parser.add_argument("--periodic-peak", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = _load_json(RESULT)
    _require(_sha256(RESULT) == RESULT_SHA256, "result hash changed")
    _require(
        result.get("status")
        == "eight_vertex_cancellation_and_local_heat_persistence_certified"
        and result.get("all_positive_checks_pass") is True,
        "annular eight-vertex heat-window result did not pass",
    )
    flags = result["certification_flags"]
    _require(
        flags["exact_equal_weight_eight_vertex_cancellation_proved"]
        is True
        and flags["annular_leading_response_vector_survives_proved"]
        is True
        and flags["six_nonzero_Walsh_channels_proved"] is True
        and flags["all_vertex_Fisher_partition_identity_proved"] is True
        and flags["vertex_dependent_Fisher_scaling_classified"] is True
        and flags["heat_viscosity_preserves_local_N4_obstruction_proved"]
        is True
        and flags["small_amplitude_NS_shadowing_transfer_proved"] is True
        and flags["arbitrary_weighted_eight_vertex_flux_controlled"]
        is False
        and flags["Navier_Stokes_global_regularity_proved"] is False,
        "annular eight-vertex certification scope changed",
    )
    _require(
        len(result["static_family_rows"]) == 6
        and len(result["heat_window_rows"]) == 5
        and all(
            row["all_checks_pass"]
            for row in result["static_family_rows"]
        )
        and all(
            row["all_checks_pass"]
            for row in result["heat_window_rows"]
        )
        and result["dictionary_replay"]["all_checks_pass"] is True,
        "annular eight-vertex finite replay changed",
    )

    regression = _load_json(FULL_REGRESSION)
    _require(
        _sha256(FULL_REGRESSION) == FULL_REGRESSION_SHA256,
        "full regression report hash changed",
    )
    _require(
        regression.get("schema_version") == 2
        and regression.get("configuration", {}).get("test_engine")
        == "pytest"
        and regression.get("discovered_test_count")
        == arguments.discovered_test_count
        and regression.get("tests_run")
        == arguments.discovered_test_count
        and regression.get("successful") is True
        and regression.get("exit_code") == 0
        and not regression.get("failures")
        and not regression.get("errors"),
        "full pytest regression did not pass",
    )
    _require(
        _sha256(
            "work/ns_collision/scripts/run_full_regression_checkpoint.py"
        )
        == RUNNER_SHA256,
        "full regression runner hash changed",
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
        len(bookmark.get("completed_obligations", [])) == 155
        and len(bookmark.get("primary_artifacts", [])) == 549
        and principal.get(
            "separable_annular_pressure_schur_no_go_audit_v1_sha256"
        )
        == PREDECESSOR_SHA256
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 156
        and len(bookmark.get("primary_artifacts", [])) == 554
        and principal.get(
            "annular_eight_vertex_heat_window_gate_audit_v1_sha256"
        )
        == RESULT_SHA256
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed eight-vertex checkpoint matches",
    )

    summary = result["numerical_summary"]
    continuum = result["continuum_response_certificate"]
    regression_seconds = float(regression["duration_seconds"])
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The explicit annular family has now been propagated through the "
        "true compatible eight-vertex identity and a parabolic heat "
        "window. Equal vertex weights cancel the complete HHL load exactly "
        "at every finite N and every time, but six nonconstant Walsh "
        "channels survive at order N; only the pure-x character vanishes. "
        "The exact vertex Fisher identity uses a difference operator for "
        "each plus sign and a sum operator for each minus sign, yielding "
        "the four scales N^-3, N^-1, N, and N^3. Global dissipation pays "
        "the load, but local +++ Fisher energy does not. Heat damping over "
        "T=0.1 preserves a strictly negative +++ limit and the N^4 "
        "integrated loss. A small-amplitude mild-solution shadowing lemma "
        "transfers the obstruction to universal homogeneous trajectory "
        "estimates near zero. The 26 focused tests and all 383 corpus tests "
        "pass. Adaptive edge-weight control, large-amplitude compensation, "
        "critical L3, and global regularity remain open."
    )
    principal.update(
        {
            "annular_eight_vertex_equal_weight_cancellation_proved": True,
            "annular_eight_vertex_surviving_Walsh_channels": [
                "y",
                "xy",
                "z",
                "xz",
                "yz",
                "xyz",
            ],
            "annular_eight_vertex_pure_x_channel_zero": True,
            "annular_eight_vertex_nonconstant_weight_response_survives": True,
            "annular_eight_vertex_Fisher_scaling_exponents": [
                -3,
                -1,
                1,
                3,
            ],
            "annular_eight_vertex_global_Fisher_partition_identity": True,
            "annular_eight_vertex_continuum_plus_static_limit": summary[
                "continuum_plus_static_limit"
            ],
            "annular_eight_vertex_continuum_plus_heat_integral": summary[
                "continuum_plus_heat_integrated_limit"
            ],
            "annular_eight_vertex_heat_pressure_lower_bound": continuum[
                "analytic_plus_heat_integral_absolute_lower_bound"
            ],
            "annular_eight_vertex_scaled_heat_window": continuum[
                "scaled_heat_window"
            ],
            "annular_eight_vertex_heat_viscosity": continuum["viscosity"],
            "annular_eight_vertex_largest_static_size": summary[
                "largest_static_size"
            ],
            "annular_eight_vertex_largest_load_over_Fisher": summary[
                "largest_plus_vertex_load_over_Fisher"
            ],
            "annular_eight_vertex_heat_local_N4_obstruction": True,
            "annular_eight_vertex_small_amplitude_NS_shadowing": True,
            "annular_eight_vertex_arbitrary_weighted_flux_controlled": False,
            "annular_eight_vertex_large_amplitude_phase_excluded": False,
            "annular_eight_vertex_cross_shell_HHL_absorbed": False,
            "annular_eight_vertex_terminal_dual_controlled": False,
            "annular_eight_vertex_critical_L3_controlled": False,
            "annular_eight_vertex_Navier_Stokes_regularity_proved": False,
            "annular_eight_vertex_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "annular_eight_vertex_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "annular_eight_vertex_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_eight_vertex_full_pytest_regression_passed": True,
            "annular_eight_vertex_full_pytest_runtime_seconds": (
                regression_seconds
            ),
            "annular_eight_vertex_resource_mode": arguments.resource_mode,
            "annular_eight_vertex_worker_count": arguments.worker_count,
            "annular_eight_vertex_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "annular_eight_vertex_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "annular_eight_vertex_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "annular_eight_vertex_cpu_periodic_peak_percent": (
                arguments.periodic_peak
            ),
            "annular_eight_vertex_result_status": result["status"],
            "annular_eight_vertex_heat_window_gate_audit_v1_sha256": (
                RESULT_SHA256
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
            "Derived the exact eight-vertex Walsh response and "
            "difference/sum Fisher geometry of the separable annular "
            "family, proved equal-weight cancellation but six-channel "
            "nonconstant survival, and proved persistence of the local "
            "N^4 obstruction through heat flow and perturbative "
            "Navier-Stokes shadowing."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "Global eight-cell cancellation and linear viscosity are now "
        "fully classified for the annular family. It remains to determine "
        "whether admissible nonnegative adaptive coefficients can retain "
        "the six-channel load without paying the much larger neighboring "
        "Fisher energies. Large-amplitude phase compensation, cross-shell "
        "HHL absorption, terminal dual control, critical L3, "
        "exceptional-set removal, and global regularity remain open."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_eight_vertex_heat_window_gate_audit.py"
    )
    bookmark["next_action"] = (
        "Form the twelve-edge compatible coefficient problem using the "
        "exact six Walsh load limits and vertex Fisher asymptotics "
        "E_v~N^(2r(v)-3). Derive the sharp homogeneous weighted-edge "
        "inequality for nonnegative coefficient vectors modulo constants. "
        "First solve the asymptotic finite-dimensional optimization "
        "exactly; then replay finite N. If every retaining coefficient "
        "sequence pays an order-N or larger neighboring Fisher cost, "
        "install that coercive graph theorem. Otherwise exhibit the "
        "escaping coefficient sequence and quantify its divergence."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 156, "unexpected completed count")
    _require(len(primary) == 554, "unexpected artifact count")
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
                "full_regression_sha256": _sha256(FULL_REGRESSION),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
