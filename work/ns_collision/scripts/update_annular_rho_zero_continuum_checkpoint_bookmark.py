"""Install the annular fixed-output continuum checkpoint."""

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
    "a9d4bfa2fceceb6eef599ce0938538864278889e11d776cbb88c873a57c21cef"
)
SUPERSEDED_INSTALLED_BOOKMARK_SHA256 = (
    "43d7f3f012a7c0f2138fcac7632f4129e05c5be0ca5f0188d70d75a857ea0d10"
)
FIXED_RESULT = (
    "work/ns_collision/results/"
    "annular_rho_zero_fixed_output_continuum_gate_audit_v1.json"
)
FIXED_RESULT_SHA256 = (
    "6b29ef28146f86d87ba4eeb22de596083d8b18fa451394b5f3ade69b1353d072"
)
QUADRATURE_RESULT = (
    "work/ns_collision/results/"
    "annular_rho_zero_continuum_convolution_quadrature_v1.json"
)
QUADRATURE_RESULT_SHA256 = (
    "b83ff6a1a5994bd7153027d734787e3cf29c1bff53be020423599506ce5b8f0e"
)
FULL_REGRESSION = (
    "work/ns_collision/results/full_regression_checkpoint_v1.json"
)
FULL_REGRESSION_SHA256 = (
    "c9de048e9914cbad3b37817b5455565b45ae89820a71e9ef9412b705aae7eb41"
)
RUNNER_SHA256 = (
    "4fd9cca40a4133bfce8bba21161dd827dc98ccbeee9c13f70f7b3718167a4609"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "annular_rho_zero_fixed_output_continuum_gate_audit.py",
    "work/ns_collision/tests/"
    "test_annular_rho_zero_fixed_output_continuum_gate.py",
    "work/ns_collision/notes/"
    "annular_rho_zero_fixed_output_continuum_gate.md",
    FIXED_RESULT,
    "work/ns_collision/scripts/"
    "annular_rho_zero_continuum_convolution_quadrature.py",
    "work/ns_collision/tests/"
    "test_annular_rho_zero_continuum_convolution_quadrature.py",
    "work/ns_collision/notes/"
    "annular_rho_zero_continuum_convolution_quadrature.md",
    QUADRATURE_RESULT,
    "work/ns_collision/scripts/"
    "update_annular_rho_zero_continuum_checkpoint_bookmark.py",
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


def _fit_by_degree(
    result: dict[str, Any],
    field: str,
    degree: int,
) -> dict[str, Any]:
    rows = result["tail_inverse_N_fits"]["fits"][field]
    return next(
        row for row in rows if row["degree_in_inverse_N"] == degree
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-test-count", type=int, default=23)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=449)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument("--periodic-average", type=float, required=True)
    parser.add_argument("--periodic-peak", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    fixed = _load_json(FIXED_RESULT)
    quadrature = _load_json(QUADRATURE_RESULT)
    regression = _load_json(FULL_REGRESSION)
    _require(_sha256(FIXED_RESULT) == FIXED_RESULT_SHA256, "fixed result changed")
    _require(
        _sha256(QUADRATURE_RESULT) == QUADRATURE_RESULT_SHA256,
        "quadrature result changed",
    )
    _require(
        fixed.get("status")
        == "annular_four_high_leading_continuum_reduced_tail_sign_pending"
        and fixed.get("all_positive_checks_pass") is True,
        "fixed-output continuum result did not pass",
    )
    fixed_flags = fixed["certification_flags"]
    _require(
        fixed_flags["active_output_support_proved"] is True
        and fixed_flags["signed_projector_matrix_proved"] is True
        and fixed_flags["continuum_limit_formula_proved"] is True
        and fixed_flags[
            "dominant_fixed_output_over_N7_convergence_proved"
        ]
        is True
        and fixed_flags["full_c1_over_N7_convergence_proved"] is False
        and fixed_flags["full_c1_remainder_ledger_complete"] is False
        and fixed_flags["continuum_limit_nonzero_certified"] is False
        and fixed_flags["continuum_limit_negative_certified"] is False
        and fixed_flags["four_high_N9_coefficient_certified"] is False,
        "fixed-output continuum certification scope changed",
    )
    stencil = fixed["active_output_stencil_certificate"]
    _require(
        stencil["active_output_count"] == 36
        and stencil["maximum_active_radius_squared"] == 6
        and stencil["projector_sum_matrix"]
        == "Q=(sqrt(2)/40) diag(0,-1,1)"
        and stencil["all_checks_pass"] is True,
        "fixed-output stencil changed",
    )

    _require(
        quadrature.get("status")
        == "continuum_convolution_quadrature_complete_sign_not_interval"
        and quadrature.get("all_positive_checks_pass") is True,
        "continuum quadrature did not pass",
    )
    quadrature_flags = quadrature["certification_flags"]
    rows = quadrature["rows"]
    cross = quadrature["fixed_output_cross_replay"]
    combined_fit = _fit_by_degree(
        quadrature,
        "combined_continuum_quadrature",
        4,
    )
    first_fit = _fit_by_degree(
        quadrature,
        "first_form_continuum_quadrature",
        4,
    )
    second_fit = _fit_by_degree(
        quadrature,
        "second_form_continuum_quadrature",
        4,
    )
    _require(
        [row["size"] for row in rows] == list(range(9, 66, 4))
        and all(row["all_checks_pass"] for row in rows)
        and max(
            row["energy_trace_relative_residual"] for row in rows
        )
        < 3.0e-15
        and cross["all_checks_pass"] is True
        and cross["largest_common_absolute_difference"] < 4.0e-9
        and -3.00e-7 < combined_fit["candidate_limit"] < -2.99e-7
        and quadrature_flags["continuum_sign_numerically_stable"] is True
        and quadrature_flags["continuum_sign_interval_certified"] is False
        and quadrature_flags["four_high_N9_coefficient_certified"] is False,
        "continuum quadrature replay changed",
    )

    _require(
        _sha256(FULL_REGRESSION) == FULL_REGRESSION_SHA256,
        "full regression report changed",
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
        "full regression runner changed",
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
        _sha256(BOOKMARK) == PREDECESSOR_BOOKMARK_SHA256
        and len(bookmark.get("completed_obligations", [])) == 162
        and len(bookmark.get("primary_artifacts", [])) == 584
    )
    correcting_installed = bool(
        _sha256(BOOKMARK) == SUPERSEDED_INSTALLED_BOOKMARK_SHA256
        and len(bookmark.get("completed_obligations", [])) == 164
        and len(bookmark.get("primary_artifacts", [])) == 593
    )
    installed = bool(
        len(bookmark.get("completed_obligations", [])) == 164
        and len(bookmark.get("primary_artifacts", [])) == 593
        and principal.get(
            "annular_rho_zero_fixed_output_continuum_gate_audit_v1_sha256"
        )
        == FIXED_RESULT_SHA256
        and principal.get(
            "annular_rho_zero_continuum_convolution_quadrature_v1_sha256"
        )
        == QUADRATURE_RESULT_SHA256
    )
    _require(
        predecessor or correcting_installed or installed,
        "neither predecessor nor installed continuum checkpoint matches",
    )

    regression_seconds = float(regression["duration_seconds"])
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The two N7-saturating fixed-output contractions in the "
        "four-high/one-low inviscid coefficient now have an exact "
        "continuum reduction. Exactly 36 outputs are active, and their "
        "signed pressure projectors sum to "
        "(sqrt(2)/40)diag(0,-1,1). The proof gives D_N/N7 -> L_EE "
        "with an explicit leading-contraction remainder. Identifying "
        "D_N with the full c_1,N still requires the termwise constant "
        "ledger for seven nonleading terms and three permutations. "
        "A separate fourfold quartic convolution "
        "quadrature through N=65 closes the Euler trace below 3e-16, "
        "cross-replays the original N=29 fixed-output value within "
        "3.09e-9, and gives the stable diagnostic candidate "
        "L_EE about -2.99386e-7 from cancelling pieces about "
        "+1.72241e-7 and -4.71627e-7. This remains numerical rather than "
        "an interval sign theorem. Thus both the full tail and sign gates "
        "remain open. All 23 focused tests and all 449 corpus tests pass."
    )
    principal.update(
        {
            "annular_fixed_output_active_mode_count": 36,
            "annular_fixed_output_maximum_radius_squared": 6,
            "annular_fixed_output_projector_sum": (
                "Q=(sqrt(2)/40) diag(0,-1,1)"
            ),
            "annular_fixed_output_leading_D_over_N7_convergence_proved": (
                True
            ),
            "annular_fixed_output_full_c1_over_N7_convergence_proved": (
                False
            ),
            "annular_fixed_output_full_c1_remainder_ledger_complete": False,
            "annular_fixed_output_continuum_sign_certified": False,
            "annular_fixed_output_N9_coefficient_certified": False,
            "annular_fixed_output_result_status": fixed["status"],
            "annular_rho_zero_fixed_output_continuum_gate_audit_v1_sha256": (
                FIXED_RESULT_SHA256
            ),
            "annular_continuum_quadrature_sizes": [
                row["size"] for row in rows
            ],
            "annular_continuum_quadrature_N65_value": rows[-1][
                "combined_continuum_quadrature"
            ],
            "annular_continuum_quadrature_tail_degree4_first_candidate": (
                first_fit["candidate_limit"]
            ),
            "annular_continuum_quadrature_tail_degree4_second_candidate": (
                second_fit["candidate_limit"]
            ),
            "annular_continuum_quadrature_tail_degree4_combined_candidate": (
                combined_fit["candidate_limit"]
            ),
            "annular_continuum_quadrature_tail_degree4_replay_residual": (
                combined_fit["maximum_replay_residual"]
            ),
            "annular_continuum_quadrature_maximum_energy_trace_residual": max(
                row["energy_trace_relative_residual"] for row in rows
            ),
            "annular_continuum_quadrature_fixed_output_cross_residual": (
                cross["largest_common_absolute_difference"]
            ),
            "annular_continuum_quadrature_sign_numerically_stable": True,
            "annular_continuum_quadrature_sign_interval_certified": False,
            "annular_continuum_quadrature_result_status": quadrature["status"],
            "annular_rho_zero_continuum_convolution_quadrature_v1_sha256": (
                QUADRATURE_RESULT_SHA256
            ),
            "annular_continuum_targeted_test_count": (
                arguments.targeted_test_count
            ),
            "annular_continuum_targeted_test_runtime_seconds": (
                arguments.targeted_test_seconds
            ),
            "annular_continuum_discovered_test_count": (
                arguments.discovered_test_count
            ),
            "annular_continuum_full_pytest_passed": True,
            "annular_continuum_full_pytest_runtime_seconds": (
                regression_seconds
            ),
            "annular_continuum_resource_mode": arguments.resource_mode,
            "annular_continuum_worker_count": arguments.worker_count,
            "annular_continuum_cpu_baseline_average_percent": (
                arguments.baseline_average
            ),
            "annular_continuum_cpu_baseline_peak_percent": (
                arguments.baseline_peak
            ),
            "annular_continuum_cpu_periodic_average_percent": (
                arguments.periodic_average
            ),
            "annular_continuum_cpu_periodic_peak_percent": (
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
    superseded_obligation = (
        "Reduced the candidate four-high N7 coefficient to exactly "
        "36 fixed pressure outputs, evaluated their signed projector "
        "sum in exact arithmetic, and proved c_1,N/N7 converges to "
        "the explicit anisotropic continuum functional L_EE."
    )
    corrected_obligation = (
        "Reduced the two N7-saturating four-high contractions to exactly "
        "36 fixed pressure outputs, evaluated their signed projector sum "
        "in exact arithmetic, and proved their sum D_N/N7 converges to "
        "the explicit anisotropic continuum functional L_EE while "
        "retaining the full c_1,N tail obligation."
    )
    if superseded_obligation in completed:
        completed[completed.index(superseded_obligation)] = (
            corrected_obligation
        )
    _append_once(
        completed,
        corrected_obligation,
    )
    _append_once(
        completed,
        (
            "Built a fourfold dealiased continuum convolution quadrature "
            "through N=65, replayed the Euler energy trace and original "
            "fixed-output route, and isolated a stable negative numerical "
            "candidate while retaining the rigorous interval-sign gate."
        ),
    )
    bookmark["unfinished_obligation"] = (
        "The leading four-high sum has the proved limit formula "
        "D_N/N7 -> L_EE, and independent quadrature places L_EE near "
        "-2.99386e-7. Two proof gates remain: instantiate and check the "
        "termwise tail constants needed to prove "
        "(c_1,N-D_N)/N7 -> 0, and give a validated joint enclosure for "
        "the cancelling L_VV and L_GH integrals. Neither the full "
        "c_1,N limit nor its sign is certified. No optimized N9 law, "
        "uniform Taylor window, critical L3 estimate, "
        "exceptional-set removal, blowup result, or global regularity "
        "theorem is proved."
    )
    bookmark["resume_command"] = (
        "python work/ns_collision/scripts/"
        "annular_rho_zero_continuum_convolution_quadrature.py "
        "--sizes 9,13,17,21,25,29,33,37,41,45,49,53,57,61,65"
    )
    bookmark["next_action"] = (
        "First complete the full c_1,N tail theorem: expand the seven "
        "nonleading amplitude-one terms and three nonleading "
        "symmetrized permutations, split bounded and dyadic pressure "
        "outputs, and attach explicit N-uniform constants to the target "
        "|c_1,N-D_N|<=C N6 log(2+N). In parallel-ready form, preserve "
        "the interval plan for L_EE, but do not certify the full limit or "
        "sign from inverse-N fits and do not rerun the eightfold grid."
    )

    primary = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary, artifact)
    _require(len(completed) == 164, "unexpected completed count")
    _require(len(primary) == 593, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary),
                "fixed_result_sha256": _sha256(FIXED_RESULT),
                "quadrature_result_sha256": _sha256(QUADRATURE_RESULT),
                "full_regression_sha256": _sha256(FULL_REGRESSION),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
