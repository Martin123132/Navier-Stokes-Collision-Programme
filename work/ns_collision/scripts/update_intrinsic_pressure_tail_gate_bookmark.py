"""Install the intrinsic pressure-tail and zero-face gate checkpoint."""

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
    "intrinsic_pressure_tail_gate_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/intrinsic_pressure_tail_gate_audit.py",
    "work/ns_collision/tests/test_intrinsic_pressure_tail_gate.py",
    "work/ns_collision/notes/intrinsic_pressure_tail_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_intrinsic_pressure_tail_gate_bookmark.py",
    "work/ns_collision/README.md",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
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
    parser.add_argument(
        "--validation-mode",
        choices=("complete", "cpu_parked_incremental"),
        default="complete",
    )
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, required=True)
    parser.add_argument("--regression-test-count", type=int, default=0)
    parser.add_argument("--regression-test-seconds", type=float, default=0.0)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    parser.add_argument("--parked-test-seconds", type=float, default=0.0)
    parser.add_argument("--first-high-average", type=float, default=0.0)
    parser.add_argument("--second-high-average", type=float, default=0.0)
    parser.add_argument("--runtime-peak", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require(
        args.targeted_test_count == 5,
        "this checkpoint requires exactly five focused tests",
    )
    _require(args.targeted_test_seconds >= 0.0, "invalid test runtime")
    _require(
        args.discovered_test_count == 174,
        "expected the inherited 169 tests plus five new tests",
    )
    _require(
        args.baseline_peak >= args.baseline_average,
        "invalid CPU sample",
    )
    if args.validation_mode == "complete":
        _require(
            args.regression_test_count == args.discovered_test_count,
            "complete mode requires all discovered tests",
        )
        _require(
            args.regression_test_seconds > 0.0,
            "complete mode requires the regression runtime",
        )
        _require(
            args.baseline_average <= 60.0,
            "complete mode requires a permitted launch baseline",
        )
    else:
        _require(
            args.regression_test_count == 0,
            "parked mode cannot claim a completed regression",
        )
        _require(
            args.baseline_average <= 60.0,
            "parked run must have started from a permitted baseline",
        )
        _require(
            args.parked_test_seconds > 0.0,
            "parked mode requires the interrupted runtime",
        )
        _require(
            args.first_high_average > 75.0
            and args.second_high_average > 75.0,
            "parked mode requires two consecutive CPU threshold breaches",
        )
        _require(
            args.runtime_peak
            >= max(args.first_high_average, args.second_high_average),
            "invalid runtime CPU peak",
        )
    for artifact in ARTIFACTS:
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")

    result = _load_json(RESULT)
    flags = result.get("certification_flags")
    checks = result.get("positive_checks")
    _require(
        result.get("kind") == "intrinsic_pressure_tail_gate_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "intrinsic_tail_scaling_certified_"
            "arbitrary_weight_absolute_route_blocked"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict)
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        "pressure-tail result is not the expected passing audit",
    )
    for key in (
        "dyadic_high_pressure_tail_identity_derived",
        "unweighted_L2_pressure_tail_bound_derived",
        "positive_floor_intrinsic_absorption_derived",
        "fixed_frequency_tail_absorption_falsified_by_scaling",
        "intrinsic_frequency_necessity_reconfirmed",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "uniform_arbitrary_weight_CZ_localization_available",
        "zero_face_full_terminal_supremum_preserved",
        "floor_free_signed_pressure_edge_bound_proved",
        "intrinsic_scale_pressure_tail_bound_proved",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    tail = result["dyadic_tail_decomposition"]
    scaling = result["taylor_green_amplitude_frequency_gate"]
    zero_face = result["zero_face_weight_gate"]
    _require(
        tail.get("all_checks_pass") is True
        and scaling.get("all_checks_pass") is True
        and zero_face.get("all_checks_pass") is True,
        "one pressure-tail sub-audit failed",
    )
    _require(
        scaling.get("normalized_pressure_tail_flux") == "beta/32"
        and zero_face["rows"][-1][
            "explicit_Hilbert_norm_lower_bound"
        ]
        > 70.0,
        "pressure-tail audit lost its quantitative margins",
    )

    bookmark = _load_json(BOOKMARK)
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark",
        "refusing to update a non-NS bookmark",
    )
    _require(
        bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "bookmark is outside the standalone workspace boundary",
    )
    principal = bookmark.setdefault("principal_results", {})
    _require(
        principal.get(
            "rho_terminal_tax_monolithic_regression_passed"
        )
        is True
        and principal.get("rho_terminal_tax_regression_test_count") == 169
        and principal.get(
            "rho_positive_finite_window_advantage_excluded_in_dual_class"
        )
        is True,
        "the prerequisite 169-test terminal-tax checkpoint is absent",
    )
    _require(
        principal.get(
            "reversible_weighted_hypercircle_componentwise64128_certified"
        )
        is True
        and principal.get(
            "reversible_weighted_hypercircle_full_inertia_certified"
        )
        is False,
        "the independent finite-pencil checkpoint changed unexpectedly",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    bookmark["checkpointed_at"] = now
    bookmark["status"] = (
        "checkpointed"
        if args.validation_mode == "complete"
        else "parked"
    )
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The rho=0 pressure-tail stage derives the exact low-low-free "
        "dyadic decomposition and the rigorous estimate "
        "||Q_m p||_2<=2||u||_infinity||grad u||_2/m. It yields intrinsic "
        "absorption at m>=||u||_infinity^2/(nu lambda_*) when the terminal "
        "weight has a positive floor lambda_*. An exact co-scaled "
        "Taylor-Green family places the whole pressure above the cutoff and "
        "reconfirms that fixed-frequency absorption fails while "
        "m proportional to amplitude/nu is scale invariant. The explicit "
        "zero-face weights epsilon+sin(x/2)^2 have singular-integral norm "
        "lower bounds growing as epsilon^(-1/4), so a uniform weighted "
        "Calderon-Zygmund localization cannot preserve the full terminal "
        "supremum. This is a route no-go, not a falsification of every "
        "signed pressure-edge estimate, and no critical or regularity "
        "conclusion is claimed. "
        + (
            "The complete 174-test regression passed in "
            f"{args.regression_test_seconds:.3f}s after a permitted "
            f"{args.baseline_average:.2f}% launch baseline."
            if args.validation_mode == "complete"
            else (
                "The prior 169-test full regression and all five new "
                "focused tests pass, but the one-shot 174-test suite is "
                f"parked after {args.parked_test_seconds:.3f}s because "
                "two consecutive runtime CPU windows averaged "
                f"{args.first_high_average:.2f}% and "
                f"{args.second_high_average:.2f}%, with a sampled peak of "
                f"{args.runtime_peak:.2f}%."
            )
        )
        + " The independent 64128-pivot checkpoint remains valid."
    )

    principal.update(
        {
            "intrinsic_tail_dyadic_identity_derived": True,
            "intrinsic_tail_unweighted_L2_bound_derived": True,
            "intrinsic_tail_positive_floor_absorption_derived": True,
            "intrinsic_tail_fixed_frequency_absorption_falsified": True,
            "intrinsic_tail_frequency_necessity_reconfirmed": True,
            "intrinsic_tail_Taylor_Green_scaling_gate_passed": True,
            "intrinsic_tail_zero_face_A2_obstruction_derived": True,
            "intrinsic_tail_uniform_weighted_CZ_route_available": False,
            "intrinsic_tail_full_terminal_supremum_preserved": False,
            "intrinsic_tail_floor_free_signed_edge_bound_proved": False,
            "intrinsic_tail_uniform_bound_proved": False,
            "intrinsic_tail_critical_bound_proved": False,
            "intrinsic_tail_low_regularity_passage_proved": False,
            "intrinsic_tail_exceptional_set_upgrade_proved": False,
            "intrinsic_tail_Navier_Stokes_regularity_proved": False,
            "intrinsic_tail_targeted_test_count": args.targeted_test_count,
            "intrinsic_tail_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "intrinsic_tail_discovered_test_count": (
                args.discovered_test_count
            ),
            "intrinsic_tail_regression_test_count": (
                args.regression_test_count
            ),
            "intrinsic_tail_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "intrinsic_tail_monolithic_regression_passed": (
                args.validation_mode == "complete"
            ),
            "intrinsic_tail_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "intrinsic_tail_cpu_baseline_peak_percent": args.baseline_peak,
            "intrinsic_tail_cpu_parked_test_runtime_seconds": (
                args.parked_test_seconds
            ),
            "intrinsic_tail_cpu_first_high_average_percent": (
                args.first_high_average
            ),
            "intrinsic_tail_cpu_second_high_average_percent": (
                args.second_high_average
            ),
            "intrinsic_tail_cpu_runtime_peak_percent": args.runtime_peak,
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Derived the low-low-free dyadic pressure-tail identity and "
            "scale-correct unweighted L2 estimate, certified conditional "
            "intrinsic absorption, reconfirmed its necessity on an exact "
            "co-scaled Taylor-Green tail, and proved that a uniform weighted "
            "singular-integral localization cannot cross arbitrary "
            "zero-face terminal weights."
        ),
    )
    theorem_obligation = (
        "The live pressure obligation is a signed dyadic pressure-flux "
        "Carleson estimate on the balanced intrinsic cover. Neighboring "
        "antisymmetric transfers must be summed before absolute values, and "
        "only coefficient mismatch may be charged. The estimate must use "
        "the Lipschitz radius and 2:1 balance without inserting an A2 floor, "
        "and must preserve the full terminal supremum. Low-regularity and "
        "exceptional-set gates remain."
    )
    if args.validation_mode == "complete":
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
        bookmark["next_action"] = (
            "Formulate the signed dyadic edge-flux Carleson sum on the "
            "existing monotone 2:1 balanced cover. Derive its discrete "
            "summation-by-parts identity and test coefficient-mismatch "
            "square summability against the Taylor-Green scaling family and "
            "the zero-face limit. Do not use cellwise absolute pressure "
            "tails or a uniform weighted Calderon-Zygmund constant."
        )
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation is parked: the one-shot 174-test suite "
            "must pass when the daytime baseline CPU is at most 60%. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
        bookmark["next_action"] = (
            "Sample total CPU for at least five seconds. If the daytime "
            "average is at most 60%, run the parked 174-test command below "
            "normal priority and install a completed validation checkpoint. "
            "Then formulate the signed dyadic edge-flux Carleson sum."
        )

    primary_artifacts = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary_artifacts, artifact)

    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary_artifacts),
                "status": bookmark["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
