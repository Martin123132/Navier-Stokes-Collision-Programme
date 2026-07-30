"""Install the finite-window rho terminal-tax no-go checkpoint."""

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
    "finite_window_rho_terminal_tax_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "finite_window_rho_terminal_tax_audit.py",
    "work/ns_collision/tests/"
    "test_finite_window_rho_terminal_tax.py",
    "work/ns_collision/notes/"
    "finite_window_rho_terminal_tax_no_go.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_finite_window_rho_terminal_tax_bookmark.py",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require(
        args.targeted_test_count == 5,
        "this checkpoint requires exactly five focused tests",
    )
    _require(args.targeted_test_seconds >= 0.0, "invalid test runtime")
    _require(
        args.discovered_test_count == 169,
        "expected the inherited 164 tests plus five new tests",
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
            args.baseline_average > 60.0,
            "parked installation requires a daytime CPU gate",
        )
    for artifact in ARTIFACTS:
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")

    result = _load_json(RESULT)
    flags = result.get("certification_flags")
    checks = result.get("positive_checks")
    _require(
        result.get("kind")
        == "finite_window_rho_terminal_tax_no_go_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "canonical_positive_rho_finite_window_route_closed_by_"
            "terminal_tax_identity"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict)
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        "terminal-tax result is not the expected passing audit",
    )
    for key in (
        "terminal_tax_identity_derived",
        "terminal_tax_nonnegative_for_positive_rho",
        "rho_zero_globally_minimizes_fixed_weight_generator",
        "rho_zero_globally_minimizes_generator_supremum",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "positive_rho_finite_window_advantage_in_this_dual_class",
        "formal_quadratic_crossover_is_a_sign_target",
        "seed81_finite_window_sign_search_required",
        "random_or_path_adapted_weight_route_excluded",
        "signed_or_multi_replica_route_excluded",
        "intrinsic_scale_pressure_tail_bound_proved",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    theorem = result["theorem"]
    stress = result["weighted_chaos_stress"]
    _require(
        theorem.get("all_checks_pass") is True
        and theorem.get("difference_symbolic_residual") == "0",
        "terminal-tax endpoint algebra failed",
    )
    _require(
        stress.get("all_checks_pass") is True
        and stress.get("rho_one_tax_residual", 1.0) < 1.0e-13
        and stress.get("tax_monotone_on_sampled_grid") is True,
        "weighted chaos stress failed",
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
        principal.get("adjoint_replica_regression_test_count") == 159
        and principal.get("scale_rho_targeted_test_count") == 5
        and principal.get("scale_rho_discovered_test_count") == 164
        and principal.get("scale_rho_monolithic_regression_passed")
        is False,
        "the inherited parked 164-test checkpoint is absent",
    )
    _require(
        principal.get("rho_short_time_first_correction_derived") is True
        and principal.get("rho_Taylor_crossover_certified") is False,
        "the prerequisite short-time rho checkpoint is absent",
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
        "The canonical positive-correlation finite-window branch is closed "
        "for deterministic nonnegative backward terminal weights. The exact "
        "identity J_rho-J_0=(3/2) integral lambda_T(C_rho-C_0), together "
        "with the Wiener-chaos expansion, makes this terminal variance tax "
        "nonnegative for every 0<=rho<=1. Therefore rho=0 minimizes both "
        "the fixed-weight generator and its supremum on every common "
        "admissible class. The formal h=0.0756123 quadratic crossover is a "
        "truncation artifact, not a finite-window sign target. This does "
        "not exclude path-adapted weights, signed or multi-replica "
        "constructions, or correlated replicas as a representation, and it "
        "does not prove the critical bound or regularity. "
        + (
            "The complete 169-test regression passed in "
            f"{args.regression_test_seconds:.3f}s after a permitted "
            f"{args.baseline_average:.2f}% launch baseline."
            if args.validation_mode == "complete"
            else (
                "The prior 159-test full regression, five scale/rho tests, "
                "and five new terminal-tax tests pass incrementally. The "
                "one-shot 169-test suite remains parked because the "
                "daytime baseline averaged "
                f"{args.baseline_average:.2f}% and peaked at "
                f"{args.baseline_peak:.2f}%."
            )
        )
        + " The independent 64128-pivot checkpoint remains valid."
    )

    principal.update(
        {
            "rho_terminal_tax_identity_derived": True,
            "rho_terminal_tax_nonnegative_for_positive_rho": True,
            "rho_zero_finite_window_fixed_weight_optimal_proved": True,
            "rho_zero_finite_window_supremum_optimal_proved": True,
            "rho_positive_finite_window_advantage_excluded_in_dual_class": (
                True
            ),
            "rho_formal_crossover_route_closed": True,
            "rho_seed81_finite_window_sign_solver_required": False,
            "rho_path_adapted_weight_route_excluded": False,
            "rho_signed_or_multi_replica_route_excluded": False,
            "rho_terminal_tax_pressure_tail_bound_proved": False,
            "rho_terminal_tax_critical_bound_proved": False,
            "rho_terminal_tax_low_regularity_passage_proved": False,
            "rho_terminal_tax_exceptional_set_upgrade_proved": False,
            "rho_terminal_tax_Navier_Stokes_regularity_proved": False,
            "rho_terminal_tax_targeted_test_count": (
                args.targeted_test_count
            ),
            "rho_terminal_tax_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "rho_terminal_tax_discovered_test_count": (
                args.discovered_test_count
            ),
            "rho_terminal_tax_regression_test_count": (
                args.regression_test_count
            ),
            "rho_terminal_tax_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "rho_terminal_tax_monolithic_regression_passed": (
                args.validation_mode == "complete"
            ),
            "rho_terminal_tax_incremental_prior_full_test_count": (
                0 if args.validation_mode == "complete" else 159
            ),
            "rho_terminal_tax_incremental_prior_targeted_test_count": (
                0 if args.validation_mode == "complete" else 5
            ),
            "rho_terminal_tax_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "rho_terminal_tax_cpu_baseline_peak_percent": (
                args.baseline_peak
            ),
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Combined the exact backward weighted replica balance, cubic "
            "terminal penalty, and Wiener-chaos ordering to prove that the "
            "full finite-window positive-rho correction is exactly its "
            "nonnegative terminal variance tax. This closes the proposed "
            "h=0.0756 net-sign search for canonical replicas with "
            "deterministic nonnegative terminal weights."
        ),
    )
    theorem_obligation = (
        "The main mathematical obligation is now a rho=0 pressure-tail "
        "estimate uniform when partition frequency tracks local amplitude "
        "through Re_cell=a/(nu m), including adaptive overlap, zero-face "
        "degeneracy, preservation of the full terminal dual supremum, "
        "low-regularity passage, and an exceptional-set upgrade."
    )
    if args.validation_mode == "complete":
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
        bookmark["next_action"] = (
            "Derive the rho=0 intrinsic-frequency pressure-tail "
            "decomposition, starting with a dyadic high-frequency "
            "paraproduct identity and an amplitude-scaled counterexample "
            "gate. Require any candidate to preserve adaptive overlap, "
            "zero-face degeneracy, and the full nonnegative terminal-weight "
            "supremum. Do not run the superseded seed-81 finite-window rho "
            "sign search."
        )
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation is parked: the one-shot 169-test suite "
            "must pass when the daytime baseline CPU is at most 60%. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
        bookmark["next_action"] = (
            "Sample total CPU for at least five seconds. If the daytime "
            "average is at most 60%, run the parked 169-test command below "
            "normal priority and install a completed validation checkpoint. "
            "Then derive the rho=0 intrinsic-frequency pressure-tail "
            "decomposition. Do not run the superseded seed-81 finite-window "
            "rho sign search."
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
