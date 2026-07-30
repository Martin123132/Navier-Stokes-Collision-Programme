"""Install the compatible eight-cell cubic graph checkpoint."""

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
PRIOR_RESULT = (
    "work/ns_collision/results/"
    "cubic_zero_face_edge_envelope_audit_v1.json"
)
RESULT = (
    "work/ns_collision/results/"
    "compatible_eight_cell_cubic_graph_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "compatible_eight_cell_cubic_graph_audit.py",
    "work/ns_collision/tests/"
    "test_compatible_eight_cell_cubic_graph.py",
    "work/ns_collision/notes/compatible_eight_cell_cubic_graph.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_compatible_eight_cell_graph_bookmark.py",
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
        choices=("complete", "inherited_cpu_parked_incremental"),
        default="inherited_cpu_parked_incremental",
    )
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, required=True)
    parser.add_argument("--regression-test-count", type=int, default=0)
    parser.add_argument("--regression-test-seconds", type=float, default=0.0)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    return parser.parse_args()


def _validate_result() -> dict[str, Any]:
    result = _load_json(RESULT)
    flags = result.get("certification_flags")
    checks = result.get("positive_checks")
    _require(
        result.get("kind")
        == "compatible_eight_cell_cubic_graph_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "graph_projection_derived_uniform_compatibility_gain_"
            "falsified_PDE_load_cone_open"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict)
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        "compatible graph result is not the expected passing audit",
    )
    for key in (
        "eight_cell_pressure_load_projection_derived",
        "eight_cell_projective_cubic_reduction_derived",
        "eight_cell_cubic_energy_nonconvexity_proved",
        "compatibility_only_uniform_strict_gain_falsified",
        "Taylor_Green_compatible_pressure_load_annihilated",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "abstract_vertex_saturator_PDE_realized",
        "Navier_Stokes_pressure_load_cone_characterized",
        "pressure_L32_remainder_absorbed",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    graph = result["exact_projection_and_cubic_energy"]
    nonconvexity = result["nonconvexity_witness"]
    saturator = result["abstract_vertex_saturator"]
    taylor_green = result["taylor_green_projection"]
    _require(
        graph.get("all_checks_pass") is True
        and graph.get("translation_residual") == "0"
        and graph.get("constant_vector_energy") == "0"
        and graph.get("projective_symbolic_residual") == "0",
        "exact graph reduction changed",
    )
    _require(
        nonconvexity.get("all_checks_pass") is True
        and nonconvexity.get("exact_convexity_violation") == "39/128"
        and nonconvexity.get("midpoint_energy") == "95/4",
        "nonconvexity witness changed",
    )
    _require(
        saturator.get("all_checks_pass") is True
        and saturator.get("normalized_cubic_energy_c_equals_t_equals_one")
        == "75/256"
        and saturator.get("normalized_pressure_load") == "225/256"
        and saturator.get("normalized_objective") == "75/128"
        and saturator.get("normalized_directionwise_envelope")
        == "75/128"
        and saturator.get("normalized_load_by_Hamming_distance")
        == {
            "0": "225/256",
            "1": "-45/256",
            "2": "-27/256",
            "3": "-9/256",
        }
        and all(
            row.get("load_sum") == "0"
            for row in saturator.get("vertex_rows", [])
        ),
        "abstract vertex saturator changed",
    )
    _require(
        taylor_green.get("all_checks_pass") is True
        and taylor_green.get("maximum_numerical_load", 1.0) < 1.0e-17
        and taylor_green.get("normalized_global_graph_supremum") == 0.0
        and taylor_green.get("normalized_directionwise_envelope", 0.0)
        > 0.0,
        "Taylor-Green graph projection changed",
    )
    return result


def main() -> None:
    args = _parse_args()
    _require(
        args.targeted_test_count == 5,
        "this checkpoint requires exactly five focused tests",
    )
    _require(args.targeted_test_seconds >= 0.0, "invalid test runtime")
    _require(
        args.discovered_test_count == 184,
        "expected 179 inherited tests plus five new tests",
    )
    _require(
        0.0 <= args.baseline_average <= args.baseline_peak,
        "invalid CPU sample",
    )
    _require(
        args.baseline_average <= 60.0,
        "the focused validation requires a permitted daytime baseline",
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
    else:
        _require(
            args.regression_test_count == 0,
            "incremental mode cannot claim a completed regression",
        )
        _require(
            args.regression_test_seconds == 0.0,
            "incremental mode cannot record a regression runtime",
        )

    for artifact in ARTIFACTS:
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")
    result = _validate_result()

    prior_result = _load_json(PRIOR_RESULT)
    _require(
        prior_result.get("kind")
        == "cubic_zero_face_edge_envelope_audit"
        and prior_result.get("all_positive_checks_pass") is True
        and _sha256(PRIOR_RESULT)
        == "7d0f37b0942a3e2103bb9fcc85b9d9e8fad74eb9266ba04e156dbd5baa357202",
        "the prerequisite cubic zero-face result changed",
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
        principal.get("rho_terminal_tax_monolithic_regression_passed")
        is True
        and principal.get("rho_terminal_tax_regression_test_count") == 169,
        "the prerequisite terminal-tax checkpoint is absent",
    )
    _require(
        principal.get("cubic_zero_face_targeted_test_count") == 5
        and principal.get("cubic_zero_face_discovered_test_count") == 179
        and principal.get("cubic_zero_face_monolithic_regression_passed")
        is False
        and principal.get(
            "cubic_zero_face_edge_envelope_audit_v1_sha256"
        )
        == _sha256(PRIOR_RESULT),
        "the prerequisite parked cubic zero-face checkpoint changed",
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
    if args.validation_mode == "inherited_cpu_parked_incremental":
        _require(
            bookmark.get("status") == "parked"
            and principal.get(
                "intrinsic_tail_cpu_parked_test_runtime_seconds"
            )
            > 0.0
            and principal.get(
                "intrinsic_tail_cpu_first_high_average_percent"
            )
            > 75.0
            and principal.get(
                "intrinsic_tail_cpu_second_high_average_percent"
            )
            > 75.0,
            "the inherited CPU-parked validation state is absent",
        )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The globally compatible eight-cell pressure/Fisher problem now "
        "reduces exactly to S(w)=b.w-cQ(w), where b is a seven-dimensional "
        "zero-sum projection of the conditional pressure edges and Q is a "
        "nonnegative homogeneous cubic. Exact ray optimization gives a "
        "projective quotient, while a rational 39/128 witness proves Q is "
        "not convex. A smooth abstract vertex flux attains the complete "
        "directionwise cubic envelope, ruling out any universal strict "
        "gain from coefficient compatibility alone. Conversely, the "
        "Taylor-Green edge is pure conditional mode three and has b=0, so "
        "compatibility annihilates its positive relaxed envelope. The "
        "PDE-realizable load cone remains open. "
        + (
            f"The complete {args.regression_test_count}-test regression "
            f"passed in {args.regression_test_seconds:.3f}s."
            if complete
            else (
                "The prior 169-test monolithic regression and fifteen "
                "focused tests across the three latest stages pass. The "
                "one-shot 184-test regression was not relaunched after "
                "the inherited 174-test run crossed the daytime CPU "
                "threshold twice and parked."
            )
        )
        + " No critical estimate or Navier-Stokes regularity conclusion "
        "is claimed."
    )

    principal.update(
        {
            "compatible_graph_pressure_load_projection_derived": True,
            "compatible_graph_projective_reduction_derived": True,
            "compatible_graph_cubic_energy_nonconvexity_proved": True,
            "compatible_graph_uniform_strict_gain_falsified": True,
            "compatible_graph_vertex_load_Hamming_profile_derived": True,
            "compatible_graph_Taylor_Green_load_annihilated": True,
            "compatible_graph_vertex_saturator_PDE_realized": False,
            "compatible_graph_PDE_load_cone_characterized": False,
            "compatible_graph_pressure_L32_remainder_absorbed": False,
            "compatible_graph_critical_bound_proved": False,
            "compatible_graph_low_regularity_passage_proved": False,
            "compatible_graph_exceptional_set_upgrade_proved": False,
            "compatible_graph_Navier_Stokes_regularity_proved": False,
            "compatible_graph_targeted_test_count": (
                args.targeted_test_count
            ),
            "compatible_graph_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "compatible_graph_discovered_test_count": (
                args.discovered_test_count
            ),
            "compatible_graph_regression_test_count": (
                args.regression_test_count
            ),
            "compatible_graph_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "compatible_graph_monolithic_regression_passed": complete,
            "compatible_graph_incremental_prior_full_test_count": 169,
            "compatible_graph_incremental_prior_focused_test_count": 10,
            "compatible_graph_incremental_total_focused_test_count": 15,
            "compatible_graph_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "compatible_graph_cpu_baseline_peak_percent": (
                args.baseline_peak
            ),
            "compatible_graph_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Reduced the globally compatible eight-cell pressure problem "
            "to its exact zero-sum load and projective cubic energy, proved "
            "the energy nonconvex, constructed all eight abstract "
            "vertex-saturating load rays, and showed Taylor-Green lies at "
            "the opposite zero-load extreme without conflating abstract "
            "edge compatibility with PDE realizability."
        ),
    )
    theorem_obligation = (
        "The live theorem obligation is the Fourier-triad image of "
        "divergence-free velocity coefficients in the seven-dimensional "
        "compatible pressure-load space. Determine whether any of the "
        "eight vertex-saturating Hamming-profile rays lies in the image or "
        "its closure. A realization is a PDE-level no-go for compatibility "
        "alone; an exclusion must be quantified in a scale-invariant norm "
        "before it can support pressure absorption. Low-regularity and "
        "exceptional-set gates remain."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
        bookmark["next_action"] = (
            "Derive the finite Fourier-triad polynomial map into the load "
            "vector b, retaining every interaction that can reach "
            "conditional frequencies zero or m. Test its image against "
            "the eight exact Hamming-profile saturating rays."
        )
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation is parked: the one-shot 184-test suite "
            "must pass when the daytime baseline CPU is at most 60%. "
            + theorem_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
        bookmark["next_action"] = (
            "On a fresh turn, sample total CPU for at least five seconds. "
            "If the daytime average is at most 60%, run the parked "
            "184-test command below normal priority and install complete "
            "validation. Then derive the finite Fourier-triad map into b "
            "and test the eight exact saturating load rays."
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
