"""Install the finite-Fourier pressure-load surjectivity checkpoint."""

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
    "compatible_eight_cell_cubic_graph_audit_v1.json"
)
RESULT = (
    "work/ns_collision/results/"
    "fourier_pressure_load_surjectivity_audit_v1.json"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "fourier_pressure_load_surjectivity_audit.py",
    "work/ns_collision/tests/"
    "test_fourier_pressure_load_surjectivity.py",
    "work/ns_collision/notes/fourier_pressure_load_surjectivity.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_fourier_pressure_load_surjectivity_bookmark.py",
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
        == "fourier_pressure_load_surjectivity_audit"
        and result.get("schema_version") == 1
        and result.get("status")
        == (
            "instantaneous_zero_sum_load_surjectivity_certified_"
            "quantitative_multiscale_gate_open"
        )
        and result.get("all_positive_checks_pass") is True
        and isinstance(flags, dict)
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        "pressure-load surjectivity result is not the expected audit",
    )
    for key in (
        "Fourier_velocity_to_pressure_to_load_map_derived",
        "finite_support_cross_block_isolation_certified",
        "instantaneous_zero_sum_load_space_surjectivity_proved",
        "vertex_saturating_Hamming_load_ray_PDE_realized",
        "Taylor_Green_zero_load_reconfirmed",
        "seed81_sparse_map_cross_validated",
    ):
        _require(flags.get(key) is True, f"missing positive flag: {key}")
    for key in (
        "abstract_pointwise_edge_saturator_PDE_realized",
        "uniform_quantitative_load_bound_proved",
        "pressure_L32_remainder_absorbed",
        "critical_signed_bound_proved",
        "low_regularity_passage_proved",
        "exceptional_set_upgrade_proved",
        "Navier_Stokes_global_regularity_proved",
    ):
        _require(flags.get(key) is False, f"invalid promotion flag: {key}")

    identity = result["exact_Fourier_and_Walsh_map"]
    construction = result["lacunary_surjectivity_construction"]
    support = construction["support_certificate"]
    taylor_green = result["taylor_green_sparse_check"]
    seed81 = result["seed81_sparse_benchmark"]
    _require(
        identity.get("all_checks_pass") is True
        and identity.get("target_loads")
        == identity.get("expected_Hamming_loads")
        and identity.get("target_loads")
        == [
            "-9/256",
            "-27/256",
            "-27/256",
            "-45/256",
            "-27/256",
            "-45/256",
            "-45/256",
            "225/256",
        ],
        "exact Fourier-Walsh load identity changed",
    )
    _require(
        support.get("all_checks_pass") is True
        and support.get("signed_mode_count") == 42
        and support.get("unordered_low_triple_count") == 14
        and support.get("invalid_low_triple_count") == 0,
        "finite support certificate changed",
    )
    _require(
        construction.get("all_checks_pass") is True
        and construction.get("combined_velocity_mode_count") == 42
        and construction.get(
            "maximum_relative_divergence_residual",
            1.0,
        )
        < 1.0e-14
        and construction.get(
            "maximum_target_transport_mode_residual",
            1.0,
        )
        < 1.0e-11
        and construction.get("maximum_undesired_stencil_mode", 1.0)
        < 1.0e-12
        and construction.get("maximum_load_residual", 1.0) < 1.0e-11
        and all(
            row.get("exact_unit_coupling") != "0"
            and row.get("divergence_residuals") == [0, 0, 0]
            for row in construction.get("block_rows", [])
        ),
        "surjective 42-mode realization changed",
    )
    _require(
        taylor_green.get("all_checks_pass") is True
        and taylor_green.get("maximum_stencil_transport_mode") == 0.0
        and taylor_green.get("maximum_compatible_load") == 0.0,
        "Taylor-Green sparse benchmark changed",
    )
    _require(
        seed81.get("all_checks_pass") is True
        and seed81.get("velocity_mode_count") == 116
        and abs(seed81.get("load_sum", 1.0)) < 1.0e-12
        and abs(seed81.get("pressure_residual", 1.0)) < 1.0e-11,
        "seed-81 sparse pressure benchmark changed",
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
        args.discovered_test_count == 189,
        "expected 184 inherited tests plus five new tests",
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
        == "compatible_eight_cell_cubic_graph_audit"
        and prior_result.get("all_positive_checks_pass") is True
        and _sha256(PRIOR_RESULT)
        == "067625c4b44aa6085ff2b59cbbcef351253dcbd5b1fb9f0eac641fdd7a48682c",
        "the prerequisite compatible graph result changed",
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
        principal.get("compatible_graph_targeted_test_count") == 5
        and principal.get("compatible_graph_discovered_test_count") == 184
        and principal.get("compatible_graph_monolithic_regression_passed")
        is False
        and principal.get(
            "compatible_eight_cell_cubic_graph_audit_v1_sha256"
        )
        == _sha256(PRIOR_RESULT),
        "the prerequisite compatible graph checkpoint changed",
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
        "The instantaneous compatible pressure-load map is now derived in "
        "Fourier and Walsh coordinates. Seven lacunary divergence-free "
        "three-mode blocks have exactly fourteen low signed triples and no "
        "cross-block interaction on the partition stencil. Their nonzero "
        "cubic couplings independently tune all seven Walsh coordinates, "
        "proving surjectivity onto the zero-sum eight-load space. A "
        "42-mode real field realizes the exact Hamming-profile saturating "
        "load ray; this realizes the load, not the earlier pointwise edge "
        "extremizer. Taylor-Green remains zero-load, and sparse seed-81 "
        "pressure work matches the stored grid value within 5e-15. "
        + (
            f"The complete {args.regression_test_count}-test regression "
            f"passed in {args.regression_test_seconds:.3f}s."
            if complete
            else (
                "The prior 169-test monolithic regression and twenty "
                "focused tests across the four latest stages pass. The "
                "one-shot 189-test regression was not relaunched after "
                "the inherited 174-test run crossed the daytime CPU "
                "threshold twice and parked."
            )
        )
        + " No quantitative critical estimate or Navier-Stokes regularity "
        "conclusion is claimed."
    )

    principal.update(
        {
            "pressure_load_Fourier_map_derived": True,
            "pressure_load_Walsh_isomorphism_derived": True,
            "pressure_load_cross_block_isolation_certified": True,
            "pressure_load_zero_sum_space_surjectivity_proved": True,
            "pressure_load_Hamming_ray_PDE_realized": True,
            "pressure_load_Taylor_Green_zero_reconfirmed": True,
            "pressure_load_seed81_cross_validation_passed": True,
            "pressure_load_pointwise_edge_saturator_PDE_realized": False,
            "pressure_load_uniform_quantitative_bound_proved": False,
            "pressure_load_L32_remainder_absorbed": False,
            "pressure_load_critical_bound_proved": False,
            "pressure_load_low_regularity_passage_proved": False,
            "pressure_load_exceptional_set_upgrade_proved": False,
            "pressure_load_Navier_Stokes_regularity_proved": False,
            "pressure_load_targeted_test_count": args.targeted_test_count,
            "pressure_load_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "pressure_load_discovered_test_count": (
                args.discovered_test_count
            ),
            "pressure_load_regression_test_count": (
                args.regression_test_count
            ),
            "pressure_load_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "pressure_load_monolithic_regression_passed": complete,
            "pressure_load_incremental_prior_full_test_count": 169,
            "pressure_load_incremental_prior_focused_test_count": 15,
            "pressure_load_incremental_total_focused_test_count": 20,
            "pressure_load_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "pressure_load_cpu_baseline_peak_percent": args.baseline_peak,
            "pressure_load_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Derived the exact Fourier-to-Walsh compatible pressure-load "
            "map and proved, by seven spectrally isolated divergence-free "
            "blocks, that its instantaneous image is the full zero-sum "
            "load space; realized the Hamming saturating ray while keeping "
            "pointwise edge saturation and quantitative absorption open."
        ),
    )
    theorem_obligation = (
        "The live theorem obligation is a scale-invariant quantitative "
        "realization cost for a prescribed compatible load. Derive the "
        "amplitude and spatial scaling of least L2, H1, and critical L3 "
        "costs; optimize one Fourier block and then block assemblies; and "
        "determine whether any cost remains coercive under carrier-frequency "
        "translation. Connect only a surviving critical coercivity bound "
        "to the intrinsic pressure/Fisher budget. Low-regularity and "
        "exceptional-set gates remain."
    )
    if complete:
        bookmark["unfinished_obligation"] = theorem_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
        bookmark["next_action"] = (
            "Derive exact amplitude and spatial scaling laws for the least "
            "velocity cost of realizing a fixed load. Optimize the "
            "three-polarization single block before any broad numerical "
            "search, and compare the Hamming ray with seed-81."
        )
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation is parked: the one-shot 189-test suite "
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
            "189-test command below normal priority and install complete "
            "validation. Then derive and optimize the scale-invariant "
            "least-cost load realization problem."
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
