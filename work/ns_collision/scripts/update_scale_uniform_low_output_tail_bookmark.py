"""Install the scale-uniform low-output stress-tail checkpoint."""

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
    "smooth_galerkin_shell_response_gate_audit_v1.json"
)
PRIOR_RESULT_SHA256 = (
    "a226f430ea1518c54780671abd3f17055333770737566e6badf4eb4a8931f0ad"
)
RESULT = (
    "work/ns_collision/results/"
    "scale_uniform_low_output_tail_gate_audit_v1.json"
)
RESULT_SHA256 = (
    "3622a76234d31dcf298e0326b1be75f888fb925926df69fd65e45a9c80c6b657"
)
ARTIFACTS = (
    "work/ns_collision/scripts/"
    "scale_uniform_low_output_tail_gate_audit.py",
    "work/ns_collision/tests/"
    "test_scale_uniform_low_output_tail_gate.py",
    "work/ns_collision/notes/"
    "scale_uniform_low_output_tail_gate.md",
    RESULT,
    "work/ns_collision/scripts/"
    "update_scale_uniform_low_output_tail_bookmark.py",
    "work/ns_collision/README.md",
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
    parser.add_argument(
        "--validation-mode",
        choices=("complete", "focused_complete_full_regression_parked"),
        default="focused_complete_full_regression_parked",
    )
    parser.add_argument("--targeted-test-count", type=int, default=5)
    parser.add_argument("--dependency-test-count", type=int, default=40)
    parser.add_argument("--dependency-test-seconds", type=float, required=True)
    parser.add_argument("--discovered-test-count", type=int, default=271)
    parser.add_argument("--regression-test-count", type=int, default=0)
    parser.add_argument("--regression-test-seconds", type=float, default=0.0)
    parser.add_argument("--resource-mode", default="daytime_one_worker")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--baseline-average", type=float, required=True)
    parser.add_argument("--baseline-peak", type=float, required=True)
    return parser.parse_args()


def _validate_result(result: dict[str, Any]) -> None:
    _require(_sha256(RESULT) == RESULT_SHA256, "result hash changed")
    prerequisites = result["prerequisites"]
    _require(
        _sha256(PRIOR_RESULT) == PRIOR_RESULT_SHA256
        and prerequisites["smooth_galerkin_shell_response_gate_sha256"]
        == PRIOR_RESULT_SHA256,
        "smooth Galerkin prerequisite changed",
    )
    _require(
        result.get("all_positive_checks_pass") is True,
        "scale-uniform low-output gate did not pass",
    )
    lattice = result["lattice_Littlewood_Paley_count"]
    dyadic = result["dyadic_tail_summation"]
    endpoint = result["endpoint_channel_saturated_pulse"]
    passage = result["Galerkin_passage"]
    _require(
        lattice["maximum_formula_residual"] == 0
        and lattice["maximum_cubic_ratio"] <= 56.0
        and dyadic["monotone_convergence_for_every_s_gt_1"] is True
        and dyadic["H_minus_1_infinite_series_diverges"] is True
        and endpoint["all_checks_pass"] is True
        and endpoint["H_minus_1_limit"] == "7/(2e)"
        and passage["fixed_mode_coefficients_uniformly_equi_Lipschitz"]
        is True,
        "a scale-uniform quantitative gate changed",
    )
    flags = result["certification_flags"]
    required_true = (
        "uniform_low_output_channel_constant_extracted",
        "three_dimensional_output_multiplicity_paid",
        "H_minus_s_tail_vanishes_for_every_s_gt_1",
        "fixed_mode_Galerkin_compactness_derived",
        "H_minus_one_plus_epsilon_Galerkin_passage_proved",
        "H_minus_1_not_derivable_from_scalar_envelope_alone",
    )
    required_false = (
        "H_minus_1_endpoint_proved",
        "H_minus_1_endpoint_falsified_for_actual_Navier_Stokes",
        "complete_suitable_weak_solution_passage_proved",
        "Navier_Stokes_global_regularity_proved",
    )
    _require(
        all(flags.get(name) is True for name in required_true)
        and all(flags.get(name) is False for name in required_false),
        "scale-uniform scope flags changed",
    )


def main() -> None:
    args = _parse_args()
    result = _load_json(RESULT)
    _validate_result(result)
    bookmark = _load_json(BOOKMARK)
    principal = bookmark.setdefault("principal_results", {})
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark"
        and bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "refusing to update a bookmark outside the standalone NS workspace",
    )
    predecessor = bool(
        bookmark.get("status") == "checkpointed"
        and len(bookmark.get("completed_obligations", [])) == 145
        and len(bookmark.get("primary_artifacts", [])) == 497
        and principal.get(
            "smooth_galerkin_shell_response_gate_audit_v1_sha256"
        )
        == PRIOR_RESULT_SHA256
        and principal.get("smooth_shell_regression_test_count") == 266
        and principal.get("smooth_shell_monolithic_regression_passed")
        is True
    )
    installed = bool(
        bookmark.get("status") in ("parked", "checkpointed")
        and len(bookmark.get("completed_obligations", [])) == 146
        and len(bookmark.get("primary_artifacts", [])) == 502
        and principal.get(
            "scale_uniform_low_output_tail_gate_audit_v1_sha256"
        )
        == RESULT_SHA256
        and principal.get(
            "scale_uniform_low_output_H_minus_s_tail_vanishes"
        )
        is True
    )
    _require(
        predecessor or installed,
        "neither predecessor nor installed scale-uniform checkpoint matches",
    )

    complete = args.validation_mode == "complete"
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "checkpointed" if complete else "parked"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The fixed-channel exact shell-response theorem now extends to "
        "every low-output Fourier channel. Exact lattice multiplicity and "
        "dyadic summation prove that the complete low-output high-high "
        "stress tail vanishes in L2_t H_x^(-s) for every s>1, uniformly "
        "over smooth Galerkin truncations with common Leray bounds. The "
        "fixed-mode bound |Nhat(k)|<=|k|E_* gives equi-Lipschitz Fourier "
        "coefficients, so the stress series passes through Galerkin limits "
        "in H^(-1-epsilon). At s=1, an exact channel-saturated parabolic "
        "pulse has unit H^(-3) forcing envelope but squared LP H^(-1) "
        "response tending to 7/(2e). Thus the scalar envelope alone cannot "
        "close the endpoint. This pulse is not a Navier-Stokes solution or "
        "counterexample; complete suitable-weak passage and regularity "
        "remain open."
    )
    if complete:
        bookmark["validated_checkpoint"] += (
            f" All {args.regression_test_count} discovered tests pass in "
            f"{args.regression_test_seconds:.3f} seconds."
        )

    principal.update(
        {
            "scale_uniform_low_output_channel_constant_extracted": True,
            "scale_uniform_low_output_multiplicity_paid": True,
            "scale_uniform_low_output_H_minus_s_tail_vanishes": True,
            "scale_uniform_low_output_threshold_s_strictly_greater_than": 1.0,
            "scale_uniform_fixed_mode_compactness_derived": True,
            "scale_uniform_H_minus_one_plus_epsilon_Galerkin_passage": True,
            "scale_uniform_H_minus_1_endpoint_proved": False,
            "scale_uniform_H_minus_1_actual_NS_counterexample_proved": False,
            "scale_uniform_H_minus_1_envelope_only_route_closed": True,
            "scale_uniform_complete_suitable_weak_passage_proved": False,
            "scale_uniform_Navier_Stokes_regularity_proved": False,
            "scale_uniform_endpoint_pulse_limit": "7/(2e)",
            "scale_uniform_targeted_test_count": args.targeted_test_count,
            "scale_uniform_dependency_test_count": args.dependency_test_count,
            "scale_uniform_dependency_test_runtime_seconds": (
                args.dependency_test_seconds
            ),
            "scale_uniform_discovered_test_count": (
                args.discovered_test_count
            ),
            "scale_uniform_regression_test_count": (
                args.regression_test_count
            ),
            "scale_uniform_regression_test_runtime_seconds": (
                args.regression_test_seconds
            ),
            "scale_uniform_monolithic_regression_passed": complete,
            "scale_uniform_resource_mode": args.resource_mode,
            "scale_uniform_worker_count": args.worker_count,
            "scale_uniform_cpu_baseline_average_percent": (
                args.baseline_average
            ),
            "scale_uniform_cpu_baseline_peak_percent": args.baseline_peak,
            "scale_uniform_result_status": result["status"],
        }
    )
    for artifact in ARTIFACTS:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Closed the scale-uniform low-output stress-tail gate: paid "
            "the complete three-dimensional output multiplicity, proved "
            "L2_t H^(-1-epsilon) tail compactness and fixed-mode Galerkin "
            "passage, and isolated an exact envelope-level H^(-1) endpoint "
            "obstruction without claiming an actual Navier-Stokes "
            "counterexample."
        ),
    )
    next_obligation = (
        "The next theorem obligation is the dense-output realization gate. "
        "Extend the certified dense HHH packet from one low tensor channel "
        "to a positive-volume low-output block and determine whether its "
        "directed H^(5/2) positivity survives uniformly. If it fails, "
        "extract the output-space cancellation and retest the H^(-1) "
        "endpoint. Only after settling that endpoint should the result be "
        "coupled to every cubic local-energy defect. Suitable-weak closure, "
        "exceptional-set removal, and global regularity remain open."
    )
    if complete:
        bookmark["unfinished_obligation"] = next_obligation
        bookmark["resume_command"] = "not_applicable_no_parked_compute"
    else:
        bookmark["unfinished_obligation"] = (
            "Operational validation remains parked: the one-shot "
            f"{args.discovered_test_count}-test suite must pass in an "
            "admissible resource window. "
            + next_obligation
        )
        bookmark["resume_command"] = (
            "python -m unittest discover -s work/ns_collision/tests "
            "-p \"test_*.py\" -q"
        )
    bookmark["next_action"] = (
        "Introduce a normalized low-output variable w=q/M in the dense "
        "three-cluster packet. Count offset triples with n1+n2+n3=q over "
        "an interior output cube, then extend the directed interval "
        "positivity certificate from w=0 to |w|<=delta. In parallel, test "
        "the complete symbol numerically over increasing M before any "
        "claim of an endpoint obstruction for actual Navier-Stokes."
    )

    primary_artifacts = bookmark.setdefault("primary_artifacts", [])
    for artifact in ARTIFACTS:
        _append_once(primary_artifacts, artifact)
    _require(len(completed) == 146, "unexpected completed count")
    _require(len(primary_artifacts) == 502, "unexpected artifact count")
    _atomic_json(BOOKMARK, bookmark)
    print(
        json.dumps(
            {
                "bookmark": BOOKMARK.relative_to(ROOT).as_posix(),
                "bookmark_sha256": _sha256(BOOKMARK),
                "status": bookmark["status"],
                "completed_obligation_count": len(completed),
                "primary_artifact_count": len(primary_artifacts),
                "result_sha256": _sha256(RESULT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
