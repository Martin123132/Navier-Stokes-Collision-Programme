"""Install the validated transition-33280 checkpoint in the NS bookmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BOOKMARK = ROOT / "work/ns_collision/results/session_bookmark.json"

ARTIFACTS = (
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_transition33280_checkpoint_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_transition33280_audit_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_transition33280_p80_checkpoint_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_transition33280_p80_audit_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_directed_ldl_transition33280_"
    "precision_crosscheck_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_congruence_residual_pilot33280_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_congruence_residual_pilot33280_p100_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_congruence_residual_"
    "precision_crosscheck33280_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_transition33280_audit_v1.json",
    "work/ns_collision/notes/"
    "neutral_strip_weighted_hypercircle_transition33280_audit.md",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_transition33280_audit.py",
    "work/ns_collision/scripts/update_transition33280_bookmark.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_transition33280_audit.py",
    "work/ns_collision/README.md",
)

TRANSITION_RESULT = ARTIFACTS[8]
DIRECTED_CROSSCHECK = ARTIFACTS[4]
RESIDUAL_CROSSCHECK = ARTIFACTS[7]


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    return value


def _sha256(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
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
    parser.add_argument("--full-test-count", type=int, required=True)
    parser.add_argument("--full-test-seconds", type=float, required=True)
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--targeted-test-seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require(args.full_test_count > 0, "full test count must be positive")
    _require(args.targeted_test_count > 0, "targeted test count must be positive")
    _require(args.full_test_seconds >= 0.0, "full runtime must be nonnegative")
    _require(
        args.targeted_test_seconds >= 0.0,
        "targeted runtime must be nonnegative",
    )
    for artifact in ARTIFACTS:
        _require((ROOT / artifact).is_file(), f"missing artifact: {artifact}")

    transition = _load_json(TRANSITION_RESULT)
    directed_crosscheck = _load_json(DIRECTED_CROSSCHECK)
    residual_crosscheck = _load_json(RESIDUAL_CROSSCHECK)
    _require(
        transition.get("all_current_stage_checks_pass") is True,
        "transition audit does not pass",
    )
    _require(
        directed_crosscheck.get("all_checks_pass") is True,
        "directed precision crosscheck does not pass",
    )
    _require(
        residual_crosscheck.get("all_checks_pass") is True,
        "residual precision crosscheck does not pass",
    )
    flags = transition["certification_flags"]
    _require(
        flags["bounded_33280_inertia_certified_by_two_routes"] is True,
        "two-route bounded certificate is absent",
    )
    _require(
        flags["full_123816_pivot_inertia_certified"] is False
        and flags["continuum_spectrum_below_60_captured"] is False
        and flags["navier_stokes_regularity_certified"] is False,
        "a prohibited broad claim is true",
    )

    bookmark = _load_json(BOOKMARK)
    _require(
        bookmark.get("kind") == "navier_stokes_collision_session_bookmark",
        "refusing to update a non-NS bookmark",
    )
    _require(
        bookmark.get("workspace_root") == "."
        and bookmark.get("research_root") == "work/ns_collision",
        "bookmark workspace boundary is not standalone",
    )
    prior_count = bookmark.get("principal_results", {}).get(
        "reversible_weighted_hypercircle_transition33280_pivot_count",
        bookmark.get("principal_results", {}).get(
            "reversible_weighted_hypercircle_transition32064_pivot_count"
        ),
    )
    _require(
        int(prior_count) in (32064, 33280),
        "bookmark is not at the expected predecessor/current boundary",
    )

    current = transition["current_certified_prefix"]
    segment = transition["new_segment_32064_33279"]
    delicate = transition["delicate_pivot"]
    residual = transition["independent_residual_certificate"]
    risk = transition["risk_change_from_32064"]
    next_boundary = transition["next_boundary"]

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The stored finite-chain screen, h=0.06 weighted-hypercircle "
        "assembly, and frozen Ruiz/MMD_AT_PLUS_A preparation remain "
        "validated. Two interval-propagation routes now certify pivots "
        "0..33279: directed LDL at precisions 50 and 80 and an independent "
        "congruence-residual proof at precisions 60 and 100 both give 31971 "
        "negative and 1309 positive pivots. All 33280 pivot intervals and "
        "131628 lower intervals pass precision nesting. The new minimum "
        "margin is 0.003304358585502960725532827806 at exact-zero-input "
        "pivot 32849; its radius/margin is below 5.92e-10, while residual "
        "bound/minimum-diagonal is below 2.94e-8. The predicted "
        "six-descendant transition at 33224 is certified and remains well "
        "separated from zero. Full 123816-pivot inertia, the weighted Ritz "
        "constant, continuum capture, and every broader Navier-Stokes claim "
        "remain false. This remains far short of a regularity or Clay-prize "
        "proof."
    )

    principal = bookmark.setdefault("principal_results", {})
    principal.update(
        {
            "reversible_weighted_hypercircle_transition33280_certified": True,
            "reversible_weighted_hypercircle_transition33280_pivot_count": int(
                current["completed_pivots"]
            ),
            "reversible_weighted_hypercircle_transition33280_signs": {
                "negative": int(current["negative"]),
                "positive": int(current["positive"]),
                "zero": int(current["zero"]),
            },
            "reversible_weighted_hypercircle_transition33280_minimum_margin": (
                current["minimum_margin_decimal"]
            ),
            "reversible_weighted_hypercircle_transition33280_minimum_margin_index": (
                current["minimum_margin_index"]
            ),
            "reversible_weighted_hypercircle_transition33280_symbolic_lower_entries": (
                current["symbolic_lower_entries"]
            ),
            "reversible_weighted_hypercircle_transition33280_maximum_relative_radius": (
                current["maximum_radius_to_margin_upper_decimal"]
            ),
            "reversible_weighted_hypercircle_transition33280_p80_nesting_crosscheck": True,
            "reversible_weighted_hypercircle_transition33280_new_pivot_count": (
                segment["pivot_count"]
            ),
            "reversible_weighted_hypercircle_transition33280_new_signs": (
                segment["signs"]
            ),
            "reversible_weighted_hypercircle_transition33280_new_block_counts": (
                segment["pivot_block_counts"]
            ),
            "reversible_weighted_hypercircle_delicate_pivot": (
                delicate["index"]
            ),
            "reversible_weighted_hypercircle_delicate_pivot_exact_zero_input": (
                delicate["input_diagonal_is_exact_zero"]
            ),
            "reversible_weighted_hypercircle_delicate_cancellation_charge": (
                delicate["cancellation_charge_upper_decimal"]
            ),
            "reversible_weighted_hypercircle_residual33280_certified": True,
            "reversible_weighted_hypercircle_residual33280_bound_to_diagonal_ratio": (
                residual["precision_60_bound_to_diagonal_ratio_upper_decimal"]
            ),
            "reversible_weighted_hypercircle_residual33280_p100_nesting_crosscheck": True,
            "reversible_weighted_hypercircle_transition33280_margin_reduction_factor": (
                risk["minimum_margin_reduction_factor"]
            ),
            "reversible_weighted_hypercircle_transition33280_cancellation_growth_factor": (
                risk["maximum_cancellation_growth_factor"]
            ),
            "reversible_weighted_hypercircle_next_symbolic_transition": (
                next_boundary["next_unprocessed_symbolic_transition_pivot"]
            ),
            "reversible_weighted_hypercircle_first_state_pivot": (
                next_boundary["first_state_pivot"]
            ),
            "reversible_weighted_hypercircle_next_bounded_pivot_target": (
                next_boundary[
                    "recommended_residual_state_entry_pilot_pivots"
                ]
            ),
            "reversible_weighted_hypercircle_standalone_residual_mode_implemented": False,
            "reversible_weighted_hypercircle_full_inertia_certified": False,
            "validated_ns_collision_test_count": args.full_test_count,
            "validated_ns_collision_test_runtime_seconds": args.full_test_seconds,
            "validated_new_targeted_test_count": args.targeted_test_count,
            "validated_new_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "validated_directed_LDL_test_count": args.targeted_test_count,
        }
    )
    for artifact in ARTIFACTS[:9]:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Certified the separate 33280-pivot threshold-pencil prefix by "
            "directed LDL at precisions 50/80 and congruence residual at "
            "precisions 60/100: 31971 negative and 1309 positive, with all "
            "pivot, lower-interval, and residual upper-bound nesting checks "
            "passing while full inertia remains false."
        ),
    )
    _append_once(
        completed,
        (
            "Isolated pivot 32849 as the first strongly cancellation-"
            "sensitive exact-zero-input pivot, distinguished it from the "
            "benign six-descendant transition at 33224, and recorded that "
            "the proposed 63680 residual-only state pilot first requires a "
            "standalone residual input contract."
        ),
    )

    bookmark["unfinished_obligation"] = (
        "The finite weighted-hypercircle pencil is certified only through "
        "pivot 33279 by two routes. Pivot 32849 reduces the minimum margin "
        "by a factor of 44.32 and increases maximum cancellation by about "
        "260.22 relative to the 32064 checkpoint, although both proofs still "
        "close decisively. The next unprocessed symbolic transition is "
        "62972 and first state entry is 63644. Before running the bounded "
        "63680 pilot, implement and test a standalone congruence-residual "
        "mode that no longer requires a matching directed-LDL audit. Full "
        "inertia 61908/61908/0, strict solution-operator and weighted Ritz "
        "thresholds, positive-time source transfer, conormal transfer, "
        "polygon-circle perturbation, and Navier-Stokes closure all remain "
        "open and fail-closed."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "Implement a standalone, hash-bound congruence-residual input mode "
        "and regression-test it against the certified 2304, 32064, and 33280 "
        "prefixes. After a fresh CPU gate, use that mode for only the 63680 "
        "state-entry pilot at precisions 60 and 100. Do not launch directed "
        "LDL to 63680, the full pencil, or any continuum stage."
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
