"""Install the validated standalone 63,680 state-entry checkpoint."""

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
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_congruence_residual_pilot.py",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_congruence_residual_precision_crosscheck.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_congruence_residual_pilot.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_congruence_residual_precision_crosscheck.py",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_standalone_residual.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_standalone_residual.py",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_standalone_residual_regression.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_standalone_residual_regression.py",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual2304_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual32064_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual33280_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual_regression_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual63680_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual63680_p100_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual_"
    "precision_crosscheck63680_v1.json",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_state_entry63680_audit.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_state_entry63680_audit.py",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_state_entry63680_audit_v1.json",
    "work/ns_collision/notes/"
    "neutral_strip_weighted_hypercircle_standalone_state_entry63680.md",
    "work/ns_collision/scripts/update_state_entry63680_bookmark.py",
    "work/ns_collision/README.md",
)

REGRESSION_RESULT = ARTIFACTS[11]
LOWER_RESULT = ARTIFACTS[12]
HIGHER_RESULT = ARTIFACTS[13]
CROSSCHECK_RESULT = ARTIFACTS[14]
STATE_AUDIT_RESULT = ARTIFACTS[17]


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
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")

    regression = _load_json(REGRESSION_RESULT)
    lower = _load_json(LOWER_RESULT)
    higher = _load_json(HIGHER_RESULT)
    crosscheck = _load_json(CROSSCHECK_RESULT)
    state_audit = _load_json(STATE_AUDIT_RESULT)
    _require(
        regression.get("all_checks_pass") is True
        and regression.get("state_entry_pilot_admitted") is True,
        "standalone historical regression does not pass",
    )
    for label, result in (("precision 60", lower), ("precision 100", higher)):
        _require(
            result.get("status") == "standalone_prefix_inertia_certified"
            and result.get("all_current_stage_checks_pass") is True
            and result["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is True,
            f"{label} standalone certificate does not close",
        )
        _require(
            result["directed_LDL_dependency"]["required"] is False
            and result["directed_LDL_dependency"]["audit_loaded"] is False,
            f"{label} result has a directed-audit dependency",
        )
    _require(
        crosscheck.get("all_checks_pass") is True,
        "standalone precision crosscheck does not pass",
    )
    _require(
        state_audit.get("all_current_stage_checks_pass") is True
        and state_audit["certification_flags"][
            "first_state_entry_certified"
        ]
        is True,
        "state-entry structural audit does not pass",
    )
    _require(
        state_audit["certification_flags"][
            "full_123816_pivot_inertia_certified"
        ]
        is False
        and state_audit["certification_flags"][
            "navier_stokes_regularity_certified"
        ]
        is False,
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
    principal = bookmark.setdefault("principal_results", {})
    predecessor = principal.get(
        "reversible_weighted_hypercircle_standalone63680_pivot_count",
        principal.get(
            "reversible_weighted_hypercircle_transition33280_pivot_count"
        ),
    )
    _require(
        int(predecessor) in (33280, 63680),
        "bookmark is not at the expected predecessor/current boundary",
    )

    certificate = state_audit["certificate_summary"]
    state = state_audit["state_entry_profile"]
    risk = state_audit["risk_change_from_33280"]
    next_boundary = state_audit["next_boundary"]
    full_blocks = state_audit["full_prefix_block_profile"]
    incremental = state_audit["incremental_33280_63679_block_profile"]

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The frozen h=0.06 interval pencil and prior directed-LDL boundary "
        "remain validated. A new standalone congruence-residual runner now "
        "binds directly to source, ordered-prefix, scale, permutation, and "
        "binary reference-factor hashes and loads no directed audit. It "
        "exactly reproduces the historical 2304, 32064, and 33280 residual "
        "certificates. At precisions 60 and 100 it independently certifies "
        "the first 63680 reference diagonals as 32392 negative and 31288 "
        "positive, with bound/minimum-diagonal ratio below 0.000246175 and "
        "precision-100 nesting. Eleven state pivots enter from 63644; all "
        "are positive and their minimum absolute diagonal exceeds 0.6366. "
        "The global minimum 0.0008150796928894088 instead occurs at "
        "edge-metric pivot 63629. Full 123816-pivot inertia, the weighted "
        "Ritz constant, continuum capture, and every broader Navier-Stokes "
        "claim remain false. This remains far short of a regularity or "
        "Clay-prize proof."
    )

    principal.update(
        {
            "reversible_weighted_hypercircle_standalone_residual_mode_implemented": True,
            "reversible_weighted_hypercircle_standalone_residual_regression_passes": True,
            "reversible_weighted_hypercircle_standalone_residual_regression_prefixes": [
                2304,
                32064,
                33280,
            ],
            "reversible_weighted_hypercircle_standalone63680_certified": True,
            "reversible_weighted_hypercircle_standalone63680_pivot_count": (
                certificate["maximum_pivots"]
            ),
            "reversible_weighted_hypercircle_standalone63680_signs": (
                certificate["reference_signs"]
            ),
            "reversible_weighted_hypercircle_standalone63680_minimum_diagonal": (
                certificate["minimum_absolute_reference_diagonal_decimal"]
            ),
            "reversible_weighted_hypercircle_standalone63680_minimum_diagonal_index": (
                certificate["minimum_absolute_reference_diagonal_index"]
            ),
            "reversible_weighted_hypercircle_standalone63680_minimum_diagonal_block": (
                certificate["minimum_absolute_reference_diagonal_block"]
            ),
            "reversible_weighted_hypercircle_standalone63680_residual_ratio": (
                certificate[
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ]
            ),
            "reversible_weighted_hypercircle_standalone63680_safety_factor": (
                certificate["certified_safety_factor_lower"]
            ),
            "reversible_weighted_hypercircle_standalone63680_p100_nesting_crosscheck": True,
            "reversible_weighted_hypercircle_first_state_pivot": (
                state["first_state_pivot"]
            ),
            "reversible_weighted_hypercircle_state_pivots_certified": (
                state["state_pivot_count"]
            ),
            "reversible_weighted_hypercircle_state_pivot_signs": (
                state["state_signs"]
            ),
            "reversible_weighted_hypercircle_state_minimum_diagonal": (
                state["minimum_absolute_state_diagonal_decimal"]
            ),
            "reversible_weighted_hypercircle_full_prefix_block_profile63680": (
                full_blocks
            ),
            "reversible_weighted_hypercircle_incremental_block_profile33280_63680": (
                incremental
            ),
            "reversible_weighted_hypercircle_inverse_one_growth33280_63680": (
                risk["inverse_one_norm_growth_factor"]
            ),
            "reversible_weighted_hypercircle_inverse_infinity_growth33280_63680": (
                risk["inverse_infinity_norm_growth_factor"]
            ),
            "reversible_weighted_hypercircle_residual_ratio_growth33280_63680": (
                risk["bound_to_diagonal_ratio_growth_factor"]
            ),
            "reversible_weighted_hypercircle_next_symbolic_transition_cluster": (
                next_boundary["next_transition_cluster_pivots"]
            ),
            "reversible_weighted_hypercircle_next_bounded_pivot_target": (
                next_boundary["recommended_bounded_pivot_count"]
            ),
            "reversible_weighted_hypercircle_full_inertia_certified": False,
            "validated_ns_collision_test_count": args.full_test_count,
            "validated_ns_collision_test_runtime_seconds": args.full_test_seconds,
            "validated_new_targeted_test_count": args.targeted_test_count,
            "validated_new_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
            ),
            "validated_standalone_residual_test_count": (
                args.targeted_test_count
            ),
        }
    )
    for artifact in ARTIFACTS[8:18]:
        key = Path(artifact).stem.replace("-", "_")
        principal[f"{key}_sha256"] = _sha256(artifact)

    completed = bookmark.setdefault("completed_obligations", [])
    _append_once(
        completed,
        (
            "Removed the residual wrapper's logical dependence on directed "
            "LDL by binding source files, ordered interval prefixes, scale, "
            "permutation, and binary reference factors directly; exact "
            "standalone replays at 2304, 32064, and 33280 reproduce every "
            "historical certificate value."
        ),
    )
    _append_once(
        completed,
        (
            "Certified the standalone 63680-pivot first-state-entry prefix "
            "at precisions 60 and 100 as 32392 negative and 31288 positive "
            "with safety factor above 4062. Eleven state pivots are all "
            "positive; the new minimum is instead edge-metric pivot 63629, "
            "and inverse-majorant growth is recorded as the dominant risk."
        ),
    )

    bookmark["unfinished_obligation"] = (
        "The finite pencil is independently certified only through pivot "
        "63679. First state entry is benign at this boundary, but the "
        "bound/minimum-diagonal ratio has grown by about 8373 times from "
        "33280 because the inverse one/infinity majorants grew by factors "
        "38.92 and 45.17. The next compact transition cluster ends at pivot "
        "64056, so only a bounded 64064 standalone residual replay is "
        "admitted. Full inertia 61908/61908/0, strict solution-operator and "
        "weighted Ritz thresholds, source/conormal/domain transfer, and "
        "Navier-Stokes closure remain open and fail-closed."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "After a fresh daytime CPU gate, run the hash-bound standalone "
        "residual certificate through exactly 64064 pivots at precisions 60 "
        "and 100, cross-check all upper bounds and provenance, and audit the "
        "compact state-region transition cluster at 63733, 63735, 63900, "
        "64043, 64044, 64049, and 64056. Do not launch directed LDL, the full "
        "pencil, or any continuum stage."
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
