"""Install the validated componentwise 64,064-pivot checkpoint."""

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
    "neutral_strip_h006_hypercircle_standalone_residual63901_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual63982_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64023_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64033_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64038_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64039_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64040_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64043_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64039_p100_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64040_p100_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual_"
    "precision_crosscheck64039_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual_"
    "precision_crosscheck64040_v1.json",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_closure_boundary64039_audit.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_closure_boundary64039_audit.py",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_closure_boundary64039_audit_v1.json",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_componentwise_residual.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_componentwise_residual.py",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64040_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64040_p100_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_componentwise_residual_"
    "precision_crosscheck64040_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64064_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64064_p100_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_componentwise_residual_"
    "precision_crosscheck64064_v1.json",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_"
    "componentwise_state_region64064_audit.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_componentwise_state_region64064_audit.py",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_"
    "componentwise_state_region64064_audit_v1.json",
    "work/ns_collision/notes/"
    "neutral_strip_weighted_hypercircle_"
    "componentwise_recovery64064.md",
    "work/ns_collision/notes/"
    "neutral_strip_weighted_hypercircle_"
    "state_region64064_obstruction.md",
    "work/ns_collision/scripts/"
    "update_componentwise_state_region64064_bookmark.py",
    "work/ns_collision/README.md",
)

CLOSURE_AUDIT = ARTIFACTS[14]
BOUNDARY_LOWER = ARTIFACTS[17]
BOUNDARY_HIGHER = ARTIFACTS[18]
BOUNDARY_CROSSCHECK = ARTIFACTS[19]
ENDPOINT_LOWER = ARTIFACTS[20]
ENDPOINT_HIGHER = ARTIFACTS[21]
ENDPOINT_CROSSCHECK = ARTIFACTS[22]
COMPONENTWISE_AUDIT = ARTIFACTS[25]


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
    _require(
        args.targeted_test_count > 0,
        "targeted test count must be positive",
    )
    _require(args.full_test_seconds >= 0.0, "full runtime must be nonnegative")
    _require(
        args.targeted_test_seconds >= 0.0,
        "targeted runtime must be nonnegative",
    )
    for artifact in ARTIFACTS:
        _require(_resolve(artifact).is_file(), f"missing artifact: {artifact}")

    closure_audit = _load_json(CLOSURE_AUDIT)
    boundary_lower = _load_json(BOUNDARY_LOWER)
    boundary_higher = _load_json(BOUNDARY_HIGHER)
    boundary_crosscheck = _load_json(BOUNDARY_CROSSCHECK)
    endpoint_lower = _load_json(ENDPOINT_LOWER)
    endpoint_higher = _load_json(ENDPOINT_HIGHER)
    endpoint_crosscheck = _load_json(ENDPOINT_CROSSCHECK)
    componentwise_audit = _load_json(COMPONENTWISE_AUDIT)
    _require(
        closure_audit.get("all_current_stage_checks_pass") is True
        and closure_audit["boundary"][
            "last_certified_prefix_pivots"
        ]
        == 64039
        and closure_audit["boundary"][
            "first_nonclosing_prefix_pivots"
        ]
        == 64040,
        "separated closure-loss boundary audit does not pass",
    )
    for label, result in (
        ("boundary precision 60", boundary_lower),
        ("boundary precision 100", boundary_higher),
        ("endpoint precision 60", endpoint_lower),
        ("endpoint precision 100", endpoint_higher),
    ):
        _require(
            result.get("status") == "standalone_prefix_inertia_certified"
            and result.get("all_current_stage_checks_pass") is True
            and result["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is True,
            f"{label} componentwise certificate does not close",
        )
        _require(
            result["directed_LDL_dependency"]["required"] is False
            and result["directed_LDL_dependency"]["audit_loaded"] is False,
            f"{label} has a directed-audit dependency",
        )
        _require(
            result["checks"][
                "separated_reference_and_residual_reproduced"
            ]
            is True
            and result["checks"]["separated_bound_reproduced"] is True,
            f"{label} does not reproduce the separated reference",
        )
    _require(
        boundary_crosscheck.get("all_checks_pass") is True
        and endpoint_crosscheck.get("all_checks_pass") is True,
        "componentwise precision crosscheck does not pass",
    )
    _require(
        componentwise_audit.get("all_current_stage_checks_pass") is True
        and componentwise_audit.get("status")
        == "componentwise_state_region_certified",
        "componentwise state-region audit does not pass",
    )
    _require(
        componentwise_audit["certification_flags"][
            "standalone_componentwise_64064_inertia_certified"
        ]
        is True
        and componentwise_audit["certification_flags"][
            "full_123816_pivot_inertia_certified"
        ]
        is False
        and componentwise_audit["certification_flags"][
            "navier_stokes_regularity_certified"
        ]
        is False,
        "componentwise audit has an invalid certification scope",
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
    _require(
        principal.get(
            "reversible_weighted_hypercircle_"
            "standalone64064_diagnostic_complete"
        )
        is True
        and principal.get(
            "reversible_weighted_hypercircle_"
            "standalone64064_certified"
        )
        is False,
        "bookmark is not at the separated 64,064 obstruction checkpoint",
    )

    boundary = componentwise_audit["boundary_recovery"]
    endpoint = componentwise_audit["state_region_recovery"]
    next_boundary = componentwise_audit["next_boundary"]
    closure_boundary = closure_audit["boundary"]
    added_pivot = closure_audit["added_pivot_profile"]
    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The original separated standalone residual majorant was localized "
        "by hash-bound bisection: prefix 64039 closes, while appending "
        "edge-metric pivot 64039 (original index 21) makes prefix 64040 fail. "
        "The new pivot has a large positive reference diagonal and a "
        "-3623.21 coupling to the earlier delicate edge pivot 63629, so the "
        "failure is global norm decoupling rather than a near-zero new "
        "pivot. A directed componentwise theorem now propagates the same "
        "residual magnitude as max(Q R Q^T 1), retains the exact frozen "
        "source/order/scale/binary-reference hashes, and reproduces every "
        "old separated quantity. Precision-60/100 runs and independent "
        "nesting checks certify prefix 64040 with ratio 0.021056 and prefix "
        "64064 with ratio 0.270366. The full state-region checkpoint through "
        "pivot 64063 is therefore independently certified as 32500 negative "
        "and 31564 positive. Full 123816-pivot inertia, weighted Ritz "
        "closure, continuum transfer, and every Navier-Stokes regularity "
        "claim remain false."
    )

    principal.update(
        {
            "reversible_weighted_hypercircle_separated_last_certified_prefix": (
                closure_boundary["last_certified_prefix_pivots"]
            ),
            "reversible_weighted_hypercircle_separated_first_nonclosing_prefix": (
                closure_boundary["first_nonclosing_prefix_pivots"]
            ),
            "reversible_weighted_hypercircle_separated_first_nonclosing_added_pivot": (
                closure_boundary["first_nonclosing_added_pivot"]
            ),
            "reversible_weighted_hypercircle_separated_closure_ratio_jump": (
                closure_boundary["ratio_jump_factor"]
            ),
            "reversible_weighted_hypercircle_closure_added_pivot_profile": (
                added_pivot
            ),
            "reversible_weighted_hypercircle_componentwise_residual_mode_implemented": True,
            "reversible_weighted_hypercircle_componentwise64040_certified": True,
            "reversible_weighted_hypercircle_componentwise64040_signs": (
                boundary["reference_signs"]
            ),
            "reversible_weighted_hypercircle_componentwise64040_ratio": (
                boundary[
                    "bound_to_minimum_diagonal_ratio_upper_decimal"
                ]
            ),
            "reversible_weighted_hypercircle_componentwise64040_safety_factor": (
                boundary["safety_factor_lower_decimal"]
            ),
            "reversible_weighted_hypercircle_componentwise64040_improvement": (
                boundary[
                    "improvement_over_separated_bound_lower_decimal"
                ]
            ),
            "reversible_weighted_hypercircle_componentwise64064_certified": True,
            "reversible_weighted_hypercircle_componentwise64064_pivot_count": (
                endpoint["maximum_pivots"]
            ),
            "reversible_weighted_hypercircle_componentwise64064_signs": (
                endpoint["reference_signs"]
            ),
            "reversible_weighted_hypercircle_componentwise64064_minimum_diagonal": (
                endpoint[
                    "minimum_absolute_reference_diagonal_decimal"
                ]
            ),
            "reversible_weighted_hypercircle_componentwise64064_bound": (
                endpoint["componentwise_bound_upper_decimal"]
            ),
            "reversible_weighted_hypercircle_componentwise64064_ratio": (
                endpoint[
                    "bound_to_minimum_diagonal_ratio_upper_decimal"
                ]
            ),
            "reversible_weighted_hypercircle_componentwise64064_safety_factor": (
                endpoint["safety_factor_lower_decimal"]
            ),
            "reversible_weighted_hypercircle_componentwise64064_improvement": (
                endpoint[
                    "improvement_over_separated_bound_lower_decimal"
                ]
            ),
            "reversible_weighted_hypercircle_componentwise_growth64040_64064": (
                endpoint["componentwise_bound_growth_from_64040"]
            ),
            "reversible_weighted_hypercircle_next_bounded_pivot_target": (
                next_boundary["recommended_bounded_pivot_count"]
            ),
            "reversible_weighted_hypercircle_next_symbolic_transition_pivot": (
                next_boundary["next_symbolic_transition_pivot"]
            ),
            "reversible_weighted_hypercircle_larger_prefix_admitted": True,
            "reversible_weighted_hypercircle_full_inertia_certified": False,
            "validated_ns_collision_test_count": args.full_test_count,
            "validated_ns_collision_test_runtime_seconds": (
                args.full_test_seconds
            ),
            "validated_new_targeted_test_count": args.targeted_test_count,
            "validated_new_targeted_test_runtime_seconds": (
                args.targeted_test_seconds
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
            "Localized the first failure of the separated residual majorant "
            "to appending edge pivot 64039: prefix 64039 closes at both "
            "precisions while prefix 64040 fails, with nested factors and "
            "a monotone 7.93-fold one-pivot ratio jump."
        ),
    )
    _append_once(
        completed,
        (
            "Replaced the separated global norm product by the rigorous "
            "componentwise row-sum bound max(Q R Q^T 1). Independent "
            "precision-60/100 checks certify prefixes 64040 and 64064, "
            "recovering the full seven-transition state region through "
            "pivot 64063 as 32500 negative and 31564 positive."
        ),
    )

    bookmark["unfinished_obligation"] = (
        "The interval pencil is now independently certified only through "
        "pivot 64063 by the componentwise standalone residual theorem. Its "
        "ratio grows by factor 12.84 over the final 24-pivot extension and "
        "leaves safety factor 3.70, so only a 64128-pivot local growth test "
        "is admitted; the distant next symbolic transition at 76921 is not. "
        "Full inertia 61908/61908/0, strict solution-operator and weighted "
        "Ritz thresholds, source/conormal/domain transfer, and Navier-Stokes "
        "closure remain open and fail-closed."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "After a fresh daytime CPU gate, test exactly 64128 pivots and no "
        "farther. First produce the separated hash-bound residual reference "
        "at precision 60, then run the componentwise certificate against it. "
        "Only if the componentwise ratio remains below one should both "
        "calculations be replayed at precision 100 and crosschecked for "
        "provenance and upper-bound nesting. Do not jump to pivot 76921, "
        "directed LDL, the full pencil, or any continuum stage."
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
