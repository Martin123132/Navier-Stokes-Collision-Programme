"""Install the validated componentwise 64,128-pivot checkpoint."""

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
    "neutral_strip_h006_hypercircle_standalone_residual64128_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64128_p100_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64128_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64128_p100_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_componentwise_residual_"
    "precision_crosscheck64128_v1.json",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_"
    "componentwise_growth64128_audit.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_componentwise_growth64128_audit.py",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_"
    "componentwise_growth64128_audit_v1.json",
    "work/ns_collision/notes/"
    "neutral_strip_weighted_hypercircle_componentwise_growth64128.md",
    "work/ns_collision/scripts/"
    "update_componentwise_growth64128_bookmark.py",
    "work/ns_collision/README.md",
)
SEPARATED_LOWER = ARTIFACTS[0]
SEPARATED_HIGHER = ARTIFACTS[1]
COMPONENTWISE_LOWER = ARTIFACTS[2]
COMPONENTWISE_HIGHER = ARTIFACTS[3]
CROSSCHECK = ARTIFACTS[4]
AUDIT = ARTIFACTS[7]


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

    separated_lower = _load_json(SEPARATED_LOWER)
    separated_higher = _load_json(SEPARATED_HIGHER)
    componentwise_lower = _load_json(COMPONENTWISE_LOWER)
    componentwise_higher = _load_json(COMPONENTWISE_HIGHER)
    crosscheck = _load_json(CROSSCHECK)
    audit = _load_json(AUDIT)
    for label, result in (
        ("separated precision 60", separated_lower),
        ("separated precision 100", separated_higher),
    ):
        _require(
            result.get("all_current_stage_checks_pass") is True
            and result.get("status") == "standalone_route_does_not_close"
            and result["certificate"]["dimension"] == 64128,
            f"{label} is not the expected valid nonclosing reference",
        )
    for label, result in (
        ("componentwise precision 60", componentwise_lower),
        ("componentwise precision 100", componentwise_higher),
    ):
        _require(
            result.get("all_current_stage_checks_pass") is True
            and result.get("status")
            == "standalone_prefix_inertia_certified"
            and result["certificate"]["dimension"] == 64128
            and result["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is True,
            f"{label} does not certify the target prefix",
        )
        _require(
            result["directed_LDL_dependency"]["required"] is False
            and result["directed_LDL_dependency"]["audit_loaded"] is False,
            f"{label} has a directed-audit dependency",
        )
    _require(
        crosscheck.get("all_checks_pass") is True,
        "componentwise precision crosscheck does not pass",
    )
    _require(
        audit.get("all_current_stage_checks_pass") is True
        and audit.get("status")
        == "componentwise_prefix64128_certified"
        and audit["certification_flags"][
            "standalone_componentwise_64128_inertia_certified"
        ]
        is True
        and audit["certification_flags"][
            "standalone_componentwise_64256_inertia_certified"
        ]
        is False
        and audit["certification_flags"][
            "full_123816_pivot_inertia_certified"
        ]
        is False
        and audit["certification_flags"][
            "navier_stokes_regularity_certified"
        ]
        is False,
        "64,128 growth audit has an invalid scope or status",
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
            "componentwise64064_certified"
        )
        is True,
        "bookmark is not at the componentwise 64,064 checkpoint",
    )
    deferred = bookmark.get(
        "hypercircle_componentwise64128_deferred_calculation"
    )
    _require(
        isinstance(deferred, dict),
        "the recorded 64,128 CPU deferral is missing",
    )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    extension = audit["extension"]
    certificate = audit["certificate"]
    next_boundary = audit["next_boundary"]
    bookmark["checkpointed_at"] = now
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The standalone componentwise residual theorem now certifies the "
        "frozen interval pencil through pivot 64127. Precision-60 and "
        "precision-100 runs certify the 64128-pivot prefix as 32564 "
        "negative and 31564 positive, with ratio 0.270366080339945 and "
        "safety factor 3.69868882495. Independent reconstruction proves "
        "that the ordered source prefixes nest bitwise and the 64064 "
        "binary reference factor is the exact leading block. The 64 added "
        "pivots are all edge-metric and negative, with smallest absolute "
        "diagonal 3.02330695021; every precision-100 control metric is "
        "exactly flat. Full 123816-pivot inertia, weighted Ritz closure, "
        "continuum transfer, and every Navier-Stokes regularity claim "
        "remain false."
    )

    principal.update(
        {
            "reversible_weighted_hypercircle_componentwise64128_certified": True,
            "reversible_weighted_hypercircle_componentwise64128_pivot_count": (
                extension["target_maximum_pivots"]
            ),
            "reversible_weighted_hypercircle_componentwise64128_last_certified_pivot": (
                extension["last_certified_pivot"]
            ),
            "reversible_weighted_hypercircle_componentwise64128_signs": (
                extension["reference_signs"]
            ),
            "reversible_weighted_hypercircle_componentwise64128_added_signs": (
                extension["added_reference_signs"]
            ),
            "reversible_weighted_hypercircle_componentwise64128_block_profile": (
                extension["incremental_block_profile"]
            ),
            "reversible_weighted_hypercircle_componentwise64128_minimum_diagonal": (
                certificate[
                    "minimum_absolute_reference_diagonal_decimal"
                ]
            ),
            "reversible_weighted_hypercircle_componentwise64128_bound": (
                certificate["componentwise_bound_upper_decimal"]
            ),
            "reversible_weighted_hypercircle_componentwise64128_ratio": (
                certificate[
                    "bound_to_minimum_diagonal_ratio_upper_decimal"
                ]
            ),
            "reversible_weighted_hypercircle_componentwise64128_safety_factor": (
                certificate["safety_factor_lower_decimal"]
            ),
            "reversible_weighted_hypercircle_componentwise64128_improvement": (
                certificate[
                    "improvement_over_separated_bound_lower_decimal"
                ]
            ),
            "reversible_weighted_hypercircle_componentwise_growth64064_64128": (
                certificate["componentwise_bound_growth_from_64064"]
            ),
            "reversible_weighted_hypercircle_next_bounded_pivot_target": (
                next_boundary["recommended_bounded_pivot_count"]
            ),
            "reversible_weighted_hypercircle_next_symbolic_transition_pivot": (
                next_boundary["next_symbolic_transition_pivot"]
            ),
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
            "Certified the 64128-pivot standalone componentwise interval "
            "family at precisions 60 and 100 as 32564 negative and 31564 "
            "positive. Exact source/factor nesting and a formal precision "
            "crosscheck show a one-fold componentwise-bound plateau from "
            "64064; all 64 added edge-metric pivots are negative and "
            "comfortably separated from zero."
        ),
    )

    deferred["resolved_at"] = now
    deferred["resolution"] = (
        "The user explicitly made CPU capacity available for one "
        "below-normal worker. The separated and componentwise precision-60 "
        "runs completed, the ratio remained below one, and precision-100 "
        "replays plus the independent growth audit passed. The 64128 "
        "componentwise certificate is installed; unrelated jobs were not "
        "stopped or reprioritized."
    )
    bookmark["unfinished_obligation"] = (
        "The interval pencil is independently certified only through pivot "
        "64127 by the componentwise standalone residual theorem. The exact "
        "flat 64064-to-64128 extension admits only a 64256-pivot local "
        "growth test; the distant next symbolic transition at 76921 is "
        "not admitted. Full inertia 61908/61908/0, strict solution-operator "
        "and weighted Ritz thresholds, source/conormal/domain transfer, "
        "and Navier-Stokes closure remain open and fail-closed."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "After a fresh resource decision, test exactly 64256 pivots and no "
        "farther. First produce the separated hash-bound precision-60 "
        "reference, then run the componentwise precision-60 certificate. "
        "Only if its ratio remains strictly below one should both runs be "
        "replayed at precision 100 and nesting-checked. Do not jump to "
        "pivot 76921, directed LDL, the full pencil, or any continuum stage."
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
