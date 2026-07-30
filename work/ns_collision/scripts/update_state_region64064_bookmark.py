"""Install the validated standalone 64,064 obstruction checkpoint."""

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
    "neutral_strip_weighted_hypercircle_standalone_residual.py",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_"
    "congruence_residual_precision_crosscheck.py",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64064_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual64064_p100_v1.json",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_standalone_residual_"
    "precision_crosscheck64064_v1.json",
    "work/ns_collision/scripts/"
    "neutral_strip_weighted_hypercircle_state_region64064_audit.py",
    "work/ns_collision/tests/"
    "test_weighted_hypercircle_state_region64064_audit.py",
    "work/ns_collision/results/"
    "neutral_strip_h006_hypercircle_state_region64064_audit_v1.json",
    "work/ns_collision/notes/"
    "neutral_strip_weighted_hypercircle_state_region64064_obstruction.md",
    "work/ns_collision/scripts/update_state_region64064_bookmark.py",
    "work/ns_collision/README.md",
)

LOWER_RESULT = ARTIFACTS[2]
HIGHER_RESULT = ARTIFACTS[3]
CROSSCHECK_RESULT = ARTIFACTS[4]
STATE_AUDIT_RESULT = ARTIFACTS[7]


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

    lower = _load_json(LOWER_RESULT)
    higher = _load_json(HIGHER_RESULT)
    crosscheck = _load_json(CROSSCHECK_RESULT)
    state_audit = _load_json(STATE_AUDIT_RESULT)
    for label, result in (("precision 60", lower), ("precision 100", higher)):
        _require(
            result.get("status") == "standalone_route_does_not_close"
            and result.get("all_current_stage_checks_pass") is True
            and result["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is False,
            f"{label} result does not record the expected obstruction",
        )
        _require(
            result["directed_LDL_dependency"]["required"] is False
            and result["directed_LDL_dependency"]["audit_loaded"] is False,
            f"{label} result has a directed-audit dependency",
        )
    crosscheck_other_checks = {
        key: value
        for key, value in crosscheck["checks"].items()
        if key != "both_residual_certificates_close"
    }
    _require(
        crosscheck.get("all_checks_pass") is False
        and crosscheck["checks"]["both_residual_certificates_close"] is False
        and all(crosscheck_other_checks.values()),
        "precision crosscheck has a failure other than expected non-closure",
    )
    _require(
        state_audit.get("all_current_stage_checks_pass") is True
        and state_audit.get("status")
        == "pass_with_certification_obstruction",
        "state-region structural audit does not pass",
    )
    _require(
        state_audit["checks"]["leading_factor_is_bitwise_unchanged"] is True
        and state_audit["checks"][
            "inverse_majorants_reconstructed_at_100_digits"
        ]
        is True,
        "state-region obstruction was not independently reconstructed",
    )
    _require(
        state_audit["certification_flags"][
            "standalone_64064_inertia_certified"
        ]
        is False
        and state_audit["certification_flags"][
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
    _require(
        int(
            principal.get(
                "reversible_weighted_hypercircle_"
                "standalone63680_pivot_count",
                0,
            )
        )
        == 63680
        and principal.get(
            "reversible_weighted_hypercircle_standalone63680_certified"
        )
        is True,
        "bookmark does not preserve the certified 63,680 predecessor",
    )

    certificate = state_audit["certificate_summary"]
    risk = state_audit["risk_change_from_63680"]
    transition = state_audit["transition_cluster"]
    diagnostics = state_audit["inverse_majorant_diagnostics"]
    incremental = state_audit[
        "incremental_63680_64063_block_profile"
    ]

    bookmark["checkpointed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    bookmark["status"] = "checkpointed"
    bookmark["codex_owned_processes_running"] = False
    bookmark["validated_checkpoint"] = (
        "The standalone 63680-pivot interval-family inertia certificate "
        "remains the last closed finite prefix. Hash-bound precision-60 and "
        "precision-100 replays through 64064 pivots agree on provenance, "
        "the binary reference factor, 32500 negative and 31564 positive "
        "reference diagonals, and all directed upper-bound nesting. Both "
        "correctly fail to certify the interval family because the "
        "transformed-residual/minimum-diagonal ratio is about 95.5495. An "
        "independent audit proves the entire 63680 leading factor and its "
        "minimum diagonal are bitwise unchanged. From 63680 to 64064, the "
        "inverse one/infinity majorants grow by about 130.70 and 99.56 and "
        "the residual norm by 29.83. The dominant chain and largest residual "
        "entry localize to edge pivots 64039--64040 through the earlier "
        "small edge pivot 63629. The symbolic transitions through 64056 are "
        "crossed, but those after 64043 add no further majorant growth. Full "
        "inertia, the weighted Ritz constant, continuum capture, and every "
        "Navier-Stokes regularity claim remain false."
    )

    principal.update(
        {
            "reversible_weighted_hypercircle_standalone64064_diagnostic_complete": True,
            "reversible_weighted_hypercircle_standalone64064_certified": False,
            "reversible_weighted_hypercircle_standalone64064_pivot_count": (
                certificate["maximum_pivots"]
            ),
            "reversible_weighted_hypercircle_standalone64064_reference_signs": (
                certificate["reference_signs"]
            ),
            "reversible_weighted_hypercircle_standalone64064_minimum_diagonal": (
                certificate["minimum_absolute_reference_diagonal_decimal"]
            ),
            "reversible_weighted_hypercircle_standalone64064_minimum_diagonal_index": (
                certificate["minimum_absolute_reference_diagonal_index"]
            ),
            "reversible_weighted_hypercircle_standalone64064_minimum_diagonal_block": (
                certificate["minimum_absolute_reference_diagonal_block"]
            ),
            "reversible_weighted_hypercircle_standalone64064_residual_ratio": (
                certificate[
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ]
            ),
            "reversible_weighted_hypercircle_standalone64064_p100_nesting_crosscheck": True,
            "reversible_weighted_hypercircle_standalone64064_expected_crosscheck_nonclosure": True,
            "reversible_weighted_hypercircle_standalone64064_leading_factor_unchanged": True,
            "reversible_weighted_hypercircle_incremental_block_profile63680_64064": (
                incremental
            ),
            "reversible_weighted_hypercircle_inverse_one_growth63680_64064": (
                risk["inverse_one_norm_growth_factor"]
            ),
            "reversible_weighted_hypercircle_inverse_infinity_growth63680_64064": (
                risk["inverse_infinity_norm_growth_factor"]
            ),
            "reversible_weighted_hypercircle_residual_growth63680_64064": (
                risk["residual_infinity_norm_growth_factor"]
            ),
            "reversible_weighted_hypercircle_residual_ratio_growth63680_64064": (
                risk["bound_to_diagonal_ratio_growth_factor"]
            ),
            "reversible_weighted_hypercircle_state_region_transition_cluster": (
                transition["observed_pivots"]
            ),
            "reversible_weighted_hypercircle_inverse_majorant_worst_bracket": (
                diagnostics["worst_profile_jump"]
            ),
            "reversible_weighted_hypercircle_larger_prefix_admitted": False,
            "reversible_weighted_hypercircle_next_bounded_pivot_target": None,
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
            "Crossed the seven-transition state-region cluster through "
            "pivot 64063 with standalone precision-60/100 replays. The "
            "reference factor and all provenance checks agree, while both "
            "certificates fail closed at ratio 95.5495. Independent "
            "reconstruction preserves the certified 63680 leading factor "
            "bitwise and localizes the inverse-majorant/residual obstruction "
            "to the edge chain through pivots 63629 and 64039--64040."
        ),
    )

    bookmark["unfinished_obligation"] = (
        "The interval pencil remains independently certified only through "
        "pivot 63679. The 64064 central reference has no zero diagonal, but "
        "its standalone transformed-residual bound exceeds the minimum "
        "diagonal by factor 95.55, so no interval-family inertia conclusion "
        "is available there and no larger prefix is admitted. The earliest "
        "closure-loss boundary inside 63680..64064 and a rigorous sharper "
        "inverse/residual treatment of the localized 64039--64040 edge chain "
        "remain open. Full inertia 61908/61908/0, strict solution-operator "
        "and weighted Ritz thresholds, source/conormal/domain transfer, and "
        "Navier-Stokes closure remain open and fail-closed."
    )
    bookmark["resume_command"] = "not_applicable_no_parked_compute"
    bookmark["next_action"] = (
        "After a fresh daytime CPU gate, do not enlarge the prefix. Run the "
        "hash-bound standalone residual at 63901 with precision 60 to choose "
        "the pass/fail half of 63680..64064, then bisect only that interval "
        "until the first closure-loss prefix is adjacent to the last passing "
        "prefix. Replay the adjacent pair at precision 100 before testing a "
        "local high-precision tail correction or sharper componentwise "
        "inverse bound. Do not launch directed LDL, the full pencil, or any "
        "continuum stage."
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
