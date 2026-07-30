#!/usr/bin/env python3
"""Cross-check two directed-LDL prefix certificates by interval nesting."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("work/ns_collision/results")
DEFAULT_LOWER_AUDIT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition2304_audit_v1.json"
)
DEFAULT_LOWER_CHECKPOINT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition2304_checkpoint_v1.json"
)
DEFAULT_HIGHER_AUDIT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition2304_p80_audit_v1.json"
)
DEFAULT_HIGHER_CHECKPOINT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition2304_p80_checkpoint_v1.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition2304_"
    "precision_crosscheck_v1.json"
)


def _load_prefix_module():
    path = (
        SCRIPT_DIR
        / "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "directed_ldl_precision_crosscheck_base",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _nested(
    inner_lower: str,
    inner_upper: str,
    outer_lower: str,
    outer_upper: str,
) -> bool:
    return (
        Decimal(outer_lower)
        <= Decimal(inner_lower)
        <= Decimal(inner_upper)
        <= Decimal(outer_upper)
    )


def _common_contract(contract: dict[str, Any]) -> dict[str, Any]:
    ignored = {"precision_schedule"}
    return {
        key: value
        for key, value in contract.items()
        if key not in ignored
    }


def _structural_interaction_profile(
    profile: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "pivot_count",
        "first_pivot",
        "last_pivot",
        "negative_pivot_count",
        "positive_pivot_count",
        "maximum_diagonal_term_count",
        "maximum_off_diagonal_recurrence_term_count",
        "maximum_symbolic_descendant_count",
        "block_counts",
    )
    return {key: profile[key] for key in keys}


def run_crosscheck(
    lower_audit_path: Path = DEFAULT_LOWER_AUDIT,
    lower_checkpoint_path: Path = DEFAULT_LOWER_CHECKPOINT,
    higher_audit_path: Path = DEFAULT_HIGHER_AUDIT,
    higher_checkpoint_path: Path = DEFAULT_HIGHER_CHECKPOINT,
    lower_precision: int = 50,
    higher_precision: int = 80,
    label: str = "transition2304",
) -> dict[str, Any]:
    if higher_precision <= lower_precision:
        raise ValueError("higher precision must exceed lower precision")
    prefix = _load_prefix_module()
    lower_audit = _load_json(lower_audit_path)
    higher_audit = _load_json(higher_audit_path)
    lower_raw = _load_json(lower_checkpoint_path)
    higher_raw = _load_json(higher_checkpoint_path)
    lower_checkpoint = prefix._load_checkpoint(
        lower_checkpoint_path,
        lower_raw["contract"],
        tuple(lower_raw["precision_schedule"]),
    )
    higher_checkpoint = prefix._load_checkpoint(
        higher_checkpoint_path,
        higher_raw["contract"],
        tuple(higher_raw["precision_schedule"]),
    )
    lower_attempt = lower_checkpoint["current_attempt"]
    higher_attempt = higher_checkpoint["current_attempt"]

    lower_pivots = list(
        zip(
            lower_attempt["pivot_lower_decimal"],
            lower_attempt["pivot_upper_decimal"],
            strict=True,
        )
    )
    higher_pivots = list(
        zip(
            higher_attempt["pivot_lower_decimal"],
            higher_attempt["pivot_upper_decimal"],
            strict=True,
        )
    )
    pivot_lengths_equal = len(lower_pivots) == len(higher_pivots)
    pivot_intervals_nested = pivot_lengths_equal and all(
        _nested(
            inner_lower,
            inner_upper,
            outer_lower,
            outer_upper,
        )
        for (outer_lower, outer_upper), (inner_lower, inner_upper) in zip(
            lower_pivots,
            higher_pivots,
            strict=True,
        )
    )

    lower_entries = lower_attempt["lower_entries"]
    higher_entries = higher_attempt["lower_entries"]
    symbolic_coordinates_equal = (
        len(lower_entries) == len(higher_entries)
        and all(
            outer[0] == inner[0] and outer[1] == inner[1]
            for outer, inner in zip(
                lower_entries,
                higher_entries,
                strict=True,
            )
        )
    )
    lower_intervals_nested = symbolic_coordinates_equal and all(
        _nested(
            inner[2],
            inner[3],
            outer[2],
            outer[3],
        )
        for outer, inner in zip(
            lower_entries,
            higher_entries,
            strict=True,
        )
    )

    lower_prefix = lower_audit["directed_LDL_prefix"]
    higher_prefix = higher_audit["directed_LDL_prefix"]
    lower_profile = lower_prefix["interaction_profile"]
    higher_profile = higher_prefix["interaction_profile"]
    contracts_match = (
        _common_contract(lower_checkpoint["contract"])
        == _common_contract(higher_checkpoint["contract"])
        and lower_audit["contract"] == lower_checkpoint["contract"]
        and higher_audit["contract"] == higher_checkpoint["contract"]
    )
    checks = {
        "checkpoint_state_hashes_replay": (
            lower_checkpoint == lower_raw
            and higher_checkpoint == higher_raw
        ),
        "audit_checkpoint_hashes_match": (
            lower_audit["artifacts"]["checkpoint_sha256"]
            == prefix._sha256_file(lower_checkpoint_path)
            and higher_audit["artifacts"]["checkpoint_sha256"]
            == prefix._sha256_file(higher_checkpoint_path)
        ),
        "common_contract_and_provenance_equal": contracts_match,
        "requested_precisions_match": (
            lower_attempt["precision"] == lower_precision
            and higher_attempt["precision"] == higher_precision
        ),
        "both_prefixes_certified": (
            lower_audit["status"] == "certified_bounded_prefix"
            and higher_audit["status"] == "certified_bounded_prefix"
            and lower_audit["certification_flags"][
                "bounded_prefix_directed_LDL_certified"
            ]
            and higher_audit["certification_flags"][
                "bounded_prefix_directed_LDL_certified"
            ]
        ),
        "endpoint_lengths_equal": pivot_lengths_equal,
        "pivot_intervals_higher_nested_in_lower": pivot_intervals_nested,
        "pivot_signs_equal": (
            lower_attempt["pivot_sign"] == higher_attempt["pivot_sign"]
        ),
        "symbolic_coordinates_equal": symbolic_coordinates_equal,
        "lower_intervals_higher_nested_in_lower": lower_intervals_nested,
        "interaction_structural_profiles_equal": (
            _structural_interaction_profile(lower_profile)
            == _structural_interaction_profile(higher_profile)
        ),
        "full_inertia_claim_remains_false": (
            not lower_audit["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
            and not higher_audit["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "kind": f"hypercircle-directed-ldl-{label}-precision-crosscheck",
        "status": "pass" if passed else "fail_closed",
        "all_checks_pass": passed,
        "checks": checks,
        "comparison": {
            "pivot_count": len(lower_pivots),
            "symbolic_lower_entry_count": len(lower_entries),
            "lower_precision": lower_precision,
            "higher_precision": higher_precision,
            "lower_minimum_margin_decimal": lower_prefix[
                "minimum_pivot_margin_decimal"
            ],
            "higher_minimum_margin_decimal": higher_prefix[
                "minimum_pivot_margin_decimal"
            ],
            "lower_maximum_relative_radius_decimal": lower_prefix[
                "maximum_pivot_radius_to_margin_upper_decimal"
            ],
            "higher_maximum_relative_radius_decimal": higher_prefix[
                "maximum_pivot_radius_to_margin_upper_decimal"
            ],
            "lower_interaction_profile": lower_profile,
            "higher_interaction_profile": higher_profile,
        },
        "artifacts": {
            "lower_audit": str(lower_audit_path).replace("\\", "/"),
            "lower_audit_sha256": prefix._sha256_file(lower_audit_path),
            "lower_checkpoint": str(lower_checkpoint_path).replace("\\", "/"),
            "lower_checkpoint_sha256": prefix._sha256_file(
                lower_checkpoint_path
            ),
            "higher_audit": str(higher_audit_path).replace("\\", "/"),
            "higher_audit_sha256": prefix._sha256_file(higher_audit_path),
            "higher_checkpoint": str(higher_checkpoint_path).replace(
                "\\",
                "/",
            ),
            "higher_checkpoint_sha256": prefix._sha256_file(
                higher_checkpoint_path
            ),
        },
        "scope": (
            "Independent replay of stored interval endpoints and checkpoint "
            "integrity at two Decimal precisions. This comparison certifies "
            "no pivot beyond the two bounded input prefixes."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower-audit", type=Path, default=DEFAULT_LOWER_AUDIT)
    parser.add_argument(
        "--lower-checkpoint",
        type=Path,
        default=DEFAULT_LOWER_CHECKPOINT,
    )
    parser.add_argument(
        "--higher-audit",
        type=Path,
        default=DEFAULT_HIGHER_AUDIT,
    )
    parser.add_argument(
        "--higher-checkpoint",
        type=Path,
        default=DEFAULT_HIGHER_CHECKPOINT,
    )
    parser.add_argument("--lower-precision", type=int, default=50)
    parser.add_argument("--higher-precision", type=int, default=80)
    parser.add_argument("--label", default="transition2304")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_crosscheck(
        lower_audit_path=args.lower_audit,
        lower_checkpoint_path=args.lower_checkpoint,
        higher_audit_path=args.higher_audit,
        higher_checkpoint_path=args.higher_checkpoint,
        lower_precision=args.lower_precision,
        higher_precision=args.higher_precision,
        label=args.label,
    )
    prefix = _load_prefix_module()
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
