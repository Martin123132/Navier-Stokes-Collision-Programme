#!/usr/bin/env python3
"""Cross-check two congruence-residual certificates by precision nesting."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("work/ns_collision/results")
DEFAULT_LOWER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_congruence_residual_pilot2304_v1.json"
)
DEFAULT_HIGHER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_congruence_residual_pilot2304_p100_v1.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_congruence_residual_"
    "precision_crosscheck2304_v1.json"
)


def _load_prefix_module():
    path = (
        SCRIPT_DIR
        / "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "congruence_residual_precision_crosscheck_base",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decimal_not_larger(
    higher: dict[str, Any],
    lower: dict[str, Any],
    key: str,
) -> bool:
    return Decimal(higher[key]) <= Decimal(lower[key])


def run_crosscheck(
    lower_path: Path = DEFAULT_LOWER,
    higher_path: Path = DEFAULT_HIGHER,
    lower_precision: int = 60,
    higher_precision: int = 100,
) -> dict[str, Any]:
    if higher_precision <= lower_precision:
        raise ValueError("higher precision must exceed lower precision")
    lower = _load_json(lower_path)
    higher = _load_json(higher_path)
    lower_certificate = lower["certificate"]
    higher_certificate = higher["certificate"]
    lower_mode = lower.get("validation_mode", "directed_crosscheck")
    higher_mode = higher.get("validation_mode", "directed_crosscheck")
    upper_bound_keys = (
        "maximum_residual_entry_upper_decimal",
        "residual_infinity_norm_upper_decimal",
        "absolute_L_inverse_infinity_norm_upper_decimal",
        "absolute_L_inverse_one_norm_upper_decimal",
        "transformed_residual_two_norm_upper_decimal",
        "transformed_bound_to_minimum_diagonal_upper_decimal",
    )
    nested_rows = {
        key: _decimal_not_larger(
            higher_certificate,
            lower_certificate,
            key,
        )
        for key in upper_bound_keys
    }
    source_hash_keys = (
        "scale_sha256",
        "raw_permutation_sha256",
        "order_sha256",
        "factor_pattern_sha256",
        "U_diagonal_sha256",
    )
    lower_hashes = lower["preparation"]["hashes"]
    higher_hashes = higher["preparation"]["hashes"]
    modes_equal = lower_mode == higher_mode
    if modes_equal and lower_mode == "standalone_hash_bound":
        validation_provenance_equal = (
            lower["standalone_contract"]["contract_sha256"]
            == higher["standalone_contract"]["contract_sha256"]
            and lower["standalone_contract"]["contract"]
            == higher["standalone_contract"]["contract"]
        )
    elif modes_equal and lower_mode == "directed_crosscheck":
        validation_provenance_equal = (
            lower["artifacts"]["directed_LDL_audit_sha256"]
            == higher["artifacts"]["directed_LDL_audit_sha256"]
        )
    else:
        validation_provenance_equal = False
    reference_hash_keys = (
        "reference_L_sha256",
        "reference_D_sha256",
        "reference_factor_sha256",
    )
    reference_hashes_equal = all(
        (
            key not in lower_certificate
            and key not in higher_certificate
        )
        or lower_certificate.get(key) == higher_certificate.get(key)
        for key in reference_hash_keys
    )
    accepted_statuses = {
        "independent_prefix_inertia_certified",
        "standalone_prefix_inertia_certified",
    }
    checks = {
        "both_residual_certificates_close": (
            lower["status"] in accepted_statuses
            and higher["status"] in accepted_statuses
            and lower["certification_flags"][
                "independent_bounded_prefix_inertia_certified"
            ]
            and higher["certification_flags"][
                "independent_bounded_prefix_inertia_certified"
            ]
        ),
        "requested_precisions_match": (
            lower_certificate["decimal_precision"] == lower_precision
            and higher_certificate["decimal_precision"] == higher_precision
        ),
        "dimension_and_reference_structure_equal": (
            lower_certificate["dimension"]
            == higher_certificate["dimension"]
            and lower_certificate["reference_L_nnz"]
            == higher_certificate["reference_L_nnz"]
            and lower_certificate["reference_product_lower_entry_count"]
            == higher_certificate["reference_product_lower_entry_count"]
            and lower_certificate["residual_lower_entry_count"]
            == higher_certificate["residual_lower_entry_count"]
        ),
        "reference_signs_equal": (
            lower_certificate["reference_diagonal_signs"]
            == higher_certificate["reference_diagonal_signs"]
        ),
        "minimum_reference_diagonal_equal": (
            lower_certificate[
                "minimum_absolute_reference_diagonal_decimal"
            ]
            == higher_certificate[
                "minimum_absolute_reference_diagonal_decimal"
            ]
        ),
        "maximum_residual_coordinate_equal": (
            lower_certificate["maximum_residual_entry_coordinate"]
            == higher_certificate["maximum_residual_entry_coordinate"]
        ),
        "proof_basis_and_validated_assumptions_equal": (
            lower_certificate["proof_basis"]
            == higher_certificate["proof_basis"]
            and lower_certificate["validated_assumptions"]
            == higher_certificate["validated_assumptions"]
        ),
        "frozen_preparation_hashes_equal": all(
            lower_hashes[key] == higher_hashes[key]
            for key in source_hash_keys
        ),
        "validation_modes_equal": modes_equal,
        "validation_provenance_equal": validation_provenance_equal,
        "reference_factor_hashes_equal": reference_hashes_equal,
        "all_higher_precision_upper_bounds_not_larger": all(
            nested_rows.values()
        ),
        "full_inertia_claim_remains_false": (
            not lower["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
            and not higher["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "kind": "hypercircle-congruence-residual-precision-crosscheck",
        "status": "pass" if passed else "fail_closed",
        "all_checks_pass": passed,
        "checks": checks,
        "upper_bound_nesting_checks": nested_rows,
        "comparison": {
            "validation_mode": lower_mode,
            "dimension": lower_certificate["dimension"],
            "lower_precision": lower_precision,
            "higher_precision": higher_precision,
            "reference_diagonal_signs": lower_certificate[
                "reference_diagonal_signs"
            ],
            "minimum_absolute_reference_diagonal_decimal": (
                lower_certificate[
                    "minimum_absolute_reference_diagonal_decimal"
                ]
            ),
            "lower_transformed_residual_two_norm_upper_decimal": (
                lower_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ]
            ),
            "higher_transformed_residual_two_norm_upper_decimal": (
                higher_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ]
            ),
            "lower_bound_to_diagonal_ratio_upper_decimal": (
                lower_certificate[
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ]
            ),
            "higher_bound_to_diagonal_ratio_upper_decimal": (
                higher_certificate[
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ]
            ),
        },
        "artifacts": {
            "lower_precision_result": str(lower_path).replace("\\", "/"),
            "lower_precision_result_sha256": _sha256_file(lower_path),
            "higher_precision_result": str(higher_path).replace("\\", "/"),
            "higher_precision_result_sha256": _sha256_file(higher_path),
        },
        "scope": (
            "Cross-precision replay of the independent bounded-prefix "
            "congruence-residual certificate. It does not certify any "
            "unprocessed pivot, full inertia, or continuum transfer."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower", type=Path, default=DEFAULT_LOWER)
    parser.add_argument("--higher", type=Path, default=DEFAULT_HIGHER)
    parser.add_argument("--lower-precision", type=int, default=60)
    parser.add_argument("--higher-precision", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_crosscheck(
        lower_path=args.lower,
        higher_path=args.higher,
        lower_precision=args.lower_precision,
        higher_precision=args.higher_precision,
    )
    prefix = _load_prefix_module()
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
