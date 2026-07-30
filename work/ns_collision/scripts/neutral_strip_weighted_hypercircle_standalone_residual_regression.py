#!/usr/bin/env python3
"""Compare standalone residual certificates with the historical route."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("work/ns_collision/results")
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual_regression_v1.json"
)
PREFIXES = (2304, 32064, 33280)


def _load_prefix_module():
    path = (
        SCRIPT_DIR
        / "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "standalone_residual_regression_prefix_base",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(prefix: int) -> tuple[Path, Path]:
    standalone = RESULTS_DIR / (
        "neutral_strip_h006_hypercircle_standalone_residual"
        f"{prefix}_v1.json"
    )
    legacy = RESULTS_DIR / (
        "neutral_strip_h006_hypercircle_congruence_residual_pilot"
        f"{prefix}_v1.json"
    )
    return standalone, legacy


def run_regression() -> dict[str, Any]:
    exact_certificate_keys = (
        "dimension",
        "identity_factor_permutations",
        "reference_L_nnz",
        "reference_product_lower_entry_count",
        "residual_lower_entry_count",
        "maximum_residual_entry_upper_decimal",
        "maximum_residual_entry_coordinate",
        "residual_infinity_norm_upper_decimal",
        "absolute_L_inverse_infinity_norm_upper_decimal",
        "absolute_L_inverse_one_norm_upper_decimal",
        "transformed_residual_two_norm_upper_decimal",
        "minimum_absolute_reference_diagonal_decimal",
        "transformed_bound_to_minimum_diagonal_upper_decimal",
        "reference_diagonal_signs",
        "interval_family_inertia_certified",
    )
    rows: list[dict[str, Any]] = []
    all_checks: dict[str, bool] = {}
    for maximum_pivots in PREFIXES:
        standalone_path, legacy_path = _paths(maximum_pivots)
        standalone = _load_json(standalone_path)
        legacy = _load_json(legacy_path)
        standalone_certificate = standalone["certificate"]
        legacy_certificate = legacy["certificate"]
        certificate_values_equal = all(
            standalone_certificate[key] == legacy_certificate[key]
            for key in exact_certificate_keys
        )
        frozen_hashes_equal = (
            standalone["preparation"]["hashes"]
            == legacy["preparation"]["hashes"]
        )
        row_checks = {
            "standalone_integrity_checks_pass": (
                standalone["all_current_stage_checks_pass"] is True
            ),
            "standalone_certificate_closes": (
                standalone["status"]
                == "standalone_prefix_inertia_certified"
                and standalone["certification_flags"][
                    "standalone_bounded_prefix_inertia_certified"
                ]
                is True
            ),
            "directed_audit_is_not_required_or_loaded": (
                standalone["directed_LDL_dependency"]
                == {
                    "required": False,
                    "audit_loaded": False,
                    "sign_comparison_used_for_certification": False,
                }
            ),
            "legacy_certificate_closes": (
                legacy["status"] == "independent_prefix_inertia_certified"
                and legacy["certification_flags"][
                    "independent_bounded_prefix_inertia_certified"
                ]
                is True
            ),
            "certificate_values_reproduce_exactly": (
                certificate_values_equal
            ),
            "frozen_preparation_hashes_equal": frozen_hashes_equal,
            "standalone_contract_dimension_matches": (
                standalone["standalone_contract"]["contract"][
                    "maximum_pivots"
                ]
                == maximum_pivots
            ),
            "full_inertia_claim_remains_false": (
                standalone["certification_flags"][
                    "full_123816_pivot_inertia_certified"
                ]
                is False
            ),
        }
        for name, value in row_checks.items():
            all_checks[f"prefix_{maximum_pivots}_{name}"] = bool(value)
        rows.append(
            {
                "maximum_pivots": maximum_pivots,
                "checks": row_checks,
                "standalone_contract_sha256": standalone[
                    "standalone_contract"
                ]["contract_sha256"],
                "standalone_reference_factor_sha256": (
                    standalone_certificate["reference_factor_sha256"]
                ),
                "reference_signs": standalone_certificate[
                    "reference_diagonal_signs"
                ],
                "bound_to_diagonal_ratio_upper_decimal": (
                    standalone_certificate[
                        "transformed_bound_to_minimum_diagonal_upper_decimal"
                    ]
                ),
                "artifacts": {
                    "standalone": str(standalone_path).replace("\\", "/"),
                    "standalone_sha256": _sha256(standalone_path),
                    "legacy": str(legacy_path).replace("\\", "/"),
                    "legacy_sha256": _sha256(legacy_path),
                },
            }
        )
    passed = bool(all(all_checks.values()))
    return {
        "kind": "hypercircle-standalone-residual-regression",
        "status": "pass" if passed else "fail_closed",
        "all_checks_pass": passed,
        "checks": all_checks,
        "prefix_comparisons": rows,
        "state_entry_pilot_admitted": passed,
        "certification_flags": {
            "standalone_residual_implementation_validated": passed,
            "full_123816_pivot_inertia_certified": False,
            "continuum_spectrum_below_60_captured": False,
            "navier_stokes_regularity_certified": False,
        },
        "scope": (
            "Exact regression comparison at three already-certified prefixes. "
            "It validates removal of the directed-audit dependency but "
            "certifies no new pivot or broader theorem."
        ),
        "next_required_step": (
            "Run only the bounded standalone 63680 state-entry pilot at "
            "precisions 60 and 100."
            if passed
            else "Do not launch the state-entry pilot; resolve the failed "
            "standalone regression first."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_regression()
    prefix = _load_prefix_module()
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
