#!/usr/bin/env python3
"""Audit componentwise residual recovery through 64,064 pivots."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


RESULTS_DIR = Path("work/ns_collision/results")
DEFAULT_BOUNDARY_LOWER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64040_v1.json"
)
DEFAULT_BOUNDARY_HIGHER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64040_p100_v1.json"
)
DEFAULT_BOUNDARY_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_componentwise_residual_"
    "precision_crosscheck64040_v1.json"
)
DEFAULT_ENDPOINT_LOWER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64064_v1.json"
)
DEFAULT_ENDPOINT_HIGHER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64064_p100_v1.json"
)
DEFAULT_ENDPOINT_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_componentwise_residual_"
    "precision_crosscheck64064_v1.json"
)
DEFAULT_CLOSURE_AUDIT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_closure_boundary64039_audit_v1.json"
)
DEFAULT_STATE_REGION_AUDIT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_state_region64064_audit_v1.json"
)
DEFAULT_SYMBOLIC_MAP = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_symbolic_transition_map123816_v1.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "componentwise_state_region64064_audit_v1.json"
)
BOUNDARY_PIVOTS = 64040
ENDPOINT_PIVOTS = 64064
NEXT_BOUNDED_TARGET = 64128
EXPECTED_NEXT_SYMBOLIC_TRANSITION = 76921


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


def _ratio(numerator: Any, denominator: Any) -> str:
    with localcontext() as context:
        context.prec = 100
        return str(Decimal(str(numerator)) / Decimal(str(denominator)))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="ascii",
            newline="\n",
        ) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _contract_core(result: dict[str, Any]) -> dict[str, Any]:
    contract = result["standalone_contract"]["contract"]
    return {
        "algorithm_version": contract["algorithm_version"],
        "full_dimension": contract["full_dimension"],
        "interval_family": contract["interval_family"],
        "reference_rule": contract["reference_rule"],
        "source_artifacts": contract["source_artifacts"],
        "frozen_preparation_hashes": contract[
            "frozen_preparation_hashes"
        ],
        "validation_mode": contract["validation_mode"],
    }


def _crosscheck_hashes_match(
    crosscheck: dict[str, Any],
    lower_path: Path,
    higher_path: Path,
) -> bool:
    return bool(
        crosscheck["artifacts"]["lower_precision_result_sha256"]
        == _sha256(lower_path)
        and crosscheck["artifacts"]["higher_precision_result_sha256"]
        == _sha256(higher_path)
    )


def run_audit(
    boundary_lower_path: Path = DEFAULT_BOUNDARY_LOWER,
    boundary_higher_path: Path = DEFAULT_BOUNDARY_HIGHER,
    boundary_crosscheck_path: Path = DEFAULT_BOUNDARY_CROSSCHECK,
    endpoint_lower_path: Path = DEFAULT_ENDPOINT_LOWER,
    endpoint_higher_path: Path = DEFAULT_ENDPOINT_HIGHER,
    endpoint_crosscheck_path: Path = DEFAULT_ENDPOINT_CROSSCHECK,
    closure_audit_path: Path = DEFAULT_CLOSURE_AUDIT,
    state_region_audit_path: Path = DEFAULT_STATE_REGION_AUDIT,
    symbolic_map_path: Path = DEFAULT_SYMBOLIC_MAP,
) -> dict[str, Any]:
    boundary_lower = _load_json(boundary_lower_path)
    boundary_higher = _load_json(boundary_higher_path)
    boundary_crosscheck = _load_json(boundary_crosscheck_path)
    endpoint_lower = _load_json(endpoint_lower_path)
    endpoint_higher = _load_json(endpoint_higher_path)
    endpoint_crosscheck = _load_json(endpoint_crosscheck_path)
    closure_audit = _load_json(closure_audit_path)
    state_region_audit = _load_json(state_region_audit_path)
    symbolic_map = _load_json(symbolic_map_path)

    boundary_certificate = boundary_higher["certificate"]
    endpoint_certificate = endpoint_higher["certificate"]
    next_symbolic_transition = min(
        int(record["pivot"])
        for record in symbolic_map[
            "new_transitions_at_or_after_prior_boundary"
        ]
        if int(record["pivot"]) >= ENDPOINT_PIVOTS
    )
    componentwise_results = (
        boundary_lower,
        boundary_higher,
        endpoint_lower,
        endpoint_higher,
    )
    checks = {
        "all_componentwise_constructions_validate": all(
            result.get("all_current_stage_checks_pass") is True
            for result in componentwise_results
        ),
        "all_componentwise_certificates_close": all(
            result["status"] == "standalone_prefix_inertia_certified"
            and result["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is True
            for result in componentwise_results
        ),
        "both_precision_crosschecks_pass": (
            boundary_crosscheck.get("all_checks_pass") is True
            and endpoint_crosscheck.get("all_checks_pass") is True
        ),
        "precision_crosscheck_hashes_match": (
            _crosscheck_hashes_match(
                boundary_crosscheck,
                boundary_lower_path,
                boundary_higher_path,
            )
            and _crosscheck_hashes_match(
                endpoint_crosscheck,
                endpoint_lower_path,
                endpoint_higher_path,
            )
        ),
        "componentwise_method_is_constant": (
            boundary_lower["certificate"]["proof_basis"]
            == boundary_higher["certificate"]["proof_basis"]
            == endpoint_lower["certificate"]["proof_basis"]
            == endpoint_higher["certificate"]["proof_basis"]
            and boundary_lower["certificate"]["validated_assumptions"]
            == boundary_higher["certificate"]["validated_assumptions"]
            == endpoint_lower["certificate"]["validated_assumptions"]
            == endpoint_higher["certificate"]["validated_assumptions"]
        ),
        "source_and_preparation_provenance_is_constant": (
            _contract_core(boundary_lower)
            == _contract_core(boundary_higher)
            == _contract_core(endpoint_lower)
            == _contract_core(endpoint_higher)
        ),
        "no_directed_audit_dependency": all(
            result["directed_LDL_dependency"]["required"] is False
            and result["directed_LDL_dependency"]["audit_loaded"] is False
            for result in componentwise_results
        ),
        "separated_reference_and_residual_reproduced": all(
            result["checks"][
                "separated_reference_and_residual_reproduced"
            ]
            is True
            and result["checks"]["separated_bound_reproduced"] is True
            for result in componentwise_results
        ),
        "componentwise_bound_improves_separated_bound": all(
            Decimal(
                result["certificate"][
                    "transformed_residual_two_norm_upper_decimal"
                ]
            )
            < Decimal(
                result["certificate"][
                    "separated_transformed_residual_two_norm_upper_decimal"
                ]
            )
            for result in componentwise_results
        ),
        "closure_boundary_obstruction_is_recovered": (
            closure_audit.get("all_current_stage_checks_pass") is True
            and closure_audit["boundary"][
                "first_nonclosing_prefix_pivots"
            ]
            == BOUNDARY_PIVOTS
            and boundary_certificate["dimension"] == BOUNDARY_PIVOTS
            and boundary_certificate[
                "interval_family_inertia_certified"
            ]
            is True
        ),
        "state_region_endpoint_obstruction_is_recovered": (
            state_region_audit.get("all_current_stage_checks_pass") is True
            and state_region_audit["certificate_summary"][
                "maximum_pivots"
            ]
            == ENDPOINT_PIVOTS
            and state_region_audit["route_obstruction"][
                "standalone_64064_inertia_certified"
            ]
            is False
            and endpoint_certificate["dimension"] == ENDPOINT_PIVOTS
            and endpoint_certificate[
                "interval_family_inertia_certified"
            ]
            is True
        ),
        "endpoint_reference_signs_match_prior_reconstruction": (
            endpoint_certificate["reference_diagonal_signs"]
            == state_region_audit["certificate_summary"][
                "reference_signs"
            ]
        ),
        "minimum_diagonal_is_unchanged": (
            boundary_certificate[
                "minimum_absolute_reference_diagonal_decimal"
            ]
            == endpoint_certificate[
                "minimum_absolute_reference_diagonal_decimal"
            ]
        ),
        "next_target_precedes_next_symbolic_transition": (
            next_symbolic_transition
            == EXPECTED_NEXT_SYMBOLIC_TRANSITION
            and NEXT_BOUNDED_TARGET < next_symbolic_transition
        ),
        "full_and_continuum_claims_remain_false": all(
            result["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
            is False
            and result["certification_flags"][
                "continuum_spectrum_below_60_captured"
            ]
            is False
            and result["certification_flags"][
                "navier_stokes_regularity_certified"
            ]
            is False
            for result in componentwise_results
        ),
    }
    all_checks = bool(all(checks.values()))

    artifacts = {
        "boundary_precision_60_result": {
            "path": str(boundary_lower_path).replace("\\", "/"),
            "sha256": _sha256(boundary_lower_path),
        },
        "boundary_precision_100_result": {
            "path": str(boundary_higher_path).replace("\\", "/"),
            "sha256": _sha256(boundary_higher_path),
        },
        "boundary_precision_crosscheck": {
            "path": str(boundary_crosscheck_path).replace("\\", "/"),
            "sha256": _sha256(boundary_crosscheck_path),
        },
        "endpoint_precision_60_result": {
            "path": str(endpoint_lower_path).replace("\\", "/"),
            "sha256": _sha256(endpoint_lower_path),
        },
        "endpoint_precision_100_result": {
            "path": str(endpoint_higher_path).replace("\\", "/"),
            "sha256": _sha256(endpoint_higher_path),
        },
        "endpoint_precision_crosscheck": {
            "path": str(endpoint_crosscheck_path).replace("\\", "/"),
            "sha256": _sha256(endpoint_crosscheck_path),
        },
        "closure_boundary_audit": {
            "path": str(closure_audit_path).replace("\\", "/"),
            "sha256": _sha256(closure_audit_path),
        },
        "separated_state_region_audit": {
            "path": str(state_region_audit_path).replace("\\", "/"),
            "sha256": _sha256(state_region_audit_path),
        },
        "symbolic_transition_map": {
            "path": str(symbolic_map_path).replace("\\", "/"),
            "sha256": _sha256(symbolic_map_path),
        },
    }
    boundary_ratio = boundary_certificate[
        "transformed_bound_to_minimum_diagonal_upper_decimal"
    ]
    endpoint_ratio = endpoint_certificate[
        "transformed_bound_to_minimum_diagonal_upper_decimal"
    ]
    return {
        "kind": (
            "hypercircle-standalone-componentwise-state-region64064-audit"
        ),
        "status": (
            "componentwise_state_region_certified"
            if all_checks
            else "fail_closed"
        ),
        "scope": (
            "Audit of the directed componentwise transformed-residual "
            "certificate through the already bounded 64,064-pivot state "
            "region. It certifies no later prefix, full inertia, continuum "
            "transfer, or Navier-Stokes regularity statement."
        ),
        "all_current_stage_checks_pass": all_checks,
        "checks": checks,
        "artifacts": artifacts,
        "boundary_recovery": {
            "maximum_pivots": BOUNDARY_PIVOTS,
            "reference_signs": boundary_certificate[
                "reference_diagonal_signs"
            ],
            "componentwise_bound_upper_decimal": boundary_certificate[
                "transformed_residual_two_norm_upper_decimal"
            ],
            "bound_to_minimum_diagonal_ratio_upper_decimal": (
                boundary_ratio
            ),
            "safety_factor_lower_decimal": _ratio(1, boundary_ratio),
            "improvement_over_separated_bound_lower_decimal": (
                boundary_certificate[
                    "componentwise_improvement_factor_lower_decimal"
                ]
            ),
        },
        "state_region_recovery": {
            "maximum_pivots": ENDPOINT_PIVOTS,
            "last_certified_pivot": ENDPOINT_PIVOTS - 1,
            "reference_signs": endpoint_certificate[
                "reference_diagonal_signs"
            ],
            "minimum_absolute_reference_diagonal_decimal": (
                endpoint_certificate[
                    "minimum_absolute_reference_diagonal_decimal"
                ]
            ),
            "componentwise_bound_upper_decimal": endpoint_certificate[
                "transformed_residual_two_norm_upper_decimal"
            ],
            "bound_to_minimum_diagonal_ratio_upper_decimal": (
                endpoint_ratio
            ),
            "safety_factor_lower_decimal": _ratio(1, endpoint_ratio),
            "improvement_over_separated_bound_lower_decimal": (
                endpoint_certificate[
                    "componentwise_improvement_factor_lower_decimal"
                ]
            ),
            "componentwise_bound_growth_from_64040": _ratio(
                endpoint_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ],
                boundary_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ],
            ),
            "incremental_block_profile_63680_64064": (
                state_region_audit[
                    "incremental_63680_64063_block_profile"
                ]
            ),
        },
        "next_boundary": {
            "recommended_bounded_pivot_count": NEXT_BOUNDED_TARGET,
            "next_symbolic_transition_pivot": next_symbolic_transition,
            "crosses_new_symbolic_transition": False,
            "full_run_admitted": False,
        },
        "certification_flags": {
            "standalone_componentwise_64040_inertia_certified": (
                all_checks
            ),
            "standalone_componentwise_64064_inertia_certified": (
                all_checks
            ),
            "full_123816_pivot_inertia_certified": False,
            "weighted_global_Ritz_projection_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
            "navier_stokes_regularity_certified": False,
        },
        "next_required_step": (
            f"After a fresh CPU gate, test only the componentwise standalone "
            f"prefix {NEXT_BOUNDED_TARGET} at precisions 60 and 100 as a "
            "local growth diagnostic. Do not jump to the next symbolic "
            "transition, full pencil, or continuum stage."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary-lower",
        type=Path,
        default=DEFAULT_BOUNDARY_LOWER,
    )
    parser.add_argument(
        "--boundary-higher",
        type=Path,
        default=DEFAULT_BOUNDARY_HIGHER,
    )
    parser.add_argument(
        "--boundary-crosscheck",
        type=Path,
        default=DEFAULT_BOUNDARY_CROSSCHECK,
    )
    parser.add_argument(
        "--endpoint-lower",
        type=Path,
        default=DEFAULT_ENDPOINT_LOWER,
    )
    parser.add_argument(
        "--endpoint-higher",
        type=Path,
        default=DEFAULT_ENDPOINT_HIGHER,
    )
    parser.add_argument(
        "--endpoint-crosscheck",
        type=Path,
        default=DEFAULT_ENDPOINT_CROSSCHECK,
    )
    parser.add_argument(
        "--closure-audit",
        type=Path,
        default=DEFAULT_CLOSURE_AUDIT,
    )
    parser.add_argument(
        "--state-region-audit",
        type=Path,
        default=DEFAULT_STATE_REGION_AUDIT,
    )
    parser.add_argument(
        "--symbolic-map",
        type=Path,
        default=DEFAULT_SYMBOLIC_MAP,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_audit(
        boundary_lower_path=args.boundary_lower,
        boundary_higher_path=args.boundary_higher,
        boundary_crosscheck_path=args.boundary_crosscheck,
        endpoint_lower_path=args.endpoint_lower,
        endpoint_higher_path=args.endpoint_higher,
        endpoint_crosscheck_path=args.endpoint_crosscheck,
        closure_audit_path=args.closure_audit,
        state_region_audit_path=args.state_region_audit,
        symbolic_map_path=args.symbolic_map,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_current_stage_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
