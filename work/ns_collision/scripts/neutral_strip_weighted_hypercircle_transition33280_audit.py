"""Audit the certified 32,064-to-33,280 hypercircle transition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "work/ns_collision/results"

DEFAULT_PRIOR_AUDIT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition32064_audit_v1.json"
)
DEFAULT_CURRENT_AUDIT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition33280_audit_v1.json"
)
DEFAULT_CURRENT_P80_AUDIT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition33280_p80_audit_v1.json"
)
DEFAULT_DIRECTED_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_directed_ldl_transition33280_"
    "precision_crosscheck_v1.json"
)
DEFAULT_PRIOR_RESIDUAL = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_congruence_residual_pilot32064_v1.json"
)
DEFAULT_CURRENT_RESIDUAL = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_congruence_residual_pilot33280_v1.json"
)
DEFAULT_CURRENT_P100_RESIDUAL = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_congruence_residual_pilot33280_p100_v1.json"
)
DEFAULT_RESIDUAL_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_congruence_residual_"
    "precision_crosscheck33280_v1.json"
)
DEFAULT_SYMBOLIC_MAP = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_symbolic_transition_map123816_v1.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_transition33280_audit_v1.json"
)

PRIOR_PIVOTS = 32064
CURRENT_PIVOTS = 33280
EXPECTED_TRANSITION_PIVOT = 33224
EXPECTED_FIRST_STATE_PIVOT = 63644


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _ratio(numerator: Any, denominator: Any) -> str:
    with localcontext() as context:
        context.prec = 50
        return str(_decimal(numerator) / _decimal(denominator))


def _extreme(
    rows: list[dict[str, Any]],
    field: str,
    *,
    minimum: bool,
) -> dict[str, Any]:
    selector = min if minimum else max
    row = selector(rows, key=lambda item: _decimal(item[field]))
    return {
        "index": int(row["index"]),
        "value_decimal": str(row[field]),
    }


def _summarize_segment(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty pivot segment")
    signs = [int(row["sign"]) for row in rows]
    return {
        "first_pivot": int(rows[0]["index"]),
        "last_pivot": int(rows[-1]["index"]),
        "pivot_count": len(rows),
        "signs": {
            "negative": signs.count(-1),
            "positive": signs.count(1),
            "zero": signs.count(0),
        },
        "minimum_margin": _extreme(
            rows,
            "pivot_margin_decimal",
            minimum=True,
        ),
        "maximum_cancellation_charge": _extreme(
            rows,
            "cancellation_charge_upper_decimal",
            minimum=False,
        ),
        "maximum_lower_interval_width": _extreme(
            rows,
            "maximum_lower_interval_width_decimal",
            minimum=False,
        ),
        "maximum_pivot_radius_to_margin": _extreme(
            rows,
            "pivot_radius_to_margin_upper_decimal",
            minimum=False,
        ),
        "maximum_diagonal_term_count": max(
            int(row["diagonal_term_count"]) for row in rows
        ),
        "maximum_off_diagonal_recurrence_term_count": max(
            int(row["off_diagonal_recurrence_term_count"]) for row in rows
        ),
        "maximum_symbolic_descendant_count": max(
            int(row["symbolic_descendant_count"]) for row in rows
        ),
        "zero_input_diagonal_count": sum(
            _decimal(row["input_diagonal_center_decimal"]) == 0
            and _decimal(row["input_diagonal_radius_decimal"]) == 0
            for row in rows
        ),
    }


def _block_delta(
    prior: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, int]:
    if set(prior) != set(current):
        raise ValueError("pivot block inventories use different keys")
    delta = {
        key: int(current[key]) - int(prior[key])
        for key in sorted(current)
    }
    if any(value < 0 for value in delta.values()):
        raise ValueError("pivot block inventory decreased")
    return delta


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def run_audit(
    *,
    prior_audit_path: Path = DEFAULT_PRIOR_AUDIT,
    current_audit_path: Path = DEFAULT_CURRENT_AUDIT,
    current_p80_audit_path: Path = DEFAULT_CURRENT_P80_AUDIT,
    directed_crosscheck_path: Path = DEFAULT_DIRECTED_CROSSCHECK,
    prior_residual_path: Path = DEFAULT_PRIOR_RESIDUAL,
    current_residual_path: Path = DEFAULT_CURRENT_RESIDUAL,
    current_p100_residual_path: Path = DEFAULT_CURRENT_P100_RESIDUAL,
    residual_crosscheck_path: Path = DEFAULT_RESIDUAL_CROSSCHECK,
    symbolic_map_path: Path = DEFAULT_SYMBOLIC_MAP,
) -> dict[str, Any]:
    paths = {
        "prior_directed_audit": prior_audit_path,
        "current_directed_audit": current_audit_path,
        "current_p80_directed_audit": current_p80_audit_path,
        "directed_precision_crosscheck": directed_crosscheck_path,
        "prior_residual": prior_residual_path,
        "current_residual": current_residual_path,
        "current_p100_residual": current_p100_residual_path,
        "residual_precision_crosscheck": residual_crosscheck_path,
        "full_symbolic_map": symbolic_map_path,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    prior = _load_json(prior_audit_path)
    current = _load_json(current_audit_path)
    current_p80 = _load_json(current_p80_audit_path)
    directed_crosscheck = _load_json(directed_crosscheck_path)
    prior_residual = _load_json(prior_residual_path)
    current_residual = _load_json(current_residual_path)
    current_p100_residual = _load_json(current_p100_residual_path)
    residual_crosscheck = _load_json(residual_crosscheck_path)
    symbolic_map = _load_json(symbolic_map_path)

    prior_prefix = prior["directed_LDL_prefix"]
    current_prefix = current["directed_LDL_prefix"]
    current_p80_prefix = current_p80["directed_LDL_prefix"]
    prior_rows = prior_prefix["pivot_diagnostics"]
    current_rows = current_prefix["pivot_diagnostics"]
    current_p80_rows = current_p80_prefix["pivot_diagnostics"]

    new_rows = current_rows[PRIOR_PIVOTS:CURRENT_PIVOTS]
    transition_rows = current_rows[
        EXPECTED_TRANSITION_PIVOT:CURRENT_PIVOTS
    ]
    new_segment = _summarize_segment(new_rows)
    transition_segment = _summarize_segment(transition_rows)
    block_delta = _block_delta(
        prior_prefix["pivot_block_counts"],
        current_prefix["pivot_block_counts"],
    )

    delicate_index = int(new_segment["minimum_margin"]["index"])
    delicate = current_rows[delicate_index]
    delicate_p80 = current_p80_rows[delicate_index]
    next_unprocessed_transition = min(
        int(record["pivot"])
        for record in symbolic_map["new_transitions_at_or_after_prior_boundary"]
        if int(record["pivot"]) > EXPECTED_TRANSITION_PIVOT
    )

    directed_artifacts = directed_crosscheck["artifacts"]
    residual_artifacts = residual_crosscheck["artifacts"]
    checks = {
        "prior_directed_prefix_certified": (
            prior.get("all_current_stage_checks_pass") is True
            and prior["certification_flags"][
                "bounded_prefix_directed_LDL_certified"
            ]
            is True
        ),
        "current_directed_prefix_certified_at_both_precisions": (
            current.get("all_current_stage_checks_pass") is True
            and current_p80.get("all_current_stage_checks_pass") is True
            and current["certification_flags"][
                "bounded_prefix_directed_LDL_certified"
            ]
            is True
            and current_p80["certification_flags"][
                "bounded_prefix_directed_LDL_certified"
            ]
            is True
        ),
        "directed_precision_crosscheck_passes": (
            directed_crosscheck.get("all_checks_pass") is True
        ),
        "directed_crosscheck_hashes_match": (
            directed_artifacts["lower_audit_sha256"]
            == _sha256(current_audit_path)
            and directed_artifacts["higher_audit_sha256"]
            == _sha256(current_p80_audit_path)
        ),
        "current_prefix_lengths_match": (
            len(current_rows) == CURRENT_PIVOTS
            and len(current_p80_rows) == CURRENT_PIVOTS
        ),
        "fresh_current_run_reproduces_prior_p50_prefix": (
            len(prior_rows) == PRIOR_PIVOTS
            and prior_rows == current_rows[:PRIOR_PIVOTS]
        ),
        "new_block_delta_matches_segment_length": (
            sum(block_delta.values()) == len(new_rows)
        ),
        "symbolic_transition_and_target_match": (
            int(symbolic_map["next_transition_pivot"])
            == EXPECTED_TRANSITION_PIVOT
            and int(symbolic_map["recommended_next_bounded_pivot_count"])
            == CURRENT_PIVOTS
        ),
        "transition_introduces_six_descendants": (
            int(current_rows[EXPECTED_TRANSITION_PIVOT][
                "symbolic_descendant_count"
            ])
            == 6
        ),
        "delicate_pivot_is_before_fill_transition": (
            delicate_index < EXPECTED_TRANSITION_PIVOT
        ),
        "delicate_p80_interval_nests": (
            _decimal(delicate["pivot_interval_decimal"][0])
            <= _decimal(delicate_p80["pivot_interval_decimal"][0])
            <= _decimal(delicate_p80["pivot_interval_decimal"][1])
            <= _decimal(delicate["pivot_interval_decimal"][1])
        ),
        "residual_certifies_current_prefix_at_both_precisions": (
            current_residual.get("all_current_stage_checks_pass") is True
            and current_p100_residual.get("all_current_stage_checks_pass")
            is True
            and current_residual["certificate"][
                "interval_family_inertia_certified"
            ]
            is True
            and current_p100_residual["certificate"][
                "interval_family_inertia_certified"
            ]
            is True
        ),
        "residual_precision_crosscheck_passes": (
            residual_crosscheck.get("all_checks_pass") is True
        ),
        "residual_crosscheck_hashes_match": (
            residual_artifacts["lower_precision_result_sha256"]
            == _sha256(current_residual_path)
            and residual_artifacts["higher_precision_result_sha256"]
            == _sha256(current_p100_residual_path)
        ),
        "two_routes_report_identical_signs": (
            current_residual["certificate"]["reference_diagonal_signs"]
            == {
                "negative": int(current_prefix["negative_pivot_count"]),
                "positive": int(current_prefix["positive_pivot_count"]),
                "zero": 0,
            }
        ),
        "full_inertia_and_continuum_claims_remain_false": (
            current["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
            is False
            and current_p80["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
            is False
            and current_residual["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
            is False
            and symbolic_map["certification_flags"][
                "continuum_spectrum_below_60_captured"
            ]
            is False
        ),
    }
    all_checks = bool(all(checks.values()))

    prior_residual_ratio = prior_residual["certificate"][
        "transformed_bound_to_minimum_diagonal_upper_decimal"
    ]
    current_residual_ratio = current_residual["certificate"][
        "transformed_bound_to_minimum_diagonal_upper_decimal"
    ]
    prior_interaction = prior_prefix["interaction_profile"]
    current_interaction = current_prefix["interaction_profile"]

    return {
        "kind": "hypercircle-transition33280-audit",
        "status": "pass" if all_checks else "fail",
        "scope": (
            "Audit of the newly certified pivots 32064..33279 under the "
            "frozen interval input, positive Ruiz scaling, and elimination "
            "order. It certifies no later pivot, full inertia, continuum "
            "transfer, or Navier-Stokes statement."
        ),
        "all_current_stage_checks_pass": all_checks,
        "checks": checks,
        "artifacts": {
            name: _artifact_record(path)
            for name, path in paths.items()
        },
        "current_certified_prefix": {
            "completed_pivots": int(
                current_prefix["completed_pivot_count"]
            ),
            "negative": int(current_prefix["negative_pivot_count"]),
            "positive": int(current_prefix["positive_pivot_count"]),
            "zero": 0,
            "minimum_margin_decimal": str(
                current_prefix["minimum_pivot_margin_decimal"]
            ),
            "minimum_margin_index": int(
                current_prefix["minimum_pivot_margin_index"]
            ),
            "symbolic_lower_entries": int(
                current_prefix["symbolic_lower_entry_count"]
            ),
            "maximum_radius_to_margin_upper_decimal": str(
                current_prefix[
                    "maximum_pivot_radius_to_margin_upper_decimal"
                ]
            ),
        },
        "new_segment_32064_33279": {
            **new_segment,
            "pivot_block_counts": block_delta,
        },
        "delicate_pivot": {
            "index": delicate_index,
            "input_diagonal_is_exact_zero": (
                _decimal(delicate["input_diagonal_center_decimal"]) == 0
                and _decimal(delicate["input_diagonal_radius_decimal"]) == 0
            ),
            "diagonal_term_count": int(delicate["diagonal_term_count"]),
            "off_diagonal_recurrence_term_count": int(
                delicate["off_diagonal_recurrence_term_count"]
            ),
            "symbolic_descendant_count": int(
                delicate["symbolic_descendant_count"]
            ),
            "pivot_interval_decimal_p50": list(
                delicate["pivot_interval_decimal"]
            ),
            "pivot_interval_decimal_p80": list(
                delicate_p80["pivot_interval_decimal"]
            ),
            "pivot_margin_decimal": str(delicate["pivot_margin_decimal"]),
            "cancellation_charge_upper_decimal": str(
                delicate["cancellation_charge_upper_decimal"]
            ),
            "maximum_lower_interval_width_decimal": str(
                delicate["maximum_lower_interval_width_decimal"]
            ),
            "pivot_radius_to_margin_upper_decimal": str(
                delicate["pivot_radius_to_margin_upper_decimal"]
            ),
            "central_reference_pivot": delicate["central_superlu_pivot"],
            "central_reference_pivot_contained": bool(
                delicate["central_superlu_pivot_contained"]
            ),
        },
        "fill_transition_33224_33279": {
            **transition_segment,
            "transition_pivot": EXPECTED_TRANSITION_PIVOT,
            "first_transition_pivot_diagnostic": {
                key: current_rows[EXPECTED_TRANSITION_PIVOT][key]
                for key in (
                    "diagonal_term_count",
                    "off_diagonal_recurrence_term_count",
                    "pivot_margin_decimal",
                    "pivot_radius_to_margin_upper_decimal",
                    "sign",
                    "symbolic_descendant_count",
                )
            },
        },
        "independent_residual_certificate": {
            "precision_60_bound_to_diagonal_ratio_upper_decimal": str(
                current_residual_ratio
            ),
            "precision_100_bound_to_diagonal_ratio_upper_decimal": str(
                current_p100_residual["certificate"][
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ]
            ),
            "minimum_absolute_reference_diagonal_decimal": str(
                current_residual["certificate"][
                    "minimum_absolute_reference_diagonal_decimal"
                ]
            ),
            "absolute_L_inverse_one_norm_upper_decimal": str(
                current_residual["certificate"][
                    "absolute_L_inverse_one_norm_upper_decimal"
                ]
            ),
            "absolute_L_inverse_infinity_norm_upper_decimal": str(
                current_residual["certificate"][
                    "absolute_L_inverse_infinity_norm_upper_decimal"
                ]
            ),
            "precision_crosscheck_passes": bool(
                residual_crosscheck["all_checks_pass"]
            ),
        },
        "risk_change_from_32064": {
            "minimum_margin_reduction_factor": _ratio(
                prior_prefix["minimum_pivot_margin_decimal"],
                current_prefix["minimum_pivot_margin_decimal"],
            ),
            "maximum_cancellation_growth_factor": _ratio(
                current_interaction[
                    "maximum_cancellation_charge_upper_decimal"
                ],
                prior_interaction[
                    "maximum_cancellation_charge_upper_decimal"
                ],
            ),
            "maximum_radius_to_margin_growth_factor": _ratio(
                current_prefix[
                    "maximum_pivot_radius_to_margin_upper_decimal"
                ],
                prior_prefix[
                    "maximum_pivot_radius_to_margin_upper_decimal"
                ],
            ),
            "residual_ratio_growth_factor": _ratio(
                current_residual_ratio,
                prior_residual_ratio,
            ),
            "interpretation": (
                "Both proofs still close decisively, but pivot 32849 marks "
                "a real cancellation-sensitive regime. Runtime and error "
                "growth must not be extrapolated from the 32064 prefix."
            ),
        },
        "next_boundary": {
            "next_unprocessed_symbolic_transition_pivot": (
                next_unprocessed_transition
            ),
            "first_state_pivot": EXPECTED_FIRST_STATE_PIVOT,
            "recommended_residual_state_entry_pilot_pivots": 63680,
            "residual_only_mode_currently_requires_implementation": True,
        },
        "certification_flags": {
            "bounded_33280_inertia_certified_by_two_routes": all_checks,
            "full_123816_pivot_inertia_certified": False,
            "weighted_global_Ritz_projection_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
            "navier_stokes_regularity_certified": False,
        },
        "next_required_step": (
            "Implement and test a standalone residual mode that does not "
            "require a matching directed-LDL audit, then run only the "
            "bounded 63680 state-entry pilot. Keep full directed LDL and all "
            "continuum claims false."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-audit", type=Path, default=DEFAULT_PRIOR_AUDIT)
    parser.add_argument(
        "--current-audit",
        type=Path,
        default=DEFAULT_CURRENT_AUDIT,
    )
    parser.add_argument(
        "--current-p80-audit",
        type=Path,
        default=DEFAULT_CURRENT_P80_AUDIT,
    )
    parser.add_argument(
        "--directed-crosscheck",
        type=Path,
        default=DEFAULT_DIRECTED_CROSSCHECK,
    )
    parser.add_argument(
        "--prior-residual",
        type=Path,
        default=DEFAULT_PRIOR_RESIDUAL,
    )
    parser.add_argument(
        "--current-residual",
        type=Path,
        default=DEFAULT_CURRENT_RESIDUAL,
    )
    parser.add_argument(
        "--current-p100-residual",
        type=Path,
        default=DEFAULT_CURRENT_P100_RESIDUAL,
    )
    parser.add_argument(
        "--residual-crosscheck",
        type=Path,
        default=DEFAULT_RESIDUAL_CROSSCHECK,
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
        prior_audit_path=args.prior_audit,
        current_audit_path=args.current_audit,
        current_p80_audit_path=args.current_p80_audit,
        directed_crosscheck_path=args.directed_crosscheck,
        prior_residual_path=args.prior_residual,
        current_residual_path=args.current_residual,
        current_p100_residual_path=args.current_p100_residual,
        residual_crosscheck_path=args.residual_crosscheck,
        symbolic_map_path=args.symbolic_map,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
