#!/usr/bin/env python3
"""Diagnose the standalone residual obstruction at 64,064 pivots."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("work/ns_collision/results")
DEFAULT_LOWER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual64064_v1.json"
)
DEFAULT_HIGHER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual64064_p100_v1.json"
)
DEFAULT_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual_"
    "precision_crosscheck64064_v1.json"
)
DEFAULT_PRIOR_HIGHER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual63680_p100_v1.json"
)
DEFAULT_PRIOR_AUDIT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_state_entry63680_audit_v1.json"
)
DEFAULT_SYMBOLIC_MAP = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_symbolic_transition_map123816_v1.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_state_region64064_audit_v1.json"
)
MAXIMUM_PIVOTS = 64064
PRIOR_PIVOTS = 63680
EXPECTED_TRANSITION_PIVOTS = [
    63733,
    63735,
    63900,
    64043,
    64044,
    64049,
    64056,
]
DAYTIME_BASELINE_CPU_LIMIT = 60.0


def _load_module(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
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


def _ratio(numerator: Decimal | str, denominator: Decimal | str) -> str:
    with localcontext() as context:
        context.prec = 100
        return str(Decimal(numerator) / Decimal(denominator))


def _sparse_exactly_equal(left: csc_matrix, right: csc_matrix) -> bool:
    left = left.tocsc(copy=True)
    right = right.tocsc(copy=True)
    left.sort_indices()
    right.sort_indices()
    return bool(
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data)
    )


def _classify_original(
    original_index: int,
    boundaries: tuple[tuple[str, int, int], ...],
) -> str:
    for name, start, stop in boundaries:
        if start <= original_index < stop:
            return name
    raise RuntimeError(
        f"original index is outside block inventory: {original_index}"
    )


def _inverse_majorant_vectors(
    lower: csc_matrix,
    arithmetic,
) -> tuple[list[Decimal], list[Decimal]]:
    dimension = lower.shape[0]
    lower_csr = lower.tocsr()
    lower_csr.sort_indices()
    row_sums = [arithmetic.one] * dimension
    for row in range(dimension):
        value = arithmetic.one
        for pointer in range(
            int(lower_csr.indptr[row]),
            int(lower_csr.indptr[row + 1]),
        ):
            column = int(lower_csr.indices[pointer])
            if column >= row:
                continue
            coefficient = Decimal.from_float(
                abs(float(lower_csr.data[pointer]))
            )
            value = arithmetic.upper.add(
                value,
                arithmetic.upper.multiply(
                    coefficient,
                    row_sums[column],
                ),
            )
        row_sums[row] = value

    column_sums = [arithmetic.one] * dimension
    for column in range(dimension - 1, -1, -1):
        value = arithmetic.one
        for pointer in range(
            int(lower.indptr[column]),
            int(lower.indptr[column + 1]),
        ):
            row = int(lower.indices[pointer])
            if row <= column:
                continue
            coefficient = Decimal.from_float(
                abs(float(lower.data[pointer]))
            )
            value = arithmetic.upper.add(
                value,
                arithmetic.upper.multiply(
                    coefficient,
                    column_sums[row],
                ),
            )
        column_sums[column] = value
    return row_sums, column_sums


def _pivot_metadata(
    pivot: int,
    value: Decimal,
    original_indices: np.ndarray,
    boundaries: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    original_index = int(original_indices[pivot])
    return {
        "pivot": pivot,
        "original_index": original_index,
        "block": _classify_original(
            original_index,
            boundaries,
        ),
        "value_upper_decimal": str(value),
    }


def _dominant_row_path(
    lower: csc_matrix,
    row_sums: list[Decimal],
    start: int,
    arithmetic,
    original_indices: np.ndarray,
    boundaries: tuple[tuple[str, int, int], ...],
) -> list[dict[str, Any]]:
    lower_csr = lower.tocsr()
    lower_csr.sort_indices()
    path: list[dict[str, Any]] = []
    pivot = start
    while len(path) < 64:
        candidates: list[tuple[Decimal, int, Decimal]] = []
        for pointer in range(
            int(lower_csr.indptr[pivot]),
            int(lower_csr.indptr[pivot + 1]),
        ):
            column = int(lower_csr.indices[pointer])
            if column >= pivot:
                continue
            coefficient = Decimal.from_float(
                abs(float(lower_csr.data[pointer]))
            )
            contribution = arithmetic.upper.multiply(
                coefficient,
                row_sums[column],
            )
            candidates.append((contribution, column, coefficient))
        if not candidates:
            break
        contribution, child, coefficient = max(candidates)
        record = _pivot_metadata(
            pivot,
            row_sums[pivot],
            original_indices,
            boundaries,
        )
        record.update(
            {
                "dominant_predecessor_pivot": child,
                "coefficient_absolute_decimal": str(coefficient),
                "contribution_upper_decimal": str(contribution),
            }
        )
        path.append(record)
        pivot = child
    return path


def _dominant_column_path(
    lower: csc_matrix,
    column_sums: list[Decimal],
    start: int,
    arithmetic,
    original_indices: np.ndarray,
    boundaries: tuple[tuple[str, int, int], ...],
) -> list[dict[str, Any]]:
    path: list[dict[str, Any]] = []
    pivot = start
    while len(path) < 64:
        candidates: list[tuple[Decimal, int, Decimal]] = []
        for pointer in range(
            int(lower.indptr[pivot]),
            int(lower.indptr[pivot + 1]),
        ):
            row = int(lower.indices[pointer])
            if row <= pivot:
                continue
            coefficient = Decimal.from_float(
                abs(float(lower.data[pointer]))
            )
            contribution = arithmetic.upper.multiply(
                coefficient,
                column_sums[row],
            )
            candidates.append((contribution, row, coefficient))
        if not candidates:
            break
        contribution, child, coefficient = max(candidates)
        record = _pivot_metadata(
            pivot,
            column_sums[pivot],
            original_indices,
            boundaries,
        )
        record.update(
            {
                "dominant_successor_pivot": child,
                "coefficient_absolute_decimal": str(coefficient),
                "contribution_upper_decimal": str(contribution),
            }
        )
        path.append(record)
        pivot = child
    return path


def _top_majorants(
    values: list[Decimal],
    original_indices: np.ndarray,
    boundaries: tuple[tuple[str, int, int], ...],
    count: int = 12,
) -> list[dict[str, Any]]:
    pivots = sorted(
        range(len(values)),
        key=values.__getitem__,
        reverse=True,
    )[:count]
    return [
        _pivot_metadata(
            pivot,
            values[pivot],
            original_indices,
            boundaries,
        )
        for pivot in pivots
    ]


def run_audit(
    lower_path: Path = DEFAULT_LOWER,
    higher_path: Path = DEFAULT_HIGHER,
    crosscheck_path: Path = DEFAULT_CROSSCHECK,
    prior_higher_path: Path = DEFAULT_PRIOR_HIGHER,
    prior_audit_path: Path = DEFAULT_PRIOR_AUDIT,
    symbolic_map_path: Path = DEFAULT_SYMBOLIC_MAP,
    enforce_cpu_policy: bool = True,
) -> dict[str, Any]:
    prefix = _load_module(
        "state_region64064_prefix_base",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )
    residual = _load_module(
        "state_region64064_residual_base",
        "neutral_strip_weighted_hypercircle_congruence_residual_pilot.py",
    )
    state_base = _load_module(
        "state_region64064_state_base",
        "neutral_strip_weighted_hypercircle_state_entry63680_audit.py",
    )
    lower = _load_json(lower_path)
    higher = _load_json(higher_path)
    crosscheck = _load_json(crosscheck_path)
    prior_higher = _load_json(prior_higher_path)
    prior_audit = _load_json(prior_audit_path)
    symbolic_map = _load_json(symbolic_map_path)

    priority_set = prefix._set_below_normal_priority()
    baseline = (
        prefix._sample_cpu_baseline(5) if enforce_cpu_policy else []
    )
    baseline_mean = (
        sum(baseline) / len(baseline) if baseline else None
    )
    if (
        baseline_mean is not None
        and baseline_mean > DAYTIME_BASELINE_CPU_LIMIT
    ):
        raise RuntimeError(
            "daytime baseline CPU exceeds the one-worker launch limit: "
            f"{baseline_mean:.3f}%"
        )

    problem, preparation = prefix._prepare_production_problem(
        Path(
            "work/ns_collision/results/"
            "neutral_strip_h006_positive_exponential_complete_assembly_v1.json"
        ),
        Path(
            "work/ns_collision/results/"
            "neutral_strip_h006_positive_exponential_assembly_matrices_v1.npz"
        ),
        Path(
            "work/ns_collision/results/"
            "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
        ),
        Path(
            "work/ns_collision/results/"
            "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
        ),
    )
    prior_factor = splu(
        residual._scaled_central_prefix(problem, PRIOR_PIVOTS),
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    factor = splu(
        residual._scaled_central_prefix(problem, MAXIMUM_PIVOTS),
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    prior_lower = prior_factor.L.tocsc()
    prior_lower.sort_indices()
    lower_factor = factor.L.tocsc()
    lower_factor.sort_indices()
    prior_diagonal = np.asarray(prior_factor.U.diagonal(), dtype=float)
    diagonal = np.asarray(factor.U.diagonal(), dtype=float)
    identity = np.arange(MAXIMUM_PIVOTS, dtype=factor.perm_r.dtype)
    prior_identity = np.arange(PRIOR_PIVOTS, dtype=prior_factor.perm_r.dtype)

    inventory = preparation["matrix_inventory"]
    boundaries = state_base._block_boundaries(inventory)
    original_indices = np.asarray(
        problem.order[:MAXIMUM_PIVOTS],
        dtype=np.int64,
    )
    transition_events = [
        record
        for record in symbolic_map[
            "new_transitions_at_or_after_prior_boundary"
        ]
        if PRIOR_PIVOTS <= int(record["pivot"]) < MAXIMUM_PIVOTS
    ]
    transition_pivots = sorted(
        {int(record["pivot"]) for record in transition_events}
    )

    arithmetic = prefix.DirectedDecimal(100)
    profile_boundaries = sorted(
        {
            PRIOR_PIVOTS,
            MAXIMUM_PIVOTS,
            *(pivot + 1 for pivot in transition_pivots),
        }
    )
    boundary_profiles: list[dict[str, Any]] = []
    final_row_sums: list[Decimal] | None = None
    final_column_sums: list[Decimal] | None = None
    previous_product: Decimal | None = None
    for boundary in profile_boundaries:
        leading_lower = lower_factor[:boundary, :boundary].tocsc()
        leading_lower.sort_indices()
        row_sums, column_sums = _inverse_majorant_vectors(
            leading_lower,
            arithmetic,
        )
        infinity_bound = max(row_sums)
        one_bound = max(column_sums)
        product = arithmetic.upper.multiply(
            infinity_bound,
            one_bound,
        )
        infinity_pivot = int(np.argmax(np.asarray(row_sums, dtype=object)))
        one_pivot = int(np.argmax(np.asarray(column_sums, dtype=object)))
        record = {
            "maximum_pivots": boundary,
            "included_transition_pivot": (
                boundary - 1 if boundary - 1 in transition_pivots else None
            ),
            "inverse_infinity_norm_upper_decimal": str(infinity_bound),
            "inverse_infinity_maximizing_pivot": _pivot_metadata(
                infinity_pivot,
                infinity_bound,
                original_indices,
                boundaries,
            ),
            "inverse_one_norm_upper_decimal": str(one_bound),
            "inverse_one_maximizing_pivot": _pivot_metadata(
                one_pivot,
                one_bound,
                original_indices,
                boundaries,
            ),
            "inverse_norm_product_upper_decimal": str(product),
            "growth_from_previous_profile_boundary": (
                _ratio(product, previous_product)
                if previous_product is not None
                else None
            ),
        }
        boundary_profiles.append(record)
        previous_product = product
        if boundary == MAXIMUM_PIVOTS:
            final_row_sums = row_sums
            final_column_sums = column_sums

    if final_row_sums is None or final_column_sums is None:
        raise RuntimeError("final inverse-majorant profile is missing")
    final_infinity_pivot = max(
        range(MAXIMUM_PIVOTS),
        key=final_row_sums.__getitem__,
    )
    final_one_pivot = max(
        range(MAXIMUM_PIVOTS),
        key=final_column_sums.__getitem__,
    )

    transition_rows = []
    lower_csr = lower_factor.tocsr()
    lower_csr.sort_indices()
    for pivot in transition_pivots:
        original_index = int(original_indices[pivot])
        transition_rows.append(
            {
                "pivot": pivot,
                "original_index": original_index,
                "block": state_base._classify_original(
                    original_index,
                    boundaries,
                ),
                "metrics": [
                    {
                        "metric": str(record["metric"]),
                        "new_value": int(record["new_value"]),
                    }
                    for record in transition_events
                    if int(record["pivot"]) == pivot
                ],
                "reference_diagonal_decimal": str(
                    Decimal.from_float(float(diagonal[pivot]))
                ),
                "reference_diagonal_sign": (
                    -1 if diagonal[pivot] < 0.0 else 1
                ),
                "strict_lower_row_nnz": int(
                    np.count_nonzero(
                        lower_csr.indices[
                            lower_csr.indptr[pivot]:
                            lower_csr.indptr[pivot + 1]
                        ]
                        < pivot
                    )
                ),
                "inverse_row_majorant_upper_decimal": str(
                    final_row_sums[pivot]
                ),
                "inverse_column_majorant_upper_decimal": str(
                    final_column_sums[pivot]
                ),
            }
        )

    lower_certificate = lower["certificate"]
    higher_certificate = higher["certificate"]
    prior_certificate = prior_higher["certificate"]
    reconstructed_prior_hash = prefix._sha256_arrays(
        prior_lower.indptr,
        prior_lower.indices,
        prior_lower.data,
        prior_diagonal,
    )
    reconstructed_hash = prefix._sha256_arrays(
        lower_factor.indptr,
        lower_factor.indices,
        lower_factor.data,
        diagonal,
    )
    crosscheck_other_checks = {
        key: value
        for key, value in crosscheck["checks"].items()
        if key != "both_residual_certificates_close"
    }
    checks = {
        "prior_63680_audit_passes": (
            prior_audit.get("all_current_stage_checks_pass") is True
        ),
        "prior_63680_standalone_certificate_closes": (
            prior_higher["status"] == "standalone_prefix_inertia_certified"
            and prior_higher["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is True
        ),
        "both_64064_certificates_fail_closed": (
            lower["status"] == "standalone_route_does_not_close"
            and higher["status"] == "standalone_route_does_not_close"
            and lower["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is False
            and higher["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is False
        ),
        "both_64064_certificate_constructions_validate": (
            lower.get("all_current_stage_checks_pass") is True
            and higher.get("all_current_stage_checks_pass") is True
        ),
        "precision_crosscheck_has_only_expected_closure_failure": (
            crosscheck.get("all_checks_pass") is False
            and crosscheck["checks"][
                "both_residual_certificates_close"
            ]
            is False
            and all(crosscheck_other_checks.values())
        ),
        "precision_crosscheck_hashes_match": (
            crosscheck["artifacts"]["lower_precision_result_sha256"]
            == state_base._sha256(lower_path)
            and crosscheck["artifacts"]["higher_precision_result_sha256"]
            == state_base._sha256(higher_path)
        ),
        "standalone_contracts_equal": (
            lower["standalone_contract"] == higher["standalone_contract"]
        ),
        "directed_audit_not_required_or_loaded": (
            lower["directed_LDL_dependency"]["required"] is False
            and lower["directed_LDL_dependency"]["audit_loaded"] is False
            and higher["directed_LDL_dependency"]["required"] is False
            and higher["directed_LDL_dependency"]["audit_loaded"] is False
        ),
        "both_reconstructed_factors_use_identity_permutations": (
            np.array_equal(prior_factor.perm_r, prior_identity)
            and np.array_equal(prior_factor.perm_c, prior_identity)
            and np.array_equal(factor.perm_r, identity)
            and np.array_equal(factor.perm_c, identity)
        ),
        "reconstructed_prior_factor_hash_matches": (
            reconstructed_prior_hash
            == prior_certificate["reference_factor_sha256"]
        ),
        "reconstructed_64064_factor_hash_matches_both_precisions": (
            reconstructed_hash
            == lower_certificate["reference_factor_sha256"]
            == higher_certificate["reference_factor_sha256"]
        ),
        "leading_factor_is_bitwise_unchanged": (
            np.array_equal(diagonal[:PRIOR_PIVOTS], prior_diagonal)
            and _sparse_exactly_equal(
                lower_factor[:PRIOR_PIVOTS, :PRIOR_PIVOTS],
                prior_lower,
            )
        ),
        "reconstructed_signs_match_certificate": (
            state_base._sign_summary(diagonal)
            == lower_certificate["reference_diagonal_signs"]
            == higher_certificate["reference_diagonal_signs"]
        ),
        "minimum_reference_diagonal_reconstructed": (
            str(
                min(
                    abs(Decimal.from_float(float(value)))
                    for value in diagonal
                )
            )
            == lower_certificate[
                "minimum_absolute_reference_diagonal_decimal"
            ]
        ),
        "inverse_majorants_reconstructed_at_100_digits": (
            str(max(final_row_sums))
            == higher_certificate[
                "absolute_L_inverse_infinity_norm_upper_decimal"
            ]
            and str(max(final_column_sums))
            == higher_certificate[
                "absolute_L_inverse_one_norm_upper_decimal"
            ]
        ),
        "transition_cluster_matches_symbolic_map": (
            transition_pivots == EXPECTED_TRANSITION_PIVOTS
        ),
        "all_reference_diagonals_are_nonzero": (
            bool(np.all(diagonal != 0.0))
        ),
        "full_inertia_and_continuum_claims_remain_false": (
            lower["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
            is False
            and lower["certification_flags"][
                "continuum_spectrum_below_60_captured"
            ]
            is False
            and lower["certification_flags"][
                "navier_stokes_regularity_certified"
            ]
            is False
        ),
    }
    all_checks = bool(all(checks.values()))
    minimum_index = int(np.argmin(np.abs(diagonal)))
    worst_profile_jump_index = max(
        range(1, len(boundary_profiles)),
        key=lambda index: Decimal(
            boundary_profiles[index][
                "growth_from_previous_profile_boundary"
            ]
        ),
    )
    worst_profile_jump = {
        "lower_maximum_pivots": boundary_profiles[
            worst_profile_jump_index - 1
        ]["maximum_pivots"],
        "upper_maximum_pivots": boundary_profiles[
            worst_profile_jump_index
        ]["maximum_pivots"],
        "inverse_norm_product_growth_factor": boundary_profiles[
            worst_profile_jump_index
        ]["growth_from_previous_profile_boundary"],
    }

    artifacts = {
        "precision_60_result": {
            "path": str(lower_path).replace("\\", "/"),
            "sha256": state_base._sha256(lower_path),
        },
        "precision_100_result": {
            "path": str(higher_path).replace("\\", "/"),
            "sha256": state_base._sha256(higher_path),
        },
        "precision_crosscheck": {
            "path": str(crosscheck_path).replace("\\", "/"),
            "sha256": state_base._sha256(crosscheck_path),
        },
        "prior_precision_100_result": {
            "path": str(prior_higher_path).replace("\\", "/"),
            "sha256": state_base._sha256(prior_higher_path),
        },
        "prior_state_entry_audit": {
            "path": str(prior_audit_path).replace("\\", "/"),
            "sha256": state_base._sha256(prior_audit_path),
        },
        "full_symbolic_map": {
            "path": str(symbolic_map_path).replace("\\", "/"),
            "sha256": state_base._sha256(symbolic_map_path),
        },
    }
    return {
        "kind": "hypercircle-standalone-state-region64064-audit",
        "status": (
            "pass_with_certification_obstruction"
            if all_checks
            else "fail_closed"
        ),
        "scope": (
            "Independent structural and inverse-majorant audit of the "
            "hash-bound standalone residual route from 63,680 through "
            "64,064 pivots. It certifies no 64,064-pivot inertia, later "
            "prefix, full inertia, continuum transfer, or Navier-Stokes "
            "statement."
        ),
        "all_current_stage_checks_pass": all_checks,
        "checks": checks,
        "artifacts": artifacts,
        "certificate_summary": {
            "maximum_pivots": MAXIMUM_PIVOTS,
            "reference_signs": lower_certificate[
                "reference_diagonal_signs"
            ],
            "minimum_absolute_reference_diagonal_decimal": (
                lower_certificate[
                    "minimum_absolute_reference_diagonal_decimal"
                ]
            ),
            "minimum_absolute_reference_diagonal_index": minimum_index,
            "minimum_absolute_reference_diagonal_original_index": int(
                original_indices[minimum_index]
            ),
            "minimum_absolute_reference_diagonal_block": (
                state_base._classify_original(
                    int(original_indices[minimum_index]),
                    boundaries,
                )
            ),
            "transformed_residual_two_norm_upper_decimal": (
                higher_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ]
            ),
            "transformed_bound_to_minimum_diagonal_upper_decimal": (
                higher_certificate[
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ]
            ),
            "reference_L_nnz": int(
                lower_certificate["reference_L_nnz"]
            ),
            "reference_factor_sha256": reconstructed_hash,
            "standalone_contract_sha256": lower[
                "standalone_contract"
            ]["contract_sha256"],
        },
        "route_obstruction": {
            "standalone_64064_inertia_certified": False,
            "bound_exceeds_minimum_diagonal": True,
            "bound_to_minimum_diagonal_factor": higher_certificate[
                "transformed_bound_to_minimum_diagonal_upper_decimal"
            ],
            "precision_pair_agrees": True,
            "leading_63680_factor_unchanged": checks[
                "leading_factor_is_bitwise_unchanged"
            ],
            "interpretation": (
                "The frozen reference factor remains valid and bitwise "
                "compatible with the certified 63,680 prefix, but the "
                "absolute inverse-majorant amplification makes the current "
                "residual bound too large. This invalidates this certificate "
                "at 64,064; it is not evidence of a zero diagonal, an "
                "inertia change, or Navier-Stokes regularity failure."
            ),
        },
        "full_prefix_block_profile": state_base._block_profile(
            diagonal,
            original_indices,
            boundaries,
            0,
            MAXIMUM_PIVOTS,
        ),
        "incremental_63680_64063_block_profile": state_base._block_profile(
            diagonal,
            original_indices,
            boundaries,
            PRIOR_PIVOTS,
            MAXIMUM_PIVOTS,
        ),
        "transition_cluster": {
            "expected_pivots": EXPECTED_TRANSITION_PIVOTS,
            "observed_pivots": transition_pivots,
            "rows": transition_rows,
        },
        "risk_change_from_63680": {
            "minimum_diagonal_reduction_factor": _ratio(
                prior_certificate[
                    "minimum_absolute_reference_diagonal_decimal"
                ],
                higher_certificate[
                    "minimum_absolute_reference_diagonal_decimal"
                ],
            ),
            "inverse_one_norm_growth_factor": _ratio(
                higher_certificate[
                    "absolute_L_inverse_one_norm_upper_decimal"
                ],
                prior_certificate[
                    "absolute_L_inverse_one_norm_upper_decimal"
                ],
            ),
            "inverse_infinity_norm_growth_factor": _ratio(
                higher_certificate[
                    "absolute_L_inverse_infinity_norm_upper_decimal"
                ],
                prior_certificate[
                    "absolute_L_inverse_infinity_norm_upper_decimal"
                ],
            ),
            "residual_infinity_norm_growth_factor": _ratio(
                higher_certificate[
                    "residual_infinity_norm_upper_decimal"
                ],
                prior_certificate[
                    "residual_infinity_norm_upper_decimal"
                ],
            ),
            "transformed_residual_growth_factor": _ratio(
                higher_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ],
                prior_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ],
            ),
            "bound_to_diagonal_ratio_growth_factor": _ratio(
                higher_certificate[
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ],
                prior_certificate[
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ],
            ),
        },
        "inverse_majorant_diagnostics": {
            "profile_boundaries": boundary_profiles,
            "worst_profile_jump": worst_profile_jump,
            "top_inverse_infinity_majorants": _top_majorants(
                final_row_sums,
                original_indices,
                boundaries,
            ),
            "top_inverse_one_majorants": _top_majorants(
                final_column_sums,
                original_indices,
                boundaries,
            ),
            "dominant_inverse_infinity_path": _dominant_row_path(
                lower_factor,
                final_row_sums,
                final_infinity_pivot,
                arithmetic,
                original_indices,
                boundaries,
            ),
            "dominant_inverse_one_path": _dominant_column_path(
                lower_factor,
                final_column_sums,
                final_one_pivot,
                arithmetic,
                original_indices,
                boundaries,
            ),
        },
        "runtime": {
            "below_normal_priority_set": priority_set,
            "baseline_cpu_samples_percent": baseline,
            "baseline_cpu_mean_percent": baseline_mean,
        },
        "certification_flags": {
            "standalone_63680_inertia_certified": True,
            "standalone_64064_inertia_certified": False,
            "full_123816_pivot_inertia_certified": False,
            "weighted_global_Ritz_projection_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
            "navier_stokes_regularity_certified": False,
        },
        "next_boundary": {
            "larger_prefix_run_admitted": False,
            "full_run_admitted": False,
            "diagnostic_bracket": worst_profile_jump,
        },
        "next_required_step": (
            "Do not enlarge the prefix. First localize the earliest "
            "standalone closure loss inside 63,680..64,064, beginning with "
            "the inverse-majorant jump bracket reported here; then test "
            "whether a sharper certified inverse bound or a different "
            "hash-bound reference can remove the majorant obstruction."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower", type=Path, default=DEFAULT_LOWER)
    parser.add_argument("--higher", type=Path, default=DEFAULT_HIGHER)
    parser.add_argument("--crosscheck", type=Path, default=DEFAULT_CROSSCHECK)
    parser.add_argument(
        "--prior-higher",
        type=Path,
        default=DEFAULT_PRIOR_HIGHER,
    )
    parser.add_argument(
        "--prior-audit",
        type=Path,
        default=DEFAULT_PRIOR_AUDIT,
    )
    parser.add_argument(
        "--symbolic-map",
        type=Path,
        default=DEFAULT_SYMBOLIC_MAP,
    )
    parser.add_argument("--skip-cpu-policy", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_audit(
        lower_path=args.lower,
        higher_path=args.higher,
        crosscheck_path=args.crosscheck,
        prior_higher_path=args.prior_higher,
        prior_audit_path=args.prior_audit,
        symbolic_map_path=args.symbolic_map,
        enforce_cpu_policy=not args.skip_cpu_policy,
    )
    prefix = _load_module(
        "state_region64064_output_base",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_current_stage_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
