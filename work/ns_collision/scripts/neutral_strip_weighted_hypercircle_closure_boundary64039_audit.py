#!/usr/bin/env python3
"""Audit the first standalone residual closure loss at pivot 64,039."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.sparse.linalg import splu


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("work/ns_collision/results")
PASS_PIVOTS = 64039
FAIL_PIVOTS = 64040
ADDED_PIVOT = 64039
DAYTIME_BASELINE_CPU_LIMIT = 60.0

BISECTION_RESULTS = (
    (
        63680,
        "neutral_strip_h006_hypercircle_"
        "standalone_residual63680_v1.json",
        True,
    ),
    (
        63901,
        "neutral_strip_h006_hypercircle_"
        "standalone_residual63901_v1.json",
        True,
    ),
    (
        63982,
        "neutral_strip_h006_hypercircle_"
        "standalone_residual63982_v1.json",
        True,
    ),
    (
        64023,
        "neutral_strip_h006_hypercircle_"
        "standalone_residual64023_v1.json",
        True,
    ),
    (
        64033,
        "neutral_strip_h006_hypercircle_"
        "standalone_residual64033_v1.json",
        True,
    ),
    (
        64038,
        "neutral_strip_h006_hypercircle_"
        "standalone_residual64038_v1.json",
        True,
    ),
    (
        64039,
        "neutral_strip_h006_hypercircle_"
        "standalone_residual64039_v1.json",
        True,
    ),
    (
        64040,
        "neutral_strip_h006_hypercircle_"
        "standalone_residual64040_v1.json",
        False,
    ),
    (
        64043,
        "neutral_strip_h006_hypercircle_"
        "standalone_residual64043_v1.json",
        False,
    ),
    (
        64064,
        "neutral_strip_h006_hypercircle_"
        "standalone_residual64064_v1.json",
        False,
    ),
)

DEFAULT_PASS_LOWER = RESULTS_DIR / BISECTION_RESULTS[6][1]
DEFAULT_FAIL_LOWER = RESULTS_DIR / BISECTION_RESULTS[7][1]
DEFAULT_PASS_HIGHER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "standalone_residual64039_p100_v1.json"
)
DEFAULT_FAIL_HIGHER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "standalone_residual64040_p100_v1.json"
)
DEFAULT_PASS_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual_"
    "precision_crosscheck64039_v1.json"
)
DEFAULT_FAIL_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual_"
    "precision_crosscheck64040_v1.json"
)
DEFAULT_STATE_REGION_AUDIT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_state_region64064_audit_v1.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_closure_boundary64039_audit_v1.json"
)


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


def _ratio(numerator: Any, denominator: Any) -> str:
    with localcontext() as context:
        context.prec = 100
        return str(Decimal(str(numerator)) / Decimal(str(denominator)))


def _sparse_exactly_equal(left, right) -> bool:
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


def _contract_core(result: dict[str, Any]) -> dict[str, Any]:
    contract = result["standalone_contract"]["contract"]
    return {
        "algorithm_version": contract["algorithm_version"],
        "frozen_preparation_hashes": contract[
            "frozen_preparation_hashes"
        ],
        "full_dimension": contract["full_dimension"],
        "interval_family": contract["interval_family"],
        "reference_rule": contract["reference_rule"],
        "source_artifacts": contract["source_artifacts"],
        "validation_mode": contract["validation_mode"],
    }


def run_audit(
    pass_lower_path: Path = DEFAULT_PASS_LOWER,
    fail_lower_path: Path = DEFAULT_FAIL_LOWER,
    pass_higher_path: Path = DEFAULT_PASS_HIGHER,
    fail_higher_path: Path = DEFAULT_FAIL_HIGHER,
    pass_crosscheck_path: Path = DEFAULT_PASS_CROSSCHECK,
    fail_crosscheck_path: Path = DEFAULT_FAIL_CROSSCHECK,
    state_region_audit_path: Path = DEFAULT_STATE_REGION_AUDIT,
    enforce_cpu_policy: bool = True,
) -> dict[str, Any]:
    prefix = _load_module(
        "closure_boundary64039_prefix_base",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )
    residual = _load_module(
        "closure_boundary64039_residual_base",
        "neutral_strip_weighted_hypercircle_congruence_residual_pilot.py",
    )
    state_base = _load_module(
        "closure_boundary64039_state_base",
        "neutral_strip_weighted_hypercircle_state_entry63680_audit.py",
    )
    region_base = _load_module(
        "closure_boundary64039_region_base",
        "neutral_strip_weighted_hypercircle_state_region64064_audit.py",
    )

    pass_lower = _load_json(pass_lower_path)
    fail_lower = _load_json(fail_lower_path)
    pass_higher = _load_json(pass_higher_path)
    fail_higher = _load_json(fail_higher_path)
    pass_crosscheck = _load_json(pass_crosscheck_path)
    fail_crosscheck = _load_json(fail_crosscheck_path)
    state_region_audit = _load_json(state_region_audit_path)
    bisection = [
        (
            expected_pivots,
            expected_certified,
            RESULTS_DIR / filename,
            _load_json(RESULTS_DIR / filename),
        )
        for expected_pivots, filename, expected_certified
        in BISECTION_RESULTS
    ]

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
    pass_factor = splu(
        residual._scaled_central_prefix(problem, PASS_PIVOTS),
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    fail_factor = splu(
        residual._scaled_central_prefix(problem, FAIL_PIVOTS),
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    pass_lower_factor = pass_factor.L.tocsc()
    pass_lower_factor.sort_indices()
    fail_lower_factor = fail_factor.L.tocsc()
    fail_lower_factor.sort_indices()
    pass_diagonal = np.asarray(pass_factor.U.diagonal(), dtype=float)
    fail_diagonal = np.asarray(fail_factor.U.diagonal(), dtype=float)
    pass_identity = np.arange(
        PASS_PIVOTS,
        dtype=pass_factor.perm_r.dtype,
    )
    fail_identity = np.arange(
        FAIL_PIVOTS,
        dtype=fail_factor.perm_r.dtype,
    )

    inventory = preparation["matrix_inventory"]
    boundaries = state_base._block_boundaries(inventory)
    original_indices = np.asarray(
        problem.order[:FAIL_PIVOTS],
        dtype=np.int64,
    )
    added_original_index = int(original_indices[ADDED_PIVOT])
    added_block = state_base._classify_original(
        added_original_index,
        boundaries,
    )

    arithmetic = prefix.DirectedDecimal(100)
    pass_row_sums, pass_column_sums = (
        region_base._inverse_majorant_vectors(
            pass_lower_factor,
            arithmetic,
        )
    )
    fail_row_sums, fail_column_sums = (
        region_base._inverse_majorant_vectors(
            fail_lower_factor,
            arithmetic,
        )
    )
    pass_infinity_pivot = max(
        range(PASS_PIVOTS),
        key=pass_row_sums.__getitem__,
    )
    pass_one_pivot = max(
        range(PASS_PIVOTS),
        key=pass_column_sums.__getitem__,
    )
    fail_infinity_pivot = max(
        range(FAIL_PIVOTS),
        key=fail_row_sums.__getitem__,
    )
    fail_one_pivot = max(
        range(FAIL_PIVOTS),
        key=fail_column_sums.__getitem__,
    )

    fail_lower_csr = fail_lower_factor.tocsr()
    fail_lower_csr.sort_indices()
    row_start = int(fail_lower_csr.indptr[ADDED_PIVOT])
    row_stop = int(fail_lower_csr.indptr[ADDED_PIVOT + 1])
    added_row_entries = []
    for pointer in range(row_start, row_stop):
        column = int(fail_lower_csr.indices[pointer])
        if column >= ADDED_PIVOT:
            continue
        coefficient = Decimal.from_float(
            float(fail_lower_csr.data[pointer])
        )
        column_original_index = int(original_indices[column])
        added_row_entries.append(
            {
                "column_pivot": column,
                "column_original_index": column_original_index,
                "column_block": state_base._classify_original(
                    column_original_index,
                    boundaries,
                ),
                "coefficient_decimal": str(coefficient),
                "absolute_coefficient_decimal": str(abs(coefficient)),
                "predecessor_inverse_row_majorant_upper_decimal": str(
                    fail_row_sums[column]
                ),
                "row_recurrence_contribution_upper_decimal": str(
                    arithmetic.upper.multiply(
                        abs(coefficient),
                        fail_row_sums[column],
                    )
                ),
            }
        )
    added_row_entries.sort(
        key=lambda record: Decimal(
            record["row_recurrence_contribution_upper_decimal"]
        ),
        reverse=True,
    )

    pass_certificate = pass_higher["certificate"]
    fail_certificate = fail_higher["certificate"]
    reconstructed_pass_hash = prefix._sha256_arrays(
        pass_lower_factor.indptr,
        pass_lower_factor.indices,
        pass_lower_factor.data,
        pass_diagonal,
    )
    reconstructed_fail_hash = prefix._sha256_arrays(
        fail_lower_factor.indptr,
        fail_lower_factor.indices,
        fail_lower_factor.data,
        fail_diagonal,
    )
    bisection_rows = [
        {
            "maximum_pivots": expected_pivots,
            "last_included_pivot": expected_pivots - 1,
            "expected_certified": expected_certified,
            "observed_certified": result["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ],
            "status": result["status"],
            "bound_to_minimum_diagonal_ratio_upper_decimal": result[
                "certificate"
            ]["transformed_bound_to_minimum_diagonal_upper_decimal"],
            "result_path": str(path).replace("\\", "/"),
            "result_sha256": state_base._sha256(path),
        }
        for expected_pivots, expected_certified, path, result in bisection
    ]
    bisection_ratios = [
        Decimal(
            row["bound_to_minimum_diagonal_ratio_upper_decimal"]
        )
        for row in bisection_rows
    ]
    bisection_contract_cores = [
        _contract_core(result)
        for _, _, _, result in bisection
    ]
    fail_crosscheck_other_checks = {
        key: value
        for key, value in fail_crosscheck["checks"].items()
        if key != "both_residual_certificates_close"
    }

    checks = {
        "all_bisection_constructions_validate": all(
            result.get("all_current_stage_checks_pass") is True
            for _, _, _, result in bisection
        ),
        "bisection_decisions_match": all(
            row["expected_certified"] == row["observed_certified"]
            for row in bisection_rows
        ),
        "bisection_ratios_are_nondecreasing": all(
            left <= right
            for left, right in zip(
                bisection_ratios,
                bisection_ratios[1:],
            )
        ),
        "bisection_provenance_core_is_constant": all(
            core == bisection_contract_cores[0]
            for core in bisection_contract_cores[1:]
        ),
        "adjacent_dimensions_are_exact": (
            pass_certificate["dimension"] == PASS_PIVOTS
            and fail_certificate["dimension"] == FAIL_PIVOTS
            and FAIL_PIVOTS == PASS_PIVOTS + 1
        ),
        "adjacent_60_digit_decisions_are_pass_then_fail": (
            pass_lower["status"]
            == "standalone_prefix_inertia_certified"
            and fail_lower["status"]
            == "standalone_route_does_not_close"
        ),
        "adjacent_100_digit_decisions_are_pass_then_fail": (
            pass_higher["status"]
            == "standalone_prefix_inertia_certified"
            and fail_higher["status"]
            == "standalone_route_does_not_close"
        ),
        "passing_precision_crosscheck_passes": (
            pass_crosscheck.get("all_checks_pass") is True
        ),
        "failing_crosscheck_has_only_expected_closure_failure": (
            fail_crosscheck.get("all_checks_pass") is False
            and fail_crosscheck["checks"][
                "both_residual_certificates_close"
            ]
            is False
            and all(fail_crosscheck_other_checks.values())
        ),
        "precision_crosscheck_hashes_match": (
            pass_crosscheck["artifacts"][
                "lower_precision_result_sha256"
            ]
            == state_base._sha256(pass_lower_path)
            and pass_crosscheck["artifacts"][
                "higher_precision_result_sha256"
            ]
            == state_base._sha256(pass_higher_path)
            and fail_crosscheck["artifacts"][
                "lower_precision_result_sha256"
            ]
            == state_base._sha256(fail_lower_path)
            and fail_crosscheck["artifacts"][
                "higher_precision_result_sha256"
            ]
            == state_base._sha256(fail_higher_path)
        ),
        "reconstructed_factors_use_identity_permutations": (
            np.array_equal(pass_factor.perm_r, pass_identity)
            and np.array_equal(pass_factor.perm_c, pass_identity)
            and np.array_equal(fail_factor.perm_r, fail_identity)
            and np.array_equal(fail_factor.perm_c, fail_identity)
        ),
        "reconstructed_factor_hashes_match": (
            reconstructed_pass_hash
            == pass_certificate["reference_factor_sha256"]
            and reconstructed_fail_hash
            == fail_certificate["reference_factor_sha256"]
        ),
        "passing_factor_is_bitwise_unchanged_leading_block": (
            np.array_equal(
                fail_diagonal[:PASS_PIVOTS],
                pass_diagonal,
            )
            and _sparse_exactly_equal(
                fail_lower_factor[:PASS_PIVOTS, :PASS_PIVOTS],
                pass_lower_factor,
            )
        ),
        "inverse_majorants_reconstructed_at_100_digits": (
            str(max(pass_row_sums))
            == pass_certificate[
                "absolute_L_inverse_infinity_norm_upper_decimal"
            ]
            and str(max(pass_column_sums))
            == pass_certificate[
                "absolute_L_inverse_one_norm_upper_decimal"
            ]
            and str(max(fail_row_sums))
            == fail_certificate[
                "absolute_L_inverse_infinity_norm_upper_decimal"
            ]
            and str(max(fail_column_sums))
            == fail_certificate[
                "absolute_L_inverse_one_norm_upper_decimal"
            ]
        ),
        "added_pivot_is_expected_edge_metric": (
            added_original_index == 21
            and added_block == "edge_metric"
            and len(added_row_entries) == 9
        ),
        "added_pivot_is_positive_and_only_sign_increment": (
            fail_diagonal[ADDED_PIVOT] > 0.0
            and fail_certificate["reference_diagonal_signs"]["negative"]
            == pass_certificate["reference_diagonal_signs"]["negative"]
            and fail_certificate["reference_diagonal_signs"]["positive"]
            == pass_certificate["reference_diagonal_signs"]["positive"] + 1
            and fail_certificate["reference_diagonal_signs"]["zero"] == 0
        ),
        "minimum_diagonal_is_unchanged": (
            pass_certificate[
                "minimum_absolute_reference_diagonal_decimal"
            ]
            == fail_certificate[
                "minimum_absolute_reference_diagonal_decimal"
            ]
        ),
        "prior_state_region_audit_supports_nested_factor": (
            state_region_audit.get("all_current_stage_checks_pass") is True
            and state_region_audit["checks"][
                "leading_factor_is_bitwise_unchanged"
            ]
            is True
        ),
        "full_and_continuum_claims_remain_false": (
            fail_higher["certification_flags"][
                "full_123816_pivot_inertia_certified"
            ]
            is False
            and fail_higher["certification_flags"][
                "continuum_spectrum_below_60_captured"
            ]
            is False
            and fail_higher["certification_flags"][
                "navier_stokes_regularity_certified"
            ]
            is False
        ),
    }
    all_checks = bool(all(checks.values()))

    artifacts = {
        "passing_precision_60_result": {
            "path": str(pass_lower_path).replace("\\", "/"),
            "sha256": state_base._sha256(pass_lower_path),
        },
        "failing_precision_60_result": {
            "path": str(fail_lower_path).replace("\\", "/"),
            "sha256": state_base._sha256(fail_lower_path),
        },
        "passing_precision_100_result": {
            "path": str(pass_higher_path).replace("\\", "/"),
            "sha256": state_base._sha256(pass_higher_path),
        },
        "failing_precision_100_result": {
            "path": str(fail_higher_path).replace("\\", "/"),
            "sha256": state_base._sha256(fail_higher_path),
        },
        "passing_precision_crosscheck": {
            "path": str(pass_crosscheck_path).replace("\\", "/"),
            "sha256": state_base._sha256(pass_crosscheck_path),
        },
        "failing_precision_crosscheck": {
            "path": str(fail_crosscheck_path).replace("\\", "/"),
            "sha256": state_base._sha256(fail_crosscheck_path),
        },
        "state_region64064_audit": {
            "path": str(state_region_audit_path).replace("\\", "/"),
            "sha256": state_base._sha256(state_region_audit_path),
        },
    }
    return {
        "kind": "hypercircle-standalone-closure-boundary64039-audit",
        "status": (
            "pass_with_minimal_certificate_obstruction"
            if all_checks
            else "fail_closed"
        ),
        "scope": (
            "Independent audit of the first closure loss of the current "
            "hash-bound standalone congruence-residual majorant. It does "
            "not assert a zero pivot, interval-family inertia at 64,040, "
            "full inertia, continuum transfer, or Navier-Stokes regularity."
        ),
        "all_current_stage_checks_pass": all_checks,
        "checks": checks,
        "artifacts": artifacts,
        "bisection_trace": bisection_rows,
        "boundary": {
            "last_certified_prefix_pivots": PASS_PIVOTS,
            "last_certified_pivot": PASS_PIVOTS - 1,
            "first_nonclosing_prefix_pivots": FAIL_PIVOTS,
            "first_nonclosing_added_pivot": ADDED_PIVOT,
            "precision_100_passing_ratio_upper_decimal": pass_certificate[
                "transformed_bound_to_minimum_diagonal_upper_decimal"
            ],
            "precision_100_failing_ratio_upper_decimal": fail_certificate[
                "transformed_bound_to_minimum_diagonal_upper_decimal"
            ],
            "ratio_jump_factor": _ratio(
                fail_certificate[
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ],
                pass_certificate[
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ],
            ),
            "monotonicity_basis": (
                "Natural-order identity-permutation LDL factors are nested "
                "bitwise; the residual-magnitude rows and positive absolute "
                "inverse recurrences only gain nonnegative terms as a pivot "
                "is appended, while the minimum absolute diagonal cannot "
                "increase. Therefore this majorant ratio is nondecreasing "
                "over the nested prefix family."
            ),
        },
        "added_pivot_profile": {
            "pivot": ADDED_PIVOT,
            "original_index": added_original_index,
            "block": added_block,
            "reference_diagonal_decimal": str(
                Decimal.from_float(float(fail_diagonal[ADDED_PIVOT]))
            ),
            "reference_diagonal_sign": 1,
            "strict_lower_row_nnz": len(added_row_entries),
            "strict_lower_entries_by_recurrence_contribution": (
                added_row_entries
            ),
            "inverse_row_majorant_upper_decimal": str(
                fail_row_sums[ADDED_PIVOT]
            ),
            "inverse_column_majorant_upper_decimal": str(
                fail_column_sums[ADDED_PIVOT]
            ),
        },
        "majorant_change": {
            "residual_infinity_norm_growth_factor": _ratio(
                fail_certificate[
                    "residual_infinity_norm_upper_decimal"
                ],
                pass_certificate[
                    "residual_infinity_norm_upper_decimal"
                ],
            ),
            "inverse_one_norm_growth_factor": _ratio(
                fail_certificate[
                    "absolute_L_inverse_one_norm_upper_decimal"
                ],
                pass_certificate[
                    "absolute_L_inverse_one_norm_upper_decimal"
                ],
            ),
            "inverse_infinity_norm_growth_factor": _ratio(
                fail_certificate[
                    "absolute_L_inverse_infinity_norm_upper_decimal"
                ],
                pass_certificate[
                    "absolute_L_inverse_infinity_norm_upper_decimal"
                ],
            ),
            "transformed_residual_growth_factor": _ratio(
                fail_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ],
                pass_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ],
            ),
            "minimum_diagonal_reduction_factor": "1",
            "passing_maximum_residual_coordinate": pass_certificate[
                "maximum_residual_entry_coordinate"
            ],
            "failing_maximum_residual_coordinate": fail_certificate[
                "maximum_residual_entry_coordinate"
            ],
            "passing_inverse_infinity_maximizer": (
                region_base._pivot_metadata(
                    pass_infinity_pivot,
                    pass_row_sums[pass_infinity_pivot],
                    original_indices,
                    boundaries,
                )
            ),
            "failing_inverse_infinity_maximizer": (
                region_base._pivot_metadata(
                    fail_infinity_pivot,
                    fail_row_sums[fail_infinity_pivot],
                    original_indices,
                    boundaries,
                )
            ),
            "passing_inverse_one_maximizer": (
                region_base._pivot_metadata(
                    pass_one_pivot,
                    pass_column_sums[pass_one_pivot],
                    original_indices,
                    boundaries,
                )
            ),
            "failing_inverse_one_maximizer": (
                region_base._pivot_metadata(
                    fail_one_pivot,
                    fail_column_sums[fail_one_pivot],
                    original_indices,
                    boundaries,
                )
            ),
            "failing_dominant_inverse_infinity_path": (
                region_base._dominant_row_path(
                    fail_lower_factor,
                    fail_row_sums,
                    fail_infinity_pivot,
                    arithmetic,
                    original_indices,
                    boundaries,
                )
            ),
            "failing_dominant_inverse_one_path": (
                region_base._dominant_column_path(
                    fail_lower_factor,
                    fail_column_sums,
                    fail_one_pivot,
                    arithmetic,
                    original_indices,
                    boundaries,
                )
            ),
        },
        "repair_gate": {
            "larger_prefix_run_admitted": False,
            "recommended_first_test": (
                "At exactly 64,040 pivots, replace the separated product "
                "||Q||_inf ||R||_inf ||Q||_1 by the directed componentwise "
                "row-sum bound max(Q R Q^T 1), where "
                "Q=(I-|L-I|)^-1 and R bounds the residual magnitude. This "
                "retains the same hash-bound binary reference and directly "
                "tests whether the global norm product is the obstruction."
            ),
            "high_precision_tail_correction_deferred": True,
        },
        "runtime": {
            "below_normal_priority_set": priority_set,
            "baseline_cpu_samples_percent": baseline,
            "baseline_cpu_mean_percent": baseline_mean,
        },
        "certification_flags": {
            "standalone_64039_inertia_certified": all_checks,
            "standalone_64040_inertia_certified": False,
            "full_123816_pivot_inertia_certified": False,
            "weighted_global_Ritz_projection_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
            "navier_stokes_regularity_certified": False,
        },
        "next_required_step": (
            "Do not enlarge the prefix. Implement and independently replay "
            "the directed componentwise transformed-residual row-sum bound "
            "at exactly 64,040 pivots and precisions 60/100. Only if that "
            "fails should a local high-precision factor-tail correction be "
            "considered."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pass-lower",
        type=Path,
        default=DEFAULT_PASS_LOWER,
    )
    parser.add_argument(
        "--fail-lower",
        type=Path,
        default=DEFAULT_FAIL_LOWER,
    )
    parser.add_argument(
        "--pass-higher",
        type=Path,
        default=DEFAULT_PASS_HIGHER,
    )
    parser.add_argument(
        "--fail-higher",
        type=Path,
        default=DEFAULT_FAIL_HIGHER,
    )
    parser.add_argument(
        "--pass-crosscheck",
        type=Path,
        default=DEFAULT_PASS_CROSSCHECK,
    )
    parser.add_argument(
        "--fail-crosscheck",
        type=Path,
        default=DEFAULT_FAIL_CROSSCHECK,
    )
    parser.add_argument(
        "--state-region-audit",
        type=Path,
        default=DEFAULT_STATE_REGION_AUDIT,
    )
    parser.add_argument("--skip-cpu-policy", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_audit(
        pass_lower_path=args.pass_lower,
        fail_lower_path=args.fail_lower,
        pass_higher_path=args.pass_higher,
        fail_higher_path=args.fail_higher,
        pass_crosscheck_path=args.pass_crosscheck,
        fail_crosscheck_path=args.fail_crosscheck,
        state_region_audit_path=args.state_region_audit,
        enforce_cpu_policy=not args.skip_cpu_policy,
    )
    prefix = _load_module(
        "closure_boundary64039_output_base",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_current_stage_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
