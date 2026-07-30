#!/usr/bin/env python3
"""Audit the componentwise certificate extension from 64,064 to 64,128."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("work/ns_collision/results")
DEFAULT_COMPLETE_RESULT = RESULTS_DIR / (
    "neutral_strip_h006_positive_exponential_complete_assembly_v1.json"
)
DEFAULT_MATRICES = RESULTS_DIR / (
    "neutral_strip_h006_positive_exponential_assembly_matrices_v1.npz"
)
DEFAULT_GAUSSIAN_RESULT = RESULTS_DIR / (
    "neutral_strip_h006_gaussian_assembly_interval_audit_v1.json"
)
DEFAULT_GAUSSIAN_CHECKPOINT = RESULTS_DIR / (
    "neutral_strip_h006_gaussian_assembly_interval_checkpoint_v1.npz"
)
DEFAULT_PRIOR_COMPONENTWISE = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64064_p100_v1.json"
)
DEFAULT_PRIOR_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_componentwise_residual_"
    "precision_crosscheck64064_v1.json"
)
DEFAULT_PRIOR_AUDIT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "componentwise_state_region64064_audit_v1.json"
)
DEFAULT_SEPARATED_LOWER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual64128_v1.json"
)
DEFAULT_SEPARATED_HIGHER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual64128_p100_v1.json"
)
DEFAULT_COMPONENTWISE_LOWER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64128_v1.json"
)
DEFAULT_COMPONENTWISE_HIGHER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "standalone_componentwise_residual64128_p100_v1.json"
)
DEFAULT_COMPONENTWISE_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_componentwise_residual_"
    "precision_crosscheck64128_v1.json"
)
DEFAULT_SYMBOLIC_MAP = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_symbolic_transition_map123816_v1.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_"
    "componentwise_growth64128_audit_v1.json"
)
PRIOR_PIVOTS = 64064
TARGET_PIVOTS = 64128
NEXT_BOUNDED_TARGET = 64256
EXPECTED_NEXT_SYMBOLIC_TRANSITION = 76921
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


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path).replace("\\", "/"),
        "sha256": _sha256(path),
    }


def run_audit(
    complete_result_path: Path = DEFAULT_COMPLETE_RESULT,
    matrices_path: Path = DEFAULT_MATRICES,
    gaussian_result_path: Path = DEFAULT_GAUSSIAN_RESULT,
    gaussian_checkpoint_path: Path = DEFAULT_GAUSSIAN_CHECKPOINT,
    prior_componentwise_path: Path = DEFAULT_PRIOR_COMPONENTWISE,
    prior_crosscheck_path: Path = DEFAULT_PRIOR_CROSSCHECK,
    prior_audit_path: Path = DEFAULT_PRIOR_AUDIT,
    separated_lower_path: Path = DEFAULT_SEPARATED_LOWER,
    separated_higher_path: Path = DEFAULT_SEPARATED_HIGHER,
    componentwise_lower_path: Path = DEFAULT_COMPONENTWISE_LOWER,
    componentwise_higher_path: Path = DEFAULT_COMPONENTWISE_HIGHER,
    componentwise_crosscheck_path: Path = (
        DEFAULT_COMPONENTWISE_CROSSCHECK
    ),
    symbolic_map_path: Path = DEFAULT_SYMBOLIC_MAP,
    enforce_cpu_policy: bool = True,
) -> dict[str, Any]:
    prefix = _load_module(
        "componentwise_growth64128_prefix_base",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )
    residual = _load_module(
        "componentwise_growth64128_residual_base",
        "neutral_strip_weighted_hypercircle_congruence_residual_pilot.py",
    )
    standalone = _load_module(
        "componentwise_growth64128_standalone_base",
        "neutral_strip_weighted_hypercircle_standalone_residual.py",
    )
    state_base = _load_module(
        "componentwise_growth64128_state_base",
        "neutral_strip_weighted_hypercircle_state_entry63680_audit.py",
    )

    priority_set = prefix._set_below_normal_priority()
    baseline = (
        prefix._sample_cpu_baseline(5) if enforce_cpu_policy else []
    )
    baseline_mean = sum(baseline) / len(baseline) if baseline else None
    if (
        baseline_mean is not None
        and baseline_mean > DAYTIME_BASELINE_CPU_LIMIT
    ):
        raise RuntimeError(
            "daytime baseline CPU exceeds the one-worker launch limit: "
            f"{baseline_mean:.3f}%"
        )

    prior_componentwise = _load_json(prior_componentwise_path)
    prior_crosscheck = _load_json(prior_crosscheck_path)
    prior_audit = _load_json(prior_audit_path)
    separated_lower = _load_json(separated_lower_path)
    separated_higher = _load_json(separated_higher_path)
    componentwise_lower = _load_json(componentwise_lower_path)
    componentwise_higher = _load_json(componentwise_higher_path)
    componentwise_crosscheck = _load_json(
        componentwise_crosscheck_path
    )
    symbolic_map = _load_json(symbolic_map_path)

    problem, preparation = prefix._prepare_production_problem(
        complete_result_path,
        matrices_path,
        gaussian_result_path,
        gaussian_checkpoint_path,
    )
    prior_factor = splu(
        residual._scaled_central_prefix(problem, PRIOR_PIVOTS),
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    target_factor = splu(
        residual._scaled_central_prefix(problem, TARGET_PIVOTS),
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    prior_lower = prior_factor.L.tocsc()
    target_lower = target_factor.L.tocsc()
    prior_lower.sort_indices()
    target_lower.sort_indices()
    prior_diagonal = np.asarray(prior_factor.U.diagonal(), dtype=float)
    target_diagonal = np.asarray(target_factor.U.diagonal(), dtype=float)
    prior_identity = np.arange(
        PRIOR_PIVOTS,
        dtype=prior_factor.perm_r.dtype,
    )
    target_identity = np.arange(
        TARGET_PIVOTS,
        dtype=target_factor.perm_r.dtype,
    )
    prior_original = np.asarray(
        problem.order[:PRIOR_PIVOTS],
        dtype=np.int64,
    )
    target_original = np.asarray(
        problem.order[:TARGET_PIVOTS],
        dtype=np.int64,
    )
    prior_center = problem.center[prior_original, :][:, prior_original]
    target_center = problem.center[target_original, :][:, target_original]
    prior_radius = problem.radius[prior_original, :][:, prior_original]
    target_radius = problem.radius[target_original, :][:, target_original]
    boundaries = state_base._block_boundaries(
        preparation["matrix_inventory"]
    )

    reconstructed_prior_contract = standalone._build_standalone_contract(
        problem,
        preparation,
        complete_result_path=complete_result_path,
        matrices_path=matrices_path,
        gaussian_result_path=gaussian_result_path,
        gaussian_checkpoint_path=gaussian_checkpoint_path,
        maximum_pivots=PRIOR_PIVOTS,
        prefix_module=prefix,
    )
    reconstructed_target_contract = standalone._build_standalone_contract(
        problem,
        preparation,
        complete_result_path=complete_result_path,
        matrices_path=matrices_path,
        gaussian_result_path=gaussian_result_path,
        gaussian_checkpoint_path=gaussian_checkpoint_path,
        maximum_pivots=TARGET_PIVOTS,
        prefix_module=prefix,
    )
    reconstructed_prior_factor_hash = prefix._sha256_arrays(
        prior_lower.indptr,
        prior_lower.indices,
        prior_lower.data,
        prior_diagonal,
    )
    reconstructed_target_factor_hash = prefix._sha256_arrays(
        target_lower.indptr,
        target_lower.indices,
        target_lower.data,
        target_diagonal,
    )

    target_results = (
        separated_lower,
        separated_higher,
        componentwise_lower,
        componentwise_higher,
    )
    prior_certificate = prior_componentwise["certificate"]
    target_certificate = componentwise_higher["certificate"]
    separated_certificate = separated_higher["certificate"]
    flat_metric_keys = (
        "minimum_absolute_reference_diagonal_decimal",
        "minimum_absolute_reference_diagonal_index",
        "maximum_residual_entry_upper_decimal",
        "maximum_residual_entry_coordinate",
        "residual_infinity_norm_upper_decimal",
        "absolute_L_inverse_infinity_norm_upper_decimal",
        "absolute_L_inverse_one_norm_upper_decimal",
        "separated_transformed_residual_two_norm_upper_decimal",
        "separated_bound_to_minimum_diagonal_upper_decimal",
        "componentwise_improvement_factor_lower_decimal",
        "componentwise_maximum_row_sum_pivot",
        "componentwise_q_transpose_one_maximum_pivot",
        "componentwise_residual_propagation_maximum_pivot",
        "transformed_residual_two_norm_upper_decimal",
        "transformed_bound_to_minimum_diagonal_upper_decimal",
    )
    transition_pivots = sorted(
        {
            int(record["pivot"])
            for record in symbolic_map[
                "new_transitions_at_or_after_prior_boundary"
            ]
            if int(record["pivot"]) >= PRIOR_PIVOTS
        }
    )
    next_symbolic_transition = transition_pivots[0]
    target_crosscheck_hashes_match = bool(
        componentwise_crosscheck["artifacts"][
            "lower_precision_result_sha256"
        ]
        == _sha256(componentwise_lower_path)
        and componentwise_crosscheck["artifacts"][
            "higher_precision_result_sha256"
        ]
        == _sha256(componentwise_higher_path)
    )
    componentwise_separated_hashes_match = bool(
        componentwise_lower["artifacts"][
            "separated_residual_result"
        ]["sha256"]
        == _sha256(separated_lower_path)
        and componentwise_higher["artifacts"][
            "separated_residual_result"
        ]["sha256"]
        == _sha256(separated_higher_path)
    )
    checks = {
        "prior_componentwise_checkpoint_passes": (
            prior_audit.get("all_current_stage_checks_pass") is True
            and prior_componentwise["status"]
            == "standalone_prefix_inertia_certified"
            and prior_crosscheck.get("all_checks_pass") is True
        ),
        "all_target_constructions_validate": all(
            result.get("all_current_stage_checks_pass") is True
            for result in target_results
        ),
        "separated_target_fails_only_the_closure_gate": (
            separated_lower["status"] == "standalone_route_does_not_close"
            and separated_higher["status"]
            == "standalone_route_does_not_close"
            and separated_lower["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is False
            and separated_higher["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is False
        ),
        "componentwise_target_closes_at_both_precisions": (
            componentwise_lower["status"]
            == "standalone_prefix_inertia_certified"
            and componentwise_higher["status"]
            == "standalone_prefix_inertia_certified"
            and componentwise_lower["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is True
            and componentwise_higher["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is True
        ),
        "target_precision_crosscheck_passes": (
            componentwise_crosscheck.get("all_checks_pass") is True
            and target_crosscheck_hashes_match
        ),
        "componentwise_results_bind_the_separated_replays": (
            componentwise_separated_hashes_match
            and componentwise_lower["checks"][
                "separated_reference_and_residual_reproduced"
            ]
            is True
            and componentwise_higher["checks"][
                "separated_reference_and_residual_reproduced"
            ]
            is True
            and componentwise_lower["checks"][
                "separated_bound_reproduced"
            ]
            is True
            and componentwise_higher["checks"][
                "separated_bound_reproduced"
            ]
            is True
        ),
        "source_and_preparation_provenance_is_constant": all(
            _contract_core(result) == _contract_core(prior_componentwise)
            for result in target_results
        ),
        "stored_contracts_reconstruct_exactly": (
            reconstructed_prior_contract
            == prior_componentwise["standalone_contract"]
            and all(
                reconstructed_target_contract
                == result["standalone_contract"]
                for result in target_results
            )
        ),
        "ordered_source_prefixes_are_bitwise_nested": (
            np.array_equal(
                target_original[:PRIOR_PIVOTS],
                prior_original,
            )
            and _sparse_exactly_equal(
                target_center[:PRIOR_PIVOTS, :PRIOR_PIVOTS],
                prior_center,
            )
            and _sparse_exactly_equal(
                target_radius[:PRIOR_PIVOTS, :PRIOR_PIVOTS],
                prior_radius,
            )
        ),
        "both_factors_use_identity_permutations": (
            np.array_equal(prior_factor.perm_r, prior_identity)
            and np.array_equal(prior_factor.perm_c, prior_identity)
            and np.array_equal(target_factor.perm_r, target_identity)
            and np.array_equal(target_factor.perm_c, target_identity)
        ),
        "reconstructed_factor_hashes_match": (
            reconstructed_prior_factor_hash
            == prior_certificate["reference_factor_sha256"]
            and all(
                reconstructed_target_factor_hash
                == result["certificate"]["reference_factor_sha256"]
                for result in target_results
            )
        ),
        "leading_reference_factor_is_bitwise_unchanged": (
            np.array_equal(
                target_diagonal[:PRIOR_PIVOTS],
                prior_diagonal,
            )
            and _sparse_exactly_equal(
                target_lower[:PRIOR_PIVOTS, :PRIOR_PIVOTS],
                prior_lower,
            )
        ),
        "reconstructed_signs_match_certificates": (
            state_base._sign_summary(prior_diagonal)
            == prior_certificate["reference_diagonal_signs"]
            and state_base._sign_summary(target_diagonal)
            == target_certificate["reference_diagonal_signs"]
            == separated_certificate["reference_diagonal_signs"]
        ),
        "extension_adds_exactly_64_negative_pivots": (
            int(np.count_nonzero(target_diagonal[PRIOR_PIVOTS:] < 0.0))
            == TARGET_PIVOTS - PRIOR_PIVOTS
            and int(np.count_nonzero(target_diagonal[PRIOR_PIVOTS:] > 0.0))
            == 0
            and bool(np.all(target_diagonal[PRIOR_PIVOTS:] != 0.0))
        ),
        "componentwise_control_metrics_are_exactly_flat": all(
            target_certificate[key] == prior_certificate[key]
            for key in flat_metric_keys
        ),
        "no_symbolic_transition_is_crossed": (
            next_symbolic_transition
            == EXPECTED_NEXT_SYMBOLIC_TRANSITION
            and TARGET_PIVOTS < next_symbolic_transition
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
            for result in (
                componentwise_lower,
                componentwise_higher,
            )
        ),
    }
    all_checks = bool(all(checks.values()))
    target_ratio = target_certificate[
        "transformed_bound_to_minimum_diagonal_upper_decimal"
    ]

    return {
        "kind": (
            "hypercircle-standalone-componentwise-growth64128-audit"
        ),
        "status": (
            "componentwise_prefix64128_certified"
            if all_checks
            else "fail_closed"
        ),
        "scope": (
            "Audit of the hash-bound componentwise extension from 64,064 "
            "through 64,128 pivots. It certifies no later prefix, full "
            "inertia, continuum transfer, or Navier-Stokes regularity "
            "statement."
        ),
        "all_current_stage_checks_pass": all_checks,
        "checks": checks,
        "artifacts": {
            "prior_componentwise_precision_100": _artifact(
                prior_componentwise_path
            ),
            "prior_componentwise_crosscheck": _artifact(
                prior_crosscheck_path
            ),
            "prior_componentwise_audit": _artifact(prior_audit_path),
            "target_separated_precision_60": _artifact(
                separated_lower_path
            ),
            "target_separated_precision_100": _artifact(
                separated_higher_path
            ),
            "target_componentwise_precision_60": _artifact(
                componentwise_lower_path
            ),
            "target_componentwise_precision_100": _artifact(
                componentwise_higher_path
            ),
            "target_componentwise_crosscheck": _artifact(
                componentwise_crosscheck_path
            ),
            "symbolic_transition_map": _artifact(symbolic_map_path),
        },
        "extension": {
            "prior_maximum_pivots": PRIOR_PIVOTS,
            "target_maximum_pivots": TARGET_PIVOTS,
            "last_certified_pivot": TARGET_PIVOTS - 1,
            "added_pivot_count": TARGET_PIVOTS - PRIOR_PIVOTS,
            "added_reference_signs": state_base._sign_summary(
                target_diagonal[PRIOR_PIVOTS:]
            ),
            "reference_signs": target_certificate[
                "reference_diagonal_signs"
            ],
            "reference_L_nnz_growth": (
                int(target_lower.nnz) - int(prior_lower.nnz)
            ),
            "ordered_center_nnz_growth": (
                int(target_center.nnz) - int(prior_center.nnz)
            ),
            "ordered_radius_nnz_growth": (
                int(target_radius.nnz) - int(prior_radius.nnz)
            ),
            "incremental_block_profile": state_base._block_profile(
                target_diagonal,
                target_original,
                boundaries,
                PRIOR_PIVOTS,
                TARGET_PIVOTS,
            ),
        },
        "certificate": {
            "minimum_absolute_reference_diagonal_decimal": (
                target_certificate[
                    "minimum_absolute_reference_diagonal_decimal"
                ]
            ),
            "minimum_absolute_reference_diagonal_index": (
                target_certificate[
                    "minimum_absolute_reference_diagonal_index"
                ]
            ),
            "componentwise_bound_upper_decimal": target_certificate[
                "transformed_residual_two_norm_upper_decimal"
            ],
            "bound_to_minimum_diagonal_ratio_upper_decimal": target_ratio,
            "safety_factor_lower_decimal": _ratio(1, target_ratio),
            "improvement_over_separated_bound_lower_decimal": (
                target_certificate[
                    "componentwise_improvement_factor_lower_decimal"
                ]
            ),
            "componentwise_bound_growth_from_64064": _ratio(
                target_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ],
                prior_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ],
            ),
            "controlling_pivots": {
                "componentwise_maximum_row_sum": target_certificate[
                    "componentwise_maximum_row_sum_pivot"
                ],
                "residual_propagation_maximum": target_certificate[
                    "componentwise_residual_propagation_maximum_pivot"
                ],
                "q_transpose_one_maximum": target_certificate[
                    "componentwise_q_transpose_one_maximum_pivot"
                ],
            },
        },
        "next_boundary": {
            "recommended_bounded_pivot_count": NEXT_BOUNDED_TARGET,
            "increment_from_current": (
                NEXT_BOUNDED_TARGET - TARGET_PIVOTS
            ),
            "selection_rule": (
                "Double the last local increment after an exact "
                "precision-100 control-metric plateau, while remaining "
                "strictly before the next symbolic transition."
            ),
            "next_symbolic_transition_pivot": next_symbolic_transition,
            "crosses_new_symbolic_transition": False,
            "full_run_admitted": False,
        },
        "runtime": {
            "below_normal_priority_set": priority_set,
            "baseline_cpu_samples_percent": baseline,
            "baseline_cpu_mean_percent": baseline_mean,
        },
        "certification_flags": {
            "standalone_componentwise_64128_inertia_certified": (
                all_checks
            ),
            "standalone_componentwise_64256_inertia_certified": False,
            "full_123816_pivot_inertia_certified": False,
            "weighted_global_Ritz_projection_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
            "navier_stokes_regularity_certified": False,
        },
        "next_required_step": (
            "Test exactly 64,256 pivots and no farther: first produce the "
            "separated precision-60 reference, then run the componentwise "
            "precision-60 certificate. Replay both at precision 100 only "
            "if the componentwise ratio remains strictly below one."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complete-result",
        type=Path,
        default=DEFAULT_COMPLETE_RESULT,
    )
    parser.add_argument("--matrices", type=Path, default=DEFAULT_MATRICES)
    parser.add_argument(
        "--gaussian-result",
        type=Path,
        default=DEFAULT_GAUSSIAN_RESULT,
    )
    parser.add_argument(
        "--gaussian-checkpoint",
        type=Path,
        default=DEFAULT_GAUSSIAN_CHECKPOINT,
    )
    parser.add_argument(
        "--prior-componentwise",
        type=Path,
        default=DEFAULT_PRIOR_COMPONENTWISE,
    )
    parser.add_argument(
        "--prior-crosscheck",
        type=Path,
        default=DEFAULT_PRIOR_CROSSCHECK,
    )
    parser.add_argument(
        "--prior-audit",
        type=Path,
        default=DEFAULT_PRIOR_AUDIT,
    )
    parser.add_argument(
        "--separated-lower",
        type=Path,
        default=DEFAULT_SEPARATED_LOWER,
    )
    parser.add_argument(
        "--separated-higher",
        type=Path,
        default=DEFAULT_SEPARATED_HIGHER,
    )
    parser.add_argument(
        "--componentwise-lower",
        type=Path,
        default=DEFAULT_COMPONENTWISE_LOWER,
    )
    parser.add_argument(
        "--componentwise-higher",
        type=Path,
        default=DEFAULT_COMPONENTWISE_HIGHER,
    )
    parser.add_argument(
        "--componentwise-crosscheck",
        type=Path,
        default=DEFAULT_COMPONENTWISE_CROSSCHECK,
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
        complete_result_path=args.complete_result,
        matrices_path=args.matrices,
        gaussian_result_path=args.gaussian_result,
        gaussian_checkpoint_path=args.gaussian_checkpoint,
        prior_componentwise_path=args.prior_componentwise,
        prior_crosscheck_path=args.prior_crosscheck,
        prior_audit_path=args.prior_audit,
        separated_lower_path=args.separated_lower,
        separated_higher_path=args.separated_higher,
        componentwise_lower_path=args.componentwise_lower,
        componentwise_higher_path=args.componentwise_higher,
        componentwise_crosscheck_path=args.componentwise_crosscheck,
        symbolic_map_path=args.symbolic_map,
        enforce_cpu_policy=not args.skip_cpu_policy,
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_current_stage_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
