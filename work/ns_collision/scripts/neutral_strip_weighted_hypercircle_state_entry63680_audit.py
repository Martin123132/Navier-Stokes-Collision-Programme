#!/usr/bin/env python3
"""Audit the standalone 63,680-pivot first-state-entry certificate."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.sparse.linalg import splu


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("work/ns_collision/results")
DEFAULT_LOWER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual63680_v1.json"
)
DEFAULT_HIGHER = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual63680_p100_v1.json"
)
DEFAULT_CROSSCHECK = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual_"
    "precision_crosscheck63680_v1.json"
)
DEFAULT_REGRESSION = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual_regression_v1.json"
)
DEFAULT_PRIOR = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_standalone_residual33280_v1.json"
)
DEFAULT_SYMBOLIC_MAP = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_symbolic_transition_map123816_v1.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "neutral_strip_h006_hypercircle_state_entry63680_audit_v1.json"
)
MAXIMUM_PIVOTS = 63680
PRIOR_PIVOTS = 33280
EXPECTED_FIRST_STATE_PIVOT = 63644
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


def _decimal_ratio(numerator: Any, denominator: Any) -> str:
    with localcontext() as context:
        context.prec = 50
        return str(Decimal(str(numerator)) / Decimal(str(denominator)))


def _block_boundaries(inventory: dict[str, Any]) -> tuple[
    tuple[str, int, int],
    ...,
]:
    edge_count = int(inventory["edge_count"])
    triangle_count = int(inventory["triangle_count"])
    state_count = int(inventory["state_count"])
    return (
        ("edge_metric", 0, edge_count),
        (
            "triangle_constraint",
            edge_count,
            edge_count + triangle_count,
        ),
        (
            "state",
            edge_count + triangle_count,
            edge_count + triangle_count + state_count,
        ),
        (
            "source_triangle",
            edge_count + triangle_count + state_count,
            int(inventory["dimension"]),
        ),
    )


def _classify_original(
    original_index: int,
    boundaries: tuple[tuple[str, int, int], ...],
) -> str:
    for name, start, stop in boundaries:
        if start <= original_index < stop:
            return name
    raise RuntimeError(f"original index is outside block inventory: {original_index}")


def _sign_summary(values: np.ndarray) -> dict[str, int]:
    return {
        "negative": int(np.count_nonzero(values < 0.0)),
        "positive": int(np.count_nonzero(values > 0.0)),
        "zero": int(np.count_nonzero(values == 0.0)),
    }


def _block_profile(
    diagonal: np.ndarray,
    original_indices: np.ndarray,
    boundaries: tuple[tuple[str, int, int], ...],
    start: int,
    stop: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, _, _ in boundaries:
        positions = np.asarray(
            [
                index
                for index in range(start, stop)
                if _classify_original(
                    int(original_indices[index]),
                    boundaries,
                )
                == name
            ],
            dtype=np.int64,
        )
        values = diagonal[positions] if len(positions) else np.empty(0)
        result[name] = {
            "pivot_count": int(len(positions)),
            "signs": _sign_summary(values),
            "first_pivot": int(positions[0]) if len(positions) else None,
            "last_pivot": int(positions[-1]) if len(positions) else None,
            "minimum_absolute_reference_diagonal_decimal": (
                str(
                    min(
                        abs(Decimal.from_float(float(value)))
                        for value in values
                    )
                )
                if len(values)
                else None
            ),
        }
    return result


def _next_transition_cluster(
    symbolic_map: dict[str, Any],
) -> tuple[list[int], int]:
    pivots = sorted(
        {
            int(record["pivot"])
            for record in symbolic_map[
                "new_transitions_at_or_after_prior_boundary"
            ]
            if int(record["pivot"]) >= MAXIMUM_PIVOTS
        }
    )
    if not pivots:
        raise RuntimeError("no unprocessed symbolic transition remains")
    cluster = [pivots[0]]
    for pivot in pivots[1:]:
        if pivot - cluster[-1] > 512:
            break
        cluster.append(pivot)
    recommended = ((cluster[-1] + 64) // 64) * 64
    return cluster, recommended


def run_audit(
    lower_path: Path = DEFAULT_LOWER,
    higher_path: Path = DEFAULT_HIGHER,
    crosscheck_path: Path = DEFAULT_CROSSCHECK,
    regression_path: Path = DEFAULT_REGRESSION,
    prior_path: Path = DEFAULT_PRIOR,
    symbolic_map_path: Path = DEFAULT_SYMBOLIC_MAP,
    enforce_cpu_policy: bool = True,
) -> dict[str, Any]:
    prefix = _load_module(
        "state_entry63680_prefix_base",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )
    residual = _load_module(
        "state_entry63680_residual_base",
        "neutral_strip_weighted_hypercircle_congruence_residual_pilot.py",
    )
    lower = _load_json(lower_path)
    higher = _load_json(higher_path)
    crosscheck = _load_json(crosscheck_path)
    regression = _load_json(regression_path)
    prior = _load_json(prior_path)
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
    central_prefix = residual._scaled_central_prefix(
        problem,
        MAXIMUM_PIVOTS,
    )
    factor = splu(
        central_prefix,
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True, "Equil": False},
    )
    identity = np.arange(MAXIMUM_PIVOTS, dtype=factor.perm_r.dtype)
    identity_permutations = bool(
        np.array_equal(factor.perm_r, identity)
        and np.array_equal(factor.perm_c, identity)
    )
    lower_factor = factor.L.tocsc()
    lower_factor.sort_indices()
    diagonal = np.asarray(factor.U.diagonal(), dtype=float)
    reference_factor_sha256 = prefix._sha256_arrays(
        lower_factor.indptr,
        lower_factor.indices,
        lower_factor.data,
        diagonal,
    )

    inventory = preparation["matrix_inventory"]
    boundaries = _block_boundaries(inventory)
    original_indices = np.asarray(
        problem.order[:MAXIMUM_PIVOTS],
        dtype=np.int64,
    )
    full_block_profile = _block_profile(
        diagonal,
        original_indices,
        boundaries,
        0,
        MAXIMUM_PIVOTS,
    )
    incremental_block_profile = _block_profile(
        diagonal,
        original_indices,
        boundaries,
        PRIOR_PIVOTS,
        MAXIMUM_PIVOTS,
    )
    state_positions = np.asarray(
        [
            pivot
            for pivot in range(MAXIMUM_PIVOTS)
            if _classify_original(
                int(original_indices[pivot]),
                boundaries,
            )
            == "state"
        ],
        dtype=np.int64,
    )
    state_values = diagonal[state_positions]
    state_rows = [
        {
            "pivot": int(pivot),
            "original_index": int(original_indices[pivot]),
            "sign": -1 if diagonal[pivot] < 0.0 else 1,
            "reference_diagonal_decimal": str(
                Decimal.from_float(float(diagonal[pivot]))
            ),
            "absolute_reference_diagonal_decimal": str(
                abs(Decimal.from_float(float(diagonal[pivot])))
            ),
        }
        for pivot in state_positions
    ]
    minimum_index = int(np.argmin(np.abs(diagonal)))
    transition_cluster, recommended_target = _next_transition_cluster(
        symbolic_map
    )

    lower_certificate = lower["certificate"]
    higher_certificate = higher["certificate"]
    prior_certificate = prior["certificate"]
    cross_artifacts = crosscheck["artifacts"]
    checks = {
        "standalone_regression_admitted_state_pilot": (
            regression.get("state_entry_pilot_admitted") is True
            and regression.get("all_checks_pass") is True
        ),
        "both_standalone_certificates_close": (
            lower["status"] == "standalone_prefix_inertia_certified"
            and higher["status"] == "standalone_prefix_inertia_certified"
            and lower["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is True
            and higher["certification_flags"][
                "standalone_bounded_prefix_inertia_certified"
            ]
            is True
        ),
        "precision_crosscheck_passes": (
            crosscheck.get("all_checks_pass") is True
        ),
        "precision_crosscheck_hashes_match": (
            cross_artifacts["lower_precision_result_sha256"]
            == _sha256(lower_path)
            and cross_artifacts["higher_precision_result_sha256"]
            == _sha256(higher_path)
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
        "reconstructed_reference_uses_identity_permutations": (
            identity_permutations
        ),
        "reconstructed_reference_hash_matches_both_precisions": (
            reference_factor_sha256
            == lower_certificate["reference_factor_sha256"]
            == higher_certificate["reference_factor_sha256"]
        ),
        "reconstructed_signs_match_certificate": (
            _sign_summary(diagonal)
            == lower_certificate["reference_diagonal_signs"]
            == higher_certificate["reference_diagonal_signs"]
        ),
        "first_state_pivot_matches_symbolic_map": (
            len(state_positions) > 0
            and int(state_positions[0]) == EXPECTED_FIRST_STATE_PIVOT
            and int(
                symbolic_map["first_block_pivots_within_scan"]["state"]
            )
            == EXPECTED_FIRST_STATE_PIVOT
        ),
        "minimum_reference_diagonal_reconstructed": (
            str(abs(Decimal.from_float(float(diagonal[minimum_index]))))
            == lower_certificate[
                "minimum_absolute_reference_diagonal_decimal"
            ]
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
    ratio = lower_certificate[
        "transformed_bound_to_minimum_diagonal_upper_decimal"
    ]
    with localcontext() as context:
        context.prec = 50
        safety_factor = str(Decimal(1) / Decimal(ratio))

    artifacts = {
        "precision_60_result": {
            "path": str(lower_path).replace("\\", "/"),
            "sha256": _sha256(lower_path),
        },
        "precision_100_result": {
            "path": str(higher_path).replace("\\", "/"),
            "sha256": _sha256(higher_path),
        },
        "precision_crosscheck": {
            "path": str(crosscheck_path).replace("\\", "/"),
            "sha256": _sha256(crosscheck_path),
        },
        "standalone_regression": {
            "path": str(regression_path).replace("\\", "/"),
            "sha256": _sha256(regression_path),
        },
        "prior_33280_result": {
            "path": str(prior_path).replace("\\", "/"),
            "sha256": _sha256(prior_path),
        },
        "full_symbolic_map": {
            "path": str(symbolic_map_path).replace("\\", "/"),
            "sha256": _sha256(symbolic_map_path),
        },
    }
    return {
        "kind": "hypercircle-standalone-state-entry63680-audit",
        "status": "pass" if all_checks else "fail_closed",
        "scope": (
            "Independent structural audit of the hash-bound standalone "
            "residual certificate through the first state pivots. It "
            "certifies no later prefix, full inertia, continuum transfer, "
            "or Navier-Stokes statement."
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
                _classify_original(
                    int(original_indices[minimum_index]),
                    boundaries,
                )
            ),
            "transformed_bound_to_minimum_diagonal_upper_decimal": ratio,
            "certified_safety_factor_lower": safety_factor,
            "reference_L_nnz": int(lower_certificate["reference_L_nnz"]),
            "reference_factor_sha256": reference_factor_sha256,
            "standalone_contract_sha256": lower[
                "standalone_contract"
            ]["contract_sha256"],
        },
        "full_prefix_block_profile": full_block_profile,
        "incremental_33280_63679_block_profile": incremental_block_profile,
        "state_entry_profile": {
            "first_state_pivot": int(state_positions[0]),
            "last_state_pivot_within_prefix": int(state_positions[-1]),
            "state_pivot_count": int(len(state_positions)),
            "state_signs": _sign_summary(state_values),
            "minimum_absolute_state_diagonal_decimal": str(
                min(
                    abs(Decimal.from_float(float(value)))
                    for value in state_values
                )
            ),
            "pivots": state_rows,
        },
        "risk_change_from_33280": {
            "minimum_diagonal_reduction_factor": _decimal_ratio(
                prior_certificate[
                    "minimum_absolute_reference_diagonal_decimal"
                ],
                lower_certificate[
                    "minimum_absolute_reference_diagonal_decimal"
                ],
            ),
            "inverse_one_norm_growth_factor": _decimal_ratio(
                lower_certificate[
                    "absolute_L_inverse_one_norm_upper_decimal"
                ],
                prior_certificate[
                    "absolute_L_inverse_one_norm_upper_decimal"
                ],
            ),
            "inverse_infinity_norm_growth_factor": _decimal_ratio(
                lower_certificate[
                    "absolute_L_inverse_infinity_norm_upper_decimal"
                ],
                prior_certificate[
                    "absolute_L_inverse_infinity_norm_upper_decimal"
                ],
            ),
            "transformed_residual_growth_factor": _decimal_ratio(
                lower_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ],
                prior_certificate[
                    "transformed_residual_two_norm_upper_decimal"
                ],
            ),
            "bound_to_diagonal_ratio_growth_factor": _decimal_ratio(
                ratio,
                prior_certificate[
                    "transformed_bound_to_minimum_diagonal_upper_decimal"
                ],
            ),
            "interpretation": (
                "State entry remains certified, but inverse-majorant growth "
                "is now the dominant risk and requires another bounded "
                "state-region sample before any long extrapolation."
            ),
        },
        "next_boundary": {
            "next_transition_cluster_pivots": transition_cluster,
            "recommended_bounded_pivot_count": recommended_target,
            "full_run_admitted": False,
        },
        "runtime": {
            "below_normal_priority_set": priority_set,
            "baseline_cpu_samples_percent": baseline,
            "baseline_cpu_mean_percent": baseline_mean,
        },
        "certification_flags": {
            "standalone_63680_inertia_certified": all_checks,
            "first_state_entry_certified": all_checks,
            "full_123816_pivot_inertia_certified": False,
            "weighted_global_Ritz_projection_constant_certified": False,
            "continuum_spectrum_below_60_captured": False,
            "navier_stokes_regularity_certified": False,
        },
        "next_required_step": (
            f"Run only the bounded standalone residual prefix "
            f"{recommended_target} at precisions 60 and 100 to cross the "
            "next compact state-region transition cluster. Do not launch "
            "full inertia or a continuum stage."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower", type=Path, default=DEFAULT_LOWER)
    parser.add_argument("--higher", type=Path, default=DEFAULT_HIGHER)
    parser.add_argument("--crosscheck", type=Path, default=DEFAULT_CROSSCHECK)
    parser.add_argument("--regression", type=Path, default=DEFAULT_REGRESSION)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
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
        regression_path=args.regression,
        prior_path=args.prior,
        symbolic_map_path=args.symbolic_map,
        enforce_cpu_policy=not args.skip_cpu_policy,
    )
    prefix = _load_module(
        "state_entry63680_output_base",
        "neutral_strip_weighted_hypercircle_directed_ldl_prefix_audit.py",
    )
    prefix._atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_current_stage_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
