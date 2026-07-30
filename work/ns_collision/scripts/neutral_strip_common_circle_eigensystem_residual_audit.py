from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix


def _load_certificate_module():
    script = Path(__file__).resolve().with_name(
        "neutral_strip_common_circle_source_time_slab_certificate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "common_circle_time_slab_for_eigen_residual",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _up(value: float) -> float:
    return float(np.nextafter(float(value), math.inf))


def _gamma(operation_count: int) -> float:
    epsilon = np.finfo(float).eps
    product = operation_count * epsilon
    if product >= 0.01:
        raise RuntimeError("roundoff operation count is too large")
    return _up(product / (1.0 - product))


def _up_array(values: np.ndarray, operation_count: int = 8) -> np.ndarray:
    nonnegative = np.maximum(np.asarray(values, dtype=float), 0.0)
    inflated = nonnegative * (1.0 + _gamma(operation_count))
    return np.nextafter(inflated, math.inf)


def _nonnegative_dense_product_upper(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    operation_count = 2 * left.shape[1] + 8
    gamma = _gamma(operation_count)
    product = np.asarray(left, dtype=float) @ np.asarray(right, dtype=float)
    return np.nextafter(
        np.maximum(product, 0.0) / (1.0 - gamma),
        math.inf,
    )


def _dense_product_roundoff_upper(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    absolute_product = _nonnegative_dense_product_upper(
        np.abs(left),
        np.abs(right),
    )
    return _up_array(
        _gamma(2 * left.shape[1] + 8) * absolute_product,
        8,
    )


def _nonnegative_sparse_product_upper(
    left: csr_matrix,
    right: np.ndarray,
) -> np.ndarray:
    row_counts = np.diff(left.indptr)
    maximum_terms = int(np.max(row_counts))
    gamma = _gamma(2 * maximum_terms + 8)
    product = abs(left) @ np.abs(right)
    return np.nextafter(
        np.maximum(np.asarray(product), 0.0) / (1.0 - gamma),
        math.inf,
    )


def _sparse_action_with_error(
    matrix: csr_matrix,
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row_counts = np.diff(matrix.indptr)
    maximum_terms = int(np.max(row_counts))
    central = np.asarray(matrix @ vectors)
    absolute_product = _nonnegative_sparse_product_upper(matrix, vectors)
    error = _up_array(
        _gamma(2 * maximum_terms + 8) * absolute_product,
        8,
    )
    return central, error


def _dense_action_with_error(
    matrix: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    central = np.asarray(matrix) @ np.asarray(vectors)
    return central, _dense_product_roundoff_upper(matrix, vectors)


def _scaled_columns_with_error(
    values: np.ndarray,
    action: np.ndarray,
    action_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    central = action * values[None, :]
    error = _up_array(
        np.abs(values)[None, :] * action_error
        + _gamma(4) * np.abs(central),
        8,
    )
    return central, error


def _difference_with_error(
    first: np.ndarray,
    first_error: np.ndarray,
    second: np.ndarray,
    second_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    central = first - second
    error = _up_array(
        first_error
        + second_error
        + _gamma(4) * (np.abs(first) + np.abs(second)),
        8,
    )
    return central, error


def _column_norms_upper(entry_magnitude_upper: np.ndarray) -> np.ndarray:
    rows = entry_magnitude_upper.shape[0]
    squared = entry_magnitude_upper * entry_magnitude_upper
    sums = np.sum(squared, axis=0)
    sums = np.nextafter(
        sums / (1.0 - _gamma(2 * rows + 8)),
        math.inf,
    )
    return np.nextafter(np.sqrt(np.maximum(sums, 0.0)), math.inf)


def _frobenius_upper(entry_magnitude_upper: np.ndarray) -> float:
    flat = np.asarray(entry_magnitude_upper).ravel()
    squared = flat * flat
    total = float(np.sum(squared))
    total = _up(total / (1.0 - _gamma(2 * flat.size + 8)))
    return _up(math.sqrt(max(total, 0.0)))


def _orthogonality_audit(
    vectors: np.ndarray,
    mass_action: np.ndarray,
    mass_action_error: np.ndarray,
) -> dict[str, object]:
    central = vectors.T @ mass_action
    product_error = _dense_product_roundoff_upper(vectors.T, mass_action)
    propagated_error = _nonnegative_dense_product_upper(
        np.abs(vectors.T),
        mass_action_error,
    )
    identity = np.eye(central.shape[0])
    defect = central - identity
    entry_upper = _up_array(
        np.abs(defect)
        + product_error
        + propagated_error
        + _gamma(4) * (np.abs(central) + identity),
        16,
    )
    return {
        "central_spectral_defect": float(np.linalg.norm(defect, 2)),
        "directed_frobenius_defect_upper": _frobenius_upper(entry_upper),
        "maximum_entry_defect_upper": float(np.max(entry_upper)),
    }


def _generalized_reference_audit(
    mass: csr_matrix,
    stiffness: csr_matrix,
    values: np.ndarray,
    vectors: np.ndarray,
    mass_coercivity: dict[str, object],
) -> dict[str, object]:
    mass_asymmetry = mass - mass.transpose()
    stiffness_asymmetry = stiffness - stiffness.transpose()
    maximum_mass_asymmetry = (
        float(np.max(np.abs(mass_asymmetry.data)))
        if mass_asymmetry.nnz
        else 0.0
    )
    maximum_stiffness_asymmetry = (
        float(np.max(np.abs(stiffness_asymmetry.data)))
        if stiffness_asymmetry.nnz
        else 0.0
    )
    mass_action, mass_error = _sparse_action_with_error(mass, vectors)
    stiffness_action, stiffness_error = _sparse_action_with_error(
        stiffness,
        vectors,
    )
    scaled_mass, scaled_error = _scaled_columns_with_error(
        values,
        mass_action,
        mass_error,
    )
    residual, residual_error = _difference_with_error(
        stiffness_action,
        stiffness_error,
        scaled_mass,
        scaled_error,
    )
    residual_upper = _up_array(
        np.abs(residual) + residual_error,
        12,
    )
    residual_norms = _column_norms_upper(residual_upper)
    stiffness_norms = _column_norms_upper(
        _up_array(np.abs(stiffness_action) + stiffness_error, 8)
    )
    scaled_mass_norms = _column_norms_upper(
        _up_array(np.abs(scaled_mass) + scaled_error, 8)
    )
    scales = np.maximum(np.maximum(stiffness_norms, scaled_mass_norms), 1.0)
    relative = np.nextafter(residual_norms / scales, math.inf)
    mass_row_counts = np.diff(mass.indptr)
    mass_row_gamma = _gamma(2 * int(np.max(mass_row_counts)) + 8)
    mass_row_sums = np.asarray(mass.sum(axis=1)).ravel()
    mass_row_sum_lower = np.nextafter(
        mass_row_sums / (1.0 + mass_row_gamma),
        -math.inf,
    )
    if np.min(mass_row_sum_lower) <= 0.0:
        raise RuntimeError("stored mass row-sum lower is not positive")
    coercivity = float(
        mass_coercivity["global_row_lumped_coercivity_lower"]
    )
    if (
        not mass_coercivity["stored_mass_row_lumped_coercivity_proved"]
        or coercivity <= 0.0
    ):
        raise RuntimeError("stored mass coercivity certificate did not pass")
    weighted_squared = _up_array(
        residual_upper * residual_upper
        / mass_row_sum_lower[:, None],
        8,
    )
    inverse_mass_squared = np.sum(weighted_squared, axis=0)
    inverse_mass_squared = np.nextafter(
        inverse_mass_squared
        / (1.0 - _gamma(2 * mass.shape[0] + 8))
        / coercivity,
        math.inf,
    )
    inverse_mass_residual = np.nextafter(
        np.sqrt(np.maximum(inverse_mass_squared, 0.0)),
        math.inf,
    )
    orthogonality = _orthogonality_audit(
        vectors,
        mass_action,
        mass_error,
    )
    maximum_orthogonality_entry = float(
        orthogonality["maximum_entry_defect_upper"]
    )
    mass_norm_lower = math.sqrt(
        max(1.0 - maximum_orthogonality_entry, 0.0)
    )
    proximity_radii = np.nextafter(
        inverse_mass_residual / mass_norm_lower,
        math.inf,
    )
    proximity_separation_margins = np.nextafter(
        (values[1:] - proximity_radii[1:])
        - (values[:-1] + proximity_radii[:-1]),
        -math.inf,
    )
    pair_rows = [
        {
            "index": int(index),
            "approximate_eigenvalue": float(values[index]),
            "directed_residual_l2_upper": float(residual_norms[index]),
            "directed_inverse_mass_residual_upper": float(
                inverse_mass_residual[index]
            ),
            "eigenvalue_proximity_radius": float(proximity_radii[index]),
            "proximity_interval_lower": float(
                np.nextafter(
                    values[index] - proximity_radii[index],
                    -math.inf,
                )
            ),
            "proximity_interval_upper": float(
                np.nextafter(
                    values[index] + proximity_radii[index],
                    math.inf,
                )
            ),
        }
        for index in range(len(values))
    ]
    worst = int(np.argmax(residual_norms))
    worst_relative = int(np.argmax(relative))
    worst_inverse_mass = int(np.argmax(inverse_mass_residual))
    worst_proximity = int(np.argmax(proximity_radii))
    stiffness_row_counts = np.diff(stiffness.indptr)
    return {
        "pair_count": int(len(values)),
        "values_strictly_increasing": bool(np.all(np.diff(values) > 0.0)),
        "maximum_stored_mass_asymmetry": maximum_mass_asymmetry,
        "maximum_stored_stiffness_asymmetry": maximum_stiffness_asymmetry,
        "mass_maximum_row_nonzeros": int(np.max(mass_row_counts)),
        "stiffness_maximum_row_nonzeros": int(
            np.max(stiffness_row_counts)
        ),
        "maximum_directed_residual_l2_upper": float(
            residual_norms[worst]
        ),
        "maximum_directed_residual_column": worst,
        "maximum_directed_relative_residual_upper": float(
            relative[worst_relative]
        ),
        "maximum_directed_relative_residual_column": worst_relative,
        "residual_frobenius_upper": _frobenius_upper(residual_upper),
        "maximum_directed_inverse_mass_residual_upper": float(
            inverse_mass_residual[worst_inverse_mass]
        ),
        "maximum_directed_inverse_mass_residual_column": worst_inverse_mass,
        "inverse_mass_residual_block_frobenius_upper": _up(
            math.sqrt(
                float(
                    np.sum(
                        inverse_mass_residual * inverse_mass_residual
                    )
                )
                / (
                    1.0
                    - _gamma(2 * inverse_mass_residual.size + 8)
                )
            )
        ),
        "maximum_generalized_eigenvalue_proximity_radius": float(
            proximity_radii[worst_proximity]
        ),
        "maximum_generalized_eigenvalue_proximity_column": worst_proximity,
        "minimum_adjacent_proximity_interval_separation": float(
            np.min(proximity_separation_margins)
        ),
        "retained_cutoff_proximity_interval_separation": float(
            proximity_separation_margins[-1]
        ),
        "all_adjacent_proximity_intervals_disjoint": bool(
            np.min(proximity_separation_margins) > 0.0
        ),
        "pair_rows": pair_rows,
        "mass_row_sum_lower_minimum": float(
            np.min(mass_row_sum_lower)
        ),
        "mass_coercivity": mass_coercivity,
        "orthogonality": orthogonality,
        "generalized_eigenvalue_proximity_intervals_proved": True,
        "distinct_eigenvalue_existence_in_each_proximity_interval_proved": (
            bool(np.min(proximity_separation_margins) > 0.0)
        ),
        "generalized_eigenvalue_inclusions_proved": False,
        "reason_inclusions_remain_open": (
            "Residual proximity proves that each approximate value is close "
            "to some stored-matrix eigenvalue, but indexed inclusions and a "
            "complete eigenvalue count below the retained cutoff still need "
            "a verified inertia or equivalent block argument."
        ),
    }


def _standard_eigensystem_audit(
    matrix: np.ndarray,
    values: np.ndarray,
    vectors: np.ndarray,
) -> dict[str, object]:
    action, action_error = _dense_action_with_error(matrix, vectors)
    scaled, scaled_error = _scaled_columns_with_error(
        values,
        vectors,
        np.zeros_like(vectors),
    )
    residual, residual_error = _difference_with_error(
        action,
        action_error,
        scaled,
        scaled_error,
    )
    residual_upper = _up_array(np.abs(residual) + residual_error, 12)
    residual_norms = _column_norms_upper(residual_upper)
    identity_action = vectors
    identity_error = np.zeros_like(vectors)
    orthogonality = _orthogonality_audit(
        vectors,
        identity_action,
        identity_error,
    )
    worst = int(np.argmax(residual_norms))
    return {
        "pair_count": int(len(values)),
        "maximum_directed_residual_l2_upper": float(
            residual_norms[worst]
        ),
        "maximum_directed_residual_column": worst,
        "residual_frobenius_upper": _frobenius_upper(residual_upper),
        "orthogonality": orthogonality,
    }


def _factorization_audit(
    restricted_mass: np.ndarray,
    restricted_stiffness: np.ndarray,
    lower: np.ndarray,
    transformed_stiffness: np.ndarray,
) -> dict[str, object]:
    mass_reconstruction = lower @ lower.T
    mass_product_error = _dense_product_roundoff_upper(lower, lower.T)
    mass_residual = restricted_mass - mass_reconstruction
    mass_entry_upper = _up_array(
        np.abs(mass_residual)
        + mass_product_error
        + _gamma(4)
        * (np.abs(restricted_mass) + np.abs(mass_reconstruction)),
        12,
    )

    first = lower @ transformed_stiffness
    first_error = _dense_product_roundoff_upper(
        lower,
        transformed_stiffness,
    )
    congruence = first @ lower.T
    congruence_error = _up_array(
        _dense_product_roundoff_upper(first, lower.T)
        + _nonnegative_dense_product_upper(first_error, np.abs(lower.T)),
        12,
    )
    congruence_residual = restricted_stiffness - congruence
    congruence_entry_upper = _up_array(
        np.abs(congruence_residual)
        + congruence_error
        + _gamma(4)
        * (np.abs(restricted_stiffness) + np.abs(congruence)),
        12,
    )
    return {
        "minimum_cholesky_diagonal": float(np.min(np.diag(lower))),
        "cholesky_reconstruction_frobenius_upper": _frobenius_upper(
            mass_entry_upper
        ),
        "cholesky_reconstruction_maximum_entry_upper": float(
            np.max(mass_entry_upper)
        ),
        "congruence_reconstruction_frobenius_upper": _frobenius_upper(
            congruence_entry_upper
        ),
        "congruence_reconstruction_maximum_entry_upper": float(
            np.max(congruence_entry_upper)
        ),
    }


def audit(
    spacing: float,
    mode_count: int,
    quadrature_order: int,
    eigen_cache_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    certificate = _load_certificate_module()
    priority_set = certificate._set_below_normal_priority()
    data = certificate._assemble_frozen_block(
        spacing,
        mode_count,
        quadrature_order,
        eigen_cache_path,
    )
    reference = _generalized_reference_audit(
        data["reference_mass"].tocsr(),
        data["reference_stiffness"].tocsr(),
        np.asarray(data["reference_all_values"]),
        np.asarray(data["reference_all_vectors"]),
        data["reference_mass_coercivity"],
    )
    factorization = _factorization_audit(
        np.asarray(data["restricted_mass"]),
        np.asarray(data["restricted_stiffness"]),
        np.asarray(data["restricted_mass_cholesky"]),
        np.asarray(data["transformed_stiffness"]),
    )
    modified = _standard_eigensystem_audit(
        np.asarray(data["transformed_stiffness"]),
        np.asarray(data["modified_values"]),
        np.asarray(data["modified_vectors"]),
    )
    checks = [
        bool(data["cache_info"]["loaded"]),
        priority_set,
        int(reference["pair_count"]) == mode_count + 1,
        reference["values_strictly_increasing"],
        float(reference["maximum_stored_mass_asymmetry"]) == 0.0,
        float(reference["maximum_stored_stiffness_asymmetry"]) == 0.0,
        float(reference["maximum_directed_relative_residual_upper"])
        < 1.0e-9,
        float(reference["maximum_directed_inverse_mass_residual_upper"])
        < 1.0e-8,
        reference["mass_coercivity"][
            "stored_mass_row_lumped_coercivity_proved"
        ],
        reference["generalized_eigenvalue_proximity_intervals_proved"],
        reference[
            "distinct_eigenvalue_existence_in_each_proximity_interval_proved"
        ],
        float(
            reference["retained_cutoff_proximity_interval_separation"]
        )
        > 0.6,
        float(
            reference["orthogonality"][
                "directed_frobenius_defect_upper"
            ]
        )
        < 1.0e-8,
        float(factorization["minimum_cholesky_diagonal"]) > 0.0,
        float(
            factorization["cholesky_reconstruction_frobenius_upper"]
        )
        < 1.0e-8,
        float(
            factorization["congruence_reconstruction_frobenius_upper"]
        )
        < 1.0e-7,
        float(modified["maximum_directed_residual_l2_upper"]) < 1.0e-9,
        float(
            modified["orthogonality"][
                "directed_frobenius_defect_upper"
            ]
        )
        < 1.0e-9,
    ]
    return {
        "model": "binary-frozen generalized eigensystem residual audit",
        "spacing": spacing,
        "state_count": data["state_count"],
        "retained_mode_count": data["retained_count"],
        "reference_pair_count": int(
            np.asarray(data["reference_all_values"]).shape[0]
        ),
        "quadrature_order": quadrature_order,
        "reference_eigensystem_cache": data["cache_info"],
        "below_normal_priority_set": priority_set,
        "roundoff_model": (
            "IEEE binary64 gamma_n bounds with outward nextafter inflation"
        ),
        "reference_generalized_eigensystem": reference,
        "restricted_modified_factorization": factorization,
        "modified_standard_eigensystem": modified,
        "stored_matrix_eigenpair_residuals_directed_enclosed": True,
        "stored_matrix_orthogonality_defects_directed_enclosed": True,
        "stored_mass_row_lumped_coercivity_proved": True,
        "generalized_eigenvalue_proximity_intervals_proved": True,
        "distinct_reference_eigenvalue_proximity_intervals_proved": True,
        "finite_element_assembly_interval_enclosed": False,
        "generalized_eigenvalue_inclusions_proved": False,
        "endpoint_effect_of_eigenpair_residuals_certified": False,
        "scope": (
            "The audit encloses residual and orthogonality evaluation "
            "roundoff relative to stored binary64 matrices and cached vectors. "
            "It does not yet convert residuals into eigenvalue/eigenspace "
            "inclusions or enclose finite-element assembly."
        ),
        "all_eigensystem_residual_checks_pass": bool(all(checks)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, default=0.06)
    parser.add_argument("--mode-count", type=int, default=240)
    parser.add_argument("--quadrature-order", type=int, default=12)
    parser.add_argument("--eigen-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.spacing,
        args.mode_count,
        args.quadrature_order,
        args.eigen_cache,
    )
    if args.output is not None:
        _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
