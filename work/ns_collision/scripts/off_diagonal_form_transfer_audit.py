"""Audit relative-form bounds for off-diagonal Poisson transfers."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


def _load_boundary_l2_module():
    script = Path(__file__).resolve().with_name(
        "gaussian_boundary_l2_transfer_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "gaussian_boundary_l2_for_off_diagonal", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _positive_square_root(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = eigh(matrix)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def _operator_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def _random_trial(seed: int, alpha: float = 0.35) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    dimension = 9
    boundary_dimension = 3
    raw = rng.normal(size=(dimension, dimension))
    baseline_operator = raw.T @ raw + 2.0 * np.eye(dimension)
    baseline_root = _positive_square_root(baseline_operator)

    perturbation_raw = rng.normal(size=(dimension, dimension))
    relative_perturbation = perturbation_raw.T @ perturbation_raw
    relative_perturbation *= alpha / float(
        np.max(eigh(relative_perturbation, eigvals_only=True))
    )
    perturbation = (
        baseline_root @ relative_perturbation @ baseline_root
    )
    perturbed_operator = baseline_operator - perturbation

    inner_source = rng.normal(size=(dimension, boundary_dimension))
    outer_source = rng.normal(size=(dimension, boundary_dimension))
    baseline_resolvent = np.linalg.inv(baseline_operator)
    perturbed_resolvent = np.linalg.inv(perturbed_operator)
    resolvent_difference = perturbed_resolvent - baseline_resolvent

    baseline_cross = inner_source.T @ baseline_resolvent @ outer_source
    perturbed_cross = inner_source.T @ perturbed_resolvent @ outer_source
    cross_difference = perturbed_cross - baseline_cross
    baseline_inner_diagonal = (
        inner_source.T @ baseline_resolvent @ inner_source
    )
    baseline_outer_diagonal = (
        outer_source.T @ baseline_resolvent @ outer_source
    )
    diagonal_envelope = math.sqrt(
        _operator_norm(baseline_inner_diagonal)
        * _operator_norm(baseline_outer_diagonal)
    )
    difference_bound = alpha / (1.0 - alpha) * diagonal_envelope
    resolvent_order_residual = np.max(
        eigh(
            resolvent_difference
            - alpha / (1.0 - alpha) * baseline_resolvent,
            eigvals_only=True,
        )
    )
    return {
        "seed": seed,
        "alpha": alpha,
        "cross_difference_norm": _operator_norm(cross_difference),
        "diagonal_envelope_bound": difference_bound,
        "cross_difference_bound_holds": bool(
            _operator_norm(cross_difference) <= difference_bound + 1.0e-11
        ),
        "resolvent_Loewner_upper_residual": float(
            resolvent_order_residual
        ),
        "resolvent_Loewner_bound_holds": bool(
            resolvent_order_residual < 1.0e-11
        ),
    }


def _counterexample(alpha: float = 0.25, epsilon: float = 1.0e-3) -> dict[str, float | bool]:
    baseline_operator = np.eye(2)
    coupling_direction = np.ones(2) / math.sqrt(2.0)
    perturbation = alpha * np.outer(
        coupling_direction, coupling_direction
    )
    perturbed_operator = baseline_operator - perturbation
    inner_source = np.array([[1.0], [0.0]])
    outer_source = np.array(
        [[epsilon], [math.sqrt(1.0 - epsilon**2)]]
    )
    baseline_resolvent = np.eye(2)
    perturbed_resolvent = np.linalg.inv(perturbed_operator)
    baseline_cross = float(
        (inner_source.T @ baseline_resolvent @ outer_source)[0, 0]
    )
    perturbed_cross = float(
        (inner_source.T @ perturbed_resolvent @ outer_source)[0, 0]
    )
    naive_upper = baseline_cross / (1.0 - alpha)
    inner_diagonal = float(
        (inner_source.T @ baseline_resolvent @ inner_source)[0, 0]
    )
    outer_diagonal = float(
        (outer_source.T @ baseline_resolvent @ outer_source)[0, 0]
    )
    correct_upper = baseline_cross + alpha / (1.0 - alpha) * math.sqrt(
        inner_diagonal * outer_diagonal
    )
    return {
        "alpha": alpha,
        "epsilon": epsilon,
        "baseline_cross_transfer": baseline_cross,
        "perturbed_cross_transfer": perturbed_cross,
        "naive_relative_upper_bound": naive_upper,
        "correct_diagonal_envelope_upper_bound": correct_upper,
        "naive_relative_bound_fails": bool(
            perturbed_cross > naive_upper
        ),
        "diagonal_envelope_bound_holds": bool(
            perturbed_cross <= correct_upper
        ),
        "relative_amplification": perturbed_cross / baseline_cross,
    }


def _allowable_alpha(
    baseline_generation_criterion: float, buffer_condition_number: float
) -> float:
    allowed_visit_multiplier = 1.0 / math.sqrt(
        baseline_generation_criterion
    )
    excess_multiplier = allowed_visit_multiplier - 1.0
    return excess_multiplier / (
        buffer_condition_number + excess_multiplier
    )


def audit() -> dict[str, object]:
    random_rows = [_random_trial(seed) for seed in range(8)]
    counterexample = _counterexample()

    boundary_l2 = _load_boundary_l2_module().audit()
    geometry_rows = []
    condition_numbers = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    for visit_row in boundary_l2["visit_rows"]:
        criterion = visit_row[
            "Gaussian_L2_complete_generation_criterion"
        ]
        geometry_rows.append(
            {
                "R_star": visit_row["R_star"],
                "half_height_over_L": visit_row["half_height_over_L"],
                "baseline_Gaussian_L2_generation_criterion": criterion,
                "maximum_visit_multiplier": 1.0 / math.sqrt(criterion),
                "relative_form_alpha_budgets": {
                    str(condition_number): _allowable_alpha(
                        criterion, condition_number
                    )
                    for condition_number in condition_numbers
                },
            }
        )

    working_geometry = geometry_rows[0]
    result: dict[str, object] = {
        "relative_form_hypothesis": (
            "0<=Q<=alpha*A_0 with 0<=alpha<1"
        ),
        "resolvent_difference_order": (
            "0<=R_q-R_0<=alpha/(1-alpha)*R_0"
        ),
        "cross_transfer_definitions": (
            "B_0=F_i^*R_0F_o, B_q=F_i^*R_qF_o, "
            "D_i=F_i^*R_0F_i, D_o=F_o^*R_0F_o"
        ),
        "correct_off_diagonal_bound": (
            "||B_q-B_0||<=alpha/(1-alpha)*"
            "sqrt(||D_i||*||D_o||)"
        ),
        "buffer_condition_number": (
            "chi=sqrt(||D_i||*||D_o||)/||B_0||>=1"
        ),
        "relative_cross_bound": (
            "||B_q||/||B_0||<=1+chi*alpha/(1-alpha)"
        ),
        "counterexample": counterexample,
        "naive_relative_cross_bound_is_false": counterexample[
            "naive_relative_bound_fails"
        ],
        "correct_bound_handles_counterexample": counterexample[
            "diagonal_envelope_bound_holds"
        ],
        "random_matrix_rows": random_rows,
        "all_random_Loewner_and_cross_bounds_hold": all(
            row["resolvent_Loewner_bound_holds"]
            and row["cross_difference_bound_holds"]
            for row in random_rows
        ),
        "renewal_alpha_formula": (
            "alpha<(C_0^(-1/2)-1)/(chi+C_0^(-1/2)-1)"
        ),
        "geometry_rows": geometry_rows,
        "all_tabulated_alpha_budgets_are_positive": all(
            alpha_budget > 0.0
            for row in geometry_rows
            for alpha_budget in row[
                "relative_form_alpha_budgets"
            ].values()
        ),
        "working_geometry_chi_one_alpha_budget": (
            working_geometry["relative_form_alpha_budgets"]["1.0"]
        ),
        "working_geometry_chi_two_alpha_budget": (
            working_geometry["relative_form_alpha_budgets"]["2.0"]
        ),
        "interpretation": (
            "relative form order alone does not control an off-diagonal "
            "Poisson block relative to its baseline; the missing cylinder "
            "quantity is the diagonal-to-cross buffer condition chi"
        ),
        "next_cylinder_gate": (
            "realize the inner/outer trace-source maps for the weighted "
            "finite cylinder and certify chi at the working geometry"
        ),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
