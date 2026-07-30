"""Audit the no-go for nonconstant conservative interface gauge weights."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
import sympy as sp


def _load_branching_module():
    script = Path(__file__).resolve().with_name(
        "branching_transfer_operator_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "branching_transfer_for_weight_no_go", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _logarithmic_l1_norm(generator: np.ndarray) -> float:
    column_measures = []
    for column in range(generator.shape[1]):
        off_diagonal = np.delete(generator[:, column], column)
        column_measures.append(
            generator[column, column] + np.sum(np.abs(off_diagonal))
        )
    return float(np.max(column_measures))


def audit() -> dict[str, object]:
    branching = _load_branching_module()
    generator = branching._build_interface_generator()
    child_weights = branching._build_child_weights()
    weight_matrix = np.diag(child_weights)
    weighted_generator = (
        weight_matrix @ generator @ np.diag(1.0 / child_weights)
    )
    pair_weighted_generator = (
        np.kron(weighted_generator, np.eye(generator.shape[0]))
        + np.kron(np.eye(generator.shape[0]), weighted_generator)
    )

    left_kernel = null_space(generator.T)
    normalized_left_kernel = left_kernel[:, 0] / np.mean(left_kernel[:, 0])
    left_weight_residual = generator.T @ child_weights
    single_logarithmic_growth = _logarithmic_l1_norm(
        weighted_generator
    )
    pair_logarithmic_growth = _logarithmic_l1_norm(
        pair_weighted_generator
    )
    componentwise_growth = left_weight_residual / child_weights

    reynolds, dimension = sp.symbols(
        "R_star dimension", positive=True, real=True
    )
    gauged_pair_true_split = sp.exp(reynolds * dimension / 24) / 4
    unit_ball_pair_condition_number = sp.exp(reynolds / 2)
    physical_pair_sufficient_factor = sp.simplify(
        gauged_pair_true_split * unit_ball_pair_condition_number
    )
    physical_contraction_threshold = sp.solve_univariate_inequality(
        physical_pair_sufficient_factor.subs(dimension, 3) < 1,
        reynolds,
    )

    physical_rows = []
    for reynolds_value in (0.5, 1.0, 1.5, 2.0):
        gauged_factor = math.exp(reynolds_value * 3.0 / 24.0) / 4.0
        conversion_factor = math.exp(reynolds_value / 2.0)
        physical_factor = gauged_factor * conversion_factor
        physical_rows.append(
            {
                "R_star": reynolds_value,
                "gauged_pair_true_split_factor": gauged_factor,
                "unit_ball_pair_condition_number": conversion_factor,
                "physical_pair_sufficient_factor": physical_factor,
                "remaining_logarithmic_budget": -math.log(physical_factor),
                "maximum_additional_cycle_factor": 1.0 / physical_factor,
            }
        )

    reynolds_two_row = physical_rows[-1]
    result: dict[str, object] = {
        "closed_interface_graph": "asymmetric irreducible three-cube graph",
        "left_kernel_dimension": int(left_kernel.shape[1]),
        "normalized_left_kernel_maximum_constant_error": float(
            np.max(np.abs(normalized_left_kernel - 1.0))
        ),
        "constant_weight_is_conservative": bool(
            np.max(np.abs(generator.T @ np.ones(generator.shape[0])))
            < 1.0e-14
        ),
        "only_constant_interface_weight_is_conservative": bool(
            left_kernel.shape[1] == 1
            and np.max(np.abs(normalized_left_kernel - 1.0)) < 1.0e-12
        ),
        "nonconstant_weight_residual": left_weight_residual.tolist(),
        "nonconstant_weight_has_both_growth_and_decay_cells": bool(
            np.max(left_weight_residual) > 0.0
            and np.min(left_weight_residual) < 0.0
        ),
        "weighted_single_logarithmic_l1_growth": (
            single_logarithmic_growth
        ),
        "maximum_componentwise_weight_growth": float(
            np.max(componentwise_growth)
        ),
        "logarithmic_norm_matches_componentwise_formula": bool(
            abs(single_logarithmic_growth - np.max(componentwise_growth))
            < 1.0e-12
        ),
        "weighted_pair_logarithmic_l1_growth": pair_logarithmic_growth,
        "independent_pair_growth_is_twice_single_growth": bool(
            abs(pair_logarithmic_growth - 2.0 * single_logarithmic_growth)
            < 1.0e-12
        ),
        "closed_graph_superharmonic_no_go": (
            "K^T w<=0 and positive stationarity force K^T w=0; "
            "irreducibility then forces w constant"
        ),
        "two_norm_architecture": (
            "use constant physical mass for interface transfer and local "
            "gauges only inside buffered coherent visits"
        ),
        "gauged_pair_true_split_factor": str(gauged_pair_true_split),
        "unit_ball_pair_condition_number": str(
            unit_ball_pair_condition_number
        ),
        "physical_pair_sufficient_factor": str(
            physical_pair_sufficient_factor
        ),
        "physical_contraction_threshold_in_3d": str(
            physical_contraction_threshold
        ),
        "physical_rows": physical_rows,
        "R_two_physical_pair_sufficient_factor": reynolds_two_row[
            "physical_pair_sufficient_factor"
        ],
        "R_two_remaining_logarithmic_budget": reynolds_two_row[
            "remaining_logarithmic_budget"
        ],
        "R_two_maximum_additional_cycle_factor": reynolds_two_row[
            "maximum_additional_cycle_factor"
        ],
        "R_two_two_norm_bound_remains_contractive": bool(
            reynolds_two_row["physical_pair_sufficient_factor"] < 1.0
        ),
        "spectral_margin_not_counted_in_two_norm_bound": True,
        "remaining_two_norm_gate": (
            "construct entry/exit maps that incur the condition number once "
            "per buffered visit, not once per interface crossing"
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
