"""Audit same-scale renewal followed by a true dyadic generation split."""

from __future__ import annotations

import json
import math

from scipy.optimize import brentq
import sympy as sp


def _cycle_quantities(
    reynolds: float,
    buffer_ratio: float,
    beta: float,
    visit_action: float = 0.0,
    dimension: float = 3.0,
) -> dict[str, float | bool]:
    condition_number = math.exp(reynolds / 2.0)
    true_split_factor = math.exp(reynolds * dimension / 24.0) / 4.0
    pair_return = buffer_ratio ** (-2.0 * beta)
    physical_visit = condition_number * math.exp(-visit_action)
    renewal_product = pair_return * physical_visit
    if renewal_product < 1.0:
        renewal_block = physical_visit / (1.0 - renewal_product)
        generation_factor = true_split_factor * renewal_block
    else:
        renewal_block = math.inf
        generation_factor = math.inf
    required_action = max(
        0.0,
        math.log(condition_number * (true_split_factor + pair_return)),
    )
    renewal_only_required_action = max(
        0.0, math.log(condition_number * pair_return)
    )
    return {
        "condition_number": condition_number,
        "true_split_factor": true_split_factor,
        "pair_return": pair_return,
        "physical_visit_factor": physical_visit,
        "renewal_product": renewal_product,
        "renewal_block_norm": renewal_block,
        "generation_factor": generation_factor,
        "renewal_only_required_visit_action": (
            renewal_only_required_action
        ),
        "full_generation_required_visit_action": required_action,
        "zero_action_closes_generation": bool(generation_factor < 1.0),
    }


def audit() -> dict[str, object]:
    reynolds, dimension, eta, beta, visit_action = sp.symbols(
        "R_star dimension eta beta visit_action", positive=True, real=True
    )
    condition_number = sp.exp(reynolds / 2)
    true_split_factor = sp.exp(reynolds * dimension / 24) / 4
    pair_return = eta ** (-2 * beta)
    physical_visit = condition_number * sp.exp(-visit_action)
    renewal_block = sp.factor(
        physical_visit / (1 - pair_return * physical_visit)
    )
    generation_factor = sp.factor(true_split_factor * renewal_block)
    generation_closure_equivalent = (
        "visit_action>log(C_pair*(gamma_pair+r_pair))"
    )

    parameter_rows = []
    for reynolds_value in (0.5, 1.0, 1.5, 2.0):
        for beta_value in (0.5, 1.0):
            for buffer_ratio in (1.5, 2.0, 3.0, 5.0):
                quantities = _cycle_quantities(
                    reynolds_value, buffer_ratio, beta_value
                )
                parameter_rows.append(
                    {
                        "R_star": reynolds_value,
                        "beta": beta_value,
                        "buffer_ratio": buffer_ratio,
                        **quantities,
                        "static_brownian_deformation_budget": (
                            beta_value * (1.0 - beta_value)
                        ),
                    }
                )

    representative_row = next(
        row
        for row in parameter_rows
        if row["R_star"] == 1.0
        and row["beta"] == 1.0
        and row["buffer_ratio"] == 2.0
    )
    edge_row = next(
        row
        for row in parameter_rows
        if row["R_star"] == 2.0
        and row["beta"] == 1.0
        and row["buffer_ratio"] == 2.0
    )

    def zero_action_equation(reynolds_value: float) -> float:
        quantities = _cycle_quantities(reynolds_value, 2.0, 1.0)
        return float(
            quantities["condition_number"]
            * (
                quantities["true_split_factor"]
                + quantities["pair_return"]
            )
            - 1.0
        )

    reynolds_threshold_eta_two = float(
        brentq(zero_action_equation, 0.01, 3.0)
    )

    result: dict[str, object] = {
        "pair_condition_number": str(condition_number),
        "true_generation_split_factor": str(true_split_factor),
        "pair_return_factor": str(pair_return),
        "physical_visit_factor": str(physical_visit),
        "same_generation_renewal_block": str(renewal_block),
        "complete_generation_factor": str(generation_factor),
        "generation_closure_equivalent": generation_closure_equivalent,
        "two_level_ordering": (
            "sum all same-scale buffered returns first, then apply one true "
            "radius-halving factor"
        ),
        "parameter_rows": parameter_rows,
        "R_one_beta_one_eta_two": representative_row,
        "R_two_beta_one_eta_two": edge_row,
        "R_one_eta_two_closes_with_zero_visit_action": bool(
            representative_row["zero_action_closes_generation"]
        ),
        "R_two_eta_two_needs_positive_visit_action": bool(
            not edge_row["zero_action_closes_generation"]
            and edge_row["full_generation_required_visit_action"] > 0.0
        ),
        "zero_action_R_threshold_for_beta_one_eta_two": (
            reynolds_threshold_eta_two
        ),
        "threshold_lies_between_one_and_one_point_five": bool(
            1.0 < reynolds_threshold_eta_two < 1.5
        ),
        "all_rows_with_zero_action_closure_have_valid_renewal": all(
            (not row["zero_action_closes_generation"])
            or row["renewal_product"] < 1.0
            for row in parameter_rows
        ),
        "beta_one_static_capacity_warning": (
            "beta=1 gives strongest Brownian return contraction but no "
            "static positive-deformation allowance; shrink or radial drift "
            "must pay exterior deformation"
        ),
        "provisional_condition_number_regime": (
            "the placeholder visit model favors R_star approximately 1, "
            "beta=1, eta approximately 2; the exact buffered-visit benchmark "
            "must supersede this provisional row"
        ),
        "remaining_visit_gate": (
            "derive a uniform gauged buffered-visit action from killed "
            "spectral margin and coherence errors, including arbitrarily "
            "short visits"
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
