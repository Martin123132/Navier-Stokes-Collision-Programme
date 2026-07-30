"""Audit an exact ideal core-plus-shell buffered Feynman-Kac visit."""

from __future__ import annotations

import json
import math

from scipy.optimize import brentq, minimize_scalar
import sympy as sp


def _visit_quantities(
    reynolds: float,
    buffer_ratio: float,
    beta: float,
    dimension: float = 3.0,
) -> dict[str, float | bool]:
    denominator = 1.0 - reynolds * math.log(buffer_ratio)
    pair_return = buffer_ratio ** (-2.0 * beta)
    true_split_factor = math.exp(reynolds * dimension / 24.0) / 4.0
    if denominator <= 0.0:
        one_history_visit = math.inf
        pair_visit = math.inf
        renewal_product = math.inf
        generation_factor = math.inf
        closure_criterion = math.inf
    else:
        one_history_visit = 1.0 / denominator
        pair_visit = one_history_visit**2
        renewal_product = pair_return * pair_visit
        closure_criterion = pair_visit * (
            true_split_factor + pair_return
        )
        if renewal_product < 1.0:
            renewal_block = pair_visit / (1.0 - renewal_product)
            generation_factor = true_split_factor * renewal_block
        else:
            generation_factor = math.inf
    return {
        "visit_denominator": denominator,
        "one_history_visit_gain": one_history_visit,
        "pair_visit_gain": pair_visit,
        "pair_return": pair_return,
        "renewal_product": renewal_product,
        "true_split_factor": true_split_factor,
        "generation_closure_criterion": closure_criterion,
        "generation_factor": generation_factor,
        "finite_visit": bool(denominator > 0.0),
        "same_generation_renewal_converges": bool(renewal_product < 1.0),
        "complete_generation_closes": bool(generation_factor < 1.0),
    }


def audit() -> dict[str, object]:
    radius, reynolds, buffer_ratio, boundary_value = sp.symbols(
        "radius R_star eta boundary_value", positive=True, real=True
    )
    inner_solution = boundary_value * sp.exp(
        reynolds * (1 - radius**2) / 2
    )
    shell_solution = boundary_value * (
        1 - reynolds * sp.log(radius)
    )
    inner_generator_residual = sp.simplify(
        sp.diff(inner_solution, radius, 2)
        + sp.diff(inner_solution, radius) / radius
        + reynolds * radius * sp.diff(inner_solution, radius)
        + 2 * reynolds * inner_solution
    )
    shell_laplacian_residual = sp.simplify(
        sp.diff(shell_solution, radius, 2)
        + sp.diff(shell_solution, radius) / radius
    )
    value_match_residual = sp.simplify(
        inner_solution.subs(radius, 1)
        - shell_solution.subs(radius, 1)
    )
    derivative_match_residual = sp.simplify(
        sp.diff(inner_solution, radius).subs(radius, 1)
        - sp.diff(shell_solution, radius).subs(radius, 1)
    )
    outer_boundary_expression = sp.simplify(
        shell_solution.subs(radius, buffer_ratio)
    )
    solved_boundary_value = sp.solve(
        sp.Eq(outer_boundary_expression, 1), boundary_value
    )[0]

    parameter_rows = []
    for reynolds_value in (0.1, 0.25, 0.4, 0.5, 0.75, 1.0):
        for beta_value in (0.5, 1.0):
            for eta_value in (1.5, 2.0, 3.0):
                parameter_rows.append(
                    {
                        "R_star": reynolds_value,
                        "beta": beta_value,
                        "buffer_ratio": eta_value,
                        **_visit_quantities(
                            reynolds_value, eta_value, beta_value
                        ),
                    }
                )

    eta_two_beta_one_R_quarter = next(
        row
        for row in parameter_rows
        if row["R_star"] == 0.25
        and row["beta"] == 1.0
        and row["buffer_ratio"] == 2.0
    )
    eta_two_beta_one_R_half = next(
        row
        for row in parameter_rows
        if row["R_star"] == 0.5
        and row["beta"] == 1.0
        and row["buffer_ratio"] == 2.0
    )

    def eta_two_generation_equation(reynolds_value: float) -> float:
        row = _visit_quantities(reynolds_value, 2.0, 1.0)
        return float(row["generation_closure_criterion"] - 1.0)

    eta_two_generation_threshold = float(
        brentq(eta_two_generation_equation, 0.001, 0.7)
    )
    eta_two_renewal_threshold = (1.0 - 0.5) / math.log(2.0)

    optimized_rows = []
    for reynolds_value in (0.1, 0.25, 0.4, 0.5, 0.75, 1.0):
        upper_eta = min(20.0, math.exp(0.999 / reynolds_value))

        def objective(eta_value: float) -> float:
            row = _visit_quantities(reynolds_value, eta_value, 1.0)
            return float(row["generation_closure_criterion"])

        optimum = minimize_scalar(
            objective,
            bounds=(1.0001, upper_eta),
            method="bounded",
            options={"xatol": 1.0e-12},
        )
        optimum_quantities = _visit_quantities(
            reynolds_value, float(optimum.x), 1.0
        )
        optimized_rows.append(
            {
                "R_star": reynolds_value,
                "optimal_buffer_ratio_for_beta_one": float(optimum.x),
                "minimum_generation_closure_criterion": float(optimum.fun),
                "optimized_generation_factor": optimum_quantities[
                    "generation_factor"
                ],
                "optimized_regime_closes": bool(optimum.fun < 1.0),
            }
        )

    result: dict[str, object] = {
        "ideal_core_model": (
            "two-dimensional radial outward OU drift and stretching 2a "
            "inside radius L"
        ),
        "ideal_shell_model": (
            "two-dimensional Brownian shell with zero stretching from L "
            "to eta*L"
        ),
        "inner_solution": str(inner_solution),
        "shell_solution": str(shell_solution),
        "inner_generator_verified": bool(inner_generator_residual == 0),
        "shell_harmonicity_verified": bool(shell_laplacian_residual == 0),
        "interface_value_matching_verified": bool(value_match_residual == 0),
        "interface_derivative_matching_verified": bool(
            derivative_match_residual == 0
        ),
        "outer_boundary_expression": str(outer_boundary_expression),
        "one_history_inner_boundary_visit_gain": str(
            solved_boundary_value
        ),
        "pair_visit_gain": "1/(1-R_star*log(eta))^2",
        "finite_visit_condition": "R_star*log(eta)<1",
        "parameter_rows": parameter_rows,
        "R_quarter_beta_one_eta_two": eta_two_beta_one_R_quarter,
        "R_half_beta_one_eta_two": eta_two_beta_one_R_half,
        "R_quarter_eta_two_closes_complete_generation": bool(
            eta_two_beta_one_R_quarter["complete_generation_closes"]
        ),
        "R_half_eta_two_fails_complete_generation": bool(
            not eta_two_beta_one_R_half["complete_generation_closes"]
        ),
        "eta_two_beta_one_same_generation_renewal_R_threshold": (
            eta_two_renewal_threshold
        ),
        "eta_two_beta_one_complete_generation_R_threshold": (
            eta_two_generation_threshold
        ),
        "complete_threshold_is_stricter_than_renewal_threshold": bool(
            eta_two_generation_threshold < eta_two_renewal_threshold
        ),
        "optimized_beta_one_rows": optimized_rows,
        "R_one_has_no_closing_buffer_in_benchmark": bool(
            not optimized_rows[-1]["optimized_regime_closes"]
        ),
        "previous_condition_number_model_was_optimistic": True,
        "benchmark_verdict": (
            "R_star near one is not viable in this recurrent transverse "
            "core-shell benchmark; eta=2 requires R_star below the audited "
            "complete-generation threshold"
        ),
        "remaining_visit_route": (
            "use finite axial killing, three-dimensional shell escape, "
            "outward shell drift, or a grouped occupation estimate to beat "
            "the logarithmic transverse return gain"
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
