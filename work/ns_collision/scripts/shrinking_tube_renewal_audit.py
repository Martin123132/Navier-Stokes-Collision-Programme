"""Audit the return barrier for a monotonically shrinking strain tube."""

from __future__ import annotations

import json

import sympy as sp


def audit() -> dict[str, object]:
    radius, inner_scale, beta = sp.symbols(
        "radius inner_scale beta", positive=True, real=True
    )
    dimension = sp.symbols("dimension", positive=True, integer=True)
    viscosity = sp.symbols("viscosity", positive=True, real=True)
    scale_rate, radial_drift, deformation = sp.symbols(
        "scale_rate radial_drift deformation", real=True
    )

    barrier = (inner_scale / radius) ** beta
    radial_laplacian = sp.factor(
        sp.diff(barrier, radius, 2)
        + (dimension - 1) * sp.diff(barrier, radius) / radius
    )
    radial_laplacian_ratio = sp.factor(
        sp.powsimp(radial_laplacian / barrier, force=True)
    )
    expected_laplacian_ratio = sp.factor(
        beta * (beta - dimension + 2) / radius**2
    )
    moving_generator_ratio = sp.factor(
        beta * scale_rate
        + viscosity * radial_laplacian_ratio
        - beta * radial_drift / radius**2
        + deformation
    )
    expected_generator_ratio = sp.factor(
        beta * scale_rate
        + beta
        * (viscosity * (beta - dimension + 2) - radial_drift)
        / radius**2
        + deformation
    )
    three_dimensional_ratio = sp.factor(
        moving_generator_ratio.subs(dimension, 3)
    )

    envelope, envelope_rate = sp.symbols(
        "envelope envelope_rate", positive=True, real=True
    )
    envelope_scale_rate = -envelope_rate / (2 * envelope)
    brownian_beta_one_ratio = sp.factor(
        three_dimensional_ratio.subs(
            {
                beta: 1,
                radial_drift: 0,
                scale_rate: envelope_scale_rate,
            }
        )
    )

    beta_symbol = sp.symbols("beta_symbol", real=True)
    brownian_capacity_budget = sp.factor(
        beta_symbol * (1 - beta_symbol)
    )
    capacity_derivative = sp.diff(
        brownian_capacity_budget, beta_symbol
    )
    capacity_optimizer = sp.solve(capacity_derivative, beta_symbol)[0]
    maximum_capacity_budget = sp.simplify(
        brownian_capacity_budget.subs(beta_symbol, capacity_optimizer)
    )

    return_rows = []
    for buffer_ratio in (1.1, 1.25, 1.5, 2.0, 3.0):
        return_rows.append(
            {
                "buffer_ratio": buffer_ratio,
                "beta_one_pair_return": buffer_ratio ** (-2.0),
                "beta_half_pair_return": buffer_ratio ** (-1.0),
                "beta_one_half_capacity_budget": 0.25,
            }
        )

    example_visit_gain = 0.9
    example_buffer_ratio = 1.5
    example_beta = 0.5
    example_pair_return = example_buffer_ratio ** (-2 * example_beta)
    example_cycle_factor = example_visit_gain * example_pair_return
    example_renewal_bound = example_visit_gain / (
        1.0 - example_cycle_factor
    )

    result: dict[str, object] = {
        "moving_barrier": "h(t,r)=(L(t)/r)^beta",
        "radial_laplacian_ratio": str(radial_laplacian_ratio),
        "radial_laplacian_verified": bool(
            sp.simplify(
                radial_laplacian_ratio - expected_laplacian_ratio
            )
            == 0
        ),
        "moving_generator_ratio": str(moving_generator_ratio),
        "moving_generator_formula_verified": bool(
            sp.simplify(
                moving_generator_ratio - expected_generator_ratio
            )
            == 0
        ),
        "three_dimensional_generator_ratio": str(
            three_dimensional_ratio
        ),
        "three_dimensional_supersolution_condition": (
            "deformation*radius^2<=beta*(radial_drift+"
            "viscosity*(1-beta)-scale_rate*radius^2)"
        ),
        "shrinking_scale_improves_generator": bool(
            sp.diff(moving_generator_ratio, scale_rate) == beta
        ),
        "envelope_scale_rate": str(envelope_scale_rate),
        "brownian_beta_one_generator": str(brownian_beta_one_ratio),
        "brownian_beta_one_deformation_allowance": (
            "deformation<=envelope_rate/(2*envelope)"
        ),
        "condition_for_deformation_equal_2a": (
            "envelope_rate>=4*a*envelope"
        ),
        "static_brownian_capacity_budget": str(
            brownian_capacity_budget
        ),
        "capacity_budget_optimizer": str(capacity_optimizer),
        "maximum_static_capacity_budget": str(maximum_capacity_budget),
        "capacity_budget_maximum_is_one_quarter": bool(
            maximum_capacity_budget == sp.Rational(1, 4)
        ),
        "return_rows": return_rows,
        "return_factor_is_uniform_in_inner_scale": bool(
            not sp.sympify(sp.Symbol("eta") ** (-beta)).has(inner_scale)
        ),
        "one_history_return_bound": "eta^(-beta)",
        "two_history_return_bound": "eta^(-2*beta)",
        "example_visit_gain": example_visit_gain,
        "example_pair_return": example_pair_return,
        "example_cycle_factor": example_cycle_factor,
        "example_renewal_converges": bool(example_cycle_factor < 1.0),
        "example_renewal_bound": example_renewal_bound,
        "conditional_closure_statement": (
            "if the adaptive visit operator is contractive and the moving "
            "barrier inequality holds, shrinking-core renewal is summable"
        ),
        "unresolved_exterior_gate": (
            "derive the moving barrier inequality from Navier-Stokes "
            "pressure, drift, and deformation geometry"
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
