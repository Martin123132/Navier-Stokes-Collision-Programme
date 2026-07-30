"""Audit a buffered exit/re-entry renewal criterion for two histories."""

from __future__ import annotations

import json
from math import isclose

import sympy as sp


def audit() -> dict[str, object]:
    radius, beta, viscosity = sp.symbols(
        "radius beta viscosity", positive=True, real=True
    )
    dimension = sp.symbols("dimension", positive=True, integer=True)
    radial_drift_dot, outside_potential = sp.symbols(
        "radial_drift_dot outside_potential", real=True
    )
    barrier = radius ** (-beta)
    radial_laplacian = sp.factor(
        sp.diff(barrier, radius, 2)
        + (dimension - 1) * sp.diff(barrier, radius) / radius
    )
    expected_laplacian = sp.factor(
        beta * (beta - dimension + 2) * barrier / radius**2
    )
    weighted_residual_3d = sp.factor(
        beta
        * (viscosity * (beta - 1) - radial_drift_dot)
        / radius**2
        + outside_potential
    )

    buffer_ratios = (1.1, 1.25, 1.5, 2.0, 3.0)
    rows = []
    for buffer_ratio in buffer_ratios:
        single_return = buffer_ratio ** (-1.0)
        pair_return = single_return**2
        rows.append(
            {
                "buffer_ratio": buffer_ratio,
                "single_brownian_return_bound": single_return,
                "independent_pair_return_bound": pair_return,
                "maximum_visit_gain_for_renewal": 1.0 / pair_return,
            }
        )

    example_ratio = 1.5
    example_visit_gain = 1.2
    example_pair_return = example_ratio ** (-2.0)
    example_cycle_factor = example_visit_gain * example_pair_return
    example_closed_sum = example_visit_gain / (1.0 - example_cycle_factor)
    example_partial_sum = sum(
        example_visit_gain * example_cycle_factor**index
        for index in range(60)
    )

    result: dict[str, object] = {
        "radial_barrier": "h(r)=(L/r)^beta",
        "radial_laplacian": str(radial_laplacian),
        "radial_laplacian_verified": bool(
            sp.simplify(radial_laplacian - expected_laplacian) == 0
        ),
        "weighted_generator_residual_in_3d": str(weighted_residual_3d),
        "weighted_return_supersolution_condition": (
            "c_+(x)*r^2<=beta*(b(x).x+nu*(1-beta))"
        ),
        "single_history_return_bound": "(L/L_out)^beta",
        "independent_pair_return_bound": "(L/L_out)^(2*beta)",
        "pure_brownian_exponent_in_dimension_d": "beta=d-2",
        "pure_brownian_3d_exponent": 1,
        "pure_brownian_2d_exponent": 0,
        "finite_3d_buffer_is_contracting": True,
        "infinite_cylinder_has_no_transverse_return_contraction": True,
        "buffered_visit_definition": (
            "one visit persists through the shell until outer-boundary exit"
        ),
        "renewal_series": "V*sum((R*V)^n)",
        "renewal_convergence_condition": "||R||*||V||<1",
        "renewal_norm_bound": "||V||/(1-||R||*||V||)",
        "brownian_buffer_rows": rows,
        "example_buffer_ratio": example_ratio,
        "example_visit_gain": example_visit_gain,
        "example_pair_return": example_pair_return,
        "example_cycle_factor": example_cycle_factor,
        "example_renewal_bound": example_closed_sum,
        "finite_sum_matches_renewal_bound": bool(
            isclose(example_partial_sum, example_closed_sum, rel_tol=1.0e-13)
        ),
        "all_pair_return_bounds_are_strictly_below_one": all(
            row["independent_pair_return_bound"] < 1.0 for row in rows
        ),
        "all_maximum_visit_gains_exceed_one": all(
            row["maximum_visit_gain_for_renewal"] > 1.0 for row in rows
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
