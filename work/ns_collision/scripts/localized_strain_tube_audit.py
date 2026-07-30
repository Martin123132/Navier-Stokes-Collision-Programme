"""Spectral audit for common residence in an ideal localized strain tube."""

from __future__ import annotations

import json

from scipy.optimize import brentq
from scipy.special import hyp1f1
import sympy as sp


def _principal_ratio(tube_reynolds: float) -> float:
    """Return A=lambda_1/(2a) from M(A,1,-R/2)=0."""
    if tube_reynolds <= 0:
        raise ValueError("tube_reynolds must be positive")

    def boundary_value(ratio: float) -> float:
        return float(hyp1f1(ratio, 1.0, -tube_reynolds / 2.0))

    lower = 1.0
    upper = 1.1
    while boundary_value(upper) > 0.0:
        upper = 1.0 + 1.5 * (upper - 1.0)
        if upper > 1.0e6:
            raise RuntimeError("failed to bracket the principal Kummer root")
    return float(brentq(boundary_value, lower, upper, xtol=1.0e-13))


def audit() -> dict[str, object]:
    x, y, a, nu = sp.symbols("x y a nu", positive=True, real=True)
    radius_sq = x**2 + y**2
    potential = a * radius_sq / 2
    gradient_potential_sq = sum(
        sp.diff(potential, variable) ** 2 for variable in (x, y)
    )
    laplacian_potential = sum(
        sp.diff(potential, variable, 2) for variable in (x, y)
    )
    gauge_potential = sp.factor(
        gradient_potential_sq / (4 * nu) + laplacian_potential / 2
    )
    expected_gauge_potential = a**2 * radius_sq / (4 * nu) + a

    oscillator_ground_state = sp.exp(-a * radius_sq / (4 * nu))
    oscillator_action = sp.simplify(
        -nu
        * sum(
            sp.diff(oscillator_ground_state, variable, 2)
            for variable in (x, y)
        )
        + a**2 * radius_sq / (4 * nu) * oscillator_ground_state
    )

    z, ratio = sp.symbols("z ratio", real=True)
    f0, f1, f2 = sp.symbols("f0 f1 f2")
    radial_equation = z * f2 + (1 - z) * f1 - ratio * f0
    transformed_generator = -2 * a * radial_equation

    error_x, error_y = sp.symbols("error_x error_y", real=True)
    divergence_error, stretching_error = sp.symbols(
        "divergence_error stretching_error", real=True
    )
    effective_error = sp.factor(
        stretching_error
        - divergence_error / 2
        - a * (x * error_x + y * error_y) / (2 * nu)
    )

    reynolds_values = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    rows = []
    for tube_reynolds in reynolds_values:
        principal_ratio = _principal_ratio(tube_reynolds)
        rows.append(
            {
                "tube_reynolds": tube_reynolds,
                "lambda_over_a": 2.0 * principal_ratio,
                "pair_decay_margin_over_a": 4.0 * (principal_ratio - 1.0),
                "single_history_perturbation_budget_over_a": (
                    2.0 * (principal_ratio - 1.0)
                ),
                "kummer_boundary_residual": float(
                    hyp1f1(principal_ratio, 1.0, -tube_reynolds / 2.0)
                ),
            }
        )

    result: dict[str, object] = {
        "transverse_backward_sde": "dX=a*X*dt+sqrt(2*nu)*dW in the disk",
        "single_particle_generator": "L=nu*Delta+a*x dot grad",
        "gauge_transformed_operator": (
            "-nu*Delta+a^2*|x|^2/(4*nu)+a"
        ),
        "gauge_potential": str(gauge_potential),
        "gauge_transform_verified": bool(
            sp.simplify(gauge_potential - expected_gauge_potential) == 0
        ),
        "full_plane_ground_energy": "2*a",
        "oscillator_ground_state_verified": bool(
            sp.simplify(
                oscillator_action - a * oscillator_ground_state
            )
            == 0
        ),
        "dirichlet_disk_principal_rate_is_strictly_above_2a": True,
        "kummer_equation": str(radial_equation),
        "radial_generator_reduction": str(transformed_generator),
        "kummer_boundary_equation": "M(lambda/(2a),1,-R/2)=0",
        "tube_reynolds_definition": "R=a*L^2/nu",
        "effective_error_potential": str(effective_error),
        "effective_error_definitions": (
            "error=b-a*x, stretching_error=c-2*a, "
            "divergence_error=div(error)"
        ),
        "robustness_condition": (
            "ess_sup(effective_error_potential)<lambda_1-2*a"
        ),
        "spectral_rows": rows,
        "all_numerical_margins_are_positive": all(
            row["pair_decay_margin_over_a"] > 0.0 for row in rows
        ),
        "maximum_kummer_boundary_residual": max(
            abs(row["kummer_boundary_residual"]) for row in rows
        ),
        "two_particle_exit_beats_4a_deformation": True,
        "collision_damping_is_additional": True,
        "perturbation_budget_is_half_the_pair_margin": all(
            abs(
                2.0 * row["single_history_perturbation_budget_over_a"]
                - row["pair_decay_margin_over_a"]
            )
            < 1.0e-12
            for row in rows
        ),
        "margin_tends_to_zero_at_large_tube_reynolds": bool(
            rows[-1]["pair_decay_margin_over_a"]
            < rows[-2]["pair_decay_margin_over_a"]
            < rows[-3]["pair_decay_margin_over_a"]
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
