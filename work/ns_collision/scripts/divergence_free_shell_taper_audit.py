"""Audit and optimize a divergence-free streamfunction shell taper."""

from __future__ import annotations

import json
import math

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy.optimize import linprog
import sympy as sp


def _taper_basis(maximum_degree: int) -> tuple[Polynomial, list[Polynomial]]:
    if maximum_degree < 3:
        raise ValueError("maximum_degree must be at least three")
    base = Polynomial([1.0, 0.0, -3.0, 2.0])
    endpoint_envelope = Polynomial([0.0, 0.0, 1.0]) * Polynomial(
        [1.0, -2.0, 1.0]
    )
    basis = [
        endpoint_envelope
        * Chebyshev.basis(index, domain=[0.0, 1.0]).convert(
            kind=Polynomial
        )
        for index in range(maximum_degree - 3)
    ]
    return base, basis


def _strain_operators(
    polynomial: Polynomial,
    normalized_radius: np.ndarray,
    taper_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = taper_radius - 1.0
    radius = 1.0 + width * normalized_radius
    value = polynomial(normalized_radius)
    first = polynomial.deriv(1)(normalized_radius) / width
    second = polynomial.deriv(2)(normalized_radius) / width**2
    axis_operator = value + radius * first
    diagonal_operator = (
        value
        + 0.75 * radius * first
        + 0.25 * radius**2 * second
    )
    return axis_operator, diagonal_operator


def optimize_taper(
    taper_radius: float,
    maximum_degree: int = 16,
    constraint_points: int = 4_001,
    validation_points: int = 200_001,
) -> tuple[Polynomial, dict[str, object]]:
    if taper_radius <= 1.0:
        raise ValueError("taper_radius must exceed one")
    base, basis = _taper_basis(maximum_degree)
    constraint_grid = np.linspace(0.0, 1.0, constraint_points)
    base_axis, base_diagonal = _strain_operators(
        base, constraint_grid, taper_radius
    )
    if basis:
        basis_axis = np.column_stack(
            [
                _strain_operators(item, constraint_grid, taper_radius)[0]
                for item in basis
            ]
        )
        basis_diagonal = np.column_stack(
            [
                _strain_operators(item, constraint_grid, taper_radius)[1]
                for item in basis
            ]
        )
    else:
        basis_axis = np.empty((constraint_points, 0))
        basis_diagonal = np.empty((constraint_points, 0))
    ones = np.ones(constraint_points)
    constraint_matrix = np.vstack(
        [
            np.column_stack([basis_axis, -ones]),
            np.column_stack([-basis_axis, -ones]),
            np.column_stack([basis_diagonal, -ones]),
            np.column_stack([-basis_diagonal, -ones]),
        ]
    )
    constraint_vector = np.concatenate(
        [-base_axis, base_axis, -base_diagonal, base_diagonal]
    )
    objective = np.zeros(len(basis) + 1)
    objective[-1] = 1.0
    optimization = linprog(
        objective,
        A_ub=constraint_matrix,
        b_ub=constraint_vector,
        bounds=[(None, None)] * len(basis) + [(0.0, None)],
        method="highs",
    )
    if not optimization.success:
        raise RuntimeError(f"taper optimization failed: {optimization.message}")
    polynomial = base + sum(
        (
            coefficient * item
            for coefficient, item in zip(optimization.x[:-1], basis)
        ),
        Polynomial([0.0]),
    )
    validation_grid = np.linspace(0.0, 1.0, validation_points)
    axis_values, diagonal_values = _strain_operators(
        polynomial, validation_grid, taper_radius
    )
    envelope = np.maximum(np.abs(axis_values), np.abs(diagonal_values))
    maximum_index = int(np.argmax(envelope))
    values = polynomial(validation_grid)
    physical_derivatives = polynomial.deriv(1)(validation_grid) / (
        taper_radius - 1.0
    )
    diagnostics: dict[str, object] = {
        "taper_radius": taper_radius,
        "maximum_polynomial_degree": maximum_degree,
        "constraint_points": constraint_points,
        "validation_points": validation_points,
        "sampled_L_infinity_objective": float(optimization.fun),
        "dense_validated_strain_amplification": float(
            envelope[maximum_index]
        ),
        "maximizing_normalized_radius": float(
            validation_grid[maximum_index]
        ),
        "minimum_taper_value": float(np.min(values)),
        "maximum_taper_value": float(np.max(values)),
        "minimum_physical_taper_derivative": float(
            np.min(physical_derivatives)
        ),
        "maximum_physical_taper_derivative": float(
            np.max(physical_derivatives)
        ),
        "chebyshev_envelope_coefficients": optimization.x[:-1].tolist(),
        "power_basis_coefficients": polynomial.coef.tolist(),
        "endpoint_value_residual": float(
            max(abs(polynomial(0.0) - 1.0), abs(polynomial(1.0)))
        ),
        "endpoint_slope_residual": float(
            max(
                abs(polynomial.deriv(1)(0.0)),
                abs(polynomial.deriv(1)(1.0)),
            )
        ),
    }
    return polynomial, diagnostics


def _symbolic_audit() -> dict[str, object]:
    x, y, z, strength = sp.symbols("x y z strength", real=True)
    radius = sp.sqrt(x**2 + y**2)
    taper = sp.Function("f")
    streamfunction = strength * taper(radius) * x * y
    drift = sp.Matrix(
        [
            x / 2 + sp.diff(streamfunction, y),
            y / 2 - sp.diff(streamfunction, x),
            -z,
        ]
    )
    divergence = sp.simplify(
        sp.diff(drift[0], x)
        + sp.diff(drift[1], y)
        + sp.diff(drift[2], z)
    )

    theta, radial_symbol = sp.symbols(
        "theta radial_symbol", real=True, positive=True
    )
    coefficients = sp.symbols("c0:5", real=True)
    polynomial_taper = sum(
        coefficient * radius**power
        for power, coefficient in enumerate(coefficients)
    )
    unit_streamfunction = polynomial_taper * x * y
    mixed = sp.diff(unit_streamfunction, x, y)
    diagonal_difference = (
        sp.diff(unit_streamfunction, y, 2)
        - sp.diff(unit_streamfunction, x, 2)
    ) / 2
    polar_substitution = {
        x: radial_symbol * sp.cos(theta),
        y: radial_symbol * sp.sin(theta),
    }
    mixed_polar = sp.trigsimp(mixed.subs(polar_substitution).doit())
    difference_polar = sp.trigsimp(
        diagonal_difference.subs(polar_substitution).doit()
    )
    radial_polynomial = sum(
        coefficient * radial_symbol**power
        for power, coefficient in enumerate(coefficients)
    )
    first = sp.diff(radial_polynomial, radial_symbol)
    second = sp.diff(radial_polynomial, radial_symbol, 2)
    centre = sp.simplify(
        radial_polynomial
        + sp.Rational(7, 8) * radial_symbol * first
        + sp.Rational(1, 8) * radial_symbol**2 * second
    )
    oscillation = sp.simplify(
        (radial_symbol**2 * second - radial_symbol * first) / 8
    )
    expected_mixed = centre - oscillation * sp.cos(4 * theta)
    expected_difference = -oscillation * sp.sin(4 * theta)
    return {
        "streamfunction": str(streamfunction),
        "backward_drift": [str(component) for component in drift],
        "divergence": str(divergence),
        "divergence_free_for_every_radial_taper": bool(divergence == 0),
        "mixed_Hessian_polar_identity_verified": bool(
            sp.simplify(mixed_polar - expected_mixed) == 0
        ),
        "diagonal_Hessian_polar_identity_verified": bool(
            sp.simplify(difference_polar - expected_difference) == 0
        ),
        "worst_angular_strain_amplification": (
            "max(|f+r*f'|,|f+3*r*f'/4+r^2*f''/4|)"
        ),
        "worst_spectrum_stretching_excess": (
            "(3/2)*max(0,sigma_star-1) at t=1"
        ),
    }


def audit() -> dict[str, object]:
    symbolic = _symbolic_audit()
    degree_rows = []
    for degree in (3, 4, 6, 8, 10, 12, 14, 16):
        _, row = optimize_taper(
            1.91,
            maximum_degree=degree,
            constraint_points=1_501,
            validation_points=50_001,
        )
        degree_rows.append(row)

    radius_rows = []
    for taper_radius in (1.91, 2.0, 2.25, 2.5, 2.6, 2.65, 2.7, 3.0):
        constraint_points = 10_001 if taper_radius == 2.65 else 2_001
        validation_points = 200_001 if taper_radius == 2.65 else 50_001
        _, row = optimize_taper(
            taper_radius,
            maximum_degree=16,
            constraint_points=constraint_points,
            validation_points=validation_points,
        )
        row["worst_t1_stretching_excess"] = 1.5 * max(
            0.0, row["dense_validated_strain_amplification"] - 1.0
        )
        radius_rows.append(row)
    selected_row = next(
        row for row in radius_rows if row["taper_radius"] == 2.65
    )
    result: dict[str, object] = {
        **symbolic,
        "optimization_status": (
            "sampled polynomial minimax pilot; not an interval proof"
        ),
        "rigorous_polynomial_taper_enclosure_certified": False,
        "degree_convergence_at_taper_radius_1p91": degree_rows,
        "taper_radius_rows": radius_rows,
        "selected_taper_radius": 2.65,
        "selected_outer_buffer_radius": 2.75,
        "selected_taper_row": selected_row,
        "narrow_eta_2_shell_has_large_strain_overshoot": bool(
            next(
                row for row in radius_rows if row["taper_radius"] == 2.0
            )["worst_t1_stretching_excess"]
            > 1.3
        ),
        "selected_taper_is_near_theoretical_strain_floor": bool(
            selected_row["dense_validated_strain_amplification"]
            < 1.00001
        ),
        "selected_taper_is_numerically_monotone": bool(
            selected_row["maximum_physical_taper_derivative"] < 1.0e-8
            and selected_row["minimum_taper_value"] > -1.0e-7
        ),
        "selected_taper_endpoint_conditions_hold": bool(
            selected_row["endpoint_value_residual"] < 1.0e-7
            and selected_row["endpoint_slope_residual"] < 1.0e-7
        ),
        "theoretical_amplification_lower_bound": 1.0,
        "lower_bound_reason": (
            "core matching f(1)=1 and f'(1)=0 forces |f+r*f'|=1"
        ),
        "remaining_gate": (
            "solve the nonsymmetric Poisson visit for the optimized "
            "divergence-free taper and replace dense sampling by a "
            "verified polynomial enclosure"
        ),
    }
    positive_checks = (
        result["divergence_free_for_every_radial_taper"],
        result["mixed_Hessian_polar_identity_verified"],
        result["diagonal_Hessian_polar_identity_verified"],
        result["narrow_eta_2_shell_has_large_strain_overshoot"],
        result["selected_taper_is_near_theoretical_strain_floor"],
        result["selected_taper_is_numerically_monotone"],
        result["selected_taper_endpoint_conditions_hold"],
    )
    result["all_positive_taper_checks_pass"] = all(positive_checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
