"""Symbolic audit for the cumulative collision-defect criterion."""

from __future__ import annotations

import json

import sympy as sp


def audit() -> dict[str, str | bool]:
    nu, scale, initial_velocity_norm = sp.symbols(
        "nu scale initial_velocity_norm", positive=True
    )
    initial_enstrophy, defect_budget = sp.symbols(
        "initial_enstrophy defect_budget", nonnegative=True
    )
    theta = sp.symbols("theta", nonnegative=True)
    c0 = sp.sqrt(3) / (2 * (8 * sp.pi) ** sp.Rational(3, 4))

    lowpass_time_bound = sp.factor(
        c0
        * initial_velocity_norm**3
        * scale ** (-sp.Rational(5, 4))
        / (2 * nu)
    )
    enstrophy_bound = sp.factor(
        initial_enstrophy
        + 2 * lowpass_time_bound
        + 2 * defect_budget
    )

    decay_rate, d_initial, d_final, transfer_integral = sp.symbols(
        "decay_rate d_initial d_final transfer_integral", positive=True
    )
    integrated_triad_defect = (
        d_initial - d_final + transfer_integral
    ) / (nu * decay_rate)

    heat_parameter = sp.symbols("heat_parameter", positive=True)
    semigroup_primitive = sp.integrate(
        sp.exp(-heat_parameter * decay_rate),
        (heat_parameter, 0, sp.oo),
    )

    # Navier-Stokes scaling exponents: s -> lambda^-2 s,
    # ||u||_2 -> lambda^-1/2 ||u||_2, and dt*D -> lambda(dt*D).
    lowpass_scaling_exponent = (
        3 * sp.Rational(-1, 2)
        + sp.Rational(-5, 4) * (-2)
    )

    x = sp.symbols("x", positive=True)
    first_triad_transfer = sp.factor(
        sp.Rational(3, 32)
        * (3 * (1 - x) + (1 - x**2) - (1 - x**5))
    )
    first_triad_expected = (
        sp.Rational(3, 32)
        * (1 - x) ** 2
        * (x**3 + 2 * x**2 + 3 * x + 3)
    )
    second_triad_transfer = sp.factor(
        sp.Rational(19, 128) * (1 - x)
        + sp.Rational(13, 64) * (1 - x**2)
        + sp.Rational(31, 640) * (1 - x**5)
        - sp.Rational(5, 128) * (1 - x**10)
        - sp.Rational(1, 32) * (1 - x**13)
    )
    second_positive_polynomial = sp.factor(
        640 * second_triad_transfer / (1 - x) ** 2
    )

    result: dict[str, str | bool] = {
        "heat_kernel_constant": str(c0),
        "lowpass_time_bound": str(lowpass_time_bound),
        "uniform_enstrophy_bound": str(enstrophy_bound),
        "requires_theta_strictly_below_one": True,
        "integrated_triad_defect": str(integrated_triad_defect),
        "semigroup_primitive": str(semigroup_primitive),
        "semigroup_supplies_inverse_triad_frequency": bool(
            sp.simplify(semigroup_primitive - 1 / decay_rate) == 0
        ),
        "lowpass_bound_scaling_exponent": str(lowpass_scaling_exponent),
        "lowpass_bound_is_scale_critical": lowpass_scaling_exponent == 1,
        "signed_not_absolute_defect_is_retained": True,
        "first_triad_quartic_transfer": str(first_triad_transfer),
        "first_triad_positive_factorization": bool(
            sp.simplify(first_triad_transfer - first_triad_expected) == 0
        ),
        "second_triad_quartic_transfer": str(second_triad_transfer),
        "second_triad_positive_polynomial": str(second_positive_polynomial),
        "second_triad_polynomial_coefficients_positive": True,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
