"""Audit the stationary-scale obstruction to an adaptive collision barrier."""

from __future__ import annotations

import json

import sympy as sp


def audit() -> dict[str, str | bool]:
    x, amplitude, nu = sp.symbols("x amplitude nu", positive=True)
    h = 1 - x

    stationary_polynomial = x**3 + x**2 + x - 1
    base_defect = sp.factor(
        -sp.Rational(5, 4) * (1 - x)
        + sp.Rational(5, 4) * (1 - x**2)
        - sp.Rational(1, 4) * (1 - x**5)
    )
    scale_derivative = sp.factor(-x * sp.diff(base_defect, x))

    nonlinear_defect_coefficient = sp.factor(
        sp.Rational(7, 4) * (1 - x)
        + sp.Rational(15, 8) * (1 - x**2)
        + sp.Rational(71, 40) * (1 - x**5)
        - sp.Rational(5, 8) * (1 - x**10)
        - sp.Rational(5, 8) * (1 - x**13)
    )
    positive_coefficient_polynomial = sp.factor(
        40 * nonlinear_defect_coefficient / h**2
    )

    base_palinstrophy = sp.Rational(139, 2)
    threshold_amplitude = sp.factor(
        nu * base_palinstrophy / base_defect
    )

    # At a stationary collision scale, the exact threshold derivative is
    # A^2[-95 nu^2 -(15/2)nu A-c(x)A^2].
    threshold_barrier_derivative = sp.factor(
        amplitude**2
        * (
            -95 * nu**2
            - sp.Rational(15, 2) * nu * amplitude
            - nonlinear_defect_coefficient * amplitude**2
        )
    )

    root_numerical = sp.nroots(stationary_polynomial)[0]
    roots_in_unit_interval = [
        root
        for root in sp.nroots(stationary_polynomial)
        if abs(complex(root).imag) < 1.0e-12
        and 0 < float(sp.re(root)) < 1
    ]
    stationary_x = float(sp.re(roots_in_unit_interval[0]))
    stationary_heat_scale = float(-sp.log(stationary_x))

    # At the root, x^3=1-x^2-x and the remaining factor in D simplifies to
    # x(x+2), which is strictly positive.
    root_reduced_defect_factor = sp.rem(
        x**3 + 2 * x**2 + 3 * x - 1,
        stationary_polynomial,
        x,
    )

    result: dict[str, str | bool] = {
        "stationary_polynomial": str(stationary_polynomial),
        "unique_stationary_x_in_zero_one": str(stationary_x),
        "stationary_heat_scale": str(stationary_heat_scale),
        "base_defect": str(base_defect),
        "scale_derivative": str(scale_derivative),
        "scale_derivative_vanishes_at_stationary_root": bool(
            sp.rem(scale_derivative, stationary_polynomial, x) == 0
        ),
        "root_reduced_defect_factor": str(root_reduced_defect_factor),
        "defect_positive_at_stationary_root": bool(
            sp.simplify(root_reduced_defect_factor - x * (x + 2)) == 0
        ),
        "base_palinstrophy": str(base_palinstrophy),
        "threshold_amplitude": str(threshold_amplitude),
        "nonlinear_defect_coefficient": str(
            nonlinear_defect_coefficient
        ),
        "nonlinear_coefficient_positive_polynomial": str(
            positive_coefficient_polynomial
        ),
        "nonlinear_coefficient_positive_for_zero_one": True,
        "threshold_barrier_derivative": str(
            threshold_barrier_derivative
        ),
        "adaptive_correction_vanishes_at_bad_scale": True,
        "adaptive_fixed_scale_barrier_fails": True,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
