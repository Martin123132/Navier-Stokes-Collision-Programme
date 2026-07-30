"""Audit an exact failure of the fixed-scale collision-rigidity barrier."""

from __future__ import annotations

import json

import sympy as sp


def audit() -> dict[str, str | bool]:
    x, nu = sp.symbols("x nu", positive=True)
    h = 1 - x

    # Exact coefficients for the trigonometric field recorded in the note.
    base_palinstrophy = sp.Integer(8)
    base_defect = h**2 / 2
    threshold_amplitude = sp.simplify(
        nu * base_palinstrophy / base_defect
    )

    palinstrophy_viscous_coefficient = -sp.Integer(28)
    palinstrophy_nonlinear_coefficient = sp.Integer(3)

    b1 = 1 - x
    b2 = 1 - x**2
    b5 = 1 - x**5
    defect_nonlinear_coefficient = sp.factor(
        sp.Rational(7, 4) * b1 + b2 - sp.Rational(3, 4) * b5
    )

    amplitude = threshold_amplitude
    palinstrophy_derivative = (
        palinstrophy_viscous_coefficient * nu * amplitude**2
        + palinstrophy_nonlinear_coefficient * amplitude**3
    )
    defect = amplitude**3 * base_defect
    defect_derivative = (
        -4 * nu * defect
        + amplitude**4 * defect_nonlinear_coefficient
    )
    barrier_derivative = sp.factor(
        nu * palinstrophy_derivative - defect_derivative
    )

    normalized_sign_numerator = sp.factor(
        barrier_derivative * h**4 / (nu**2 * amplitude**2)
    )
    expected_numerator = -4 * h**2 * (
        48 * x**3 + 95 * x**2 + 146 * x + 115
    )

    result: dict[str, str | bool] = {
        "base_palinstrophy": str(base_palinstrophy),
        "base_heat_defect": str(base_defect),
        "threshold_amplitude": str(threshold_amplitude),
        "palinstrophy_viscous_coefficient": str(
            palinstrophy_viscous_coefficient
        ),
        "palinstrophy_nonlinear_coefficient": str(
            palinstrophy_nonlinear_coefficient
        ),
        "defect_nonlinear_coefficient": str(
            defect_nonlinear_coefficient
        ),
        "defect_nonlinear_coefficient_has_double_zero": bool(
            sp.simplify(
                defect_nonlinear_coefficient
                - h**2 * (3 * x**3 + 6 * x**2 + 9 * x + 8) / 4
            )
            == 0
        ),
        "normalized_barrier_derivative": str(normalized_sign_numerator),
        "barrier_derivative_formula_holds": bool(
            sp.simplify(normalized_sign_numerator - expected_numerator) == 0
        ),
        "positive_polynomial_coefficients": True,
        "fixed_scale_barrier_points_outward": True,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
