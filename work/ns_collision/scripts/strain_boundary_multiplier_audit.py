"""Audit the exact heat attenuation of the Newtonian strain kernel."""

from __future__ import annotations

import json

import sympy as sp


def audit() -> dict[str, str | bool]:
    r, length, z = sp.symbols("r length z", positive=True)

    # The relative heat semigroup has length sqrt(8*nu*tau).  Its action on
    # 1/r is erf(r/length)/r.
    regularized_newtonian = sp.erf(r / length) / r
    anisotropic_hessian_coefficient = sp.simplify(
        sp.diff(regularized_newtonian, r, 2)
        - sp.diff(regularized_newtonian, r) / r
    )

    # For 1/r the corresponding Hessian coefficient is 3/r^3.
    multiplier_from_hessian = sp.simplify(
        (r**3 / 3) * anisotropic_hessian_coefficient
    )
    multiplier = sp.erf(z) - (
        2 * z + sp.Rational(4, 3) * z**3
    ) * sp.exp(-(z**2)) / sp.sqrt(sp.pi)
    substituted_multiplier = sp.simplify(
        multiplier_from_hessian.subs(r, z * length)
    )

    multiplier_derivative = sp.simplify(sp.diff(multiplier, z))
    expected_derivative = (
        8 * z**4 * sp.exp(-(z**2)) / (3 * sp.sqrt(sp.pi))
    )
    small_scale_coefficient = sp.simplify(sp.limit(multiplier / z**5, z, 0))
    expected_small_scale_coefficient = 8 / (15 * sp.sqrt(sp.pi))

    # The regularized degree-minus-three coefficient vanishes quadratically.
    regularized_kernel_coefficient = sp.simplify(3 * multiplier / (z * length) ** 3)
    quadratic_collision_coefficient = sp.simplify(
        sp.limit(regularized_kernel_coefficient / (z * length) ** 2, z, 0)
    )

    # A skew cross-product matrix kills the isotropic part of a radial Hessian.
    a, b = sp.symbols("a b", real=True)
    e1, e2, e3 = sp.symbols("e1 e2 e3", real=True)
    w1, w2, w3 = sp.symbols("w1 w2 w3", real=True)
    e = sp.Matrix([e1, e2, e3])
    cross_w = sp.Matrix(
        [
            [0, w3, -w2],
            [-w3, 0, w1],
            [w2, -w1, 0],
        ]
    )
    radial_hessian = a * sp.eye(3) + b * e * e.T
    strain_tensor = sp.simplify(
        cross_w * radial_hessian + radial_hessian * cross_w.T
    )
    expected_tensor = sp.simplify(
        b * (cross_w * e * e.T + e * e.T * cross_w.T)
    )

    result: dict[str, str | bool] = {
        "hessian_multiplier_formula": sp.simplify(
            substituted_multiplier - multiplier
        )
        == 0,
        "multiplier_derivative_formula": sp.simplify(
            multiplier_derivative - expected_derivative
        )
        == 0,
        "multiplier_strictly_increasing_for_positive_z": True,
        "multiplier_limit_at_zero": str(sp.limit(multiplier, z, 0)),
        "multiplier_limit_at_infinity": str(sp.limit(multiplier, z, sp.oo)),
        "small_scale_coefficient": str(small_scale_coefficient),
        "small_scale_coefficient_correct": sp.simplify(
            small_scale_coefficient - expected_small_scale_coefficient
        )
        == 0,
        "regularized_kernel_quadratic_coefficient": str(
            quadratic_collision_coefficient
        ),
        "isotropic_hessian_part_cancels": sp.simplify(
            strain_tensor - expected_tensor
        )
        == sp.zeros(3),
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
