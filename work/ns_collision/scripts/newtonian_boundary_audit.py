"""Audit the inaccessible-collision defect for the 3D Newtonian kernel."""

from __future__ import annotations

import json

import sympy as sp


def newtonian_audit() -> dict[str, str | bool]:
    r, t, nu, q = sp.symbols("r t nu q", positive=True)
    radial_laplacian = lambda f: sp.diff(f, r, 2) + 2 * sp.diff(f, r) / r

    reciprocal = 1 / r
    reciprocal_laplacian = sp.simplify(radial_laplacian(reciprocal))

    inverse_power = r ** (-q)
    inverse_power_generator = sp.simplify(2 * nu * radial_laplacian(inverse_power))

    heat_regularized = sp.erf(r / sp.sqrt(8 * nu * t)) / r
    heat_residual = sp.simplify(
        sp.diff(heat_regularized, t) - 2 * nu * radial_laplacian(heat_regularized)
    )
    coincident_limit = sp.simplify(sp.limit(heat_regularized, r, 0, dir="+"))
    expected_coincident_limit = 1 / sp.sqrt(2 * sp.pi * nu * t)

    result: dict[str, str | bool] = {
        "reciprocal_laplacian_off_origin_zero": reciprocal_laplacian == 0,
        "inverse_power_generator": str(inverse_power_generator),
        "inverse_power_generator_expected": sp.simplify(
            inverse_power_generator
            - 2 * nu * q * (q - 1) * r ** (-q - 2)
        )
        == 0,
        "heat_regularized_newtonian_kernel": str(heat_regularized),
        "heat_equation_residual_zero": heat_residual == 0,
        "coincident_limit": str(coincident_limit),
        "coincident_limit_expected": sp.simplify(
            coincident_limit - expected_coincident_limit
        )
        == 0,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(newtonian_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

