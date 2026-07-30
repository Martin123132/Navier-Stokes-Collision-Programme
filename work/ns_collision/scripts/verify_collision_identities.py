"""Symbolic audit for the first Navier-Stokes collision identities."""

from __future__ import annotations

import json

import sympy as sp


def verify_identities() -> dict[str, str | bool]:
    x = sp.symbols("x0:3", real=True)
    y = sp.symbols("y0:3", real=True)
    nu = sp.symbols("nu", positive=True)
    r = [x_i - y_i for x_i, y_i in zip(x, y)]
    h = sp.expand(sum(r_i**2 for r_i in r))
    g = sp.sqrt(h)

    joint_laplacian_g = sp.simplify(
        sum(sp.diff(g, q, 2) for q in (*x, *y))
    )
    joint_laplacian_h = sp.simplify(
        sum(sp.diff(h, q, 2) for q in (*x, *y))
    )
    radial_qv = sp.simplify(
        2 * nu * sum(sp.diff(g, q) ** 2 for q in (*x, *y))
    )
    squared_qv = sp.simplify(
        2 * nu * sum(sp.diff(h, q) ** 2 for q in (*x, *y))
    )

    t, T = sp.symbols("t T", real=True)
    a = sp.Function("a")(t)
    A = sp.diag(-a, -a, 2 * a)
    pressure_hessian = -(sp.diff(A, t) + A**2)
    affine_residual = sp.simplify(sp.diff(A, t) + A**2 + pressure_hessian)

    c = sp.symbols("c", positive=True)
    phi = x[0] ** 2 - c**2 + 2 * nu * t
    heat_residual = sp.simplify(sp.diff(phi, t) - nu * sp.diff(phi, x[0], 2))
    gap = 2 * sp.sqrt(c**2 - 2 * nu * t)
    heat_gap_residual = sp.simplify(sp.diff(gap, t) + 4 * nu / gap)

    blowup_a = 1 / (T - t)
    blowup_A = A.subs(a, blowup_a)
    blowup_pressure_hessian = sp.simplify(
        -(sp.diff(blowup_A, t) + blowup_A**2)
    )

    checks = {
        "joint_laplacian_distance_is_4_over_g": sp.simplify(
            joint_laplacian_g - 4 / g
        )
        == 0,
        "radial_ito_drift_is_4nu_over_g": sp.simplify(
            nu * joint_laplacian_g - 4 * nu / g
        )
        == 0,
        "radial_martingale_qv_is_4nu": sp.simplify(radial_qv - 4 * nu) == 0,
        "joint_laplacian_squared_distance_is_12": joint_laplacian_h == 12,
        "squared_distance_ito_drift_is_12nu": sp.simplify(
            nu * joint_laplacian_h - 12 * nu
        )
        == 0,
        "squared_distance_martingale_qv_is_16nu_h": sp.simplify(
            squared_qv - 16 * nu * h
        )
        == 0,
        "affine_navier_stokes_residual_zero": affine_residual == sp.zeros(3),
        "heat_polynomial_residual_zero": heat_residual == 0,
        "ordinary_heat_gap_is_attractive": heat_gap_residual == 0,
        "blowup_pressure_hessian": str(blowup_pressure_hessian),
    }
    checks["all_boolean_checks_pass"] = all(
        value for value in checks.values() if isinstance(value, bool)
    )
    return checks


def main() -> None:
    print(json.dumps(verify_identities(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

