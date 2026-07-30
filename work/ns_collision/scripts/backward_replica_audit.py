"""Exact covariance audit for backward replicas in affine strain."""

from __future__ import annotations

import json

import sympy as sp


def affine_covariance_audit() -> dict[str, str | bool]:
    tau, nu, a = sp.symbols("tau nu a", positive=True)

    c_perp = 2 * nu / a * (sp.exp(2 * a * tau) - 1)
    c_parallel = nu / a * (1 - sp.exp(-4 * a * tau))

    residual_perp = sp.simplify(sp.diff(c_perp, tau) - (2 * a * c_perp + 4 * nu))
    residual_parallel = sp.simplify(
        sp.diff(c_parallel, tau) - (-4 * a * c_parallel + 4 * nu)
    )
    initial_radial_slope = sp.simplify(
        sp.limit(sp.diff(2 * c_perp + c_parallel, tau), tau, 0, dir="+")
    )
    parallel_limit = sp.simplify(sp.limit(c_parallel, tau, sp.oo))

    result: dict[str, str | bool] = {
        "transverse_covariance": str(c_perp),
        "parallel_covariance": str(c_parallel),
        "transverse_lyapunov_residual_zero": residual_perp == 0,
        "parallel_lyapunov_residual_zero": residual_parallel == 0,
        "initial_radial_variance_slope_is_12nu": sp.simplify(
            initial_radial_slope - 12 * nu
        )
        == 0,
        "parallel_variance_limit": str(parallel_limit),
        "parallel_limit_is_nu_over_a": sp.simplify(parallel_limit - nu / a)
        == 0,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(affine_covariance_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

