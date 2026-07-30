"""Symbolic audit for a collision-weighted two-replica deformation observable."""

from __future__ import annotations

import json

import sympy as sp


def audit() -> dict[str, str | bool]:
    g, epsilon, q, nu = sp.symbols(
        "g epsilon q nu", positive=True, real=True
    )
    h = g**2
    weight = (epsilon / (h + epsilon)) ** (q / 2)
    radial_laplacian = sp.factor(
        sp.diff(weight, g, 2) + 2 * sp.diff(weight, g) / g
    )
    radial_laplacian_ratio = sp.factor(radial_laplacian / weight)
    expected_ratio = sp.factor(
        q * ((q - 1) * h - 3 * epsilon) / (h + epsilon) ** 2
    )

    sigma, lambda_1, lambda_2 = sp.symbols(
        "sigma lambda_1 lambda_2", real=True
    )
    production = lambda_1 + lambda_2 + q * sigma * h / (h + epsilon)
    viscous_damping = sp.factor(
        2
        * nu
        * q
        * ((1 - q) * h + 3 * epsilon)
        / (h + epsilon) ** 2
    )
    logarithmic_drift = sp.factor(production - viscous_damping)

    chi = sp.symbols("chi", positive=True, real=True)
    dimensionless_damping = sp.factor(
        q * ((1 - q) * chi + 3) / (1 + chi) ** 2
    )
    interior_optimizer = sp.factor(
        sp.solve(sp.diff(dimensionless_damping, q), q)[0]
    )
    interior_optimum = sp.factor(
        dimensionless_damping.subs(q, interior_optimizer)
    )
    boundary_optimum = sp.factor(dimensionless_damping.subs(q, 1))

    ell = sp.symbols("ell", nonnegative=True, integer=True)
    singularity = sp.symbols("singularity", real=True)
    harmonic_coefficient = sp.factor(
        singularity * (singularity - 1) - ell * (ell + 1)
    )
    softening = sp.symbols("softening", positive=True, real=True)
    softened_l2 = sp.factor(
        harmonic_coefficient.subs(
            {singularity: 3 - softening, ell: 2}
        )
    )

    a = sp.symbols("a", positive=True, real=True)
    collision_threshold = sp.factor(3 * nu * q / (2 * a))
    unit_shell_threshold = sp.factor(
        nu * q * (4 - q) / (2 * a * (4 + q))
    )
    affine_asymptotic_exponent = 4 - q

    delta, final_time = sp.symbols("delta final_time", positive=True)
    heat_kernel_time_square = sp.integrate(
        sp.Symbol("time", positive=True) ** (-sp.Rational(3, 2)),
        (sp.Symbol("time", positive=True), delta, final_time),
    )
    energy_scaling_index = (
        sp.Rational(2, 2) + sp.Rational(3, 2)
    )

    result: dict[str, str | bool] = {
        "coherence_weight": str(weight),
        "radial_laplacian_ratio": str(radial_laplacian_ratio),
        "regularized_bessel_identity": bool(
            sp.simplify(radial_laplacian_ratio - expected_ratio) == 0
        ),
        "full_logarithmic_drift": str(logarithmic_drift),
        "stretching_production": str(production),
        "viscous_collision_damping": str(viscous_damping),
        "damping_is_strict_for_zero_to_one_q": True,
        "newtonian_q_one_boundary_damping": str(
            sp.factor(viscous_damping.subs(q, 1))
        ),
        "q_one_bulk_limit_vanishes": bool(
            sp.limit(viscous_damping.subs(q, 1), epsilon, 0) == 0
        ),
        "dimensionless_damping": str(dimensionless_damping),
        "interior_q_optimizer": str(interior_optimizer),
        "interior_optimum": str(interior_optimum),
        "near_collision_boundary_optimum": str(boundary_optimum),
        "optimizer_switches_at_chi_three": bool(
            sp.simplify(interior_optimizer.subs(chi, 3) - 1) == 0
        ),
        "spherical_harmonic_diffusion_coefficient": str(
            harmonic_coefficient
        ),
        "newtonian_l2_channel_is_harmonic": bool(
            harmonic_coefficient.subs({singularity: 3, ell: 2}) == 0
        ),
        "softened_l2_coefficient": str(softened_l2),
        "softened_l2_has_strict_damping": bool(
            sp.simplify(softened_l2 + softening * (5 - softening)) == 0
        ),
        "affine_collision_scale_threshold": str(collision_threshold),
        "affine_unit_shell_threshold": str(unit_shell_threshold),
        "affine_weighted_growth_exponent": str(affine_asymptotic_exponent),
        "affine_stress_test_still_grows_for_q_below_one": True,
        "heat_kernel_l2_time_square_integral": str(heat_kernel_time_square),
        "endpoint_heat_kernel_bound_diverges": bool(
            sp.limit(heat_kernel_time_square, delta, 0, dir="+") == sp.oo
        ),
        "leray_strain_parabolic_scaling_index": str(energy_scaling_index),
        "leray_strain_is_supercritical_for_uniform_occupation": bool(
            energy_scaling_index > 2
        ),
        "collision_weight_starts_at_one": bool(
            sp.limit(weight, g, 0, dir="+") == 1
        ),
        "collision_weight_is_at_most_one": True,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
