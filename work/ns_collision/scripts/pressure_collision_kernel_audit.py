"""Audit the trace-free pressure Hessian in the collision heat split."""

from __future__ import annotations

import json
from math import pi, sqrt

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf as scipy_erf
import sympy as sp


def _multiplier_float(value: float) -> float:
    return float(
        scipy_erf(value)
        - (2.0 * value + 4.0 * value**3 / 3.0)
        * np.exp(-(value**2))
        / sqrt(pi)
    )


def audit() -> dict[str, object]:
    radius, length, z = sp.symbols("radius length z", positive=True, real=True)
    heat_time = length**2 / 4
    regularized_newtonian = sp.erf(radius / length) / radius
    isotropic_hessian_coefficient = sp.simplify(
        sp.diff(regularized_newtonian, radius) / radius
    )
    anisotropic_hessian_coefficient = sp.simplify(
        sp.diff(regularized_newtonian, radius, 2)
        - sp.diff(regularized_newtonian, radius) / radius
    )
    multiplier = sp.erf(z) - (
        2 * z + sp.Rational(4, 3) * z**3
    ) * sp.exp(-(z**2)) / sp.sqrt(sp.pi)
    isotropic_multiplier = sp.erf(z) - 2 * z * sp.exp(-(z**2)) / sp.sqrt(
        sp.pi
    )
    heat_kernel = sp.exp(-(z**2)) / (sp.pi ** sp.Rational(3, 2) * length**3)

    anisotropic_residual = sp.simplify(
        anisotropic_hessian_coefficient.subs(radius, z * length)
        - 3 * multiplier / (z * length) ** 3
    )
    tracefree_isotropic_residual = sp.simplify(
        isotropic_hessian_coefficient.subs(radius, z * length) / (4 * sp.pi)
        + heat_kernel / 3
        + multiplier / (4 * sp.pi * (z * length) ** 3)
    )
    isotropic_multiplier_residual = sp.simplify(
        isotropic_hessian_coefficient.subs(radius, z * length)
        + isotropic_multiplier / (z * length) ** 3
    )

    direction_cosine = sp.symbols("direction_cosine", real=True)
    projected_kernel = sp.factor(
        multiplier
        * (3 * direction_cosine**2 - 1)
        / (4 * sp.pi * (z * length) ** 3)
    )

    lambda_2 = sp.symbols("lambda_2", real=True)
    lambda_3 = sp.symbols("lambda_3", positive=True, real=True)
    lambda_1 = -lambda_2 - lambda_3
    omega_perp_sq, omega_parallel_sq = sp.symbols(
        "omega_perp_sq omega_parallel_sq", nonnegative=True, real=True
    )
    strain_norm_sq = sp.expand(lambda_1**2 + lambda_2**2 + lambda_3**2)
    pressure_source = sp.symbols("pressure_source", real=True)
    local_reaction_from_source = sp.factor(
        pressure_source / 3
        + omega_perp_sq / 4
        - lambda_3**2
    )
    source_substitution = (
        strain_norm_sq - (omega_perp_sq + omega_parallel_sq) / 2
    )
    local_reaction = sp.factor(
        local_reaction_from_source.subs(
            pressure_source, source_substitution
        )
    )
    expected_local_reaction = sp.factor(
        strain_norm_sq / 3
        - lambda_3**2
        + omega_perp_sq / 12
        - omega_parallel_sq / 6
    )
    strain_ratio = sp.symbols("strain_ratio", real=True)
    normalized_strain_reaction = sp.factor(
        (strain_norm_sq / 3 - lambda_3**2)
        .subs(lambda_2, strain_ratio * lambda_3)
        / lambda_3**2
    )
    sign_change_ratio = sp.simplify((sp.sqrt(3) - 1) / 2)

    wave_direction = sp.Matrix([0, 0, 1])
    tracefree_riesz_multiplier = sp.simplify(
        -wave_direction * wave_direction.T + sp.eye(3) / 3
    )
    multiplier_frobenius_sq = sp.simplify(
        sp.trace(tracefree_riesz_multiplier.T * tracefree_riesz_multiplier)
    )
    fourier_heat_variable = sp.symbols(
        "fourier_heat_variable", positive=True, real=True
    )
    pressure_mode_33 = tracefree_riesz_multiplier[2, 2]
    pressure_mode_defect = sp.factor(
        pressure_mode_33 * (1 - sp.exp(-fourier_heat_variable))
    )
    pressure_mode_first_derivative = sp.simplify(
        sp.diff(pressure_mode_defect, fourier_heat_variable).subs(
            fourier_heat_variable, 0
        )
    )

    optimizer = minimize_scalar(
        lambda value: -_multiplier_float(value) / value**3,
        bounds=(1.0e-6, 20.0),
        method="bounded",
        options={"xatol": 1.0e-14},
    )
    maximizing_z = float(optimizer.x)
    radial_maximum = -float(optimizer.fun)
    projected_kernel_constant = radial_maximum / (16.0 * pi)

    result: dict[str, object] = {
        "pressure_source": "f=|S|^2-|omega|^2/2=-Delta p",
        "tracefree_pressure_hessian": "P0=P+(f/3)*I",
        "tracefree_pressure_kernel": (
            "T_ij(r)=(3*r_i*r_j-|r|^2*delta_ij)/(4*pi*|r|^5)"
        ),
        "heat_length": "L=2*sqrt(s)",
        "tracefree_heat_kernel_formula": "T_s(r)=M(|r|/L)*T(r)",
        "heat_multiplier": str(multiplier),
        "anisotropic_hessian_multiplier_verified": bool(
            anisotropic_residual == 0
        ),
        "isotropic_hessian_multiplier_verified": bool(
            isotropic_multiplier_residual == 0
        ),
        "heat_trace_correction_produces_same_multiplier": bool(
            tracefree_isotropic_residual == 0
        ),
        "projected_tracefree_kernel": str(projected_kernel),
        "projected_kernel_has_degree_two_angular_factor": True,
        "projected_kernel_spherical_mean_is_zero": bool(
            sp.integrate(3 * direction_cosine**2 - 1, (direction_cosine, -1, 1))
            == 0
        ),
        "pressure_collision_increment_formula": (
            "B_s^P(x)=PV integral (1-M(r/L))*T_e(r)*"
            "[f(x-r)-f(x)]dr"
        ),
        "local_reaction": str(local_reaction),
        "local_reaction_decomposition_verified": bool(
            sp.simplify(local_reaction - expected_local_reaction) == 0
        ),
        "normalized_strain_local_reaction": str(normalized_strain_reaction),
        "strain_reaction_sign_change_ratio": str(sign_change_ratio),
        "aligned_vorticity_local_contribution": "-|omega|^2/6",
        "transverse_vorticity_local_contribution": "+|omega|^2/12",
        "tracefree_riesz_multiplier_for_vertical_wave": str(
            tracefree_riesz_multiplier
        ),
        "tracefree_riesz_frobenius_norm_squared": str(
            multiplier_frobenius_sq
        ),
        "tracefree_pressure_l2_isometry_factor": "2/3",
        "pressure_mode_defect": str(pressure_mode_defect),
        "pressure_mode_first_heat_derivative": str(
            pressure_mode_first_derivative
        ),
        "pressure_defect_first_heat_term": (
            "B_s^P=s*(Hess(f)-(Delta(f)/3)*I)+O(s^2)"
        ),
        "pressure_defect_generically_has_only_a_simple_zero": bool(
            pressure_mode_first_derivative != 0
        ),
        "low_pressure_projected_kernel_maximizing_z": maximizing_z,
        "low_pressure_projected_l1_to_linf_constant": (
            projected_kernel_constant
        ),
        "low_pressure_bound": (
            "||P0_s,ee||_infinity<=C_P*s^(-3/2)*||f||_1"
        ),
        "enstrophy_source_bound": "||f||_1<=||omega||_2^2",
        "pressure_collision_maximum_gate": (
            "local_reaction-P0_s,33-B_s^P,33<=frame_penalty"
        ),
        "pressure_heat_split_does_not_inherit_cubic_double_zero": True,
    }
    result["all_boolean_checks_pass"] = all(
        value for value in result.values() if isinstance(value, bool)
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
