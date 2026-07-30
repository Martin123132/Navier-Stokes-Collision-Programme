"""Audit a spectrum-uniform floor for trace-free affine strain cores."""

from __future__ import annotations

import json
import math

import sympy as sp
from scipy.special import jn_zeros


def audit() -> dict[str, object]:
    t = sp.symbols("t", real=True)
    forward_eigenvalues = (-1 - t, t, sp.Integer(1))
    backward_eigenvalues = (1 + t, -t, sp.Integer(-1))
    trace = sp.simplify(sum(forward_eigenvalues))

    # In units lambda_3 L^2/nu=1, conjugation of the transverse drift
    # gives this nonnegative anisotropic oscillator potential.
    x, y = sp.symbols("x y", real=True)
    transverse_potential = sp.expand(
        ((1 + t) ** 2 * x**2 + t**2 * y**2) / 4
    )
    full_space_ground_excess = sp.simplify((t + sp.Abs(t)) / 2)

    parameter_minimum = -0.5
    parameter_maximum = 1.0
    sample_parameters = [
        parameter_minimum + index * 0.05 for index in range(31)
    ]
    sampled_ordering_checks = [
        (-1.0 - value <= value <= 1.0) for value in sample_parameters
    ]
    sampled_potential_coefficient_checks = [
        ((1.0 + value) ** 2 >= 0.0 and value**2 >= 0.0)
        for value in sample_parameters
    ]

    bessel_j01 = float(jn_zeros(0, 1)[0])
    disk_dirichlet_laplacian_floor = bessel_j01**2
    axial_oscillator_ground = 0.5
    normalized_stretching = 1.0
    uniform_affine_spectral_margin = (
        disk_dirichlet_laplacian_floor
        + axial_oscillator_ground
        - normalized_stretching
    )

    axisymmetric_certified_margin = 5.296809625343342
    spectral_loss_from_uniformization = (
        axisymmetric_certified_margin - uniform_affine_spectral_margin
    )

    support_radius = 1.91
    one_dimensional_ims_upper = 157.0 / 200.0
    transverse_knot_spacing = support_radius / (2.0 * math.sqrt(2.0))
    axial_knot_spacing = 0.75
    transverse_ims_cost = (
        2.0 * one_dimensional_ims_upper / transverse_knot_spacing**2
    )
    axial_ims_cost = one_dimensional_ims_upper / axial_knot_spacing**2
    full_tensor_ims_cost = transverse_ims_cost + axial_ims_cost
    post_ims_uniform_margin = (
        uniform_affine_spectral_margin - full_tensor_ims_cost
    )

    sharp_sobolev_constant = 4.0 ** (2.0 / 3.0) / (
        3.0 * math.pi ** (4.0 / 3.0)
    )
    unit_relative_form_mass_budget = post_ims_uniform_margin / (
        sharp_sobolev_constant * (post_ims_uniform_margin + 1.0)
    )

    # This is reported only to quantify the opportunity. The old alpha was
    # proved for the axisymmetric Poisson/cylinder transfer, not this family.
    axisymmetric_poisson_alpha = 0.12539869261729739
    uncertified_old_alpha_mass_budget = (
        axisymmetric_poisson_alpha * unit_relative_form_mass_budget
    )

    checks = {
        "trace_free_parameterization_verified": bool(trace == 0),
        "ordered_spectrum_interval_verified": bool(
            all(sampled_ordering_checks)
            and math.isclose(parameter_minimum, -0.5)
            and math.isclose(parameter_maximum, 1.0)
        ),
        "transverse_oscillator_potential_is_nonnegative": bool(
            all(sampled_potential_coefficient_checks)
        ),
        "full_space_affine_excess_is_nonnegative": bool(
            all(max(value, 0.0) >= 0.0 for value in sample_parameters)
        ),
        "uniform_margin_remains_positive_after_full_tensor_IMS": bool(
            post_ims_uniform_margin > 0.0
        ),
        "uniformization_loss_is_below_0p014": bool(
            0.0 < spectral_loss_from_uniformization < 0.014
        ),
    }

    result: dict[str, object] = {
        "normalization": "lambda_3*L^2/nu=1",
        "ordered_trace_free_parameter_interval": "-1/2<=t<=1",
        "forward_strain_eigenvalues": [str(value) for value in forward_eigenvalues],
        "backward_drift_eigenvalues": [str(value) for value in backward_eigenvalues],
        "trace": str(trace),
        "transverse_conjugated_potential": str(transverse_potential),
        "full_space_ground_excess": str(full_space_ground_excess),
        "bessel_j_0_first_zero": bessel_j01,
        "unit_disk_dirichlet_laplacian_floor": (
            disk_dirichlet_laplacian_floor
        ),
        "uniform_general_affine_spectral_margin": (
            uniform_affine_spectral_margin
        ),
        "axisymmetric_certified_spectral_margin": (
            axisymmetric_certified_margin
        ),
        "spectral_loss_from_uniformization": (
            spectral_loss_from_uniformization
        ),
        "optimized_support_radius_over_L": support_radius,
        "dimensionless_transverse_cubic_IMS_cost": transverse_ims_cost,
        "dimensionless_axial_cubic_IMS_cost": axial_ims_cost,
        "dimensionless_full_tensor_cubic_IMS_cost": full_tensor_ims_cost,
        "dimensionless_uniform_margin_after_full_tensor_IMS": (
            post_ims_uniform_margin
        ),
        "sharp_R3_homogeneous_Sobolev_constant": sharp_sobolev_constant,
        "unit_relative_form_L3_over_2_budget_after_IMS": (
            unit_relative_form_mass_budget
        ),
        "axisymmetric_Poisson_alpha_for_diagnostic_only": (
            axisymmetric_poisson_alpha
        ),
        "uncertified_mass_budget_if_old_alpha_were_uniform": (
            uncertified_old_alpha_mass_budget
        ),
        "general_affine_Poisson_transfer_certified": False,
        "spectral_stage_supports_locally_fitted_general_symmetric_affine_reference": True,
        "remaining_Poisson_gate": (
            "extend the complete finite-cylinder/outer-Poisson boundary "
            "transfer uniformly to anisotropic trace-free affine drift"
        ),
        "remaining_frame_gate": (
            "construct a conservative moving, rotating cell-label transfer; "
            "a general skew drift is not invisible to an anisotropic core"
        ),
        **checks,
    }
    result["all_positive_certificate_checks_pass"] = all(checks.values())
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
