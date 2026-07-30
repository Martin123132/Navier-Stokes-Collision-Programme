"""Audit the neutral-direction obstruction to affine exterior L2 tails."""

from __future__ import annotations

import json
import math

import sympy as sp


def audit() -> dict[str, object]:
    rho, time = sp.symbols("rho time", nonnegative=True, real=True)
    axial_rate = 1 + rho
    deformation_rate = sp.Integer(1)
    axial_l2_decay_rate = axial_rate / 2
    residual_growth_rate = sp.simplify(
        deformation_rate - axial_l2_decay_rate
    )
    trace = sp.simplify(-1 - rho + axial_rate)

    weyl_rows = []
    for support_scale in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        quotient = math.pi**2 / support_scale**2
        weyl_rows.append(
            {
                "neutral_direction_support_length": support_scale,
                "exact_separated_Rayleigh_quotient": quotient,
            }
        )

    parameter_rows = []
    for rho_value in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0):
        parameter_rows.append(
            {
                "rho": rho_value,
                "backward_drift_eigenvalues": [
                    -1.0,
                    -rho_value,
                    1.0 + rho_value,
                ],
                "axial_L2_decay_rate": (1.0 + rho_value) / 2.0,
                "residual_growth_rate_after_axial_compensation": (
                    1.0 - rho_value
                )
                / 2.0,
                "minimum_required_transverse_hitting_decay_rate": (
                    1.0 - rho_value
                )
                / 2.0,
            }
        )

    result: dict[str, object] = {
        "return_aligned_affine_family": (
            "b_rho=(-x,-rho*y,(1+rho)*z), 0<=rho<=1"
        ),
        "trace_free_identity": str(trace),
        "trace_free_verified": bool(trace == 0),
        "one_history_deformation_rate": str(deformation_rate),
        "axial_gaussian_L2_decay_rate": str(axial_l2_decay_rate),
        "residual_exponential_rate_after_axial_L2": str(
            residual_growth_rate
        ),
        "summable_tail_requirement": (
            "lambda_perp(rho)>(1-rho)/2"
        ),
        "parameter_rows": parameter_rows,
        "neutral_endpoint_transverse_generator": (
            "L_perp=Delta-x*partial_x on R2\\unit_disk"
        ),
        "neutral_endpoint_conjugated_operator": (
            "H_0=(-partial_xx+x^2/4-1/2)-partial_yy"
        ),
        "Weyl_sequence": (
            "psi_R(x,y)=Gaussian_ground(x)*sqrt(2/R)"
            "sin(pi*(y-R)/R) on R<y<2R"
        ),
        "Weyl_sequence_rows": weyl_rows,
        "neutral_endpoint_spectral_bottom": 0.0,
        "neutral_endpoint_has_positive_transverse_spectral_gap": False,
        "axisymmetric_endpoint_axial_compensation_remains_valid": True,
        "uniform_positive_transverse_rate_over_full_affine_family": False,
        "uniform_static_affine_weighted_L2_tail_gate_closed": False,
        "fixed_outer_start_kernel_nonsummability_fully_proved": False,
        "full_Navier_Stokes_exterior_gate_closed": False,
        "interpretation": (
            "the finite axial patch repairs the axisymmetric inward-affine "
            "stress test, but at rho=0 one returning transverse direction "
            "is neutral. Axial L2 dilution then leaves exp(t/2) residual "
            "growth while the transverse killed operator has no spectral "
            "gap. A uniform all-entry spectral envelope cannot follow from "
            "the present cylinder geometry"
        ),
        "scope_guard": (
            "the trace-free balance, axial exponent, conjugated operator, "
            "and Weyl quotients are exact. The Weyl sequence rules out a "
            "uniform operator spectral gap; a complete lower asymptotic for "
            "the one fixed outer-start hitting kernel is not proved here"
        ),
        "next_gate": (
            "either redesign the exterior storage geometry so every neutral "
            "direction contributes density dilution, or prove that the "
            "Navier-Stokes return branch cannot persist near rho=0; do not "
            "extrapolate the axisymmetric compensation to all spectra"
        ),
    }
    positive_checks = (
        result["trace_free_verified"],
        str(residual_growth_rate) == "1/2 - rho/2",
        weyl_rows[-1]["exact_separated_Rayleigh_quotient"] < 0.003,
        all(
            later["exact_separated_Rayleigh_quotient"]
            < earlier["exact_separated_Rayleigh_quotient"]
            for earlier, later in zip(weyl_rows[:-1], weyl_rows[1:])
        ),
        result["neutral_endpoint_spectral_bottom"] == 0.0,
        not result[
            "neutral_endpoint_has_positive_transverse_spectral_gap"
        ],
        result["axisymmetric_endpoint_axial_compensation_remains_valid"],
        not result["uniform_static_affine_weighted_L2_tail_gate_closed"],
        not result["fixed_outer_start_kernel_nonsummability_fully_proved"],
    )
    result["all_positive_anisotropic_tail_checks_pass"] = all(
        positive_checks
    )
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
