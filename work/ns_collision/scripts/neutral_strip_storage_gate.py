"""Audit a neutral-strip repair of the affine exterior spectral tail."""

from __future__ import annotations

import json
import math

import sympy as sp


def audit() -> dict[str, object]:
    rho, y_half_width = sp.symbols(
        "rho Y", nonnegative=True, real=True
    )
    entry_radius = sp.Integer(2)
    target_radius = sp.Integer(1)

    transverse_floor = sp.pi**2 / (4 * y_half_width**2) - rho / 2
    residual_growth = (1 - rho) / 2
    net_tail_margin = sp.simplify(transverse_floor - residual_growth)
    maximum_half_width = sp.pi / sp.sqrt(2)
    maximum_entry_buffer = sp.simplify(maximum_half_width - entry_radius)

    working_half_width = sp.Rational(21, 10)
    working_margin = sp.simplify(
        net_tail_margin.subs(y_half_width, working_half_width)
    )

    parameter_rows = []
    for rho_value in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0):
        floor_value = (
            math.pi**2 / (4.0 * float(working_half_width) ** 2)
            - rho_value / 2.0
        )
        residual_value = (1.0 - rho_value) / 2.0
        parameter_rows.append(
            {
                "rho": rho_value,
                "transverse_killed_rate_lower_bound": floor_value,
                "residual_growth_after_axial_L2": residual_value,
                "net_weighted_tail_margin_lower_bound": (
                    floor_value - residual_value
                ),
            }
        )

    result: dict[str, object] = {
        "affine_subfamily": "b_rho=(-x,-rho*y,(1+rho)*z), 0<=rho<=1",
        "stopped_transverse_domain": (
            "Omega_Y={(x,y):x^2+y^2>1 and |y|<Y}"
        ),
        "stopping_partition": (
            "first hit of r=1 is return; first hit of |y|=Y is outer exit"
        ),
        "conjugated_transverse_operator": (
            "H_rho=(-d_xx+x^2/4-1/2)"
            "+(-d_yy+rho^2*y^2/4-rho/2)"
        ),
        "sliced_Poincare_bound": (
            "int|partial_y psi|^2>=pi^2/(4Y^2) int|psi|^2"
        ),
        "transverse_killed_rate_lower_bound": str(transverse_floor),
        "residual_growth_after_axial_L2": str(residual_growth),
        "uniform_net_weighted_tail_margin": str(net_tail_margin),
        "admissible_half_width_interval": "2<Y<pi/sqrt(2)",
        "maximum_admissible_half_width": float(maximum_half_width),
        "maximum_buffer_beyond_full_r2_entry_circle": float(
            maximum_entry_buffer
        ),
        "working_half_width": float(working_half_width),
        "working_exact_net_margin": str(working_margin),
        "working_net_margin": float(working_margin),
        "parameter_rows": parameter_rows,
        "full_outer_entry_circle_strictly_inside_working_strip": bool(
            working_half_width > entry_radius
        ),
        "inner_target_fits_inside_working_strip": bool(
            working_half_width > target_radius
        ),
        "admissible_storage_window_nonempty": bool(
            maximum_half_width > entry_radius
        ),
        "neutral_endpoint_zero_gap_repaired_at_survival_semigroup_level": bool(
            working_margin > 0
        ),
        "uniform_rho_0_to_1_weighted_survival_tail_margin_positive": bool(
            working_margin > 0
        ),
        "outer_wall_exit_identified_with_physical_true_split": False,
        "boundary_flux_space_time_density_envelope_certified": False,
        "negative_parameter_half_of_full_affine_spectrum_covered": False,
        "full_Navier_Stokes_storage_gate_closed": False,
        "interpretation": (
            "capping the neutral transverse coordinate converts the rho=0 "
            "continuous-spectrum obstruction into a uniform positive "
            "weighted survival margin for 0<=rho<=1"
        ),
        "scope_guard": (
            "the conjugation, sliced Poincare lower bound, admissible-width "
            "window, and net exponent are exact. This does not prove a "
            "surface hitting-density envelope, handle time-dependent "
            "eigenframes or rho<0, or prove that a geometric wall exit is "
            "a genuine Navier-Stokes scale split"
        ),
        "next_gate": (
            "derive the two unnormalized boundary kernels on Omega_Y and "
            "prove that the |y|=Y law enters the physical cubic scale-split "
            "mechanism; then test their complete gains in a_S^2+a_R^2<1"
        ),
    }

    margins = [
        row["net_weighted_tail_margin_lower_bound"]
        for row in parameter_rows
    ]
    checks = (
        sp.simplify(net_tail_margin - (sp.pi**2 / (4 * y_half_width**2) - sp.Rational(1, 2))) == 0,
        result["admissible_storage_window_nonempty"],
        result["full_outer_entry_circle_strictly_inside_working_strip"],
        result["inner_target_fits_inside_working_strip"],
        result[
            "neutral_endpoint_zero_gap_repaired_at_survival_semigroup_level"
        ],
        max(margins) - min(margins) < 1e-14,
        min(margins) > 0.059,
        not result["outer_wall_exit_identified_with_physical_true_split"],
        not result["boundary_flux_space_time_density_envelope_certified"],
        not result["full_Navier_Stokes_storage_gate_closed"],
    )
    result["all_positive_neutral_strip_checks_pass"] = all(checks)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
